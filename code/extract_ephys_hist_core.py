"""
Dataclasses for passing around pipeline parameters and paths.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from aind_s3_cache.json_utils import get_json
from aind_s3_cache.uri_utils import as_pathlike
from aind_zarr_utils.neuroglancer import (
    get_image_sources,
)
from aind_zarr_utils.pipeline_transformed import (
    _asset_from_zarr_pathlike,
    alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike,
    pipeline_transforms_local_paths,
)
from ibl_preprocess_types import (
    Args,
    AssetInfo,
    InputPaths,
    OutputDirs,
    PipelineRegistrationInfo,
    ZarrPaths,
)
from validate_inputs import PipelineValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        dest="annotation_manifest",
        default="729293/Manifest_Day1_2_729293 1.csv",
        help="Probe Annotations",
    )

    parser.add_argument(
        "--neuroglancer",
        dest="neuroglancer",
        default="Probes_561_729293_Day1and2.json",
        help="Directory containing probe annotations",
    )

    parser.add_argument(
        "--skip-ephys",
        type=int,
        choices=[0, 1],
        default=0,
        help="Skip ephys extraction (extract_continuous and extract_spikes). "
        "Only process histology and probe tracks.",
    )

    parser.add_argument(
        "--validate-only",
        type=int,
        choices=[0, 1],
        default=0,
        help="Run validation checks only without processing. "
        "Exits after reporting validation results.",
    )

    parser.add_argument(
        "--run-async",
        type=int,
        choices=[0, 1],
        default=1,
        help="Run the processing asynchronously.",
    )

    args = parser.parse_args()
    return args


def parse_and_normalize_args() -> Args:
    """
    Parse CLI args
    """
    a = parse_args()
    return Args(
        neuroglancer=a.neuroglancer,
        annotation_manifest=a.annotation_manifest,
        skip_ephys=bool(a.skip_ephys),
        validate_only=bool(a.validate_only),
        run_async=bool(a.run_async),
    )


def resolve_paths(args: Args) -> InputPaths:
    """
    Resolve inputs to absolute /data and /results paths.

    Returns
    -------
    InputPaths
        Object with resolved neuroglancer file, manifest CSV, roots.
    """
    data_root = Path("/data")
    results_root = Path("/results")

    def under_data(p: str | Path, data_root: Path = Path("/data")) -> Path:
        pth = Path(p).expanduser()
        return pth if pth.is_absolute() else (data_root / pth)

    return InputPaths(
        neuroglancer_file=under_data(args.neuroglancer),
        manifest_csv=under_data(args.annotation_manifest),
        data_root=data_root,
        results_root=results_root,
    )


def find_asset_info(paths: InputPaths) -> AssetInfo:
    ng_data = get_json(str(paths.neuroglancer_file))
    sources = get_image_sources(ng_data)
    a_zarr_uri = next(iter(sources.values()), None)
    if a_zarr_uri is None:
        raise ValueError("No image sources found in neuroglancer data")
    _, _, a_zarr_pathlike = as_pathlike(a_zarr_uri)
    asset_pathlike = _asset_from_zarr_pathlike(a_zarr_pathlike)
    asset_path = paths.data_root / asset_pathlike
    if not asset_path.exists():
        raise FileNotFoundError(f"Asset path not found: {asset_path}")
    asset_path_str = str(asset_path)
    zarr_path = asset_path / "image_tile_fusing" / "OMEZarr"
    image_channel_zarrs = [
        p for p in zarr_path.iterdir() if p.is_dir() and p.suffix == ".zarr"
    ]
    alignment_zarr_uri, metadata, processing_data = (
        alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike(
            asset_uri=asset_path_str
        )
    )
    other_channels = list(
        {p.as_posix() for p in image_channel_zarrs} - {alignment_zarr_uri}
    )
    zarr_paths = ZarrPaths(
        registration=alignment_zarr_uri,
        additional=other_channels,
        metadata=metadata,
        processing=processing_data,
    )

    pt_tx_str, pt_tx_inverted, img_tx_str, img_tx_inverted = (
        pipeline_transforms_local_paths(
            alignment_zarr_uri,
            processing_data,
            anonymous=True,
        )
    )
    pipeline_reg_info = PipelineRegistrationInfo(
        pt_tx_str=pt_tx_str,
        pt_tx_inverted=pt_tx_inverted,
        img_tx_str=img_tx_str,
        img_tx_inverted=img_tx_inverted,
    )

    alignment_zarr_path = Path(alignment_zarr_uri)
    registration_channel_stem = alignment_zarr_path.stem
    registration_dir_path = (
        asset_path / "image_atlas_alignment" / f"{registration_channel_stem}"
    )
    registration_in_ccf_precomputed = (
        registration_dir_path / "moved_ls_to_ccf.nii.gz"
    )
    return AssetInfo(
        asset_path=asset_path,
        zarr_volumes=zarr_paths,
        pipeline_registration_chains=pipeline_reg_info,
        registration_dir_path=registration_dir_path,
        registration_in_ccf_precomputed=registration_in_ccf_precomputed,
    )


# ---- Stage 3: outputs --------------------------------------------------------


def prepare_result_dirs(mouse_id: str, results_root: Path) -> OutputDirs:
    """
    Create directory structure for histology and track outputs.
    """
    histology_ccf = results_root / mouse_id / "ccf_space_histology"
    histology_img = results_root / mouse_id / "image_space_histology"
    tracks_root = results_root / mouse_id / "track_data"
    spim = tracks_root / "spim"
    template = tracks_root / "template"
    ccf = tracks_root / "ccf"
    bregma = tracks_root / "bregma_xyz"
    for d in (histology_ccf, histology_img, spim, template, ccf, bregma):
        d.mkdir(parents=True, exist_ok=True)
    return OutputDirs(
        histology_ccf, histology_img, tracks_root, spim, template, ccf, bregma
    )


# ---- Stage 4–6: batch operations (no per-row state) -------------------------


def determine_desired_level(
    zarr_metadata, desired_voxel_size_um: float = 25.0
) -> int:
    # Get the minimum voxel spatial dimension for each multiscale level
    scales = np.array(
        [
            np.array(x[0]["scale"][2:]).min()
            for x in zarr_metadata["coordinateTransformations"]
        ]
    )
    # Find the highest-resolution level not exceeding desired_voxel_size_um
    level = np.maximum(
        np.searchsorted(scales, desired_voxel_size_um, side="right") - 1,
        0,
    )
    return level


def handle_validation(paths: InputPaths, args: Args) -> None:
    # Run validation checks
    logger.info("Running validation checks...")
    validator = PipelineValidator(args, paths, skip_resource_checks=False)
    validation_results = validator.validate_all()

    # If --validate-only flag is set, print summary and exit
    if args.validate_only:
        validator.print_summary(validation_results)
        sys.exit(0 if not validator.has_errors(validation_results) else 1)

    # If validation has errors, fail early
    if validator.has_errors(validation_results):
        validator.print_summary(validation_results)
        logger.error(
            "Validation failed. Please fix the errors above before running the pipeline."
        )
        sys.exit(1)

    # Print warnings if any (but continue processing)
    warnings = [r for r in validation_results if r.severity == "warning"]
    if warnings:
        logger.warning(
            f"Validation passed with {len(warnings)} warning(s). See details above."
        )
    return
