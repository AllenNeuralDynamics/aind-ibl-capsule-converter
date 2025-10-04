"""
SmartSPIM → Template → CCF probe/histology export pipeline.

This script ingests a Neuroglancer session, a manifest of probe annotations,
and a registration output (either inferred from the pipeline layout or provided
via --legacy-registration). It produces per-mouse histology volumes in both
CCF and image (SmartSPIM) spaces, per-probe tracks in SPIM/template/CCF spaces,
and IBL-style xyz-picks JSONs suitable for the ephys alignment GUI.

Inputs
------
CLI arguments:
  --neuroglancer
      Path to a Neuroglancer layer JSON (absolute path is used as-is; a
      relative path is anchored under /data).
  --annotation-manifest
      CSV manifest describing probes and associated recordings. Relative paths
      are anchored under /data. The CSV is snapshotted to /results/manifest.csv.
  --legacy-registration (optional)
      A registration results directory. If omitted, the script infers the
      SmartSPIM session and alignment channel from the Neuroglancer layer and
      uses the pipeline directory layout.

Manifest CSV schema (one row per probe):
  mouseid              Mouse identifier (string or int). Used to place outputs
                       under /results/<mouseid>/...
  sorted_recording     Name of the spike-sorting folder under /data; the
                       recording ID is derived by stripping a trailing "_sorted".
  probe_file           Basename (no extension) of the probe annotation file.
                       The script locates "*/{probe_file}.<ext>" under /data.
  probe_id             Identifier used to name outputs (e.g., "PRB1").
  probe_name           Subfolder name for GUI artifacts under the per-recording
                       results directory.
  annotation_format    Optional; currently only "json" is supported (default).
  probe_shank          Optional integer (0-based). If present, per-probe outputs
                       are also emitted with "_shank{probe_shank+1}" suffixes.
  surface_finding      Optional path fragment under /data, passed to
                       extract_continuous when present.

Reference data expected under /data:
  - smartspim_lca_template/smartspim_lca_template_25.nii.gz
  - allen_mouse_ccf/average_template/average_template_25.nii.gz
  - allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz
  - spim_template_to_ccf/syn_*.{nii.gz,mat} (template↔CCF transforms)


Outputs
-------


Processing overview
-------------------
1) Determine registration layout:
   - Pipeline mode: infer SmartSPIM session and alignment channel from
     the Neuroglancer file and use the pipeline directory structure.
   - Legacy mode: use the provided --legacy-registration directory.

2) Registration-channel outputs:
   - Write the registration channel moved to CCF:
       /results/<mouseid>/ccf_space_histology/histology_registration.nrrd
   - Copy the preprocessed image (image space):
       /results/<mouseid>/image_space_histology/histology_registration.nii.gz

3) Additional channels:
   - Legacy mode: lift each moved template-space channel to CCF and write
     histology_<channel>.nrrd in the CCF histology folder.
   - Pipeline mode: load non-alignment OME-Zarr channels, write image-space
     NIfTIs (<channel>.nii.gz), then transform each to CCF and write
     histol

"""

import argparse
import json
import os
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd
import SimpleITK as sitk
from aind_anatomical_utils.coordinate_systems import convert_coordinate_system
from aind_ephys_ibl_gui_conversion.ephys import (
    extract_continuous,
    extract_spikes,
)
from aind_ephys_ibl_gui_conversion.histology import (
    create_slicer_fcsv,
    get_additional_channel_image_at_highest_level,
    order_annotation_pts,
    read_json_as_dict,
)
from aind_registration_utils.ants import apply_ants_transforms_to_point_arr
from aind_s3_cache.json_utils import get_json
from aind_s3_cache.uri_utils import as_pathlike
from aind_zarr_utils.neuroglancer import (
    get_image_sources,
    neuroglancer_annotations_to_anatomical,
)
from aind_zarr_utils.pipeline_transformed import (
    _asset_from_zarr_pathlike,
    alignment_zarr_uri_and_metadata_from_zarr_or_asset_pathlike,
    mimic_pipeline_zarr_to_anatomical_stub,
    mimic_pipeline_zarr_to_ants,
    pipeline_transforms_local_paths,
)
from aind_zarr_utils.zarr import (
    _open_zarr,
    zarr_to_ants,
    zarr_to_sitk_stub,
)
from ants.core import ANTsImage
from iblatlas import atlas


@dataclass(frozen=True)
class Args:
    neuroglancer: str
    annotation_manifest: str
    legacy_registration: str | None


@dataclass(frozen=True)
class InputPaths:
    neuroglancer_file: Path
    manifest_csv: Path
    data_root: Path
    results_root: Path


@dataclass(frozen=True)
class ReferenceVolumes:
    template_25: ants.ANTsImage
    ccf_25: ants.ANTsImage
    ccf_labels_25: ants.ANTsImage
    # Object exposing ccf2xyz; keep it typed as Any if needed
    brain_atlas: atlas.AllenAtlas


@dataclass(frozen=True)
class ZarrPaths:
    registration: str
    additional: list[str]
    metadata: dict[str, Any]
    processing: dict[str, Any]


@dataclass(frozen=True)
class RegistrationInfo:
    registration_root: Path
    prep_image_folder: Path
    moved_image_folder: Path
    alignment_channel: str | None  # None in legacy mode


@dataclass(frozen=True)
class PipelineRegistrationInfo:
    pt_tx_str: list[str]
    pt_tx_inverted: list[bool]
    img_tx_str: list[str]
    img_tx_inverted: list[bool]


@dataclass(frozen=True)
class AssetInfo:
    asset_path: Path
    zarr_volumes: ZarrPaths
    pipeline_registration_chains: PipelineRegistrationInfo
    registration_dir_path: Path
    registration_in_ccf_precomputed: Path


@dataclass(frozen=True)
class Geometry:
    extrema_mm: np.ndarray  # shape (3,)
    origin_mm: Sequence[float]
    exemplar_image: ants.ANTsImage


@dataclass(frozen=True)
class OutputDirs:
    histology_ccf: Path
    histology_img: Path
    tracks_root: Path
    spim: Path
    template: Path
    ccf: Path
    bregma_xyz: Path


@dataclass(frozen=True)
class ProcessResult:
    probe_id: str
    recording_id: str
    wrote_files: bool
    skipped_reason: str | None = None


# ---- Stage 1: args and input resolution -------------------------------------


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
        "--legacy",
        dest="legacy_registration",
        default=None,
        help="Use old registration from Di capsual",
    )

    parser.add_argument(
        "--update_packages_from_source",
        default=None,
        help="Unused in capsule run script",
    )

    args = parser.parse_args()
    return args


def parse_and_normalize_args() -> Args:
    """
    Parse CLI args and normalize empty legacy flag to None.
    """
    a = parse_args()
    legacy = a.legacy_registration or None
    return Args(
        neuroglancer=a.neuroglancer,
        annotation_manifest=a.annotation_manifest,
        legacy_registration=legacy,
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


# ---- Stage 2: references, registration layout, geometry ---------------------


def load_references() -> ReferenceVolumes:
    """
    Load template, CCF volumes, and atlas helper.
    """
    template = ants.image_read(
        "/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz"
    )
    ccf = ants.image_read(
        "/data/allen_mouse_ccf/average_template/average_template_25.nii.gz"
    )
    labels = ants.image_read(
        "/data/allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz"
    )
    brain_atlas = atlas.AllenAtlas(25, hist_path="/scratch/")
    return ReferenceVolumes(template, ccf, labels, brain_atlas)


def determine_registration_layout_legacy(
    legacy_registration: str, paths: InputPaths
) -> RegistrationInfo:
    """
    Return key folders for legacy registration.

    Notes
    -----
    In pipeline mode, discovers SmartSPIM session ID and alignment channel.
    """
    legacy_root = (
        paths.data_root / legacy_registration
        if not str(legacy_registration).startswith(str(paths.data_root))
        else Path(legacy_registration)
    )
    return RegistrationInfo(
        registration_root=legacy_root,
        prep_image_folder=legacy_root / "registration",
        moved_image_folder=legacy_root / "registration",
        alignment_channel=None,
    )


def inspect_image_geometry(reg: RegistrationInfo) -> Geometry:
    """
    Read a representative image to compute spacing, origin, and extent.

    Returns
    -------
    Geometry
        Physical extent and origin in millimeters.
    """
    img = ants.image_read(str(reg.prep_image_folder / "prep_n4bias.nii.gz"))
    extrema = np.array(img.shape) * np.array(img.spacing)
    return Geometry(
        extrema_mm=extrema, origin_mm=img.origin, exemplar_image=img
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


def write_registration_channel_outputs(
    asset_info: AssetInfo,
    outputs: OutputDirs,
    *,
    level: int = 3,
    opened_zarr: tuple[Any, dict[str, Any]] | None = None,
) -> ANTsImage:
    """
    Write registration-channel outputs to CCF and image space.
    """
    if not asset_info.registration_in_ccf_precomputed.exists():
        raise FileNotFoundError(
            "Precomputed registration in CCF not found: "
            f"{asset_info.registration_in_ccf_precomputed}"
        )
    # Save the precomputed CCF-space image as a nrrd
    ccf_img = sitk.ReadImage(str(asset_info.registration_in_ccf_precomputed))
    sitk.WriteImage(
        ccf_img,
        str(outputs.histology_ccf / "histology_registration.nrrd"),
        useCompression=True,
    )
    reg_zarr = asset_info.zarr_volumes.registration
    if opened_zarr is None:
        zarr_node, zarr_metadata = _open_zarr(reg_zarr)
    else:
        zarr_node, zarr_metadata = opened_zarr
    # Get the minimum voxel spatial dimension for each multiscale level
    raw_img = zarr_to_ants(
        reg_zarr,
        asset_info.zarr_volumes.metadata,
        level=level,
        opened_zarr=(zarr_node, zarr_metadata),
    )
    # Copy the preprocessed image (image space)
    raw_img_dst = outputs.histology_img / "histology_registration.nii.gz"
    ants.image_write(raw_img, str(raw_img_dst))
    return raw_img


def write_registration_channel_outputs_legacy(
    reg: RegistrationInfo, outputs: OutputDirs
) -> None:
    """
    Write registration-channel outputs to CCF and image space.
    """
    outimg = ants.image_read(
        str(reg.moved_image_folder / "moved_ls_to_ccf.nii.gz")
    )
    ants.image_write(
        outimg, str(outputs.histology_ccf / "histology_registration.nrrd")
    )
    src = reg.prep_image_folder / "prep_n4bias.nii.gz"
    dst = outputs.histology_img / "histology_registration.nii.gz"
    shutil.copy(src, dst)


def process_additional_channels_legacy(
    reg: RegistrationInfo, refs: ReferenceVolumes, outputs: OutputDirs
) -> None:
    """
    Legacy channel handling: template -> CCF for each moved channel.

    Reads files named "moved_ls_to_template_<channel>.nii.gz" from
    reg.moved_image_folder, applies template→CCF transforms, and writes:

      - CCF-space (on CCF grid):  outputs.histology_ccf / "histology_<channel>.nrrd"
      - Template-space copy:      outputs.histology_img / "histology_<channel>.nii.gz"

    Notes
    -----
    - The template-space NIfTI is copied byte-for-byte for reference.
    """
    moved_dir: Path = reg.moved_image_folder

    # Template → CCF transforms (forward direction; no inversion needed)
    xforms_template_to_ccf = [
        "/data/spim_template_to_ccf/syn_1Warp.nii.gz",
        "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
    ]

    for f in moved_dir.iterdir():
        name = f.name
        if not (
            name.startswith("moved_ls_to_template_")
            and name.endswith(".nii.gz")
        ):
            continue

        chname = name.split("moved_ls_to_template_")[-1].split(".nii.gz")[0]

        # Load the template-space image for this channel
        template_img_path = reg.registration_root / "registration" / name
        image_in_template = ants.image_read(str(template_img_path))

        # Map to CCF space on the CCF grid (fixed = refs.ccf_25)
        outimg = ants.apply_transforms(
            refs.ccf_25,
            image_in_template,
            xforms_template_to_ccf,
        )

        # Write outputs
        ants.image_write(
            outimg, str(outputs.histology_ccf / f"histology_{chname}.nrrd")
        )
        (outputs.histology_img / f"histology_{chname}.nii.gz").write_bytes(
            template_img_path.read_bytes()
        )


def process_additional_channels_pipeline(
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
    level: int = 3,
) -> ANTsImage | None:
    """
    Pipeline channel handling:
      1) Load non-alignment OME-Zarr channels at highest level on the template grid,
         write image-space NIfTI (<channel>.nii.gz) into outputs.histology_img.
      2) Chain ls->template then template->CCF transforms; write CCF-space
         histology_<channel>.nrrd into outputs.histology_ccf (fixed grid = CCF).
    """
    metadata = asset_info.zarr_volumes.metadata
    processing = asset_info.zarr_volumes.processing

    a_bugged_img = None
    for zarr_path in asset_info.zarr_volumes.additional:
        # Load the image in the vanilla space
        ch_str = Path(zarr_path).stem
        img_raw = zarr_to_ants(
            zarr_path, asset_info.zarr_volumes.metadata, level=level
        )
        ants.image_write(
            img_raw, str(outputs.histology_img / f"{ch_str}.nii.gz")
        )

        # Map to CCF
        # First mimic the pipeline's histology loading, buggy though it may be
        img_bugged = mimic_pipeline_zarr_to_ants(
            zarr_path, metadata, processing, level=level
        )
        # So we can use the existing pipeline transforms
        ch_in_ccf = ants.apply_transforms(
            refs.ccf_25,
            img_bugged,
            asset_info.pipeline_registration_chains.img_tx_str,
            whichtoinvert=asset_info.pipeline_registration_chains.img_tx_inverted,
        )
        ants.image_write(
            ch_in_ccf, str(outputs.histology_ccf / f"histology_{ch_str}.nrrd")
        )
        a_bugged_img = (
            img_bugged  # Get the histology domain of the buggy ccf transform
        )
    return a_bugged_img


def _apply_ccf_pt_tx_buggy_domain_then_fix(
    ccf_space_img: ANTsImage,
    buggy_hist_domain_img: ANTsImage,
    hist_domain_img: ANTsImage,
    asset_info: AssetInfo,
    **kwargs: Any,
) -> ANTsImage:
    pt_tx_str = asset_info.pipeline_registration_chains.pt_tx_str
    pt_tx_inverted = asset_info.pipeline_registration_chains.pt_tx_inverted
    # This will be in the buggy domain, but we can fix that later
    ccf_space_in_hist_img: ANTsImage = ants.apply_transforms(
        fixed=buggy_hist_domain_img,
        moving=ccf_space_img,
        transformlist=pt_tx_str,
        whichtoinvert=pt_tx_inverted,
        **kwargs,
    )
    # Update the spatial domain to match the real image
    ccf_space_in_hist_img.set_spacing(hist_domain_img.spacing)
    ccf_space_in_hist_img.set_origin(hist_domain_img.origin)
    ccf_space_in_hist_img.set_direction(hist_domain_img.direction)
    return ccf_space_in_hist_img


def push_atlas_to_image_space(
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    raw_img: ANTsImage,
    raw_img_domain_bugged: ANTsImage,
    outputs: OutputDirs,
) -> None:
    """
    Transform CCF template and labels into native image space.
    """
    # point transforms are inverse of image transforms
    # Need to use buggy domain to use the ccf transforms
    ccf_in_hist_img = _apply_ccf_pt_tx_buggy_domain_then_fix(
        refs.ccf_25,
        buggy_hist_domain_img=raw_img_domain_bugged,
        hist_domain_img=raw_img,
        asset_info=asset_info,
    )
    ants.image_write(
        ccf_in_hist_img, str(outputs.histology_img / "ccf_in_mouse.nrrd")
    )
    ccf_labels_in_hist_img = _apply_ccf_pt_tx_buggy_domain_then_fix(
        refs.ccf_labels_25,
        buggy_hist_domain_img=raw_img_domain_bugged,
        hist_domain_img=raw_img,
        asset_info=asset_info,
        interpolator="genericLabel",
    )
    ants.image_write(
        ccf_labels_in_hist_img,
        str(outputs.histology_img / "labels_in_mouse.nrrd"),
    )


def push_atlas_to_image_space_legacy(
    refs: ReferenceVolumes, reg: RegistrationInfo, outputs: OutputDirs
) -> None:
    """
    Transform CCF template and labels into native image space.
    """
    xforms = [
        reg.moved_image_folder / "ls_to_template_SyN_0GenericAffine.mat",
        reg.moved_image_folder / "ls_to_template_SyN_1InverseWarp.nii.gz",
        Path("/data/spim_template_to_ccf/syn_0GenericAffine.mat"),
        Path("/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz"),
    ]
    ccf_img = ants.apply_transforms(
        reg.prep_image_folder / "prep_n4bias.nii.gz",
        refs.ccf_25,
        [str(p) for p in xforms],
        whichtoinvert=[True, False, True, False],
    )
    ants.image_write(ccf_img, str(outputs.histology_img / "ccf_in_mouse.nrrd"))

    labels_img = ants.apply_transforms(
        reg.prep_image_folder / "prep_n4bias.nii.gz",
        refs.ccf_labels_25,
        [str(p) for p in xforms],
        whichtoinvert=[True, False, True, False],
        interpolator="genericLabel",
    )
    ants.image_write(
        labels_img, str(outputs.histology_img / "labels_in_mouse.nrrd")
    )


# ---- Stage 7: per-row processing (probe-centric) -----------------------------


def process_manifest_row(
    row: pd.Series,
    asset_info: AssetInfo,
    hist_stub: sitk.Image,
    hist_stub_buggy: sitk.Image,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
) -> ProcessResult:
    """
    End-to-end processing for a single manifest row.

    Steps
    -----
    1. Locate annotation, load NG points.
    2. Convert to image-space physical (LPS) coordinates.
    3. Write FCSV for SPIM/template/CCF.
    4. Convert to IBL xyz-picks and write JSONs (with shank handling).
    5. Copy xyz-picks into GUI sorting folder.

    Returns
    -------
    ProcessResult
        Summary of success/skip for the row.
    """
    # -- find annotation file (json) --
    # -- load points -> image coords using geometry.extrema_mm & origin_mm --
    # -- order points, write FCSVs to outputs.spim / outputs.template / outputs.ccf --
    # -- transform to template/CCF via ants.apply_transforms_to_points --
    # -- convert to IBL ML/AP/DV using refs.brain_atlas.ccf2xyz --
    # -- write JSONs under outputs.bregma_xyz and copy into GUI folder --
    # 1) Locate annotation (JSON default)
    try:
        ext = "json" if str(row.annotation_format).lower() == "json" else None
    except Exception:
        ext = "json"
    if ext is None:
        return ProcessResult(
            str(row.probe_id),
            str(row.sorted_recording),
            False,
            "Only JSON annotations supported",
        )

    pattern = f"*/{row.probe_file}.{ext}"
    ann_path = next(Path("/data").glob(pattern), None)
    probe_id = str(row.probe_id)
    if ann_path is None:
        return ProcessResult(
            probe_id,
            str(row.sorted_recording),
            False,
            f"Annotation not found: {pattern}",
        )
    # 2) Load NG points in histology space
    ng_data = get_json(str(ann_path))
    anno_zarr = asset_info.zarr_volumes.registration
    metadata = asset_info.zarr_volumes.metadata
    probe_pt_dict, _ = neuroglancer_annotations_to_anatomical(
        ng_data,
        anno_zarr,
        metadata,
        layer_names=[probe_id],
        stub_image=hist_stub,
    )
    probe_pts = probe_pt_dict.get(probe_id, None)
    if probe_pts is None:
        return ProcessResult(
            probe_id,
            str(row.sorted_recording),
            False,
            f"Probe points not found: {probe_id}",
        )

    # 3) Write SPIM FCSV
    create_slicer_fcsv(
        str(outputs.spim / f"{probe_id}.fcsv"),
        probe_pts,
        direction="LPS",
    )

    # 4) Image → Template (points) via buggy pipeline transform
    # there might be a scale problem here?
    probe_pt_dict_buggy, _ = neuroglancer_annotations_to_anatomical(
        ng_data,
        anno_zarr,
        metadata,
        layer_names=[probe_id],
        stub_image=hist_stub_buggy,
    )
    probe_pts_buggy = probe_pt_dict_buggy[probe_id]
    tx_list_pt_template = [
        str(
            asset_info.registration_dir_path
            / "ls_to_template_SyN_0GenericAffine.mat"
        ),
        str(
            asset_info.registration_dir_path
            / "ls_to_template_SyN_1InverseWarp.nii.gz"
        ),
    ]
    tx_list_pt_template_invert = [True, False]
    pts_template = apply_ants_transforms_to_point_arr(
        probe_pts_buggy,
        tx_list_pt_template,
        whichtoinvert=tx_list_pt_template_invert,
    )
    create_slicer_fcsv(
        str(outputs.template / f"{probe_id}.fcsv"),
        pts_template,
        direction="LPS",
    )

    # 5) Template -> CCF (points) not sure this works
    pts_ccf = apply_ants_transforms_to_point_arr(
        probe_pts_buggy,
        asset_info.pipeline_registration_chains.pt_tx_str,
        whichtoinvert=asset_info.pipeline_registration_chains.pt_tx_inverted,
    )
    create_slicer_fcsv(
        str(outputs.ccf / f"{row.probe_id}.fcsv"),
        pts_ccf,
        direction="LPS",
    )

    # 6) IBL xyz-picks (µm) from CCF (ML/AP/DV with signed flips)

    ccf_mlapdv_um = convert_coordinate_system(
        1000.0 * pts_ccf,
        src_coord="LPS",
        dst_coord="RPI",
    )
    bregma_mlapdv_um = (
        # Returns in meters, scale to µm
        1_000_000.0
        * refs.brain_atlas.ccf2xyz(ccf_mlapdv_um, ccf_order="mlapdv")
    )

    # Image-space xyz-picks (µm), matching original math
    xyz_img = 1000.0 * convert_coordinate_system(
        probe_pts, src_coord="LPS", dst_coord="RAS"
    )

    xyz_picks_image = {"xyz_picks": xyz_img.tolist()}
    xyz_picks_ccf = {"xyz_picks": bregma_mlapdv_um.tolist()}

    # 7) Write bregma_xyz JSONs (global per-mouse)
    has_shank = ("probe_shank" in row.keys()) and (
        not pd.isna(row.probe_shank)
    )
    if has_shank:
        shank_id = int(row.probe_shank) + 1
        img_name = f"{row.probe_id}_shank{shank_id}_image_space.json"
        ccf_name = f"{row.probe_id}_shank{shank_id}_ccf.json"
    else:
        img_name = f"{row.probe_id}_image_space.json"
        ccf_name = f"{row.probe_id}_ccf.json"

    (outputs.bregma_xyz / img_name).write_text(json.dumps(xyz_picks_image))
    (outputs.bregma_xyz / ccf_name).write_text(json.dumps(xyz_picks_ccf))

    # 8) Per-recording GUI artifacts
    recording_id = str(row.sorted_recording).split("_sorted")[0]
    gui_folder = (
        outputs.tracks_root.parent / recording_id / str(row.probe_name)
    )
    gui_folder.mkdir(parents=True, exist_ok=True)
    if has_shank:
        gui_img = f"xyz_picks_shank{shank_id}_image_space.json"
        gui_ccf = f"xyz_picks_shank{shank_id}.json"
    else:
        gui_img = "xyz_picks_image_space.json"
        gui_ccf = "xyz_picks.json"
    (gui_folder / gui_img).write_text(json.dumps(xyz_picks_image))
    (gui_folder / gui_ccf).write_text(json.dumps(xyz_picks_ccf))

    return ProcessResult(
        probe_id=str(row.probe_id),
        recording_id=recording_id,
        wrote_files=True,
        skipped_reason=None,
    )


def process_manifest_row_legacy(
    row: pd.Series,
    geometry: Geometry,
    reg: RegistrationInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
) -> ProcessResult:
    """
    End-to-end processing for a single manifest row.

    Steps
    -----
    1. Locate annotation, load NG points.
    2. Convert to image-space physical (LPS) coordinates.
    3. Write FCSV for SPIM/template/CCF.
    4. Convert to IBL xyz-picks and write JSONs (with shank handling).
    5. Copy xyz-picks into GUI sorting folder.

    Returns
    -------
    ProcessResult
        Summary of success/skip for the row.
    """
    # -- find annotation file (json) --
    # -- load points -> image coords using geometry.extrema_mm & origin_mm --
    # -- order points, write FCSVs to outputs.spim / outputs.template / outputs.ccf --
    # -- transform to template/CCF via ants.apply_transforms_to_points --
    # -- convert to IBL ML/AP/DV using refs.brain_atlas.ccf2xyz --
    # -- write JSONs under outputs.bregma_xyz and copy into GUI folder --
    # 1) Locate annotation (JSON default)
    try:
        ext = "json" if str(row.annotation_format).lower() == "json" else None
    except Exception:
        ext = "json"
    if ext is None:
        return ProcessResult(
            str(row.probe_id),
            str(row.sorted_recording),
            False,
            "Only JSON annotations supported",
        )

    pattern = f"*/{row.probe_file}.{ext}"
    ann_path = next(Path("/data").glob(pattern), None)
    if ann_path is None:
        return ProcessResult(
            str(row.probe_id),
            str(row.sorted_recording),
            False,
            f"Annotation not found: {pattern}",
        )

    # 2) Load NG points → image-space physical (LPS, mm)
    probe_df, dims = load_neuroglancer_points(ann_path, row.probe_id)
    ext_mm = geometry.extrema_mm
    org_mm = geometry.origin_mm

    x = ext_mm[0] - probe_df.x.values * dims["x"][0] * 1e3 + org_mm[0]
    y = probe_df.y.values * dims["y"][0] * 1e3 + org_mm[1]
    z = -probe_df.z.values * dims["z"][0] * 1e3 + org_mm[2]
    pts_spim_mm = np.vstack([x, y, z]).T
    pts_spim_mm = order_annotation_pts(pts_spim_mm)

    # 3) Write SPIM FCSV
    create_slicer_fcsv(
        str(outputs.spim / f"{row.probe_id}.fcsv"),
        pts_spim_mm,
        direction="LPS",
    )

    # 4) Image -> Template (points)
    df_pts = pd.DataFrame(
        {
            "x": pts_spim_mm[:, 0],
            "y": pts_spim_mm[:, 1],
            "z": pts_spim_mm[:, 2],
        }
    )
    pts_template = ants.apply_transforms_to_points(
        3,
        df_pts,
        [
            str(
                reg.moved_image_folder
                / "ls_to_template_SyN_0GenericAffine.mat"
            ),
            str(
                reg.moved_image_folder
                / "ls_to_template_SyN_1InverseWarp.nii.gz"
            ),
        ],
        whichtoinvert=[True, False],
    )
    create_slicer_fcsv(
        str(outputs.template / f"{row.probe_id}.fcsv"),
        pts_template.values,
        direction="LPS",
    )

    # 5) Template -> CCF (points)
    pts_ccf = ants.apply_transforms_to_points(
        3,
        pts_template,
        [
            "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
            "/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz",
        ],
        whichtoinvert=[True, False],
    )
    create_slicer_fcsv(
        str(outputs.ccf / f"{row.probe_id}.fcsv"),
        pts_ccf.values,
        direction="LPS",
    )

    # 6) IBL xyz-picks (µm) from CCF (ML/AP/DV with signed flips)
    ccf_mlapdv_um = pts_ccf.values.copy() * 1000.0
    ccf_mlapdv_um[:, 0] = -ccf_mlapdv_um[:, 0]  # ML flip
    ccf_mlapdv_um[:, 2] = -ccf_mlapdv_um[:, 2]  # DV flip
    bregma_mlapdv_um = (
        refs.brain_atlas.ccf2xyz(ccf_mlapdv_um, ccf_order="mlapdv")
        * 1_000_000.0
    )

    # Image-space xyz-picks (µm), matching original math
    xyz_img = probe_df[["x", "y", "z"]].to_numpy().copy()
    xyz_img[:, 0] = (
        ext_mm[0] - (xyz_img[:, 0] * dims["x"][0] * 1000.0)
    ) * 1000.0
    xyz_img[:, 1] = xyz_img[:, 1] * dims["y"][0] * 1_000_000.0
    xyz_img[:, 2] = xyz_img[:, 2] * dims["z"][0] * 1_000_000.0

    xyz_picks_image = {"xyz_picks": xyz_img.tolist()}
    xyz_picks_ccf = {"xyz_picks": bregma_mlapdv_um.tolist()}

    # 7) Write bregma_xyz JSONs (global per-mouse)
    has_shank = ("probe_shank" in row.keys()) and (
        not pd.isna(row.probe_shank)
    )
    if has_shank:
        shank_id = int(row.probe_shank) + 1
        img_name = f"{row.probe_id}_shank{shank_id}_image_space.json"
        ccf_name = f"{row.probe_id}_shank{shank_id}_ccf.json"
    else:
        img_name = f"{row.probe_id}_image_space.json"
        ccf_name = f"{row.probe_id}_ccf.json"

    (outputs.bregma_xyz / img_name).write_text(json.dumps(xyz_picks_image))
    (outputs.bregma_xyz / ccf_name).write_text(json.dumps(xyz_picks_ccf))

    # 8) Per-recording GUI artifacts
    recording_id = str(row.sorted_recording).split("_sorted")[0]
    gui_folder = (
        outputs.tracks_root.parent / recording_id / str(row.probe_name)
    )
    gui_folder.mkdir(parents=True, exist_ok=True)
    if has_shank:
        gui_img = f"xyz_picks_shank{shank_id}_image_space.json"
        gui_ccf = f"xyz_picks_shank{shank_id}.json"
    else:
        gui_img = "xyz_picks_image_space.json"
        gui_ccf = "xyz_picks.json"
    (gui_folder / gui_img).write_text(json.dumps(xyz_picks_image))
    (gui_folder / gui_ccf).write_text(json.dumps(xyz_picks_ccf))

    return ProcessResult(
        probe_id=str(row.probe_id),
        recording_id=recording_id,
        wrote_files=True,
        skipped_reason=None,
    )


# ---- Stage 8: optional ephys -------------------------------------------------


def maybe_run_ephys(
    row: pd.Series, outputs: OutputDirs, processed: set[str]
) -> None:
    """
    Run ephys extraction once per unique `sorted_recording`.

    Creates (if needed) the results folder under:
        /results/<mouseid>/<recording_id>/

    Then invokes:
        extract_continuous(recording_folder, results_folder, [probe_surface_finding=...])
        extract_spikes(recording_folder, results_folder)

    On failure, emits a warning and attempts to copy any "output" artifact
    from the source recording folder into the results folder for debugging.

    Parameters
    ----------
    row : pd.Series
        A manifest row containing `mouseid`, `sorted_recording`, and optional `surface_finding`.
    outputs : OutputDirs
        Dataclass with resolved output directories. We derive the per-recording
        results root from `outputs.tracks_root.parent` (i.e., /results/<mouseid>).
    processed : set[str]
        Set of `sorted_recording` strings already processed to ensure idempotency.
    """

    # Idempotency key
    sorted_rec = str(row.sorted_recording)
    if sorted_rec in processed:
        return
    processed.add(sorted_rec)

    # Derive key paths
    recording_id = sorted_rec.split("_sorted")[0]
    mouse_root = outputs.tracks_root.parent  # /results/<mouseid>
    results_folder = (
        mouse_root / recording_id
    )  # /results/<mouseid>/<recording_id>
    results_folder.mkdir(parents=True, exist_ok=True)

    recording_folder = Path("/data") / sorted_rec  # /data/<sorted_recording>

    # Run extraction, optionally with surface finding hint
    try:
        if ("surface_finding" in row.index) and (
            not pd.isna(row.surface_finding)
        ):
            extract_continuous(
                recording_folder,
                results_folder,
                probe_surface_finding=Path("/data") / str(row.surface_finding),
            )
        else:
            extract_continuous(recording_folder, results_folder)

        extract_spikes(recording_folder, results_folder)

    except ValueError:
        warnings.warn(
            f"Missing spike sorting for {sorted_rec}. Proceeding with histology only.",
            stacklevel=1,
        )
        # Best-effort copy of any diagnostic artifact named 'output'
        src = recording_folder / "output"
        dst = results_folder / "output"
        try:
            if src.exists():
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as e:
            warnings.warn(
                f"Failed to copy diagnostic artifact from {src} -> {dst}: {e}",
                stacklevel=1,
            )


# ---- Orchestrator (new short main) ------------------------------------------


def process_histology_and_ephys(args: Args):
    paths = resolve_paths(args)

    # Keep a manifest snapshot for reproducibility
    shutil.copy(paths.manifest_csv, "/results/manifest.csv")

    manifest_df = pd.read_csv(paths.manifest_csv)
    s = manifest_df["mouseid"].astype("string")
    val = s.iat[0]
    if pd.isna(val):
        raise ValueError("mouseid is missing in first row")
    mouse_id: str = val  # now definitely a str
    refs = load_references()
    out = prepare_result_dirs(mouse_id, paths.results_root)
    if args.legacy_registration is None:
        asset_info = find_asset_info(paths)
        node, zarr_metadata = _open_zarr(asset_info.zarr_volumes.registration)
        level = determine_desired_level(
            zarr_metadata, desired_voxel_size_um=25.0
        )

        raw_img = write_registration_channel_outputs(
            asset_info, out, level=level, opened_zarr=(node, zarr_metadata)
        )
        a_bugged_img = process_additional_channels_pipeline(
            asset_info, refs, out
        )
        if a_bugged_img is None:
            raw_img_domain_bugged = mimic_pipeline_zarr_to_ants(
                asset_info.zarr_volumes.registration,
                asset_info.zarr_volumes.metadata,
                asset_info.zarr_volumes.processing,
                level=level,
            )
        else:
            raw_img_domain_bugged = a_bugged_img
        push_atlas_to_image_space(
            asset_info, refs, raw_img, raw_img_domain_bugged, out
        )
        raw_img_stub, _ = zarr_to_sitk_stub(
            asset_info.zarr_volumes.registration,
            asset_info.zarr_volumes.metadata,
            opened_zarr=(node, zarr_metadata),
        )
        raw_img_stub_buggy = mimic_pipeline_zarr_to_anatomical_stub(
            asset_info.zarr_volumes.registration,
            asset_info.zarr_volumes.metadata,
            asset_info.zarr_volumes.processing,
            opened_zarr=(node, zarr_metadata),
        )
        processed_recordings: set[str] = set()
        for _, row in manifest_df.iterrows():
            process_manifest_row(
                row, asset_info, raw_img_stub, raw_img_stub_buggy, refs, out
            )
            maybe_run_ephys(row, out, processed_recordings)

    else:
        reg = determine_registration_layout_legacy(
            args.legacy_registration, paths
        )
        process_additional_channels_legacy(reg, refs, out)
        geom = inspect_image_geometry(reg)
        write_registration_channel_outputs_legacy(reg, out)
        push_atlas_to_image_space_legacy(refs, reg, out)

        processed_recordings: set[str] = set()
        for _, row in manifest_df.iterrows():
            res = process_manifest_row_legacy(row, geom, reg, refs, out)
            maybe_run_ephys(row, out, processed_recordings)


def new_main() -> None:
    """
    Orchestrate the full processing pipeline.
    """
    args = parse_and_normalize_args()
    process_histology_and_ephys(args)


def main():
    # ------------------------------------------------------------
    # Parse CLI arguments and normalize optional flags/values
    # ------------------------------------------------------------
    args = parse_args()
    if args.legacy_registration == "":
        args.legacy_registration = None

    # ------------------------------------------------------------
    # Resolve and validate input paths (neuroglancer & manifest)
    # Always map to /data/* so downstream code has absolute paths
    # ------------------------------------------------------------
    if "/data/" not in args.neuroglancer:
        neuroglancer_file_path = os.path.join("/data/", args.neuroglancer)
    else:
        neuroglancer_file_path = args.neuroglancer

    if "/data/" not in args.annotation_manifest:
        annotation_manifest_path = os.path.join(
            "/data/", args.annotation_manifest
        )
    else:
        annotation_manifest_path = args.annotation_manifest

    # ------------------------------------------------------------
    # Snapshot the manifest for reproducibility/debugging
    # ------------------------------------------------------------
    Path("/results/manifest.csv").write_bytes(
        Path(annotation_manifest_path).read_bytes()
    )

    # ------------------------------------------------------------
    # Load annotation–ephys pairings and reference volumes/atlas
    # (template, CCF, labels, and AllenAtlas helper)
    # ------------------------------------------------------------
    manifest_df = pd.read_csv(annotation_manifest_path)
    template = ants.image_read(
        "/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz"
    )
    ccf_25 = ants.image_read(
        "/data/allen_mouse_ccf/average_template/average_template_25.nii.gz"
    )
    ccf_annotation_25 = ants.image_read(
        "/data/allen_mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz"
    )
    brain_atlas = atlas.AllenAtlas(25, hist_path="/scratch/")

    # ------------------------------------------------------------
    # Determine registration source:
    #  - Default: use pipeline outputs referenced by Neuroglancer
    #  - Legacy mode: use provided registration directory layout
    # This block resolves folders for preprocessed & moved images.
    # ------------------------------------------------------------
    #
    # Default is to use the pipeline registration.
    # However, if an alternative path is passed as "legacy_registration", we will pull the regisration from there.
    # This assumes that you used Di's conversion capsual. If you don't know what that means, you probably didnt...

    if args.legacy_registration:
        # Handle legacy path
        if "/data/" not in args.legacy_registration:
            registration_data_asset = os.path.join(
                "/data/", args.legacy_registration
            )
        else:
            registration_data_asset = args.legacy_registration

        prep_image_folder = os.path.join(
            registration_data_asset, "registration"
        )
        moved_image_folder = os.path.join(
            registration_data_asset, "registration"
        )
    else:
        # ... discover SmartSPIM session, alignment channel, and folders ...
        #
        # Image source will be in a neuroglancer layer.
        # This assumes that a matching stiched asset is attached to the file.
        sources = get_image_source(
            neuroglancer_file_path
        )  # TODO: switch to zarr_utils
        if len(sources) > 1:
            print(
                "Found multiple SmartSPIM sources in Neuroglancer file. Using source for first layer."
            )
        source = sources[0]
        print(f"Using SmartSPIM source: {source}")
        smartspim_session_id = next(
            x for x in source.split("/") if x.startswith("SmartSPIM_")
        )
        registration_data_asset = os.path.join("/data/", smartspim_session_id)

        # Find data
        alignment_files = os.listdir(
            os.path.join(registration_data_asset, "image_atlas_alignment")
        )
        alignment_channel = [
            channel_file
            for channel_file in alignment_files
            if "Ex" in channel_file
        ][0]
        prep_image_folder = os.path.join(
            registration_data_asset,
            "image_atlas_alignment",
            alignment_channel,
            "metadata",
            "registration_metadata",
        )
        moved_image_folder = os.path.join(
            registration_data_asset, "image_atlas_alignment", alignment_channel
        )

    # ------------------------------------------------------------
    # Inspect image geometry to compute physical extent and offset
    # (used later to convert NG pixel coords to physical coords)
    # ------------------------------------------------------------
    zarr_read = ants.image_read(
        os.path.join(prep_image_folder, "prep_n4bias.nii.gz")
    )
    extrema = np.array(zarr_read.shape) * np.array(zarr_read.spacing)
    offset = zarr_read.origin

    # ------------------------------------------------------------
    # Prepare result directories for histology products
    #   - CCF-space outputs
    #   - Image-space (native SmartSPIM) outputs
    # ------------------------------------------------------------
    histology_results = os.path.join(
        "/results", str(manifest_df.mouseid[0]), "ccf_space_histology"
    )
    os.makedirs(histology_results, exist_ok=True)
    image_histology_results = os.path.join(
        "/results", str(manifest_df.mouseid[0]), "image_space_histology"
    )
    os.makedirs(image_histology_results, exist_ok=True)

    # ------------------------------------------------------------
    # Write registration-channel volumes to CCF and image space
    # (pipeline already transformed the registration channel)
    # ------------------------------------------------------------
    #
    # Read the registration channel data. No need to re-transform since this was done as part of inital registration
    outimg = ants.image_read(
        os.path.join(moved_image_folder, "moved_ls_to_ccf.nii.gz")
    )
    ants.image_write(
        outimg, os.path.join(histology_results, "histology_registration.nrrd")
    )
    # Handle other channel data. Depending on legacy flag, this may still need to be computed.
    shutil.copy(
        os.path.join(prep_image_folder, "prep_n4bias.nii.gz"),
        os.path.join(image_histology_results, "histology_registration.nii.gz"),
    )

    if args.legacy_registration:
        # ... iterate moved_* files, apply template->CCF transforms, write outputs ...
        # Handle other channels: This is a work in progress
        other_files = [
            x
            for x in os.listdir(moved_image_folder)
            if "moved_ls_to_template_" in x and ".nii.gz" in x
        ]
        for fl in other_files:
            chname = fl.split("moved_ls_to_template_")[-1].split(".nii.gz")[0]
            image_in_template = ants.image_read(
                os.path.join(registration_data_asset, "registration", fl)
            )
            outimg = ants.apply_transforms(
                ccf_25,
                image_in_template,
                [
                    "/data/spim_template_to_ccf/syn_1Warp.nii.gz",
                    "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
                ],
            )

            ants.image_write(
                outimg,
                os.path.join(histology_results, f"histology_{chname}.nrrd"),
            )
            shutil.copy(
                os.path.join(registration_data_asset, "registration", fl),
                os.path.join(
                    image_histology_results, "histology_{chname}.nii.gz"
                ),
            )
    else:
        # ... enumerate OME-Zarr channels, load, write image-space NIfTI,
        #     then chain transforms (ls->template, template->CCF) and save ...
        # find channels not used for alignment
        stitched_zarrs = os.path.join(
            registration_data_asset, "image_tile_fusing", "OMEZarr"
        )
        image_channel_zarrs = [
            x
            for x in os.listdir(stitched_zarrs)
            if os.path.isdir(os.path.join(stitched_zarrs, x)) and "zarr" in x
        ]
        image_channels = [x.split(".zarr")[0] for x in image_channel_zarrs]
        image_channels.pop(image_channels.index(alignment_channel))

        acquisition_path = f"{registration_data_asset}/acquisition.json"
        acquisition_json = read_json_as_dict(acquisition_path)
        acquisition_orientation = acquisition_json.get("axes")

        for this_channel in image_channels:
            # Load the channel
            this_ants_img = get_additional_channel_image_at_highest_level(
                os.path.join(
                    registration_data_asset,
                    "image_tile_fusing",
                    "OMEZarr",
                    f"{this_channel}.zarr",
                ),
                template,
                acquisition_orientation,
            )
            ants.image_write(
                this_ants_img,
                os.path.join(
                    image_histology_results, f"{this_channel}.nii.gz"
                ),
            )
            #
            channel_in_ccf = ants.apply_transforms(
                ccf_25,
                this_ants_img,
                [
                    "/data/spim_template_to_ccf/syn_1Warp.nii.gz",
                    "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
                    os.path.join(
                        moved_image_folder, "ls_to_template_SyN_1Warp.nii.gz"
                    ),
                    os.path.join(
                        moved_image_folder,
                        "ls_to_template_SyN_0GenericAffine.mat",
                    ),
                ],
            )
            ants.image_write(
                channel_in_ccf,
                os.path.join(
                    histology_results, f"histology_{this_channel}.nrrd"
                ),
            )

    # ------------------------------------------------------------
    # Push atlas content into image space:
    # Transform the CCF template and label volumes into native image space
    # for visualization/overlay in SmartSPIM coordinates.
    # ------------------------------------------------------------
    #
    # Tranform the CCF into image space
    ccf_in_image_space = ants.apply_transforms(
        zarr_read,
        ccf_25,
        [
            os.path.join(
                moved_image_folder, "ls_to_template_SyN_0GenericAffine.mat"
            ),
            os.path.join(
                moved_image_folder, "ls_to_template_SyN_1InverseWarp.nii.gz"
            ),
            "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
            "/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz",
        ],
        whichtoinvert=[True, False, True, False],
    )
    ants.image_write(
        ccf_in_image_space,
        os.path.join(
            image_histology_results, f"ccf_in_{manifest_df.mouseid[0]}.nrrd"
        ),
    )

    ccf_labels_in_image_space = ants.apply_transforms(
        zarr_read,
        ccf_annotation_25,
        [
            os.path.join(
                moved_image_folder, "ls_to_template_SyN_0GenericAffine.mat"
            ),
            os.path.join(
                moved_image_folder, "ls_to_template_SyN_1InverseWarp.nii.gz"
            ),
            "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
            "/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz",
        ],
        whichtoinvert=[True, False, True, False],
        interpolator="genericLabel",
    )
    ants.image_write(
        ccf_labels_in_image_space,
        os.path.join(
            image_histology_results, f"labels_in_{manifest_df.mouseid[0]}.nrrd"
        ),
    )

    # ------------------------------------------------------------
    # Create per-space track-data output roots:
    #   - SmartSPIM (spim), template, CCF, and IBL bregma-XYZ
    # ------------------------------------------------------------
    #
    # Prep file save local
    track_results = (
        Path("/results/") / str(manifest_df.mouseid[0]) / "track_data"
    )
    os.makedirs(track_results, exist_ok=True)
    spim_results = os.path.join(track_results, "spim")
    os.makedirs(spim_results, exist_ok=True)
    template_results = os.path.join(track_results, "template")
    os.makedirs(template_results, exist_ok=True)
    ccf_results = os.path.join(track_results, "ccf")
    os.makedirs(ccf_results, exist_ok=True)
    bregma_results = os.path.join(track_results, "bregma_xyz")
    os.makedirs(bregma_results, exist_ok=True)

    # ------------------------------------------------------------
    # Iterate over manifest rows:
    #  - Locate per-probe annotations
    #  - Convert NG coords -> image physical (LPS) coords
    #  - Save FCSV for SmartSPIM (spim) and template/CCF spaces
    #  - Convert to IBL xyz-picks (bregma ML/AP/DV) and save JSON
    # ------------------------------------------------------------
    processed_recordings = []

    for ii, row in manifest_df.iterrows():
        # ... detect annotation format; find annotation file path ...
        # ... load NG points; map to image space using spacing/origin ...
        # ... order points tip-to-surface and write slicer FCSV for spim ...
        # ... transform points: image -> template -> CCF, write FCSV ...
        # ... convert CCF (RAS) to IBL bregma ML/AP/DV and image-space picks ...
        # ... write xyz_picks JSONs (per-shank if specified) to bregma_results ...
        # ... also copy xyz_picks into sorter’s results folder for the GUI ...
        try:
            if row.annotation_format.lower() == "json":
                extension = "json"
            else:
                raise ValueError(
                    "Currently only jsons from neuroglancer are supported!"
                )
        except:
            print("No annotation format specified. Assuming JSON.")
            extension = "json"

        # Find the sorted and original data
        recording_id = row.sorted_recording.split("_sorted")[0]
        recording_folder = Path("/data/") / row.sorted_recording
        results_folder = Path("/results/") / str(row.mouseid) / recording_id

        pattern = f"*/{row.probe_file}.{extension}"
        annotation_file_path = next(Path("/data").glob(pattern), None)
        ng_data = get_json(str(annotation_file_path))
        if annotation_file_path is None:
            print(f"Failed to find {pattern!r}")
            continue
        else:
            # TODO: switch to zarr_utils
            this_probe_data, dims = load_neuroglancer_points(
                annotation_file_path, row.probe_id
            )
            x = (
                extrema[0]
                - this_probe_data.x.values * dims["x"][0] * 1e3
                + offset[0]
            )
            y = this_probe_data.y.values * dims["y"][0] * 1e3 + offset[1]
            z = -this_probe_data.z.values * dims["z"][0] * 1e3 + offset[2]

            this_probe = np.vstack([x, y, z]).T
            this_probe = order_annotation_pts(this_probe)
            create_slicer_fcsv(
                os.path.join(spim_results, f"{row.probe_id}.fcsv"),
                this_probe,
                direction="LPS",
            )

            # Move probe into template space.
            this_probe_df = pd.DataFrame(
                {
                    "x": this_probe[:, 0],
                    "y": this_probe[:, 1],
                    "z": this_probe[:, 2],
                }
            )
            # Transform into template space
            this_probe_template = ants.apply_transforms_to_points(
                3,
                this_probe_df,
                [
                    os.path.join(
                        moved_image_folder,
                        "ls_to_template_SyN_0GenericAffine.mat",
                    ),
                    os.path.join(
                        moved_image_folder,
                        "ls_to_template_SyN_1InverseWarp.nii.gz",
                    ),
                ],
                whichtoinvert=[True, False],
            )
            create_slicer_fcsv(
                os.path.join(template_results, f"{row.probe_id}.fcsv"),
                this_probe_template.values,
                direction="LPS",
            )

            # Move probe into ccf space
            this_probe_ccf = ants.apply_transforms_to_points(
                3,
                this_probe_template,
                [
                    "/data/spim_template_to_ccf/syn_0GenericAffine.mat",
                    "/data/spim_template_to_ccf/syn_1InverseWarp.nii.gz",
                ],
                whichtoinvert=[True, False],
            )
            create_slicer_fcsv(
                os.path.join(ccf_results, f"{row.probe_id}.fcsv"),
                this_probe_ccf.values,
                direction="LPS",
            )

            # TODO: fix this?
            # Transform into ibl x-y-z-picks space
            ccf_mlapdv = this_probe_ccf.values.copy() * 1000
            ccf_mlapdv[:, 0] = -ccf_mlapdv[:, 0]
            ccf_mlapdv[:, 1] = ccf_mlapdv[:, 1]
            ccf_mlapdv[:, 2] = -ccf_mlapdv[:, 2]
            bregma_mlapdv = (
                brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order="mlapdv") * 1000000
            )
            # xyz_picks = {'xyz_picks':bregma_mlapdv.tolist()}

            xyz_image_space = this_probe_data[["x", "y", "z"]].to_numpy()
            xyz_image_space[:, 0] = (
                extrema[0] - (xyz_image_space[:, 0] * dims["x"][0] * 1000)
            ) * 1000
            xyz_image_space[:, 1] = (
                xyz_image_space[:, 1] * dims["y"][0] * 1000000
            )
            xyz_image_space[:, 2] = (
                xyz_image_space[:, 2] * dims["z"][0] * 1000000
            )

            xyz_picks_image_space = {"xyz_picks": xyz_image_space.tolist()}
            xyz_picks_ccf = {"xyz_picks": bregma_mlapdv.tolist()}

            # assumes 1 shank unless probe shanks are specified.
            if ("probe_shank" in row.keys()) and (
                not pd.isna(row.probe_shank)
            ):
                shank_id = row.probe_shank + 1
                # Save this in two locations. First, save sorted by filename
                with open(
                    os.path.join(
                        bregma_results,
                        f"{row.probe_id}_shank{shank_id}_image_space.json",
                    ),
                    "w",
                ) as f:
                    # Serialize data to JSON format and write to file
                    json.dump(xyz_picks_image_space, f)

                with open(
                    os.path.join(
                        bregma_results,
                        f"{row.probe_id}_shank{shank_id}_ccf.json",
                    ),
                    "w",
                ) as f:
                    # Serialize data to JSON format and write to file
                    json.dump(xyz_picks_ccf, f)
            else:
                with open(
                    os.path.join(
                        bregma_results, f"{row.probe_id}_image_space.json"
                    ),
                    "w",
                ) as f:
                    # Serialize data to JSON format and write to file
                    json.dump(xyz_picks_image_space, f)

                with open(
                    os.path.join(bregma_results, f"{row.probe_id}_ccf.json"),
                    "w",
                ) as f:
                    # Serialize data to JSON format and write to file
                    json.dump(xyz_picks_ccf, f)

            # Second, save the XYZ picks to the sorting folder for the gui.
            # This step will be skipped if there was a problem with the ephys pipeline.
            folder_path = os.path.join(results_folder, str(row.probe_name))
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            if ("probe_shank" in row.keys()) and (
                not pd.isna(row.probe_shank)
            ):
                shank_id = row.probe_shank + 1
                image_space_filename = (
                    f"xyz_picks_shank{shank_id}_image_space.json"
                )
                ccf_space_filename = f"xyz_picks_shank{shank_id}.json"
            else:
                image_space_filename = "xyz_picks_image_space.json"
                ccf_space_filename = "xyz_picks.json"

            with open(
                os.path.join(folder_path, image_space_filename), "w"
            ) as f:
                json.dump(xyz_picks_image_space, f)

            with open(os.path.join(folder_path, ccf_space_filename), "w") as f:
                json.dump(xyz_picks_ccf, f)

            # --------------------------------------------------------
            # (Optional) Ephys extraction per recording:
            # Run only once per sorted recording; skip if already done.
            # Extract continuous signals and spikes; continue on failures.
            # --------------------------------------------------------
            #
            # Do ephys processing.
            # This is the last step here b.c. it is a annoyingly slow, and we need to give the the histology a chance to crash b.f. we reach it.
            if (
                row.sorted_recording not in processed_recordings
            ):  # DEBUGGING HACK TO STOP EPHYS PROCESSING!
                # ... create results folder, run extract_continuous/spikes with guards ...
                print(
                    f"Have not yet processed: {row.sorted_recording}. Doing that now."
                )
                os.makedirs(results_folder, exist_ok=True)
                try:
                    if not pd.isna(row.surface_finding):
                        extract_continuous(
                            recording_folder,
                            results_folder,
                            probe_surface_finding=Path(
                                f"/data/{row.surface_finding}"
                            ),
                        )
                    else:
                        extract_continuous(recording_folder, results_folder)
                    extract_spikes(recording_folder, results_folder)
                except ValueError:
                    warnings.warn(
                        f"Missing spike sorting for {row.sorted_recording}. Proceeding with histology only"
                    )
                    # Coppy the spike sorting error message to help future debugging.
                    shutil.copy(
                        os.path.join(recording_folder, "output"),
                        os.path.join(results_folder, "output"),
                    )
                processed_recordings.append(row.sorted_recording)


if __name__ == "__main__":
    new_main()
