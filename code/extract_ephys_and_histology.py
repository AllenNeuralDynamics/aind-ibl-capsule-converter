"""
SmartSPIM → Template → CCF probe/histology export pipeline.

This script ingests a Neuroglancer session, a manifest of probe annotations,
and a registration output inferred from the pipeline layout. It produces
per-mouse histology volumes in both CCF and image (SmartSPIM) spaces, per-probe
tracks in SPIM/template/CCF spaces, and IBL-style xyz-picks JSONs suitable for
the ephys alignment GUI.

Inputs
------
CLI arguments:
  --neuroglancer
      Path to a Neuroglancer layer JSON (absolute path is used as-is; a
      relative path is anchored under /data).
  --annotation-manifest
      CSV manifest describing probes and associated recordings. Relative paths
      are anchored under /data. The CSV is snapshotted to /results/manifest.csv.

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

2) Registration-channel outputs:
   - Write the registration channel moved to CCF:
       /results/<mouseid>/ccf_space_histology/histology_registration.nrrd
   - Copy the preprocessed image (image space):
       /results/<mouseid>/image_space_histology/histology_registration.nii.gz

3) Additional channels:
   Pipeline mode: load non-alignment OME-Zarr channels, write image-space
     NIfTIs (<channel>.nii.gz), then transform each to CCF and write
     histol

"""

import argparse
import json
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
    alignment_channel: str


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
        "--update_packages_from_source",
        default=None,
        help="Unused in capsule run script",
    )

    args = parser.parse_args()
    return args


def _parse_and_normalize_args() -> Args:
    """
    Parse CLI args
    """
    a = parse_args()
    return Args(
        neuroglancer=a.neuroglancer,
        annotation_manifest=a.annotation_manifest,
    )


def _resolve_paths(args: Args) -> InputPaths:
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


def _find_asset_info(paths: InputPaths) -> AssetInfo:
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


def _load_references() -> ReferenceVolumes:
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


def _inspect_image_geometry(reg: RegistrationInfo) -> Geometry:
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


def _prepare_result_dirs(mouse_id: str, results_root: Path) -> OutputDirs:
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


def _determine_desired_level(
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


def _write_registration_channel_outputs(
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


def _process_additional_channels_pipeline(
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


def _push_atlas_to_image_space(
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


# ---- Stage 7: per-row processing (probe-centric) -----------------------------


def _process_manifest_row(
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


# ---- Stage 8: optional ephys -------------------------------------------------


def _maybe_run_ephys(
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


def _process_histology_and_ephys(args: Args):
    paths = _resolve_paths(args)

    # Keep a manifest snapshot for reproducibility
    shutil.copy(paths.manifest_csv, "/results/manifest.csv")

    manifest_df = pd.read_csv(paths.manifest_csv)
    s = manifest_df["mouseid"].astype("string")
    val = s.iat[0]
    if pd.isna(val):
        raise ValueError("mouseid is missing in first row")
    mouse_id: str = val  # now definitely a str
    refs = _load_references()
    out = _prepare_result_dirs(mouse_id, paths.results_root)
    asset_info = _find_asset_info(paths)
    node, zarr_metadata = _open_zarr(asset_info.zarr_volumes.registration)
    level = _determine_desired_level(zarr_metadata, desired_voxel_size_um=25.0)

    raw_img = _write_registration_channel_outputs(
        asset_info, out, level=level, opened_zarr=(node, zarr_metadata)
    )
    a_bugged_img = _process_additional_channels_pipeline(asset_info, refs, out)
    if a_bugged_img is None:
        raw_img_domain_bugged = mimic_pipeline_zarr_to_ants(
            asset_info.zarr_volumes.registration,
            asset_info.zarr_volumes.metadata,
            asset_info.zarr_volumes.processing,
            level=level,
        )
    else:
        raw_img_domain_bugged = a_bugged_img
    _push_atlas_to_image_space(
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
        _process_manifest_row(
            row, asset_info, raw_img_stub, raw_img_stub_buggy, refs, out
        )
        _maybe_run_ephys(row, out, processed_recordings)


def main() -> None:
    """
    Orchestrate the full processing pipeline.
    """
    args = _parse_and_normalize_args()
    _process_histology_and_ephys(args)


if __name__ == "__main__":
    main()
