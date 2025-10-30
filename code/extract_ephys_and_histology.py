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
  --skip-ephys
      If present, skip ephys extraction (extract_continuous and extract_spikes).
  --validate-only
      If present, run validation checks only without processing. Exits after
      reporting validation results.

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

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import ants
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
from aind_zarr_utils.neuroglancer import (
    neuroglancer_annotations_to_anatomical,
)
from aind_zarr_utils.pipeline_transformed import (
    base_and_pipeline_anatomical_stub,
    base_and_pipeline_zarr_to_sitk,
)
from aind_zarr_utils.zarr import (
    _open_zarr,
    zarr_to_sitk,
)
from ants.core import ANTsImage
from extract_ephys_hist_core import (
    _BLESSED_DIRECTION,
    determine_desired_level,
    find_asset_info,
    handle_validation,
    parse_and_normalize_args,
    prepare_result_dirs,
    resolve_paths,
)
from ibl_preprocess_types import (
    Args,
    AssetInfo,
    ManifestRow,
    OutputDirs,
    ProcessResult,
    ReferencePaths,
    ReferenceVolumes,
)
from iblatlas.atlas import AllenAtlas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- Stage 1: args and input resolution -------------------------------------


def _convert_img_direction_and_write(
    img: sitk.Image, output_path: Path, direction: str = _BLESSED_DIRECTION
) -> None:
    """
    Convert an image to the specified orientation and write as compressed NRRD.
    """
    img_oriented = sitk.DICOMOrient(img, direction)
    sitk.WriteImage(img_oriented, str(output_path), useCompression=True)


def _copy_registration_channel_ccf_reorient(
    asset_info: AssetInfo, outputs: OutputDirs
) -> None:
    if not asset_info.registration_in_ccf_precomputed.exists():
        raise FileNotFoundError(
            "Precomputed registration in CCF not found: "
            f"{asset_info.registration_in_ccf_precomputed}"
        )
    # Save the precomputed CCF-space image as a nrrd
    ccf_img = sitk.ReadImage(str(asset_info.registration_in_ccf_precomputed))
    img_in_ccf_dst = outputs.histology_ccf / "histology_registration.nrrd"
    _convert_img_direction_and_write(ccf_img, img_in_ccf_dst)


def _write_registration_channel_images(
    asset_info: AssetInfo,
    outputs: OutputDirs,
    *,
    level: int = 3,
    opened_zarr: tuple[Any, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """
    Write registration-channel outputs to CCF and image space.
    """
    reg_zarr = asset_info.zarr_volumes.registration
    if opened_zarr is None:
        zarr_node, zarr_metadata = _open_zarr(reg_zarr)
    else:
        zarr_node, zarr_metadata = opened_zarr
    # Get the minimum voxel spatial dimension for each multiscale level

    metadata = asset_info.zarr_volumes.metadata
    processing = asset_info.zarr_volumes.processing
    raw_img, pipeline_raw_img = base_and_pipeline_zarr_to_sitk(
        reg_zarr,
        metadata,
        processing,
        level=level,
        opened_zarr=(zarr_node, zarr_metadata),
    )
    raw_img_dst = outputs.histology_img / "histology_registration.nrrd"
    _convert_img_direction_and_write(raw_img, raw_img_dst)
    del raw_img
    bugged_img_dst = (
        outputs.histology_img / "histology_registration_pipeline.nrrd"
    )
    _convert_img_direction_and_write(pipeline_raw_img, bugged_img_dst)
    return raw_img_dst, bugged_img_dst


def _process_additional_channels_pipeline(
    pipeline_histology_space_img: ANTsImage,
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
    level: int = 3,
) -> None:
    """
    Pipeline channel handling:
      1) Load non-alignment OME-Zarr channels at highest level on the template grid,
         write image-space NIfTI (<channel>.nii.gz) into outputs.histology_img.
      2) Chain ls->template then template->CCF transforms; write CCF-space
         histology_<channel>.nrrd into outputs.histology_ccf (fixed grid = CCF).
    """

    for zarr_path in asset_info.zarr_volumes.additional:
        # Load the image in the vanilla space
        ch_str = Path(zarr_path).stem
        img_raw = zarr_to_sitk(
            zarr_path, asset_info.zarr_volumes.metadata, level=level
        )
        # Need to save everything in IRP orientation for IBL ephys gui
        channel_dst = outputs.histology_img / f"{ch_str}.nrrd"
        _convert_img_direction_and_write(img_raw, channel_dst)
        del img_raw
        # Need this image in ANTs format for transform application
        # Unfortunately, going through disk is one of the simpler ways to do
        # this
        ants_hist_img = ants.image_read(str(channel_dst), pixeltype=None)  # type: ignore
        # Mutates in place. ants_hist_img will now be in pipeline space
        # Importantly, pipeline_histology_space_img is also IRP!
        ants.copy_image_info(pipeline_histology_space_img, ants_hist_img)

        # Map to CCF using existing pipeline transforms
        ch_in_ccf = ants.apply_transforms(
            refs.ccf_25,
            ants_hist_img,
            asset_info.pipeline_registration_chains.img_tx_str,
            whichtoinvert=asset_info.pipeline_registration_chains.img_tx_inverted,
        )
        ch_in_ccf_dst = outputs.histology_ccf / f"histology_{ch_str}.nrrd"
        ch_in_ccf_tmp_dst = Path(f"/scratch/histology-{ch_str}-ccf.nrrd")
        ants.image_write(ch_in_ccf, str(ch_in_ccf_tmp_dst))
        del ch_in_ccf
        try:
            _compress_reorient_nrrd_file(
                ch_in_ccf_tmp_dst,
                ch_in_ccf_dst,
                force_orientation=_BLESSED_DIRECTION,
            )
        finally:
            ch_in_ccf_tmp_dst.unlink(missing_ok=True)


def _apply_ccf_inverse_tx_then_fix_domain(
    ccf_space_img_moving: ANTsImage,
    pipeline_space_fixed_img: ANTsImage,
    correct_hist_domain_img: ANTsImage,
    asset_info: AssetInfo,
    **kwargs: Any,
) -> ANTsImage:
    """
    Apply the inverse pipeline (CCF->histology) transform then repair image
    domain.

    This helper maps a CCF-space image (template or labels) back into the
    mouse's native histology (SmartSPIM) space using the point
    (template->histology) transform chain inferred from the pipeline output.
    The inverse warps place the image on a buggy/intermediate spatial domain
    produced by the pipeline registration; we subsequently overwrite the
    spacing, origin, and direction with the "correct" histology image domain so
    downstream consumers (e.g. IBL ephys GUI) see consistent physical
    coordinates.

    Parameters
    ----------
    ccf_space_img_moving : ANTsImage
        Image defined in CCF/template space to be moved into histology space.
        (E.g. average template or lateralized label volume.)
    pipeline_space_fixed_img : ANTsImage
        An image in the pipeline's (buggy) histology space used as the fixed
        image for `ants.apply_transforms`. Must correspond to the same mouse
        and have the geometry expected by the pipeline transforms.
    correct_hist_domain_img : ANTsImage
        Reference histology image whose spacing, origin, and direction encode
        the desired (repaired) physical domain. These values are copied onto
        the result after transformation.
    asset_info : AssetInfo
        Container with pipeline registration chain paths; specifically
        ``asset_info.pipeline_registration_chains.pt_tx_str`` (list of
        transform filenames) and ``pt_tx_inverted`` (list of booleans
        indicating inversion state per transform).
    **kwargs : Any
        Additional keyword arguments forwarded to ``ants.apply_transforms``.
        Common values include ``interpolator="linear"`` for continuous images
        or ``interpolator="genericLabel"`` for label volumes.

    Returns
    -------
    ANTsImage
        The CCF-space input resampled into histology space with corrected
        spacing, origin, and direction (i.e., a domain-consistent image ready for
        serialization in IRP orientation if needed).

    Notes
    -----
    The pipeline domain mismatch arises because transforms were estimated on a
    preprocessed version of the histology image with altered geometry.
    """

    pt_tx_str = asset_info.pipeline_registration_chains.pt_tx_str
    pt_tx_inverted = asset_info.pipeline_registration_chains.pt_tx_inverted
    # This will be in the buggy domain, but we can fix that later
    ccf_space_img_in_hist_space: ANTsImage = ants.apply_transforms(
        fixed=pipeline_space_fixed_img,
        moving=ccf_space_img_moving,
        transformlist=pt_tx_str,
        whichtoinvert=pt_tx_inverted,
        **kwargs,
    )
    # Update the spatial domain to match the real image
    ccf_space_img_in_hist_space.set_spacing(correct_hist_domain_img.spacing)
    ccf_space_img_in_hist_space.set_origin(correct_hist_domain_img.origin)
    ccf_space_img_in_hist_space.set_direction(
        correct_hist_domain_img.direction
    )
    return ccf_space_img_in_hist_space


def _compress_reorient_nrrd_file(
    input_path: Path, output_path: Path, force_orientation: str | None = None
) -> None:
    img = sitk.ReadImage(str(input_path))
    orientation_code = (
        sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            img.GetDirection()
        )
    )
    if force_orientation is not None and orientation_code != force_orientation:
        logger.info(
            f"Reorienting {input_path} from {orientation_code} to {force_orientation}"
        )
        out_img = sitk.DICOMOrient(img, force_orientation)
    else:
        out_img = img
    # Write to temporary compressed nrrd
    temp_output_path = output_path.with_suffix(".temp.nrrd")
    sitk.WriteImage(out_img, str(temp_output_path), useCompression=True)
    # Replace original file with compressed version
    temp_output_path.replace(output_path)


def _transform_ccf_to_image_space(
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    raw_hist_img: ANTsImage,
    pipeline_hist_domain_img: ANTsImage,
    outputs: OutputDirs,
) -> None:
    """
    Transform CCF template and labels into native image space.
    """
    # point transforms are inverse of image transforms
    # Need to use buggy domain to use the ccf transforms
    # The IBL ephys gui expects IRP orientation. Reorienting the hist-domain
    # image to be IRP will ensure the transformed CCF images are also IRP.

    ccf_in_hist_img = _apply_ccf_inverse_tx_then_fix_domain(
        refs.ccf_25,
        pipeline_space_fixed_img=pipeline_hist_domain_img,
        correct_hist_domain_img=raw_hist_img,
        asset_info=asset_info,
    )
    ccf_in_hist_img_tmp_dst = Path("/scratch/histology-ccf-in-mouse.nrrd")
    ccf_in_hist_img_path = outputs.histology_img / "ccf_in_mouse.nrrd"
    ants.image_write(ccf_in_hist_img, str(ccf_in_hist_img_tmp_dst))
    del ccf_in_hist_img
    try:
        _compress_reorient_nrrd_file(
            ccf_in_hist_img_tmp_dst,
            ccf_in_hist_img_path,
            force_orientation=_BLESSED_DIRECTION,
        )
    finally:
        ccf_in_hist_img_tmp_dst.unlink(missing_ok=True)


def _transform_ccf_labels_to_image_space(
    asset_info: AssetInfo,
    ref_paths: ReferencePaths,
    raw_hist_img: ANTsImage,
    pipeline_hist_domain_img: ANTsImage,
    outputs: OutputDirs,
) -> None:
    # Load the lateralized image
    ccf_labels_lateralized_25 = ants.image_read(
        str(ref_paths.ccf_labels_lateralized_25),
        pixeltype=None,  # type: ignore
    )
    ccf_labels_in_hist_img = _apply_ccf_inverse_tx_then_fix_domain(
        ccf_labels_lateralized_25,
        pipeline_space_fixed_img=pipeline_hist_domain_img,
        correct_hist_domain_img=raw_hist_img,
        asset_info=asset_info,
        interpolator="genericLabel",
    )
    del ccf_labels_lateralized_25
    ccf_labels_in_hist_img_tmp_dst = Path(
        "/scratch/histology-ccf-labels-in-mouse.nrrd"
    )
    ccf_labels_in_hist_img_path = (
        outputs.histology_img / "labels_in_mouse.nrrd"
    )
    ants.image_write(
        ccf_labels_in_hist_img,
        str(ccf_labels_in_hist_img_tmp_dst),
    )
    del ccf_labels_in_hist_img
    try:
        _compress_reorient_nrrd_file(
            ccf_labels_in_hist_img_tmp_dst,
            ccf_labels_in_hist_img_path,
            force_orientation=_BLESSED_DIRECTION,
        )
    finally:
        ccf_labels_in_hist_img_tmp_dst.unlink(missing_ok=True)


# ---- Stage 7: per-row processing (probe-centric) -----------------------------


def _process_manifest_row(
    row: ManifestRow,
    asset_info: AssetInfo,
    hist_stub: sitk.Image,
    hist_stub_buggy: sitk.Image,
    ibl_atlas: AllenAtlas,
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
    # -- convert to IBL ML/AP/DV using ibl_atlas.ccf2xyz --
    # -- write JSONs under outputs.bregma_xyz and copy into GUI folder --
    # 1) Locate annotation (JSON default)
    ext = "json" if row.annotation_format == "json" else None
    if ext is None:
        return ProcessResult(
            row.probe_id,
            row.recording_id,
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
        1_000_000.0 * ibl_atlas.ccf2xyz(ccf_mlapdv_um, ccf_order="mlapdv")
    )

    # Image-space xyz-picks (µm), matching original math
    xyz_img = 1000.0 * convert_coordinate_system(
        probe_pts, src_coord="LPS", dst_coord="RAS"
    )

    xyz_picks_image = {"xyz_picks": xyz_img.tolist()}
    xyz_picks_ccf = {"xyz_picks": bregma_mlapdv_um.tolist()}

    gui_folder = row.gui_folder(outputs)
    if row.probe_shank is None:
        img_name = f"{row.probe_id}_image_space.json"
        ccf_name = f"{row.probe_id}_ccf.json"
        gui_img = "xyz_picks_image_space.json"
        gui_ccf = "xyz_picks.json"
    else:
        shank_id = int(row.probe_shank) + 1
        img_name = f"{row.probe_id}_shank{shank_id}_image_space.json"
        ccf_name = f"{row.probe_id}_shank{shank_id}_ccf.json"
        gui_img = f"xyz_picks_shank{shank_id}_image_space.json"
        gui_ccf = f"xyz_picks_shank{shank_id}.json"

    # 7) Write bregma_xyz JSONs (global per-mouse)
    (outputs.bregma_xyz / img_name).write_text(json.dumps(xyz_picks_image))
    (outputs.bregma_xyz / ccf_name).write_text(json.dumps(xyz_picks_ccf))

    # 8) Per-recording GUI artifacts
    gui_folder.mkdir(parents=True, exist_ok=True)
    (gui_folder / gui_img).write_text(json.dumps(xyz_picks_image))
    (gui_folder / gui_ccf).write_text(json.dumps(xyz_picks_ccf))

    return ProcessResult(
        probe_id=str(row.probe_id),
        recording_id=row.recording_id,
        wrote_files=True,
        skipped_reason=None,
    )


# ---- Stage 8: optional ephys -------------------------------------------------


def _maybe_run_ephys(
    row: ManifestRow, outputs: OutputDirs, processed: set[str]
) -> None:
    """
    Run ephys extraction once per unique `sorted_recording`.

    Creates (if needed) the results folder under:
        /results/<mouseid>/<recording_id>/

    Then invokes:
        extract_continuous(recording_folder, results_folder, [probe_surface_finding=...])
        extract_spikes(recording_folder, results_folder)

    Any errors from extract_continuous or extract_spikes will propagate to the caller.

    Parameters
    ----------
    row : ManifestRow
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
    recording_id = row.recording_id
    mouse_root = outputs.tracks_root.parent  # /results/<mouseid>
    # /results/<mouseid>/<recording_id>
    results_folder = mouse_root / recording_id
    results_folder.mkdir(parents=True, exist_ok=True)

    # /data/<sorted_recording>
    recording_folder = Path("/data") / sorted_rec

    # Run extraction, optionally with surface finding hint
    if row.surface_finding is not None:
        extract_continuous(
            recording_folder,
            results_folder,
            probe_surface_finding=Path("/data") / str(row.surface_finding),
        )
    else:
        extract_continuous(recording_folder, results_folder)

    extract_spikes(recording_folder, results_folder)


# ---- Orchestrator (new short main) ------------------------------------------


def _process_histology_and_ephys(args: Args):
    paths = resolve_paths(args)

    handle_validation(paths, args)

    # Keep a manifest snapshot for reproducibility
    shutil.copy(paths.manifest_csv, "/results/manifest.csv")

    manifest_df = pd.read_csv(paths.manifest_csv)
    mouse_id: str = str(manifest_df["mouseid"].astype("string").iat[0])
    ref_paths = ReferencePaths()
    ref_imgs = ReferenceVolumes.from_paths(ref_paths)
    out = prepare_result_dirs(mouse_id, paths.results_root)
    asset_info = find_asset_info(paths)
    node, zarr_metadata = _open_zarr(asset_info.zarr_volumes.registration)
    level = determine_desired_level(zarr_metadata, desired_voxel_size_um=25.0)

    _copy_registration_channel_ccf_reorient(asset_info, out)
    raw_img_path, pipeline_img_path = _write_registration_channel_images(
        asset_info, out, level=level, opened_zarr=(node, zarr_metadata)
    )
    pipeline_img_ants = ants.image_read(str(pipeline_img_path), pixeltype=None)  # type: ignore
    raw_img_ants = ants.image_read(str(raw_img_path), pixeltype=None)  # type: ignore
    _process_additional_channels_pipeline(
        pipeline_img_ants,
        asset_info,
        ref_imgs,
        out,
        level=level,
    )
    _transform_ccf_to_image_space(
        asset_info, ref_imgs, raw_img_ants, pipeline_img_ants, out
    )
    _transform_ccf_labels_to_image_space(
        asset_info, ref_paths, raw_img_ants, pipeline_img_ants, out
    )
    raw_img_stub, raw_img_stub_buggy, _ = base_and_pipeline_anatomical_stub(
        asset_info.zarr_volumes.registration,
        asset_info.zarr_volumes.metadata,
        asset_info.zarr_volumes.processing,
        opened_zarr=(node, zarr_metadata),
    )
    processed_recordings: set[str] = set()
    processed_results: list[ProcessResult] = []
    ibl_atlas = AllenAtlas(25, hist_path=ref_paths.ibl_atlas_histology_path)

    if args.skip_ephys:
        logger.info("Ephys processing disabled via --skip-ephys flag")

    for _, row in manifest_df.iterrows():
        mr = ManifestRow.from_series(row)
        result = _process_manifest_row(
            mr, asset_info, raw_img_stub, raw_img_stub_buggy, ibl_atlas, out
        )
        processed_results.append(result)
        if not result.wrote_files:
            logger.warning(
                f"Did not write files for {mr.sorted_recording}: "
                f"{result.skipped_reason}"
            )
            continue
        # Only run ephys processing if not skipped
        if not args.skip_ephys:
            _maybe_run_ephys(mr, out, processed_recordings)


def main() -> None:
    """
    Orchestrate the full processing pipeline.
    """
    args = parse_and_normalize_args()
    _process_histology_and_ephys(args)


if __name__ == "__main__":
    main()
