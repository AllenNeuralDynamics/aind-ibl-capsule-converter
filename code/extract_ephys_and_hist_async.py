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

import asyncio
import faulthandler
import json
import logging
import os
import shutil
import signal
import threading
from asyncio import Task
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
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
from aind_ephys_ibl_gui_conversion.histology import create_slicer_fcsv
from aind_registration_utils.ants import apply_ants_transforms_to_point_arr
from aind_zarr_utils import neuroglancer_annotations_to_anatomical
from aind_zarr_utils.pipeline_transformed import (
    base_and_pipeline_anatomical_stub,
    base_and_pipeline_zarr_to_sitk,
)
from aind_zarr_utils.zarr import _open_zarr, zarr_to_sitk
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
from filelock import FileLock
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
faulthandler.enable()  # dump stacks on fatal errors
os.environ.setdefault("PYTHONASYNCIODEBUG", "1")

EPROCS = int(os.environ.get("EPROCS", "8"))  # tune for your box
IO_THREADS = 40
# ---- Stage 1: args and input resolution -------------------------------------


def _thread_excepthook(args: threading.ExceptHookArgs):
    logging.error(
        "Thread %s crashed",
        args.thread.name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _thread_excepthook


async def to_thread_logged(fn, *a, **kw):
    def _wrap():
        try:
            return fn(*a, **kw)
        except Exception:
            logging.exception("Threaded call failed: %r%r%r", fn, a, kw)
            raise

    return await asyncio.to_thread(_wrap)


def _asyncio_exception_handler(loop, context):
    # VS Code sometimes misses background task exceptions unless surfaced
    msg = context.get("message")
    exc = context.get("exception")
    logging.error("Asyncio exception: %s", msg or exc, exc_info=exc)
    # Re-raise into the loop so the debugger treats it as uncaught:
    if exc:
        loop.call_soon_threadsafe(lambda: (_ for _ in ()).throw(exc))


def _maybe_semaphore(
    limit: int | None,
) -> asyncio.BoundedSemaphore | nullcontext:
    """Create semaphore if limit is set, else return no-op context manager."""
    return (
        asyncio.BoundedSemaphore(limit) if limit is not None else nullcontext()
    )


class IOLimits:
    def __init__(
        self,
        scratch: int | None = None,
        results: int | None = None,
        data: int | None = None,
    ):
        self._lanes = {
            "/scratch": _maybe_semaphore(scratch),  # NVMe
            "/results": _maybe_semaphore(results),  # EBS?
            "/data": _maybe_semaphore(data),  # S3/FUSE
        }

    def lane_for(self, path: str):
        # Fast path: match by prefix; fall back to the tighter lane.
        for root, sem in self._lanes.items():
            if path.startswith(root):
                return sem
        # default lane if you also touch /data or others
        return self._lanes["/results"]


class Limits:
    ephys: asyncio.BoundedSemaphore | nullcontext
    registration: asyncio.BoundedSemaphore | nullcontext
    manifest_rows: asyncio.BoundedSemaphore | nullcontext
    io: IOLimits
    # Store original numeric values for subprocess serialization
    max_ephys: int | None
    max_registration: int | None
    max_manifest_rows: int | None
    max_scratch: int | None
    max_results: int | None
    max_data: int | None

    def __init__(
        self,
        max_ephys: int | None = 2,
        max_registration: int | None = 1,
        max_manifest_rows: int | None = None,
        max_scratch: int | None = None,
        max_results: int | None = None,
        max_data: int | None = None,
    ):
        # Store numeric values
        self.max_ephys = max_ephys
        self.max_registration = max_registration
        self.max_manifest_rows = max_manifest_rows
        self.max_scratch = max_scratch
        self.max_results = max_results
        self.max_data = max_data
        # Create semaphores
        self.ephys = _maybe_semaphore(max_ephys)
        self.registration = _maybe_semaphore(max_registration)
        self.manifest_rows = _maybe_semaphore(max_manifest_rows)
        self.io = IOLimits(max_scratch, max_results, max_data)


async def io_to_thread_on(
    limits: Limits, target_path: str, fn, *args, **kwargs
):
    async with limits.io.lane_for(str(target_path)):
        return await to_thread_logged(fn, *args, **kwargs)


def _run_ephys_sync(mr: ManifestRow, out: OutputDirs) -> None:
    """
    Single recording ephys in a separate *process*.
    Idempotent via a disk marker and file lock.
    """
    sorted_rec = mr.sorted_recording
    recording_id = mr.recording_id
    results_folder = out.tracks_root.parent / recording_id
    results_folder.mkdir(parents=True, exist_ok=True)

    done = results_folder / ".ephys.done"
    lock = FileLock(str(results_folder.with_suffix(".lock")))
    with lock:
        if done.exists():
            logger.info(f"[Ephys {recording_id}] Skipping (already complete)")
            return
        logger.info(
            f"[Ephys {recording_id}] Starting extraction (process pool)"
        )
        recording_folder = Path("/data") / sorted_rec
        if mr.surface_finding is not None:
            extract_continuous(
                recording_folder,
                results_folder,
                probe_surface_finding=Path("/data") / str(mr.surface_finding),
            )
        else:
            extract_continuous(recording_folder, results_folder)
        extract_spikes(recording_folder, results_folder)
        done.write_text("ok")
        logger.info(f"[Ephys {recording_id}] Completed")


@dataclass
class EphysCoordinator:
    pool: ProcessPoolExecutor
    max_inflight: int = 4
    _tasks: dict[str, Task] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_inflight)

    async def ensure(self, key: str, run_sync, *args, **kwargs) -> None:
        """
        Single-flight: ensure exactly one process-pooled run per `key`.
        Others await the same Task. Also bounded by `_sem`.
        """
        async with self._lock:
            t = self._tasks.get(key)
            if t is None:

                async def _runner():
                    async with self._sem:
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(
                            self.pool, partial(run_sync, *args, **kwargs)
                        )

                t = asyncio.create_task(_runner(), name=f"ephys-{key}")

                # cleanup on completion or cancellation
                def _cleanup(_):
                    # keep successful tasks if you want caching; else always pop
                    self._tasks.pop(key, None)

                t.add_done_callback(_cleanup)
                self._tasks[key] = t
        # Propagate exceptions to callers
        await t


async def _compress_reorient_nrrd_file_async(
    input_path: Path,
    output_path: Path,
    limits: Limits,
    force_orientation: str | None = None,
) -> None:
    logger.info(f"Reading {input_path} for compression and reorientation")
    img = await io_to_thread_on(
        limits, str(input_path), sitk.ReadImage, str(input_path)
    )
    orientation_code = (
        sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            img.GetDirection()
        )
    )
    if force_orientation is not None and orientation_code != force_orientation:
        logger.info(
            f"Reorienting {input_path} from {orientation_code} to {force_orientation}"
        )
        out_img = await to_thread_logged(
            sitk.DICOMOrient, img, force_orientation
        )
    else:
        out_img = img
    # Write to temporary compressed nrrd
    temp_output_path = output_path.with_suffix(".temp.nrrd")

    logger.info(f"Writing {input_path} for compression and reorientation")
    await io_to_thread_on(
        limits,
        str(temp_output_path),
        sitk.WriteImage,
        out_img,
        str(temp_output_path),
        useCompression=True,
    )
    # Replace original file with compressed version

    logger.info(f"Replacing {input_path} for compression and reorientation")
    await io_to_thread_on(
        limits, str(output_path), temp_output_path.replace, output_path
    )


async def _convert_img_to_direction_and_write_async(
    img: sitk.Image,
    dst_path: Path | str,
    limits: Limits,
    direction: str = _BLESSED_DIRECTION,
) -> None:
    logger.info(
        f"[Histology] Converting image for {dst_path} to {direction} orientation"
    )
    img_oriented = await to_thread_logged(sitk.DICOMOrient, img, direction)
    logger.info(f"[Histology] Writing image for {dst_path} to disk")
    await io_to_thread_on(
        limits,
        str(dst_path),
        sitk.WriteImage,
        img_oriented,
        str(dst_path),
        useCompression=True,
    )
    logger.info(f"[Histology] Done writing image for {dst_path} to disk")


async def _copy_registration_channel_ccf_reorient_async(
    asset_info: AssetInfo,
    outputs: OutputDirs,
    limits: Limits,
) -> None:
    logger.info("[CCF Copy] Copying precomputed CCF registration to results")
    if not asset_info.registration_in_ccf_precomputed.exists():
        raise FileNotFoundError(
            "Precomputed registration in CCF not found: "
            f"{asset_info.registration_in_ccf_precomputed}"
        )
    # Save the precomputed CCF-space image as a nrrd
    ccf_img = await io_to_thread_on(
        limits,
        str(asset_info.registration_in_ccf_precomputed),
        sitk.ReadImage,
        str(asset_info.registration_in_ccf_precomputed),
    )
    logger.info("[CCF Copy] Read precomupted CCF registration image")
    ccf_img_dest = str(outputs.histology_ccf / "histology_registration.nrrd")
    await _convert_img_to_direction_and_write_async(
        ccf_img, ccf_img_dest, limits
    )
    logger.info(
        "[CCF Copy] Completed: histology_registration.nrrd in CCF space"
    )


async def _write_registration_channel_images_async(
    asset_info: AssetInfo,
    outputs: OutputDirs,
    limits: Limits,
    *,
    level: int = 3,
    opened_zarr: tuple[Any, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    reg_zarr = asset_info.zarr_volumes.registration
    zarr_name = Path(reg_zarr).stem
    logger.info(
        f"[Histology] Reading registration channel from zarr: {zarr_name} at level {level}"
    )
    if opened_zarr is None:
        zarr_node, zarr_metadata = await to_thread_logged(_open_zarr, reg_zarr)
    else:
        zarr_node, zarr_metadata = opened_zarr
    # Get the minimum voxel spatial dimension for each multiscale level

    metadata = asset_info.zarr_volumes.metadata
    processing = asset_info.zarr_volumes.processing
    raw_img, pipeline_raw_img = await to_thread_logged(
        base_and_pipeline_zarr_to_sitk,
        reg_zarr,
        metadata,
        processing,
        level=level,
        opened_zarr=(zarr_node, zarr_metadata),
    )
    logger.info(
        "[Histology] Registration channel loaded: raw + pipeline-transformed images"
    )
    raw_img_dst = outputs.histology_img / "histology_registration.nrrd"
    bugged_img_dst = (
        outputs.histology_img / "histology_registration_pipeline.nrrd"
    )
    logger.info(
        f"[Histology] Registration channel conversion to {_BLESSED_DIRECTION} + write started"
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            _convert_img_to_direction_and_write_async(
                raw_img, raw_img_dst, limits
            ),
            name="write-registration-raw",
        )
        tg.create_task(
            _convert_img_to_direction_and_write_async(
                pipeline_raw_img,
                bugged_img_dst,
                limits,
            ),
            name="write-registration-pipeline",
        )
    return raw_img_dst, bugged_img_dst


async def _apply_ccf_inverse_tx_then_fix_domain_async(
    ccf_space_img_moving: ANTsImage,
    pipeline_space_fixed_img: ANTsImage,
    correct_hist_domain_img: ANTsImage,
    asset_info: AssetInfo,
    limits: Limits,
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
    async with limits.registration:
        ccf_img_in_hist_space: ANTsImage = await to_thread_logged(
            ants.apply_transforms,
            fixed=pipeline_space_fixed_img,
            moving=ccf_space_img_moving,
            transformlist=pt_tx_str,
            whichtoinvert=pt_tx_inverted,
            **kwargs,
        )
    # Update the spatial domain to match the real image
    ccf_img_in_hist_space.set_spacing(correct_hist_domain_img.spacing)
    ccf_img_in_hist_space.set_origin(correct_hist_domain_img.origin)
    ccf_img_in_hist_space.set_direction(correct_hist_domain_img.direction)
    return ccf_img_in_hist_space


async def _transform_ccf_to_image_space_async(
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    raw_hist_img: ANTsImage,
    pipeline_hist_domain_img: ANTsImage,
    outputs: OutputDirs,
    limits: Limits,
) -> None:
    """
    Transform CCF template and labels into native image space.
    """
    # point transforms are inverse of image transforms
    # Need to use buggy domain to use the ccf transforms
    # The IBL ephys gui expects IRP orientation. Reorienting the hist-domain
    # image to be IRP will ensure the transformed CCF images are also IRP.

    logger.info(
        "[CCF Transform] Starting CCF template → image space transform"
    )
    logger.debug(
        "[CCF Transform] Applying ANTs inverse transform (this may take time)"
    )
    ccf_in_hist_img = await _apply_ccf_inverse_tx_then_fix_domain_async(
        refs.ccf_25,
        pipeline_space_fixed_img=pipeline_hist_domain_img,
        correct_hist_domain_img=raw_hist_img,
        asset_info=asset_info,
        limits=limits,
    )
    logger.debug("[CCF Transform] ANTs transform complete, writing output")
    ccf_in_hist_img_tmp_dst = Path("/scratch/histology-ccf-in-mouse.nrrd")
    ccf_in_hist_img_path = outputs.histology_img / "ccf_in_mouse.nrrd"
    await to_thread_logged(
        ants.image_write, ccf_in_hist_img, str(ccf_in_hist_img_tmp_dst)
    )
    del ccf_in_hist_img
    try:
        await _compress_reorient_nrrd_file_async(
            ccf_in_hist_img_tmp_dst,
            ccf_in_hist_img_path,
            limits,
            force_orientation=_BLESSED_DIRECTION,
        )
    finally:
        ccf_in_hist_img_tmp_dst.unlink(missing_ok=True)
    logger.info(f"[CCF Transform] Completed: {ccf_in_hist_img_path.name}")


async def _transform_ccf_labels_to_image_space_async(
    asset_info: AssetInfo,
    ref_paths: ReferencePaths,
    raw_hist_img: ANTsImage,
    pipeline_hist_domain_img: ANTsImage,
    outputs: OutputDirs,
    limits: Limits,
) -> None:
    # Load the lateralized image
    logger.info("[CCF Labels] Starting CCF labels → image space transform")
    ccf_labels_lateralized_25 = await to_thread_logged(
        ants.image_read,
        str(ref_paths.ccf_labels_lateralized_25),
        pixeltype=None,  # type: ignore
    )
    logger.debug(
        "[CCF Labels] Applying ANTs inverse transform with genericLabel interpolation"
    )
    ccf_labels_in_hist_img = await _apply_ccf_inverse_tx_then_fix_domain_async(
        ccf_labels_lateralized_25,
        pipeline_space_fixed_img=pipeline_hist_domain_img,
        correct_hist_domain_img=raw_hist_img,
        asset_info=asset_info,
        limits=limits,
        interpolator="genericLabel",
    )
    logger.debug("[CCF Labels] ANTs transform complete, writing output")
    del ccf_labels_lateralized_25
    ccf_labels_in_hist_img_tmp_dst = Path(
        "/scratch/histology-ccf-labels-in-mouse.nrrd"
    )
    ccf_labels_in_hist_img_path = (
        outputs.histology_img / "labels_in_mouse.nrrd"
    )
    await to_thread_logged(
        ants.image_write,
        ccf_labels_in_hist_img,
        str(ccf_labels_in_hist_img_tmp_dst),
    )
    del ccf_labels_in_hist_img
    try:
        await _compress_reorient_nrrd_file_async(
            ccf_labels_in_hist_img_tmp_dst,
            ccf_labels_in_hist_img_path,
            limits,
            force_orientation=_BLESSED_DIRECTION,
        )
    finally:
        ccf_labels_in_hist_img_tmp_dst.unlink(missing_ok=True)
    logger.info(f"[CCF Labels] Completed: {ccf_labels_in_hist_img_path.name}")


async def _process_additional_channel_pipeline_async(
    zarr_path: str,
    pipeline_histology_space_img: ANTsImage,
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
    limits: Limits,
    level: int = 3,
) -> None:
    # Load the image in the vanilla space
    ch_str = Path(zarr_path).stem
    logger.info(f"[Channel {ch_str}] Starting processing")
    img_raw = await to_thread_logged(
        zarr_to_sitk, zarr_path, asset_info.zarr_volumes.metadata, level=level
    )
    logger.info(f"[Channel {ch_str}] read from zarr complete")
    # Need to save everything in IRP orientation for IBL ephys gui
    channel_dst = outputs.histology_img / f"{ch_str}.nrrd"
    await _convert_img_to_direction_and_write_async(
        img_raw, channel_dst, limits
    )
    logger.info(
        f"[Channel {ch_str}] converted to {_BLESSED_DIRECTION} and written to disk"
    )
    # Need this image in ANTs format for transform application
    # Unfortunately, going through disk is one of the simpler ways to do
    # this
    ants_hist_img = await io_to_thread_on(
        limits,
        str(channel_dst),
        ants.image_read,
        str(channel_dst),
        pixeltype=None,
    )
    logger.info(f"[Channel {ch_str}] read into ANTs complete")

    # Mutates in place. ants_hist_img will now be in pipeline space
    # Importantly, pipeline_histology_space_img is also IRP!
    ants.copy_image_info(pipeline_histology_space_img, ants_hist_img)

    # Map to CCF using existing pipeline transforms
    logger.debug(f"[Channel {ch_str}] Applying ANTs transform to CCF")
    async with limits.registration:
        ch_in_ccf = await to_thread_logged(
            ants.apply_transforms,
            refs.ccf_25,
            ants_hist_img,
            asset_info.pipeline_registration_chains.img_tx_str,
            whichtoinvert=asset_info.pipeline_registration_chains.img_tx_inverted,
        )
    ch_in_ccf_dst = outputs.histology_ccf / f"histology_{ch_str}.nrrd"
    ch_in_ccf_tmp_dst = Path(f"/scratch/histology-{ch_str}-ccf.nrrd")
    logger.info(f"[Registered channel {ch_str}] writing to disk")
    await io_to_thread_on(
        limits,
        str(ch_in_ccf_tmp_dst),
        ants.image_write,
        ch_in_ccf,
        str(ch_in_ccf_tmp_dst),
    )
    del ch_in_ccf
    try:
        logger.info(
            f"[Registered channel {ch_str}] compressing and reorienting"
        )
        await _compress_reorient_nrrd_file_async(
            ch_in_ccf_tmp_dst,
            ch_in_ccf_dst,
            limits,
            force_orientation=_BLESSED_DIRECTION,
        )
    finally:
        ch_in_ccf_tmp_dst.unlink(missing_ok=True)
    logger.info(
        f"[Channel {ch_str}] Completed: {channel_dst.name} + histology_{ch_str}.nrrd"
    )


async def _process_additional_channels_pipeline_async(
    pipeline_histology_space_img: ANTsImage,
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
    limits: Limits,
    level: int = 3,
) -> ANTsImage | None:
    async with asyncio.TaskGroup() as tg:
        for zarr_path in asset_info.zarr_volumes.additional:
            ch_name = Path(zarr_path).stem
            tg.create_task(
                _process_additional_channel_pipeline_async(
                    zarr_path,
                    pipeline_histology_space_img,
                    asset_info,
                    refs,
                    outputs,
                    limits,
                    level=level,
                ),
                name=f"channel-{ch_name}",
            )


async def _create_volumes_async(
    asset_info: AssetInfo,
    ref_imgs: ReferenceVolumes,
    ref_paths: ReferencePaths,
    out: OutputDirs,
    node: Any,  # ome-zarr Node
    zarr_metadata: dict[str, Any],
    limits: Limits,
) -> None:
    logger.info("[Histology] Starting volume processing")
    level = determine_desired_level(zarr_metadata, desired_voxel_size_um=25.0)
    num_additional = len(asset_info.zarr_volumes.additional)
    logger.info(
        f"[Histology] Processing registration channel (level {level}) + {num_additional} additional channel(s)"
    )
    (
        raw_img_path,
        pipeline_img_path,
    ) = await _write_registration_channel_images_async(
        asset_info, out, limits, level=level, opened_zarr=(node, zarr_metadata)
    )
    logger.info(
        f"[Histology] Registration channel export complete: raw={raw_img_path.name}, pipeline={pipeline_img_path.name}"
    )
    async with asyncio.TaskGroup() as tg:
        pipeline_img_ants_task = tg.create_task(
            io_to_thread_on(
                limits,
                str(pipeline_img_path),
                ants.image_read,
                str(pipeline_img_path),
                pixeltype=None,
            ),
            name="load-ants-pipeline-img",
        )
        raw_img_ants_task = tg.create_task(
            io_to_thread_on(
                limits,
                str(raw_img_path),
                ants.image_read,
                str(raw_img_path),
                pixeltype=None,
            ),
            name="load-ants-raw-img",
        )
    pipeline_img_ants = pipeline_img_ants_task.result()
    raw_img_ants = raw_img_ants_task.result()
    logger.info(
        f"[Histology] Starting parallel processing: {num_additional} additional channels, CCF template transform, CCF labels transform"
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            _process_additional_channels_pipeline_async(
                pipeline_img_ants,
                asset_info,
                ref_imgs,
                out,
                limits,
                level=level,
            ),
            name="process-additional-channels",
        )
        tg.create_task(
            _transform_ccf_to_image_space_async(
                asset_info,
                ref_imgs,
                raw_img_ants,
                pipeline_img_ants,
                out,
                limits,
            ),
            name="transform-ccf-template-to-image",
        )
        tg.create_task(
            _transform_ccf_labels_to_image_space_async(
                asset_info,
                ref_paths,
                raw_img_ants,
                pipeline_img_ants,
                out,
                limits,
            ),
            name="transform-ccf-labels-to-image",
        )
    logger.info("[Histology] All volume processing complete")


async def read_json_in_thread(path: Path, limits: Limits):
    def _read():
        with open(path) as f:
            return json.load(f)

    return await io_to_thread_on(limits, str(path), _read)


async def _process_manifest_row_async(
    row: ManifestRow,
    asset_info: AssetInfo,
    hist_stub: sitk.Image,
    hist_stub_buggy: sitk.Image,
    ibl_atlas: AllenAtlas,
    outputs: OutputDirs,
    limits: Limits,
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
    logger.info(
        f"[Probe {row.probe_id}] Starting processing for recording: {row.recording_id}"
    )
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
    p_img = outputs.bregma_xyz / img_name
    p_ccf = outputs.bregma_xyz / ccf_name
    if p_img.exists() and p_ccf.exists():
        # Already done
        return ProcessResult(
            probe_id,
            str(row.sorted_recording),
            True,
            "Already processed",
        )
    # 2) Load NG points in histology space
    ng_data = await read_json_in_thread(ann_path, limits)
    anno_zarr = asset_info.zarr_volumes.registration
    metadata = asset_info.zarr_volumes.metadata
    async with asyncio.TaskGroup() as tg:
        probe_pt_dict_task = tg.create_task(
            to_thread_logged(
                neuroglancer_annotations_to_anatomical,
                ng_data,
                anno_zarr,
                metadata,
                layer_names=[probe_id],
                stub_image=hist_stub,
            ),
            name=f"ng-to-anat-{probe_id}",
        )
        probe_pt_dict_buggy_task = tg.create_task(
            to_thread_logged(
                neuroglancer_annotations_to_anatomical,
                ng_data,
                anno_zarr,
                metadata,
                layer_names=[probe_id],
                stub_image=hist_stub_buggy,
            ),
            name=f"ng-to-anat-buggy-{probe_id}",
        )
    probe_pt_dict, _ = probe_pt_dict_task.result()
    probe_pt_dict_buggy, _ = probe_pt_dict_buggy_task.result()
    probe_pts = probe_pt_dict.get(probe_id, None)
    if probe_pts is None:
        return ProcessResult(
            probe_id,
            str(row.sorted_recording),
            False,
            f"Probe points not found: {probe_id}",
        )
    probe_pts_buggy = probe_pt_dict_buggy[probe_id]
    num_points = len(probe_pts)
    logger.debug(
        f"[Probe {row.probe_id}] Annotation loaded: {num_points} points"
    )
    # 3) Write SPIM FCSV
    await io_to_thread_on(
        limits,
        str(outputs.spim),
        create_slicer_fcsv,
        str(outputs.spim / f"{probe_id}.fcsv"),
        probe_pts,
        direction="LPS",
    )

    # 4) Image → Template (points) via buggy pipeline transform
    # there might be a scale problem here?
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
    pts_template = await to_thread_logged(
        apply_ants_transforms_to_point_arr,
        probe_pts_buggy,
        tx_list_pt_template,
        whichtoinvert=tx_list_pt_template_invert,
    )
    await io_to_thread_on(
        limits,
        str(outputs.template),
        create_slicer_fcsv,
        str(outputs.template / f"{probe_id}.fcsv"),
        pts_template,
        direction="LPS",
    )

    # 5) Template -> CCF (points) not sure this works
    pts_ccf = await to_thread_logged(
        apply_ants_transforms_to_point_arr,
        probe_pts_buggy,
        asset_info.pipeline_registration_chains.pt_tx_str,
        whichtoinvert=asset_info.pipeline_registration_chains.pt_tx_inverted,
    )
    await io_to_thread_on(
        limits,
        str(outputs.ccf),
        create_slicer_fcsv,
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

    # 7) Write bregma_xyz JSONs (global per-mouse)
    img_json_str = json.dumps(xyz_picks_image)
    ccf_json_str = json.dumps(xyz_picks_ccf)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            io_to_thread_on(
                limits, str(p_img), p_img.write_text, img_json_str
            ),
            name=f"write-bregma-img-{probe_id}",
        )
        tg.create_task(
            io_to_thread_on(
                limits, str(p_ccf), p_ccf.write_text, ccf_json_str
            ),
            name=f"write-bregma-ccf-{probe_id}",
        )

    # 8) Per-recording GUI artifacts
    await io_to_thread_on(
        limits, str(gui_folder), gui_folder.mkdir, parents=True, exist_ok=True
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            io_to_thread_on(
                limits,
                str(gui_folder),
                (gui_folder / gui_img).write_text,
                img_json_str,
            ),
            name=f"write-gui-img-{probe_id}",
        )
        tg.create_task(
            io_to_thread_on(
                limits,
                str(gui_folder),
                (gui_folder / gui_ccf).write_text,
                ccf_json_str,
            ),
            name=f"write-gui-ccf-{probe_id}",
        )

    logger.info(
        f"[Probe {row.probe_id}] Completed: wrote xyz_picks to {len([p_img, p_ccf, gui_folder / gui_img, gui_folder / gui_ccf])} locations"
    )
    return ProcessResult(
        probe_id=str(row.probe_id),
        recording_id=row.recording_id,
        wrote_files=True,
        skipped_reason=None,
    )


async def _process_manifest_row_limit_async(
    row: ManifestRow,
    asset_info: AssetInfo,
    hist_stub: sitk.Image,
    hist_stub_buggy: sitk.Image,
    ibl_atlas: AllenAtlas,
    outputs: OutputDirs,
    limits: Limits,
) -> ProcessResult:
    async with limits.manifest_rows:
        return await _process_manifest_row_async(
            row,
            asset_info,
            hist_stub,
            hist_stub_buggy,
            ibl_atlas,
            outputs,
            limits,
        )


async def _process_manifest_row_safe_async(
    row: ManifestRow,
    asset_info: AssetInfo,
    hist_stub: sitk.Image,
    hist_stub_buggy: sitk.Image,
    ibl_atlas: AllenAtlas,
    outputs: OutputDirs,
    limits: Limits,
) -> ProcessResult:
    try:
        return await _process_manifest_row_limit_async(
            row,
            asset_info,
            hist_stub,
            hist_stub_buggy,
            ibl_atlas,
            outputs,
            limits,
        )
    except Exception as e:
        # keep going: log & convert to a skipped ProcessResult
        logger.exception(
            "Row failed: probe=%s recording=%s", row.probe_id, row.recording_id
        )
        return ProcessResult(
            probe_id=str(row.probe_id),
            recording_id=row.recording_id,
            wrote_files=False,
            skipped_reason=f"{type(e).__name__}: {e}",
        )


async def _process_manifest_async(
    manifest_df: pd.DataFrame,
    asset_info: AssetInfo,
    ibl_atlas: AllenAtlas,
    out: OutputDirs,
    node: Any,  # ome-zarr Node
    zarr_metadata: dict[str, Any],
    args: Args,
    ephys: EphysCoordinator,
    limits: Limits,
) -> list[ProcessResult]:
    num_probes = len(manifest_df)
    logger.info(
        f"[Manifest] Starting manifest processing: {num_probes} probe(s)"
    )
    raw_img_stub, raw_img_stub_buggy, _ = await to_thread_logged(
        base_and_pipeline_anatomical_stub,
        asset_info.zarr_volumes.registration,
        asset_info.zarr_volumes.metadata,
        asset_info.zarr_volumes.processing,
        opened_zarr=(node, zarr_metadata),
    )

    if args.skip_ephys:
        logger.info(
            "[Manifest] Ephys processing disabled via --skip-ephys flag"
        )
    row_tasks: list[tuple[ManifestRow, asyncio.Task[ProcessResult]]] = []

    logger.info(
        f"[Manifest] Creating parallel tasks for all {num_probes} probe(s)"
    )
    async with asyncio.TaskGroup() as tg:
        for _, row in manifest_df.iterrows():
            mr = ManifestRow.from_series(row)
            t = tg.create_task(
                _process_manifest_row_safe_async(
                    mr,
                    asset_info,
                    raw_img_stub,
                    raw_img_stub_buggy,
                    ibl_atlas,
                    out,
                    limits,
                ),
                name=f"probe-{mr.probe_id}-{mr.recording_id}",
            )
            row_tasks.append((mr, t))
            # Only run ephys processing if not skipped
            if not args.skip_ephys:
                tg.create_task(
                    ephys.ensure(
                        key=mr.sorted_recording,
                        run_sync=_run_ephys_sync,  # your sync worker that uses file locks + .done
                        mr=mr,
                        out=out,
                    ),
                    name=f"ephys-ensure-{mr.sorted_recording}",
                )

    processed_results: list[ProcessResult] = []
    for mr, rt in row_tasks:
        result = rt.result()
        processed_results.append(result)
        if not result.wrote_files:
            logger.warning(
                f"Did not write files for {mr.sorted_recording}: "
                f"{result.skipped_reason}"
            )
    num_succeeded = sum(1 for r in processed_results if r.wrote_files)
    num_failed = len(processed_results) - num_succeeded
    logger.info(
        f"[Manifest] Completed: {num_succeeded} succeeded, {num_failed} failed"
    )
    return processed_results


def _run_manifest_subprocess_sync(
    manifest_df: pd.DataFrame,
    asset_info: AssetInfo,
    ref_paths: ReferencePaths,
    out: OutputDirs,
    args: Args,
    max_ephys: int | None,
    max_manifest_rows: int | None,
    max_scratch: int | None,
    max_results: int | None,
    max_data: int | None,
) -> list[ProcessResult]:
    """
    Subprocess entry point for manifest processing.

    Recreates heavy objects (AllenAtlas, EphysCoordinator, Limits) in the
    subprocess, then runs the async manifest processing in a new event loop.

    This function is designed to be run via ProcessPoolExecutor from the main
    orchestrator, moving AllenAtlas and stub image creation into an isolated
    subprocess for better memory management.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Manifest rows to process
    asset_info : AssetInfo
        SmartSPIM asset metadata and registration paths
    ref_paths : ReferencePaths
        Paths to reference volumes (CCF, template, etc.)
    out : OutputDirs
        Output directory structure
    args : Args
        CLI arguments
    max_ephys : int | None
        Maximum concurrent ephys extractions
    max_manifest_rows : int | None
        Maximum concurrent manifest row processing
    max_scratch : int | None
        I/O concurrency limit for /scratch
    max_results : int | None
        I/O concurrency limit for /results
    max_data : int | None
        I/O concurrency limit for /data

    Returns
    -------
    list[ProcessResult]
        Processing results for each manifest row
    """

    async def _async_main():
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.set_exception_handler(_asyncio_exception_handler)
        loop.set_default_executor(ThreadPoolExecutor(max_workers=IO_THREADS))

        # Recreate Limits
        limits = Limits(
            max_ephys=max_ephys,
            max_registration=None,  # Not used in manifest processing
            max_manifest_rows=max_manifest_rows,
            max_scratch=max_scratch,
            max_results=max_results,
            max_data=max_data,
        )

        # Recreate AllenAtlas in subprocess
        logger.info("[Subprocess] Loading AllenAtlas in subprocess")
        ibl_atlas = await io_to_thread_on(
            limits,
            str(ref_paths.ibl_atlas_histology_path),
            AllenAtlas,
            25,
            hist_path=ref_paths.ibl_atlas_histology_path,
        )

        # Reopen zarr in subprocess
        logger.info("[Subprocess] Opening zarr in subprocess")
        node, zarr_metadata = await to_thread_logged(
            _open_zarr, asset_info.zarr_volumes.registration
        )

        # Create EphysCoordinator with new pool in subprocess
        ephys_pool = ProcessPoolExecutor(max_workers=EPROCS)
        ephys = EphysCoordinator(pool=ephys_pool, max_inflight=2)

        try:
            logger.info("[Subprocess] Starting manifest processing")
            result = await _process_manifest_async(
                manifest_df,
                asset_info,
                ibl_atlas,
                out,
                node,
                zarr_metadata,
                args,
                ephys,
                limits,
            )
            return result
        finally:
            ephys_pool.shutdown(wait=True)

    return asyncio.run(_async_main())


# ---- Orchestrator (new short main) ------------------------------------------
async def _process_histology_and_ephys_async(
    args: Args, max_workers: int = 40
) -> list[ProcessResult]:
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    loop.set_exception_handler(_asyncio_exception_handler)
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max_workers))
    paths = resolve_paths(args)

    # Run validation checks
    handle_validation(paths, args)
    limits = Limits()

    # Keep a manifest snapshot for reproducibility
    shutil.copy(paths.manifest_csv, "/results/manifest.csv")

    manifest_df = pd.read_csv(paths.manifest_csv)
    mouse_id: str = str(manifest_df["mouseid"].astype("string").iat[0])
    num_probes = len(manifest_df)
    logger.info(
        f"[Orchestrator] Starting histology and ephys processing for mouse: {mouse_id}"
    )
    logger.info(f"[Orchestrator] Manifest contains {num_probes} probe(s)")
    ref_paths = ReferencePaths()

    async with asyncio.TaskGroup() as tg:
        ref_imgs_task = tg.create_task(
            ReferenceVolumes.from_paths_async(ref_paths),
            name="load-ref-volumes",
        )
        asset_info_task = tg.create_task(
            to_thread_logged(find_asset_info, paths), name="find-asset-info"
        )
    ref_imgs = ref_imgs_task.result()
    asset_info = asset_info_task.result()

    out = prepare_result_dirs(mouse_id, paths.results_root)

    # Create process pool for manifest processing (runs in subprocess)
    manifest_pool = ProcessPoolExecutor(max_workers=1)

    # Open zarr for volume processing (manifest will reopen in subprocess)
    node, zarr_metadata = _open_zarr(asset_info.zarr_volumes.registration)

    skip_ephys_msg = " (ephys disabled)" if args.skip_ephys else ""
    logger.info(
        f"[Orchestrator] Launching 3 parallel task groups: volumes, manifest ({num_probes} probes) in subprocess, CCF copy{skip_ephys_msg}"
    )
    async with asyncio.TaskGroup() as tg:
        tg.create_task(
            _create_volumes_async(
                asset_info,
                ref_imgs,
                ref_paths,
                out,
                node,
                zarr_metadata,
                limits,
            ),
            name=f"create-volumes-{mouse_id}",
        )

        # Submit manifest processing to subprocess
        async def _run_manifest_in_subprocess():
            return await loop.run_in_executor(
                manifest_pool,
                _run_manifest_subprocess_sync,
                manifest_df,
                asset_info,
                ref_paths,
                out,
                args,
                limits.max_ephys,
                limits.max_manifest_rows,
                limits.max_scratch,
                limits.max_results,
                limits.max_data,
            )

        manifest_task = tg.create_task(
            _run_manifest_in_subprocess(),
            name=f"process-manifest-subprocess-{mouse_id}",
        )

        tg.create_task(
            _copy_registration_channel_ccf_reorient_async(
                asset_info, out, limits
            ),
            name=f"copy-ccf-registration-{mouse_id}",
        )
    logger.info("[Orchestrator] All parallel tasks completed")
    manifest_pool.shutdown(wait=True)
    processed_results = manifest_task.result()
    num_succeeded = sum(1 for r in processed_results if r.wrote_files)
    num_failed = len(processed_results) - num_succeeded
    logger.info(
        f"[Orchestrator] Pipeline complete: {num_succeeded} succeeded, {num_failed} failed"
    )
    return processed_results


def main() -> None:
    """
    Orchestrate the full processing pipeline.
    """
    faulthandler.register(signal.SIGUSR1)
    args = parse_and_normalize_args()
    asyncio.run(_process_histology_and_ephys_async(args))


if __name__ == "__main__":
    main()
