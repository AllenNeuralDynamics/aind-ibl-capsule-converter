#!/usr/bin/env python
"""
Profiling wrapper for extract_ephys_and_histology.py

This script wraps the main pipeline with timing instrumentation without
modifying the original source code. It uses function wrapping to inject
timing measurements at key pipeline stages.

Usage:
    python run_profiled.py --neuroglancer path/to/file.json --manifest path/to/manifest.csv

    # Or use the same arguments as the original script
    python run_profiled.py  # Uses defaults

Output:
    - Console log with timing information
    - /results/performance_timing.csv - Detailed timing data
    - /results/performance_summary.txt - Human-readable summary
"""

from __future__ import annotations

import atexit
import functools
import logging
from pathlib import Path

# Import the profiler before importing the module to patch
import performance_profiler as prof

# Configure logging to see timing output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Import the module we want to patch
import extract_ephys_and_histology as pipeline


# Register cleanup to save results at exit
def save_results():
    """Save profiling results before exit."""
    results_dir = Path("/results")
    if results_dir.exists():
        prof.log_summary()
        prof.save_timing_csv(results_dir / "performance_timing.csv")
    else:
        # Local run, save to current directory
        prof.log_summary()
        prof.save_timing_csv(Path("performance_timing.csv"))


atexit.register(save_results)


# Patch key functions with timing wrappers
def patch_module():
    """Wrap key functions in the pipeline module with timing instrumentation."""

    # Stage 1: Path resolution (should be fast)
    pipeline.resolve_paths = prof.timing_decorator("1.resolve_paths")(
        pipeline.resolve_paths
    )

    # Stage 2: Asset discovery (may be slow - S3 access)
    pipeline.find_asset_info = prof.timing_decorator("2.find_asset_info")(
        pipeline.find_asset_info
    )

    # Stage 3: Result directory creation (should be fast)
    pipeline.prepare_result_dirs = prof.timing_decorator(
        "3.prepare_result_dirs"
    )(pipeline.prepare_result_dirs)

    # Stage 4: Registration channel outputs (likely slow - Zarr read + write)
    orig_write_reg = pipeline._write_registration_channel_outputs

    @functools.wraps(orig_write_reg)
    def write_reg_timed(*args, **kwargs):
        with prof.timing_context(
            "4.write_registration_channel", stage="registration"
        ):
            return orig_write_reg(*args, **kwargs)

    pipeline._write_registration_channel_outputs = write_reg_timed

    # Stage 5: Additional channels processing (likely very slow - multiple channels)
    orig_add_channels = pipeline._process_additional_channels_pipeline

    @functools.wraps(orig_add_channels)
    def add_channels_timed(*args, **kwargs):
        asset_info = args[0]
        n_channels = len(asset_info.zarr_volumes.additional)
        with prof.timing_context(
            "5.process_additional_channels", num_channels=n_channels
        ):
            return orig_add_channels(*args, **kwargs)

    pipeline._process_additional_channels_pipeline = add_channels_timed

    # Stage 6: CCF to image space transform (likely slow - ANTs)
    pipeline._transform_ccf_to_image_space = prof.timing_decorator(
        "6.transform_ccf_to_image"
    )(pipeline._transform_ccf_to_image_space)

    # Stage 7: Per-probe processing
    orig_process_row = pipeline._process_manifest_row

    @functools.wraps(orig_process_row)
    def process_row_timed(row, *args, **kwargs):
        probe_id = row.probe_id
        with prof.timing_context("7.process_probe", probe_id=probe_id):
            return orig_process_row(row, *args, **kwargs)

    pipeline._process_manifest_row = process_row_timed

    # Stage 8: Ephys extraction (very slow when present)
    orig_ephys = pipeline._maybe_run_ephys

    @functools.wraps(orig_ephys)
    def ephys_timed(row, *args, **kwargs):
        recording_id = row.recording_id
        with prof.timing_context(
            "8.ephys_extraction", recording_id=recording_id
        ):
            return orig_ephys(row, *args, **kwargs)

    pipeline._maybe_run_ephys = ephys_timed

    # Wrap reference volume loading (may be slow - first time loads CCF)
    orig_from_paths = pipeline.ReferenceVolumes.from_paths

    @classmethod
    def from_paths_timed(cls, paths):
        with prof.timing_context("ref.load_ccf_volumes"):
            return orig_from_paths(paths)

    pipeline.ReferenceVolumes.from_paths = from_paths_timed

    # Wrap AllenAtlas initialization (may be slow)
    from iblatlas.atlas import AllenAtlas

    orig_atlas_init = AllenAtlas.__init__

    def atlas_init_timed(self, res_um=25, **kwargs):
        with prof.timing_context("ref.init_allen_atlas", resolution_um=res_um):
            return orig_atlas_init(self, res_um=res_um, **kwargs)

    AllenAtlas.__init__ = atlas_init_timed

    # Wrap ANTs operations if we want fine-grained timing
    try:
        import ants

        orig_apply_transforms = ants.apply_transforms

        def apply_transforms_timed(fixed, moving, transformlist, **kwargs):
            with prof.timing_context("ants.apply_transforms"):
                return orig_apply_transforms(
                    fixed, moving, transformlist, **kwargs
                )

        ants.apply_transforms = apply_transforms_timed

        orig_image_write = ants.image_write

        def image_write_timed(image, filename, **kwargs):
            size_mb = image.numpy().nbytes / (1024 * 1024)
            with prof.timing_context(
                "ants.image_write",
                file=Path(filename).name,
                size_mb=f"{size_mb:.1f}",
            ):
                return orig_image_write(image, filename, **kwargs)

        ants.image_write = image_write_timed

    except ImportError:
        logger.warning("Could not import ants for fine-grained timing")

    # Wrap zarr operations
    try:
        from aind_zarr_utils.zarr import zarr_to_ants

        orig_zarr_to_ants = zarr_to_ants

        def zarr_to_ants_timed(zarr_path, *args, **kwargs):
            path_str = str(zarr_path)
            channel = (
                Path(path_str).stem
                if isinstance(zarr_path, (str, Path))
                else "unknown"
            )
            with prof.timing_context("zarr.zarr_to_ants", channel=channel):
                return orig_zarr_to_ants(zarr_path, *args, **kwargs)

        # Monkey-patch the module
        import aind_zarr_utils.zarr

        aind_zarr_utils.zarr.zarr_to_ants = zarr_to_ants_timed

    except ImportError:
        logger.warning("Could not import aind_zarr_utils for Zarr timing")

    logger.info("✓ Pipeline functions instrumented for profiling")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("PROFILING RUN - Performance Monitoring Enabled")
    logger.info("=" * 80)

    # Patch the module
    patch_module()

    # Start overall timing
    overall_start = prof.time_start("TOTAL_PIPELINE")

    # Run the original main function
    try:
        pipeline.main()
    finally:
        prof.time_end(overall_start, "TOTAL_PIPELINE")

    logger.info("=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info("=" * 80)
