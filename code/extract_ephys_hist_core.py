"""
Capsule-only: CLI argument parsing, path resolution, and validation wrapper.

All pipeline logic has been moved to the ``aind_ibl_ephys_alignment_preprocessing``
library.  This module keeps only Code Ocean-specific concerns (argparse defaults,
``/data``/``/results`` hardcoding, ``sys.exit``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ibl_preprocess_types import Args, InputPaths

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
    """Parse CLI args."""
    a = parse_args()
    return Args(
        neuroglancer=a.neuroglancer,
        annotation_manifest=a.annotation_manifest,
        skip_ephys=bool(a.skip_ephys),
        validate_only=bool(a.validate_only),
        run_async=bool(a.run_async),
    )


def resolve_paths(args: Args) -> InputPaths:
    """Resolve inputs to absolute /data and /results paths."""
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
