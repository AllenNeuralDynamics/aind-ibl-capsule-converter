"""Code Ocean capsule entry point — thin wrapper around the library."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from aind_ibl_ephys_alignment_preprocessing.types import PipelineConfig
from extract_ephys_hist_core import parse_and_normalize_args, resolve_paths


def main() -> None:
    """Parse CLI args, build PipelineConfig, and dispatch to library."""
    print("whoami:", os.geteuid() if hasattr(os, "geteuid") else "n/a")
    print(
        "HOME:", os.environ.get("HOME"), "expanduser:", os.path.expanduser("~")
    )
    print("exe:", sys.executable)
    print("cwd:", os.getcwd())
    print("sys.path[0:3]:", sys.path[:3])

    args = parse_and_normalize_args()

    if args.annotation_manifest is None and args.neuroglancer is None:
        print(
            "No --manifest or --neuroglancer provided; nothing to do.\n"
            "\n"
            "This capsule converts AIND Neuroglancer histology + ephys data\n"
            "into IBL ALF format. To process data, attach the input assets\n"
            "and set both parameters in the app panel:\n"
            "  --manifest      path (relative to /data) to the manifest CSV\n"
            "  --neuroglancer  path (relative to /data) to the Neuroglancer\n"
            "                  probe JSON\n"
            "See README.md for the manifest format and required assets.\n"
            "\n"
            "Exiting 0 so a reproducible run with no assets attached "
            "succeeds (release gate).",
        )
        sys.exit(0)

    if args.annotation_manifest is None or args.neuroglancer is None:
        missing = (
            "--manifest" if args.annotation_manifest is None else "--neuroglancer"
        )
        print(
            f"Error: {missing} not set. Both --manifest and --neuroglancer "
            "must be provided to run the conversion.",
            file=sys.stderr,
        )
        sys.exit(2)

    paths = resolve_paths(args)

    config = PipelineConfig(
        data_root=paths.data_root,
        results_root=paths.results_root,
        scratch_root=Path("/scratch"),
        neuroglancer_file=paths.neuroglancer_file,
        manifest_csv=paths.manifest_csv,
        skip_ephys=args.skip_ephys,
    )

    from aind_ibl_ephys_alignment_preprocessing.validation import (
        PipelineValidator,
    )

    validator = PipelineValidator(config)
    results = validator.validate_all()
    validator.print_summary(results)
    if validator.has_errors(results):
        sys.exit(1)
    if args.validate_only:
        sys.exit(0)

    if args.run_async:
        from aind_ibl_ephys_alignment_preprocessing._async.pipeline import (
            run_pipeline_async,
        )

        asyncio.run(run_pipeline_async(config))
    else:
        from aind_ibl_ephys_alignment_preprocessing.pipeline import (
            run_pipeline,
        )

        run_pipeline(config)


if __name__ == "__main__":
    main()
