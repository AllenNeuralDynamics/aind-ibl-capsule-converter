from __future__ import annotations

import asyncio

from extract_ephys_and_hist_async import (
    _process_histology_and_ephys_async,
)
from extract_ephys_and_histology import _process_histology_and_ephys
from extract_ephys_hist_core import parse_and_normalize_args
import os, sys

def main() -> None:
    """
    Orchestrate the full processing pipeline.
    """
    print("whoami:", os.geteuid() if hasattr(os, "geteuid") else "n/a")
    print("HOME:", os.environ.get("HOME"), "expanduser:", os.path.expanduser("~"))
    print("exe:", sys.executable)
    print("cwd:", os.getcwd())
    print("sys.path[0:3]:", sys.path[:3])
    args = parse_and_normalize_args()
    if args.run_async:
        asyncio.run(_process_histology_and_ephys_async(args))
    else:
        _process_histology_and_ephys(args)


if __name__ == "__main__":
    main()
