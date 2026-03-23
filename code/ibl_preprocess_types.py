"""Capsule-only types and re-exports from the library.

All shared dataclasses now live in ``aind_ibl_ephys_alignment_preprocessing.types``.
This module re-exports them for backward compatibility with existing capsule code,
and keeps ``Args`` which is capsule-only.
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export everything from the library
from aind_ibl_ephys_alignment_preprocessing.types import (  # noqa: F401
    AssetInfo,
    InputPaths,
    ManifestRow,
    OutputDirs,
    PipelineRegistrationInfo,
    ProcessResult,
    ReferencePaths,
    ReferenceVolumes,
    ZarrPaths,
)


@dataclass(frozen=True)
class Args:
    """Capsule-only CLI arguments."""

    neuroglancer: str
    annotation_manifest: str
    skip_ephys: bool = False
    validate_only: bool = False
    run_async: bool = True
