"""
Dataclasses for passing around pipeline parameters and paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ants
import pandas as pd


@dataclass(frozen=True)
class Args:
    neuroglancer: str
    annotation_manifest: str
    skip_ephys: bool = False
    validate_only: bool = False


@dataclass(frozen=True)
class InputPaths:
    neuroglancer_file: Path
    manifest_csv: Path
    data_root: Path
    results_root: Path


@dataclass(frozen=True)
class ReferencePaths:
    template_25: Path = Path(
        "/data/smartspim_lca_template/smartspim_lca_template_25.nii.gz"
    )
    ccf_25: Path = Path(
        "/data/allen_mouse_ccf/average_template/average_template_25.nii.gz"
    )

    ccf_labels_lateralized_25: Path = Path(
        "/data/allen_mouse_ccf_annotations_lateralized_compact/"
        "ccf_2017_annotation_25_lateralized_compact.nrrd"
    )
    ibl_atlas_histology_path: Path = Path("/data/iblatlas_allenatlas/")


@dataclass(frozen=True)
class ReferenceVolumes:
    ccf_25: ants.ANTsImage

    @classmethod
    def from_paths(cls, paths: ReferencePaths) -> ReferenceVolumes:
        ccf = ants.image_read(str(paths.ccf_25), pixeltype=None)  # type: ignore
        return cls(ccf_25=ccf)


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


@dataclass(frozen=True)
class ManifestRow:
    probe_id: str
    probe_name: str
    probe_file: str
    sorted_recording: str
    mouseid: str
    annotation_format: str = "json"
    probe_shank: int | None = None
    surface_finding: Path | None = None
    row_index: int | None = None  # provenance/debug

    @property
    def recording_id(self) -> str:
        # centralize the split logic used in multiple places
        return self.sorted_recording.split("_sorted")[0]

    def gui_folder(self, outputs: OutputDirs) -> Path:
        return outputs.tracks_root.parent / self.recording_id / self.probe_name

    @classmethod
    def from_series(cls, s: pd.Series) -> ManifestRow:
        # normalize types and handle NA robustly
        def opt_int(x):
            try:
                return int(x) if pd.notna(x) else None
            except Exception:
                return None

        def opt_path(x):
            return Path(str(x)) if pd.notna(x) and str(x) else None

        return cls(
            probe_id=str(s.get("probe_id")),
            probe_name=str(s.get("probe_name")),
            probe_file=str(s.get("probe_file")),
            sorted_recording=str(s.get("sorted_recording")),
            mouseid=str(s.get("mouseid")),
            annotation_format=str(s.get("annotation_format", "json")).lower(),
            probe_shank=opt_int(s.get("probe_shank")),
            surface_finding=opt_path(s.get("surface_finding")),
            row_index=int(s.name) if hasattr(s, "name") else None,
        )
