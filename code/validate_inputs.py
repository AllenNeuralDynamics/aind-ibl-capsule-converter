"""
Input validation for the SmartSPIM → IBL conversion pipeline.

This module provides comprehensive pre-flight checks to detect missing files,
incorrect configurations, and resource constraints before the pipeline starts
processing. This helps avoid wasting hours on a run that will fail due to
missing data or insufficient resources.

Usage:
    from validate_inputs import PipelineValidator, ValidationResult

    validator = PipelineValidator(args, paths)
    results = validator.validate_all()

    if not all(r.passed for r in results if r.severity == "error"):
        # Handle validation errors
        validator.print_summary(results)
        sys.exit(1)
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Import the data structures from the main pipeline
try:
    from extract_ephys_and_histology import (
        Args,
        InputPaths,
        ReferencePaths,
        ManifestRow,
    )
except ImportError:
    # For standalone testing
    pass


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of a single validation check.

    Attributes
    ----------
    passed : bool
        Whether the validation check passed.
    category : str
        Category of the check (e.g., "CLI Args", "Manifest CSV", "Reference Data").
    item : str
        Specific item being checked (e.g., "template_25.nii.gz", "mouseid column").
    message : str
        Human-readable message about the result.
    severity : str
        Severity level: "error" (must fix), "warning" (should fix), or "info" (informational).
    """
    passed: bool
    category: str
    item: str
    message: str
    severity: str = "error"  # "error", "warning", "info"


class PipelineValidator:
    """
    Comprehensive validation for the SmartSPIM → IBL conversion pipeline.

    This validator performs pre-flight checks on all required inputs and resources
    before the pipeline starts processing, helping to fail fast and provide
    actionable error messages.

    Parameters
    ----------
    args : Args
        CLI arguments from the pipeline.
    paths : InputPaths
        Resolved input paths (neuroglancer file, manifest, data/results roots).
    skip_resource_checks : bool, optional
        If True, skip resource-intensive checks like disk space and RAM (default: False).
    """

    def __init__(
        self,
        args: Args,
        paths: InputPaths,
        skip_resource_checks: bool = False,
    ):
        self.args = args
        self.paths = paths
        self.skip_resource_checks = skip_resource_checks
        self.ref_paths = ReferencePaths()
        self.results: list[ValidationResult] = []

    def validate_all(self) -> list[ValidationResult]:
        """
        Run all validation checks in sequence.

        Returns
        -------
        list[ValidationResult]
            List of all validation results, including errors, warnings, and info messages.
        """
        self.results = []

        # Run checks in logical order
        self.validate_cli_args()
        self.validate_manifest_structure()
        self.validate_reference_data()
        self.validate_neuroglancer_and_asset()
        self.validate_per_probe_files()
        self.validate_output_access()

        if not self.skip_resource_checks:
            self.validate_resources()

        return self.results

    def _add_result(
        self,
        passed: bool,
        category: str,
        item: str,
        message: str,
        severity: str = "error",
    ) -> None:
        """Add a validation result to the results list."""
        self.results.append(
            ValidationResult(
                passed=passed,
                category=category,
                item=item,
                message=message,
                severity=severity,
            )
        )

    # ---- Category 1: CLI Arguments ----

    def validate_cli_args(self) -> None:
        """Validate CLI arguments for basic correctness."""
        category = "CLI Args"

        # Check neuroglancer argument
        if not self.args.neuroglancer:
            self._add_result(
                False,
                category,
                "neuroglancer",
                "Neuroglancer file path is empty",
            )
        else:
            self._add_result(
                True,
                category,
                "neuroglancer",
                f"Neuroglancer path provided: {self.args.neuroglancer}",
                severity="info",
            )

        # Check annotation_manifest argument
        if not self.args.annotation_manifest:
            self._add_result(
                False,
                category,
                "annotation_manifest",
                "Annotation manifest path is empty",
            )
        else:
            self._add_result(
                True,
                category,
                "annotation_manifest",
                f"Manifest path provided: {self.args.annotation_manifest}",
                severity="info",
            )

        # Info about ephys processing
        if self.args.skip_ephys:
            self._add_result(
                True,
                category,
                "skip_ephys",
                "Ephys processing will be skipped (--skip-ephys flag set)",
                severity="info",
            )

    # ---- Category 2: Manifest CSV ----

    def validate_manifest_structure(self) -> None:
        """Validate manifest CSV file structure and required columns."""
        category = "Manifest CSV"

        # Check file exists
        if not self.paths.manifest_csv.exists():
            self._add_result(
                False,
                category,
                "file_exists",
                f"Manifest CSV not found: {self.paths.manifest_csv}",
            )
            return

        self._add_result(
            True,
            category,
            "file_exists",
            f"Manifest CSV exists: {self.paths.manifest_csv}",
            severity="info",
        )

        # Try to read the CSV
        try:
            df = pd.read_csv(self.paths.manifest_csv)
        except Exception as e:
            self._add_result(
                False,
                category,
                "readable",
                f"Failed to read manifest CSV: {e}",
            )
            return

        self._add_result(
            True,
            category,
            "readable",
            f"Manifest CSV readable ({len(df)} rows)",
            severity="info",
        )

        # Check required columns
        required_cols = [
            "mouseid",
            "sorted_recording",
            "probe_file",
            "probe_id",
            "probe_name",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self._add_result(
                False,
                category,
                "required_columns",
                f"Missing required columns: {', '.join(missing_cols)}",
            )
        else:
            self._add_result(
                True,
                category,
                "required_columns",
                "All required columns present",
                severity="info",
            )

        # Check that mouseid is consistent
        if "mouseid" in df.columns:
            unique_mice = df["mouseid"].unique()
            if len(unique_mice) > 1:
                self._add_result(
                    False,
                    category,
                    "mouseid_consistency",
                    f"Multiple mouse IDs found: {', '.join(map(str, unique_mice))}. "
                    "Pipeline currently supports single mouse per run.",
                )
            elif len(unique_mice) == 0 or pd.isna(unique_mice[0]):
                self._add_result(
                    False,
                    category,
                    "mouseid_consistency",
                    "Mouse ID is missing or NA in all rows",
                )
            else:
                self._add_result(
                    True,
                    category,
                    "mouseid_consistency",
                    f"Single mouse ID: {unique_mice[0]}",
                    severity="info",
                )

        # Check for empty required fields
        if len(df) > 0:
            for col in required_cols:
                if col in df.columns:
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        self._add_result(
                            False,
                            category,
                            f"{col}_nulls",
                            f"Column '{col}' has {null_count} null/empty values",
                            severity="warning",
                        )

    # ---- Category 3: Reference Data ----

    def validate_reference_data(self) -> None:
        """Validate that all required reference data files exist."""
        category = "Reference Data"

        # Check template
        if self.ref_paths.template_25.exists():
            self._add_result(
                True,
                category,
                "template_25",
                f"Template found: {self.ref_paths.template_25}",
                severity="info",
            )
        else:
            self._add_result(
                False,
                category,
                "template_25",
                f"Template not found: {self.ref_paths.template_25}",
            )

        # Check CCF template
        if self.ref_paths.ccf_25.exists():
            self._add_result(
                True,
                category,
                "ccf_25",
                f"CCF template found: {self.ref_paths.ccf_25}",
                severity="info",
            )
        else:
            self._add_result(
                False,
                category,
                "ccf_25",
                f"CCF template not found: {self.ref_paths.ccf_25}",
            )

        # Check CCF labels
        if self.ref_paths.ccf_labels_lateralized_25.exists():
            self._add_result(
                True,
                category,
                "ccf_labels",
                f"CCF labels found: {self.ref_paths.ccf_labels_lateralized_25}",
                severity="info",
            )
        else:
            self._add_result(
                False,
                category,
                "ccf_labels",
                f"CCF labels not found: {self.ref_paths.ccf_labels_lateralized_25}",
            )

        # Check IBL atlas directory
        if self.ref_paths.ibl_atlas_histology_path.exists():
            self._add_result(
                True,
                category,
                "ibl_atlas",
                f"IBL atlas directory found: {self.ref_paths.ibl_atlas_histology_path}",
                severity="info",
            )
        else:
            self._add_result(
                False,
                category,
                "ibl_atlas",
                f"IBL atlas directory not found: {self.ref_paths.ibl_atlas_histology_path}",
            )

    # ---- Category 4: SmartSPIM Asset Discovery ----

    def validate_neuroglancer_and_asset(self) -> None:
        """Validate Neuroglancer file and discover SmartSPIM asset structure."""
        category = "Asset Discovery"

        # Check Neuroglancer file exists
        if not self.paths.neuroglancer_file.exists():
            self._add_result(
                False,
                category,
                "neuroglancer_file",
                f"Neuroglancer file not found: {self.paths.neuroglancer_file}",
            )
            return

        self._add_result(
            True,
            category,
            "neuroglancer_file",
            f"Neuroglancer file exists: {self.paths.neuroglancer_file}",
            severity="info",
        )

        # Try to validate asset structure (simplified check)
        # Full validation happens in _find_asset_info, but we can do basic checks
        try:
            from aind_s3_cache.json_utils import get_json
            from aind_zarr_utils.neuroglancer import get_image_sources

            ng_data = get_json(str(self.paths.neuroglancer_file))
            sources = get_image_sources(ng_data)

            if not sources:
                self._add_result(
                    False,
                    category,
                    "image_sources",
                    "No image sources found in Neuroglancer file",
                )
                return

            self._add_result(
                True,
                category,
                "image_sources",
                f"Found {len(sources)} image source(s) in Neuroglancer file",
                severity="info",
            )

            # Try to locate asset path
            from aind_s3_cache.uri_utils import as_pathlike
            from aind_zarr_utils.pipeline_transformed import _asset_from_zarr_pathlike

            a_zarr_uri = next(iter(sources.values()), None)
            if a_zarr_uri:
                _, _, a_zarr_pathlike = as_pathlike(a_zarr_uri)
                asset_pathlike = _asset_from_zarr_pathlike(a_zarr_pathlike)
                asset_path = self.paths.data_root / asset_pathlike

                if asset_path.exists():
                    self._add_result(
                        True,
                        category,
                        "asset_path",
                        f"Asset path exists: {asset_path}",
                        severity="info",
                    )

                    # Check for OME-Zarr directory
                    zarr_path = asset_path / "image_tile_fusing" / "OMEZarr"
                    if zarr_path.exists():
                        self._add_result(
                            True,
                            category,
                            "omezarr_dir",
                            f"OME-Zarr directory exists: {zarr_path}",
                            severity="info",
                        )

                        # Check for .zarr files
                        zarr_files = list(zarr_path.glob("*.zarr"))
                        if zarr_files:
                            self._add_result(
                                True,
                                category,
                                "zarr_channels",
                                f"Found {len(zarr_files)} .zarr channel(s)",
                                severity="info",
                            )
                        else:
                            self._add_result(
                                False,
                                category,
                                "zarr_channels",
                                f"No .zarr files found in {zarr_path}",
                            )
                    else:
                        self._add_result(
                            False,
                            category,
                            "omezarr_dir",
                            f"OME-Zarr directory not found: {zarr_path}",
                        )

                    # Check for registration directory
                    reg_dir = asset_path / "image_atlas_alignment"
                    if reg_dir.exists():
                        self._add_result(
                            True,
                            category,
                            "registration_dir",
                            f"Registration directory exists: {reg_dir}",
                            severity="info",
                        )

                        # Look for moved_ls_to_ccf.nii.gz in subdirectories
                        ccf_files = list(reg_dir.glob("*/moved_ls_to_ccf.nii.gz"))
                        if ccf_files:
                            self._add_result(
                                True,
                                category,
                                "precomputed_registration",
                                f"Found precomputed CCF registration: {ccf_files[0]}",
                                severity="info",
                            )
                        else:
                            self._add_result(
                                False,
                                category,
                                "precomputed_registration",
                                f"Precomputed registration (moved_ls_to_ccf.nii.gz) not found in {reg_dir}",
                            )

                        # Check for transform files
                        affine_files = list(reg_dir.glob("*/*_0GenericAffine.mat"))
                        warp_files = list(reg_dir.glob("*/*_1InverseWarp.nii.gz"))

                        if affine_files and warp_files:
                            self._add_result(
                                True,
                                category,
                                "transform_files",
                                f"Found transform files (affine: {len(affine_files)}, warp: {len(warp_files)})",
                                severity="info",
                            )
                        else:
                            missing = []
                            if not affine_files:
                                missing.append("affine (*_0GenericAffine.mat)")
                            if not warp_files:
                                missing.append("warp (*_1InverseWarp.nii.gz)")
                            self._add_result(
                                False,
                                category,
                                "transform_files",
                                f"Missing transform files: {', '.join(missing)}",
                            )
                    else:
                        self._add_result(
                            False,
                            category,
                            "registration_dir",
                            f"Registration directory not found: {reg_dir}",
                        )
                else:
                    self._add_result(
                        False,
                        category,
                        "asset_path",
                        f"Asset path not found: {asset_path}",
                    )
        except Exception as e:
            self._add_result(
                False,
                category,
                "asset_discovery",
                f"Failed to discover asset structure: {e}",
                severity="warning",
            )

    # ---- Category 5: Per-probe Files ----

    def validate_per_probe_files(self) -> None:
        """Validate that per-probe annotation files and ephys data exist."""
        category = "Per-Probe Files"

        # Read manifest to check each probe
        if not self.paths.manifest_csv.exists():
            return  # Already reported in manifest validation

        try:
            df = pd.read_csv(self.paths.manifest_csv)
        except Exception:
            return  # Already reported in manifest validation

        for idx, row in df.iterrows():
            try:
                mr = ManifestRow.from_series(row)
            except Exception as e:
                self._add_result(
                    False,
                    category,
                    f"row_{idx}",
                    f"Failed to parse manifest row {idx}: {e}",
                    severity="warning",
                )
                continue

            # Check annotation file
            ext = "json" if mr.annotation_format == "json" else None
            if ext:
                pattern = f"*/{mr.probe_file}.{ext}"
                matches = list(self.paths.data_root.glob(pattern))

                if matches:
                    self._add_result(
                        True,
                        category,
                        f"{mr.probe_id}_annotation",
                        f"Annotation found for {mr.probe_id}: {matches[0].relative_to(self.paths.data_root)}",
                        severity="info",
                    )
                else:
                    self._add_result(
                        False,
                        category,
                        f"{mr.probe_id}_annotation",
                        f"Annotation not found for {mr.probe_id} (pattern: {pattern})",
                    )

            # Check ephys recording folder (only if not skipping ephys)
            if not self.args.skip_ephys:
                recording_folder = self.paths.data_root / mr.sorted_recording
                if recording_folder.exists():
                    self._add_result(
                        True,
                        category,
                        f"{mr.probe_id}_ephys",
                        f"Ephys folder exists for {mr.sorted_recording}",
                        severity="info",
                    )
                else:
                    self._add_result(
                        False,
                        category,
                        f"{mr.probe_id}_ephys",
                        f"Ephys folder not found: {recording_folder}",
                    )

            # Check surface_finding file if specified
            if mr.surface_finding is not None:
                surface_path = self.paths.data_root / mr.surface_finding
                if surface_path.exists():
                    self._add_result(
                        True,
                        category,
                        f"{mr.probe_id}_surface_finding",
                        f"Surface finding file exists: {surface_path}",
                        severity="info",
                    )
                else:
                    self._add_result(
                        False,
                        category,
                        f"{mr.probe_id}_surface_finding",
                        f"Surface finding file not found: {surface_path}",
                        severity="warning",
                    )

    # ---- Category 6: Output Directory Access ----

    def validate_output_access(self) -> None:
        """Validate that output directory is writable and has sufficient space."""
        category = "Output Access"

        # Check if results directory exists and is writable
        results_dir = self.paths.results_root
        if not results_dir.exists():
            self._add_result(
                False,
                category,
                "results_dir_exists",
                f"Results directory does not exist: {results_dir}",
            )
            return

        if not os.access(results_dir, os.W_OK):
            self._add_result(
                False,
                category,
                "results_writable",
                f"Results directory is not writable: {results_dir}",
            )
        else:
            self._add_result(
                True,
                category,
                "results_writable",
                f"Results directory is writable: {results_dir}",
                severity="info",
            )

        # Check available disk space
        try:
            stat = shutil.disk_usage(results_dir)
            free_gb = stat.free / (1024**3)

            if free_gb < 10:
                self._add_result(
                    False,
                    category,
                    "disk_space",
                    f"Insufficient disk space: {free_gb:.1f} GB available (need at least 10 GB)",
                )
            elif free_gb < 50:
                self._add_result(
                    True,
                    category,
                    "disk_space",
                    f"Low disk space warning: {free_gb:.1f} GB available (recommend at least 50 GB)",
                    severity="warning",
                )
            else:
                self._add_result(
                    True,
                    category,
                    "disk_space",
                    f"Sufficient disk space: {free_gb:.1f} GB available",
                    severity="info",
                )
        except Exception as e:
            self._add_result(
                False,
                category,
                "disk_space",
                f"Failed to check disk space: {e}",
                severity="warning",
            )

    # ---- Category 7: Resource Checks ----

    def validate_resources(self) -> None:
        """Validate system resources (RAM, CPU, network mounts)."""
        category = "System Resources"

        # Check available RAM
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemAvailable:"):
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024**2)

                    if mem_gb < 8:
                        self._add_result(
                            False,
                            category,
                            "ram",
                            f"Insufficient RAM: {mem_gb:.1f} GB available (need at least 8 GB)",
                        )
                    elif mem_gb < 16:
                        self._add_result(
                            True,
                            category,
                            "ram",
                            f"Low RAM warning: {mem_gb:.1f} GB available (recommend at least 16 GB)",
                            severity="warning",
                        )
                    else:
                        self._add_result(
                            True,
                            category,
                            "ram",
                            f"Sufficient RAM: {mem_gb:.1f} GB available",
                            severity="info",
                        )
                    break
        except Exception as e:
            self._add_result(
                False,
                category,
                "ram",
                f"Failed to check RAM: {e}",
                severity="warning",
            )

        # Check for network mounts (potential IO bottleneck)
        try:
            result = subprocess.run(
                ["df", "-T", str(self.paths.data_root)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    fs_type = lines[1].split()[1]
                    if fs_type in ["nfs", "nfs4", "cifs", "fuse", "fuse.s3fs"]:
                        self._add_result(
                            True,
                            category,
                            "network_mount",
                            f"Data directory is on network storage ({fs_type}). "
                            "Consider using local caching for better performance.",
                            severity="warning",
                        )
                    else:
                        self._add_result(
                            True,
                            category,
                            "network_mount",
                            f"Data directory is on local storage ({fs_type})",
                            severity="info",
                        )
        except Exception as e:
            self._add_result(
                False,
                category,
                "network_mount",
                f"Failed to check filesystem type: {e}",
                severity="warning",
            )

    # ---- Output Methods ----

    def print_summary(self, results: list[ValidationResult] | None = None) -> None:
        """
        Print a formatted summary of validation results.

        Parameters
        ----------
        results : list[ValidationResult], optional
            Results to print. If None, uses self.results.
        """
        if results is None:
            results = self.results

        # Group by category
        by_category: dict[str, list[ValidationResult]] = {}
        for r in results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)

        # Count by severity
        errors = [r for r in results if not r.passed and r.severity == "error"]
        warnings = [r for r in results if r.severity == "warning"]

        # Print header
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Print each category
        for category, cat_results in by_category.items():
            print(f"\n{category}:")
            print("-" * 80)
            for r in cat_results:
                if not r.passed and r.severity == "error":
                    icon = "❌"
                elif r.severity == "warning":
                    icon = "⚠️ "
                elif not r.passed:
                    icon = "❌"
                else:
                    icon = "✓ "

                # Indent message
                message_lines = r.message.split("\n")
                first_line = f"{icon} {r.item}: {message_lines[0]}"
                print(f"  {first_line}")
                for line in message_lines[1:]:
                    print(f"    {line}")

        # Print summary
        print("\n" + "="*80)
        if errors:
            print(f"RESULT: FAILED with {len(errors)} error(s)")
            if warnings:
                print(f"        Also {len(warnings)} warning(s)")
            print("\nPlease fix the errors above before running the pipeline.")
        elif warnings:
            print(f"RESULT: PASSED with {len(warnings)} warning(s)")
            print("\nYou can proceed, but consider addressing the warnings above.")
        else:
            print("RESULT: PASSED")
            print("\nAll checks passed! Ready to run the pipeline.")
        print("="*80 + "\n")

    def has_errors(self, results: list[ValidationResult] | None = None) -> bool:
        """
        Check if there are any error-level failures.

        Parameters
        ----------
        results : list[ValidationResult], optional
            Results to check. If None, uses self.results.

        Returns
        -------
        bool
            True if there are any error-level failures.
        """
        if results is None:
            results = self.results
        return any(not r.passed and r.severity == "error" for r in results)
