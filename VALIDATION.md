# Pipeline Validation Guide

## Overview

The SmartSPIM → IBL conversion pipeline includes a comprehensive validation system that checks for required files, resources, and configurations **before** starting the time-consuming processing. This helps you:

- **Fail fast**: Detect missing data in 30 seconds instead of hours into a run
- **See all errors at once**: Get a complete checklist of what's missing, not one error at a time
- **Get actionable feedback**: Clear messages about exactly what's wrong and where to find it

## Quick Start

### Pre-flight Check (Recommended)

Before running the full pipeline, validate your inputs:

```bash
python extract_ephys_and_histology.py \
    --neuroglancer Probes_561_729293_Day1and2.json \
    --manifest 729293/Manifest_Day1_2_729293.csv \
    --validate-only
```

This will check everything and exit without processing. If validation passes, you'll see:

```
================================================================================
RESULT: PASSED
All checks passed! Ready to run the pipeline.
================================================================================
```

If there are errors:

```
================================================================================
RESULT: FAILED with 3 error(s)
Please fix the errors above before running the pipeline.
================================================================================
```

### Automatic Validation

When you run the pipeline normally (without `--validate-only`), validation runs automatically at the start. The pipeline will **exit immediately** if validation fails, before doing any processing.

```bash
python extract_ephys_and_histology.py \
    --neuroglancer Probes_561_729293_Day1and2.json \
    --manifest 729293/Manifest_Day1_2_729293.csv
```

## What Gets Validated

The validator performs 7 categories of checks:

### 1. CLI Arguments ✓

Checks that command-line arguments are provided and valid:
- Neuroglancer file path is specified
- Manifest CSV path is specified
- Flags like `--skip-ephys` are recognized

**Common issues:**
- Empty or missing arguments
- Typos in flag names

### 2. Manifest CSV ✓

Validates the manifest file structure and contents:
- File exists and is readable
- Has all required columns: `mouseid`, `sorted_recording`, `probe_file`, `probe_id`, `probe_name`
- `mouseid` is consistent (pipeline supports single mouse per run)
- Required fields are not null/empty

**Common issues:**
- CSV file not found at specified path
- Missing columns (check for typos in column names)
- Multiple mouse IDs in one manifest (split into separate runs)
- Empty cells in required columns

**How to fix:**
- Ensure CSV is under `/data/` or provide absolute path
- Check column names match exactly (case-sensitive)
- Fill in missing values or remove incomplete rows

### 3. Reference Data ✓

Checks for required Allen CCF and template files under `/data/`:
- `smartspim_lca_template/smartspim_lca_template_25.nii.gz`
- `allen_mouse_ccf/average_template/average_template_25.nii.gz`
- `allen_mouse_ccf_annotations_lateralized_compact/ccf_2017_annotation_25_lateralized_compact.nrrd`
- `iblatlas_allenatlas/` (directory)

**Common issues:**
- Reference data not mounted to `/data/`
- Incorrect directory structure
- Files renamed or in wrong location

**How to fix:**
```bash
# In Code Ocean capsule: attach reference data asset
# Check structure:
ls -R /data/smartspim_lca_template/
ls -R /data/allen_mouse_ccf/
ls -R /data/iblatlas_allenatlas/
```

### 4. Asset Discovery ✓

Validates the SmartSPIM asset structure discovered from Neuroglancer file:
- Neuroglancer file exists and is readable
- Contains valid image sources
- Asset path exists under `/data/`
- `image_tile_fusing/OMEZarr/` directory with `.zarr` channels
- `image_atlas_alignment/` directory with registration outputs
- Precomputed registration: `moved_ls_to_ccf.nii.gz`
- Transform files: `*_0GenericAffine.mat`, `*_1InverseWarp.nii.gz`

**Common issues:**
- Neuroglancer file points to S3 URIs that don't map to `/data/`
- Asset not fully transferred (missing subdirectories)
- Registration not completed (missing transform files)

**How to fix:**
```bash
# Find the asset directory mentioned in error
# Check its structure:
ls /data/<asset-id>/image_tile_fusing/OMEZarr/
ls /data/<asset-id>/image_atlas_alignment/

# Look for registration channel folder
ls /data/<asset-id>/image_atlas_alignment/*/moved_ls_to_ccf.nii.gz
```

### 5. Per-Probe Files ✓

Validates that data exists for each probe in the manifest:
- Annotation files: `*/{probe_file}.json` (glob search under `/data/`)
- Ephys recording folders: `/data/{sorted_recording}/` (if not `--skip-ephys`)
- Surface finding files: specified paths exist (if provided in manifest)

**Common issues:**
- Probe annotation file not found (check `probe_file` column in manifest)
- Ephys folder name mismatch (check `sorted_recording` column)
- Surface finding path incorrect

**How to fix:**
```bash
# Search for annotation files:
find /data -name "probe123.json"

# Check ephys folder exists:
ls /data/729293_2024-01-15_sorted/

# Verify surface finding file:
ls /data/surface_finding/probe123_surface.csv
```

**Tip:** Annotation files are searched with glob pattern `*/{probe_file}.json`, so they can be anywhere under `/data/`. If not found, check the filename in the manifest matches exactly.

### 6. Output Access ✓

Ensures the output directory is writable and has sufficient space:
- `/results/` directory exists
- `/results/` is writable
- At least 10 GB free disk space (error if less)
- At least 50 GB recommended (warning if less)

**Common issues:**
- Results directory read-only
- Disk quota exceeded
- Network storage full

**How to fix:**
```bash
# Check disk space:
df -h /results

# Check permissions:
ls -ld /results
touch /results/test.txt && rm /results/test.txt
```

### 7. System Resources ⚠️

Checks available system resources (warnings only, won't block pipeline):
- RAM: At least 8 GB required, 16 GB recommended
- Network storage detection: Warns if `/data/` is on NFS/FUSE (can be slow)

**Common issues:**
- Running on low-memory instance
- Network storage causing slow IO

**How to fix:**
- Increase instance memory if RAM warning appears
- Consider local caching for network storage (see `OPTIMIZATION_IO.md`)

## Understanding Results

### Result Types

- ✓ **Passed** (green): Check succeeded, all good
- ❌ **Error** (red): Critical issue, must fix before running pipeline
- ⚠️ **Warning** (yellow): Potential issue, pipeline can run but may have problems

### Exit Codes

When using `--validate-only`:
- `0`: Validation passed (no errors)
- `1`: Validation failed (has errors)

Warnings do **not** cause non-zero exit codes.

### Reading the Output

Validation results are grouped by category:

```
CLI Args:
--------------------------------------------------------------------------------
  ✓  neuroglancer: Neuroglancer path provided: Probes_561_729293_Day1and2.json
  ✓  annotation_manifest: Manifest path provided: 729293/Manifest.csv

Manifest CSV:
--------------------------------------------------------------------------------
  ✓  file_exists: Manifest CSV exists: /data/729293/Manifest.csv
  ❌ mouseid_consistency: Multiple mouse IDs found: 729293, 729294
     Pipeline currently supports single mouse per run.

Reference Data:
--------------------------------------------------------------------------------
  ✓  template_25: Template found: /data/smartspim_lca_template/...
  ❌ ccf_25: CCF template not found: /data/allen_mouse_ccf/...
```

Each result shows:
- **Icon**: ✓, ❌, or ⚠️
- **Item**: What was checked (e.g., `ccf_25`, `mouseid_consistency`)
- **Message**: Details about the result and how to fix

## Common Validation Failures

### "Asset path not found"

**Cause:** The asset derived from Neuroglancer file doesn't exist under `/data/`.

**Fix:**
1. Check Neuroglancer file contains correct S3 URIs
2. Ensure asset is attached/mounted to capsule
3. Verify asset ID in URI matches directory under `/data/`

### "Annotation not found for probe"

**Cause:** The `probe_file` in manifest doesn't match any JSON file under `/data/`.

**Fix:**
1. Check spelling of `probe_file` in manifest (case-sensitive)
2. Search for the file: `find /data -name "probe123.json"`
3. Ensure file extension is `.json` (not `.JSON` or `.txt`)

### "Missing required columns"

**Cause:** Manifest CSV is missing one or more required columns.

**Fix:**
1. Open manifest CSV and check column headers
2. Required columns: `mouseid`, `sorted_recording`, `probe_file`, `probe_id`, `probe_name`
3. Fix typos (e.g., `probeid` vs `probe_id`)

### "Multiple mouse IDs found"

**Cause:** Manifest contains rows with different `mouseid` values.

**Fix:**
Split manifest into separate CSVs, one per mouse:
```bash
# Filter for single mouse
awk -F',' '$1 == "729293"' manifest.csv > manifest_729293.csv
```

### "Precomputed registration not found"

**Cause:** The registration pipeline hasn't completed for this asset, or output files are missing.

**Fix:**
1. Check `image_atlas_alignment/` directory exists under asset
2. Look for subdirectories (one per channel)
3. Verify `moved_ls_to_ccf.nii.gz` exists in registration channel folder
4. If missing, re-run the registration pipeline

### "Insufficient disk space"

**Cause:** Less than 10 GB available on `/results/` volume.

**Fix:**
1. Delete old results: `rm -rf /results/old_run/`
2. Increase volume size (Code Ocean capsule settings)
3. Check disk quota: `df -h /results`

## Integration with Pipeline

### Automatic Validation

Validation runs automatically when you start the pipeline. You don't need to do anything special:

```bash
# This automatically validates before processing
python extract_ephys_and_histology.py \
    --neuroglancer file.json \
    --manifest manifest.csv
```

If validation fails, you'll see:

```
INFO:__main__:Running validation checks...
[validation output...]
ERROR:__main__:Validation failed. Please fix the errors above before running the pipeline.
```

The pipeline will exit with code 1 and **no processing will occur**.

### Skipping Validation

You **cannot** skip validation. It always runs to protect against wasted time.

If you're debugging and want to bypass validation (not recommended), you would need to modify the source code. This is intentionally difficult to prevent accidentally running with bad inputs.

### Validation Performance

Validation typically takes **30-60 seconds** depending on:
- Manifest size (number of probes)
- Network latency (if checking remote files)
- Asset structure complexity

This is much faster than waiting hours for the pipeline to fail.

## Troubleshooting

### Validation hangs or times out

**Cause:** Network filesystem is slow or unresponsive.

**Fix:**
1. Check `/data/` mount: `df -T /data`
2. Test read speed: `time ls -R /data/ > /dev/null`
3. If on NFS/FUSE, consider local caching

### False positives (validation fails but files exist)

**Cause:**
- Race condition (files being written)
- Permission issues (files exist but not readable)
- Symlinks broken

**Fix:**
```bash
# Check actual file existence
ls -lh /data/path/to/file

# Check permissions
stat /data/path/to/file

# If symlink, check target
readlink -f /data/path/to/file
```

### Validation passes but pipeline fails later

**Cause:** Validation checks file existence but not file **content**. A corrupt or empty file will pass validation.

**Examples:**
- Empty NRRD file
- Corrupt Zarr store
- Invalid JSON syntax

**What to do:** File a bug report with the specific failure. We can add content validation for that file type.

## Advanced Usage

### Skip Resource Checks

If resource checks are slow or unreliable in your environment:

```python
# In your own wrapper script:
validator = PipelineValidator(args, paths, skip_resource_checks=True)
```

Note: This is not exposed as a CLI flag to keep validation comprehensive by default.

### Custom Validation

To add your own checks:

```python
from validate_inputs import PipelineValidator, ValidationResult

class MyValidator(PipelineValidator):
    def validate_custom(self):
        # Your custom checks here
        if not my_condition():
            self._add_result(
                False,
                "Custom",
                "my_check",
                "Custom check failed",
                severity="error"
            )

validator = MyValidator(args, paths)
results = validator.validate_all()
validator.validate_custom()  # Add your checks
```

### Programmatic Usage

Use validation in your own Python scripts:

```python
from extract_ephys_and_histology import Args, _resolve_paths
from validate_inputs import PipelineValidator

args = Args(
    neuroglancer="file.json",
    annotation_manifest="manifest.csv",
    skip_ephys=False,
    validate_only=False,
)
paths = _resolve_paths(args)
validator = PipelineValidator(args, paths)
results = validator.validate_all()

# Check for errors
if validator.has_errors(results):
    print("Validation failed!")
    for r in results:
        if not r.passed and r.severity == "error":
            print(f"- {r.category}: {r.message}")
else:
    print("All good!")
```

## Getting Help

If validation is blocking you and you believe it's incorrect:

1. **Check the error message carefully**: It usually tells you exactly what's missing
2. **Run manual checks**: Use `ls`, `find`, `cat` to verify the file issue
3. **Check file paths**: Ensure relative paths are relative to `/data/`
4. **Verify manifest**: Open CSV in editor to check for typos
5. **Review asset structure**: Compare your asset to the expected layout in this doc

If you're still stuck:
- Check the project documentation: `CLAUDE.md`
- File an issue with the validation output
- Ask in Slack with the full error message

## Summary

✅ **Always run `--validate-only` first** when setting up a new pipeline run

✅ **Read the error messages** – they tell you exactly what to fix

✅ **Check file paths carefully** – most errors are typos or wrong directories

✅ **Validation saves time** – 30 seconds of validation beats hours of wasted processing

⚠️ **Warnings are informational** – pipeline will run but consider addressing them

❌ **Errors must be fixed** – pipeline will not start until resolved
