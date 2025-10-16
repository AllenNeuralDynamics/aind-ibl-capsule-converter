# Parallelization Optimization Guide

How to parallelize the SmartSPIM → IBL converter to leverage multiple CPU cores.

## Problem

The current pipeline processes items sequentially:
- Multiple imaging channels processed one-by-one (line 472)
- Multiple probes processed one-by-one (line 889)
- Multiple recordings processed one-by-one (via idempotency check)

If you have:
- 3 channels × 70 seconds each = 210 seconds total
- 4 probes × 30 seconds each = 120 seconds total

**With parallelization:**
- 3 channels on 3 cores = 70 seconds total (**3x speedup**)
- 4 probes on 4 cores = 30 seconds total (**4x speedup**)

## Quick Win: Enable ANTs Multi-Threading First

Before parallelizing Python code, enable ANTs multi-threading:

```bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
export ANTS_RANDOM_SEED=123  # For reproducibility
```

Add to your shell profile or run script:

```bash
# Add to ~/.bashrc or run script
echo 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)' >> ~/.bashrc
source ~/.bashrc
```

This alone can give **2-4x speedup** on ANTs operations with no code changes!

---

## Parallelization Opportunities

### 1. Channel Processing (Highest Impact)

**Location**: `_process_additional_channels_pipeline()` (lines 455-499)

**Current Code**:
```python
for zarr_path in asset_info.zarr_volumes.additional:
    # Load channel
    img_raw = zarr_to_ants(zarr_path, ...)
    ants.image_write(img_raw, ...)

    # Transform to CCF
    img_bugged = mimic_pipeline_zarr_to_ants(zarr_path, ...)
    ch_in_ccf = ants.apply_transforms(refs.ccf_25, img_bugged, ...)
    ants.image_write(ch_in_ccf, ...)
```

**Issue**: Channels are independent but processed serially.

**Solution**: Use `ProcessPoolExecutor` to process channels in parallel.

#### Implementation

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def _process_single_channel(
    zarr_path: str,
    metadata: dict,
    processing: dict,
    ccf_25_path: Path,  # Pass path, not ANTsImage (not picklable)
    img_tx_str: list[str],
    img_tx_inverted: list[bool],
    output_img_dir: Path,
    output_ccf_dir: Path,
    level: int = 3,
) -> tuple[str, Path, Path]:
    """
    Process a single channel (for parallel execution).

    Returns
    -------
    tuple
        (channel_name, image_output_path, ccf_output_path)
    """
    import ants
    from aind_zarr_utils.zarr import zarr_to_ants
    from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_ants

    ch_str = Path(zarr_path).stem

    # Load and write image-space volume
    img_raw = zarr_to_ants(zarr_path, metadata, level=level)
    img_output = output_img_dir / f"{ch_str}.nii.gz"
    ants.image_write(img_raw, str(img_output))

    # Load CCF reference (will be cached by OS)
    ccf_25 = ants.image_read(str(ccf_25_path))

    # Transform to CCF
    img_bugged = mimic_pipeline_zarr_to_ants(zarr_path, metadata, processing, level=level)
    ch_in_ccf = ants.apply_transforms(
        ccf_25,
        img_bugged,
        img_tx_str,
        whichtoinvert=img_tx_inverted,
    )
    ccf_output = output_ccf_dir / f"histology_{ch_str}.nrrd"
    ants.image_write(ch_in_ccf, str(ccf_output))

    return ch_str, img_output, ccf_output


def _process_additional_channels_pipeline_parallel(
    asset_info: AssetInfo,
    refs: ReferenceVolumes,
    outputs: OutputDirs,
    level: int = 3,
    max_workers: int = None,
) -> ANTsImage | None:
    """
    Parallel version of _process_additional_channels_pipeline.

    Parameters
    ----------
    max_workers : int, optional
        Maximum number of parallel processes. Defaults to min(n_channels, n_cpus).
    """
    metadata = asset_info.zarr_volumes.metadata
    processing = asset_info.zarr_volumes.processing
    n_channels = len(asset_info.zarr_volumes.additional)

    if n_channels == 0:
        return None

    # Determine number of workers
    if max_workers is None:
        import os
        max_workers = min(n_channels, os.cpu_count() or 1)

    logger.info(f"Processing {n_channels} channels with {max_workers} workers")

    # Save CCF to temp file for workers
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        ccf_temp_path = Path(f.name)
    ants.image_write(refs.ccf_25, str(ccf_temp_path))

    try:
        # Submit all channel jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_channel,
                    zarr_path,
                    metadata,
                    processing,
                    ccf_temp_path,
                    asset_info.pipeline_registration_chains.img_tx_str,
                    asset_info.pipeline_registration_chains.img_tx_inverted,
                    outputs.histology_img,
                    outputs.histology_ccf,
                    level,
                ): zarr_path
                for zarr_path in asset_info.zarr_volumes.additional
            }

            # Collect results as they complete
            a_bugged_img = None
            for future in as_completed(futures):
                zarr_path = futures[future]
                try:
                    ch_name, img_out, ccf_out = future.result()
                    logger.info(f"✓ Completed channel: {ch_name}")
                except Exception as e:
                    logger.error(f"✗ Failed channel {zarr_path}: {e}")
                    raise

            # Load one bugged image for return value (needed by caller)
            if asset_info.zarr_volumes.additional:
                from aind_zarr_utils.pipeline_transformed import mimic_pipeline_zarr_to_ants
                zarr_path = asset_info.zarr_volumes.additional[-1]
                a_bugged_img = mimic_pipeline_zarr_to_ants(
                    zarr_path, metadata, processing, level=level
                )

    finally:
        # Clean up temp file
        if ccf_temp_path.exists():
            ccf_temp_path.unlink()

    return a_bugged_img
```

**To Use**: Replace the call on line 860:

```python
# OLD:
a_bugged_img = _process_additional_channels_pipeline(asset_info, ref_imgs, out)

# NEW:
a_bugged_img = _process_additional_channels_pipeline_parallel(
    asset_info, ref_imgs, out, max_workers=4
)
```

**Expected Speedup**: 2-4x for channel processing (depends on number of channels and cores)

---

### 2. Probe Processing (Moderate Impact)

**Location**: Per-probe loop (lines 889-901)

**Current Code**:
```python
for _, row in manifest_df.iterrows():
    mr = ManifestRow.from_series(row)
    result = _process_manifest_row(mr, asset_info, ...)
    processed_results.append(result)
```

**Issue**: Each probe is independent but processed serially.

**Challenges**:
- Probes share some state (atlas, stubs)
- Need to collect results in order
- Ephys extraction has idempotency check (set of processed recordings)

**Solution**: Use ThreadPoolExecutor (lighter than processes, good for IO-bound probe processing)

#### Implementation

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def _process_histology_and_ephys_parallel(args: Args, max_workers: int = 4):
    """
    Parallel version with threaded probe processing.
    """
    paths = _resolve_paths(args)
    shutil.copy(paths.manifest_csv, "/results/manifest.csv")

    manifest_df = pd.read_csv(paths.manifest_csv)
    # ... setup code same as original ...

    # Shared state
    processed_recordings: set[str] = set()
    processed_results: list[ProcessResult] = []
    recording_lock = Lock()  # Protect set access

    def process_probe_wrapper(row_tuple):
        """Wrapper for thread-safe probe processing."""
        idx, row = row_tuple
        mr = ManifestRow.from_series(row)

        # Process probe (thread-safe, no shared state)
        result = _process_manifest_row(
            mr, asset_info, raw_img_stub, raw_img_stub_buggy, ibl_atlas, out
        )

        if result.wrote_files:
            # Ephys extraction needs lock (modifies shared set)
            with recording_lock:
                _maybe_run_ephys(mr, out, processed_recordings)

        return idx, result

    # Process probes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_probe_wrapper, (idx, row)): idx
            for idx, row in manifest_df.iterrows()
        }

        # Collect results in order
        results_dict = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_idx, result = future.result()
                results_dict[result_idx] = result
                logger.info(f"✓ Completed probe {result.probe_id}")
            except Exception as e:
                logger.error(f"✗ Failed probe at index {idx}: {e}")
                raise

        # Sort results by original index
        for idx in sorted(results_dict.keys()):
            result = results_dict[idx]
            processed_results.append(result)
            if not result.wrote_files:
                logger.warning(
                    f"Did not write files for {result.recording_id}: "
                    f"{result.skipped_reason}"
                )
```

**Expected Speedup**: 2-4x for probe processing (if IO-bound) or 1-2x (if CPU-bound)

**Caution**:
- Threads share memory, so be careful with shared state
- Use locks for `processed_recordings` set
- AllenAtlas is thread-safe for reading

---

### 3. Hybrid Approach (Maximum Performance)

Combine both strategies:
1. **Parallel channel processing** (ProcessPoolExecutor) - Different processes
2. **Parallel probe processing** (ThreadPoolExecutor) - Same process, different threads

This maximizes CPU utilization for both IO-bound and CPU-bound operations.

**Expected Total Speedup**: 4-8x depending on workload

---

## Configuration Guidelines

### Choosing max_workers

```python
import os

def optimal_workers(n_items: int, operation_type: str) -> int:
    """
    Determine optimal number of workers.

    Parameters
    ----------
    n_items : int
        Number of items to process
    operation_type : str
        'cpu' for CPU-bound, 'io' for IO-bound, 'mixed' for both
    """
    n_cpus = os.cpu_count() or 1

    if operation_type == 'cpu':
        # Don't oversubscribe CPUs for CPU-bound work
        return min(n_items, n_cpus)
    elif operation_type == 'io':
        # Can use more workers for IO-bound work (waiting on disk/network)
        return min(n_items, n_cpus * 2)
    else:  # mixed
        return min(n_items, n_cpus)
```

### For Channel Processing

- **CPU-bound** (ANTs transforms dominate)
- Use `max_workers = min(n_channels, n_cpus)`
- Example: 4 channels, 8 CPUs → use 4 workers

### For Probe Processing

- **IO-bound** (file reading/writing dominates)
- Use `max_workers = min(n_probes, n_cpus * 2)`
- Example: 10 probes, 8 CPUs → use 16 workers

### Environment Variables

Set in your environment or run script:

```bash
export OMP_NUM_THREADS=1  # Prevent numpy from nested parallelism
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2  # ANTs uses 2 threads per worker
export MAX_CHANNEL_WORKERS=4
export MAX_PROBE_WORKERS=8
```

---

## Potential Issues and Solutions

### Issue 1: Out of Memory

**Symptom**: Process killed with "Killed" message

**Cause**: Too many parallel processes loading large images

**Solution**:
```python
# Reduce max_workers
max_workers = min(n_channels, max(1, os.cpu_count() // 2))

# Or process in batches
from itertools import islice

def batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

for batch in batched(asset_info.zarr_volumes.additional, 2):
    # Process 2 channels at a time
    ...
```

### Issue 2: Slower with Parallelization

**Symptom**: Parallel version is slower than serial

**Cause**:
- Overhead of process creation
- Disk IO contention (all workers hitting same disk)
- Network IO contention (all workers downloading from S3)

**Solution**:
```python
# For small datasets, don't parallelize
if n_channels < 2:
    return _process_additional_channels_pipeline(...)  # Use serial version

# For network storage, limit workers
if is_network_storage(zarr_path):
    max_workers = min(2, n_channels)  # Limit to 2 to avoid network saturation
```

### Issue 3: Deadlock or Hanging

**Symptom**: Process hangs indefinitely

**Cause**:
- Thread lock contention
- Worker waiting for resource held by another worker

**Solution**:
```python
# Add timeouts to futures
for future in as_completed(futures, timeout=3600):  # 1 hour timeout
    try:
        result = future.result(timeout=60)  # 1 minute per result
    except TimeoutError:
        logger.error("Worker timeout - cancelling")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
```

### Issue 4: ANTs Nested Parallelism

**Symptom**: System load very high, but slow progress

**Cause**: Each parallel Python process spawns multiple ANTs threads

**Solution**:
```python
# Limit ANTs threads when using multiprocessing
import os
if max_workers > 1:
    threads_per_worker = max(1, os.cpu_count() // max_workers)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)
```

---

## Testing Parallelization

### Minimal Test Case

Create a test script to verify parallelization works:

```python
# test_parallel.py
import time
from concurrent.futures import ProcessPoolExecutor
from aind_zarr_utils.zarr import zarr_to_ants

def load_channel(zarr_path):
    start = time.time()
    img = zarr_to_ants(zarr_path, metadata, level=3)
    elapsed = time.time() - start
    print(f"Loaded {zarr_path.split('/')[-1]} in {elapsed:.1f}s")
    return img.shape

# Serial
start = time.time()
for path in zarr_paths:
    load_channel(path)
serial_time = time.time() - start

# Parallel
start = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_channel, zarr_paths))
parallel_time = time.time() - start

print(f"\nSerial: {serial_time:.1f}s")
print(f"Parallel: {parallel_time:.1f}s")
print(f"Speedup: {serial_time / parallel_time:.1f}x")
```

### Expected Results

For 4 channels on 8 cores:
- Serial: 120 seconds
- Parallel: 35 seconds
- Speedup: 3.4x

If speedup is less than 2x, there's a bottleneck (IO, memory, or misconfiguration).

---

## Implementation Roadmap

### Phase 1: Quick Win (No Code Changes)
```bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
```
**Expected**: 2-4x speedup on ANTs operations

### Phase 2: Parallel Channels (Code Changes)
Implement `_process_additional_channels_pipeline_parallel()`
**Expected**: Additional 2-3x speedup on channel processing

### Phase 3: Parallel Probes (Code Changes)
Implement `_process_histology_and_ephys_parallel()`
**Expected**: Additional 2-3x speedup on probe processing

### Total Potential Speedup
Sequential: 2x → 6x → 12x
Realistic: 2x → 4x → 6x (accounting for overheads and non-parallelizable parts)

---

## Next Steps

1. **Measure current performance** with `python run_profiled.py`
2. **Identify bottleneck** - Are channels or probes the slowest part?
3. **Enable ANTs threading first** - Easiest, biggest win
4. **Implement parallel channels** if channel processing is slow
5. **Implement parallel probes** if probe processing is slow
6. **Re-measure** and verify speedup

Remember: Premature optimization is the root of all evil. Profile first, then parallelize!
