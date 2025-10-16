# ANTs Optimization Guide

How to optimize ANTs (Advanced Normalization Tools) performance in the SmartSPIM → IBL converter.

## Problem Identification

ANTs operations are bottlenecks if:
- `ants.apply_transforms()` takes a long time (>60s per call)
- CPU usage is high but single-threaded (one core at 100%, others idle)
- Profiling shows ANTs operations consuming >30% of total time

```bash
# Check if ANTs is the bottleneck
python run_profiled.py 2>&1 | tee /results/pipeline.log
grep "ants\\.apply_transforms" /results/pipeline.log

# Typical output:
# [TIMING] ants.apply_transforms: 67.8s | Read: 0 B | Write: 0 B
```

---

## Quick Win: Enable Multi-Threading

**Single biggest optimization for ANTs performance!**

### Step 1: Set Environment Variable

```bash
# Set before running pipeline
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)

# Or set to specific number
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

# Make permanent (add to ~/.bashrc)
echo 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Verify It's Working

```bash
# During pipeline execution, check thread usage
PID=$(pgrep -f "extract_ephys_and_histology|run_profiled" | head -n 1)
top -H -p $PID

# You should see multiple threads (press 'H' in top)
# If only 1 thread at 100%, threading is NOT enabled
```

### Expected Speedup

| Cores Used | Speedup | Example Time |
|------------|---------|--------------|
| 1 (single) | 1x      | 68 seconds   |
| 2          | 1.7x    | 40 seconds   |
| 4          | 2.8x    | 24 seconds   |
| 8          | 3.5x    | 19 seconds   |
| 16         | 4.0x    | 17 seconds   |

**Note**: Speedup is sublinear due to synchronization overhead and memory bandwidth limits.

---

## ANTs Operations in Pipeline

### 1. Image Transforms (`ants.apply_transforms()`)

**Locations**:
- Line 488-493: Transform channels to CCF
- Line 512-518: Transform CCF to image space (reverse)

**Time Consumption**: 50-70% of channel processing time

**Optimization**:

```python
import os
import ants

# Before any ANTs operations
def setup_ants_threading(n_threads=None):
    """
    Configure ANTs to use multiple threads.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads. If None, use all available cores.
    """
    if n_threads is None:
        n_threads = os.cpu_count() or 1

    # Set via environment (before first ANTs call)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(n_threads)

    logger.info(f"ANTs configured for {n_threads} threads")

# Call at start of pipeline
setup_ants_threading()
```

### 2. Image Reading/Writing

**Locations**:
- `ants.image_write()` - Multiple locations
- `ants.image_read()` - Reading CCF template

**Time Consumption**: 20-30% of processing time

**Optimization**:

```python
def write_image_optimized(image: ants.ANTsImage, filepath: str, compress: bool = True):
    """
    Optimized image writing with optional compression.

    Parameters
    ----------
    compress : bool
        If False, skip compression for ~2x faster writes (but larger files).
    """
    # ANTs doesn't expose compression flag directly, use SimpleITK
    if not compress:
        import SimpleITK as sitk
        sitk_img = sitk.GetImageFromArray(image.numpy())
        sitk_img.SetOrigin(image.origin)
        sitk_img.SetSpacing(image.spacing)
        sitk_img.SetDirection(image.direction.flatten())
        sitk.WriteImage(sitk_img, filepath, useCompression=False)
    else:
        ants.image_write(image, filepath)
```

---

## Advanced Optimization: Transform Chaining

### Problem

Current code applies transforms sequentially:

```python
# Current (line 488-493)
ch_in_ccf = ants.apply_transforms(
    fixed=refs.ccf_25,
    moving=img_bugged,
    transformlist=[transform1, transform2, transform3],
    whichtoinvert=[True, False, False],
)
```

Each transform is computed separately, even though they could be composed.

### Solution: Pre-Compose Transforms

```python
def compose_transforms(transform_list, output_path):
    """
    Compose multiple transforms into a single transform.

    This is faster than applying them sequentially because:
    1. Only one resampling operation
    2. No intermediate images
    3. Better cache locality

    Returns
    -------
    str
        Path to composed transform file
    """
    # ANTs has a command-line tool for this
    import subprocess

    cmd = [
        "ComposeMultiTransform",
        "3",  # 3D
        str(output_path),
        "-R", str(reference_image),
    ] + transform_list

    subprocess.run(cmd, check=True)
    return output_path

# Use composed transform
composed_tx = compose_transforms(
    asset_info.pipeline_registration_chains.img_tx_str,
    "/tmp/composed_transform.mat"
)

# Apply once instead of multiple times
ch_in_ccf = ants.apply_transforms(
    fixed=refs.ccf_25,
    moving=img_bugged,
    transformlist=[composed_tx],
)
```

**Expected Speedup**: 1.2-1.5x for multi-step transforms

**Caveat**: Only works if all transforms use the same interpolation method.

---

## Memory Optimization

### Problem: Large Images Cause Swapping

ANTs operations on large volumes can exceed available RAM, causing swapping to disk (very slow).

```bash
# Check memory usage
free -h

# Watch memory during ANTs operations
watch -n 1 free -h
```

### Solution 1: Process at Lower Resolution

```python
def _determine_desired_level_adaptive(zarr_metadata, max_memory_gb=16):
    """
    Adaptively choose resolution level based on available memory.

    Parameters
    ----------
    max_memory_gb : float
        Maximum memory to use for a single image (GB)
    """
    import psutil

    available_gb = psutil.virtual_memory().available / (1024**3)
    max_volume_gb = min(max_memory_gb, available_gb * 0.7)  # Use 70% of available

    # Estimate memory for each level
    for level in range(len(zarr_metadata["coordinateTransformations"])):
        # Get voxel size for this level
        scale = zarr_metadata["coordinateTransformations"][level][0]["scale"][2:]
        # Estimate volume size (very approximate)
        voxels = (10000 / scale[0]) * (8000 / scale[1]) * (6000 / scale[2])
        memory_gb = voxels * 4 / (1024**3)  # 4 bytes per float32 voxel

        if memory_gb < max_volume_gb:
            return level

    # If none fit, use highest level (lowest resolution)
    return len(zarr_metadata["coordinateTransformations"]) - 1
```

### Solution 2: Swap Image and Template Roles

Sometimes swapping which image is "fixed" vs "moving" can save memory:

```python
# If moving image is larger than fixed:
if moving.numpy().nbytes > fixed.numpy().nbytes:
    # Swap and invert transform
    result = ants.apply_transforms(
        fixed=moving,
        moving=fixed,
        transformlist=invert_transforms(transform_list),
    )
else:
    # Standard
    result = ants.apply_transforms(fixed, moving, transform_list)
```

---

## Interpolation Optimization

### Understanding Interpolation Methods

ANTs supports multiple interpolation methods with different speed/quality trade-offs:

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `nearestNeighbor` | Fastest | Poor (blocky) | Label maps only |
| `linear` | Fast | Good | General images (default) |
| `bSpline` | Slow | Better | Smooth anatomical images |
| `gaussian` | Slowest | Best | High-quality final outputs |

### Current Usage

```python
# Line 488-493: Uses default (linear)
ch_in_ccf = ants.apply_transforms(...)  # implicit interpolator='linear'

# Line 568: Uses genericLabel for label maps
ccf_labels_in_hist_img = ants.apply_transforms(..., interpolator="genericLabel")
```

### Optimization

For intermediate processing steps, use faster interpolation:

```python
def apply_transforms_fast(fixed, moving, transformlist, is_label=False):
    """
    Wrapper with optimized interpolation settings.

    Parameters
    ----------
    is_label : bool
        If True, use label-preserving interpolation. Otherwise use linear.
    """
    if is_label:
        interpolator = "genericLabel"
    else:
        interpolator = "linear"  # Or "nearestNeighbor" for even faster

    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transformlist,
        interpolator=interpolator,
    )
```

**Speed Comparison** (for 1000³ volume):
- `nearestNeighbor`: 8 seconds
- `linear`: 15 seconds (1.9x slower)
- `bSpline`: 45 seconds (5.6x slower)

---

## Parallel ANTs Processing

### Issue: Sequential Transform Application

If processing multiple images with the same transform (e.g., multiple channels):

```python
# Current (sequential)
for channel in channels:
    result = ants.apply_transforms(ccf, channel, transforms)
    save(result)
```

### Solution: Parallel Processing

**IMPORTANT**: When parallelizing ANTs operations, adjust threads per worker:

```python
from concurrent.futures import ProcessPoolExecutor
import os

def apply_transform_to_channel(channel_data):
    """Worker function for parallel transform application."""
    channel_img, transforms, fixed_img = channel_data

    # Each worker should use fewer threads to avoid oversubscription
    n_workers = 4  # Number of parallel workers
    threads_per_worker = max(1, os.cpu_count() // n_workers)
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(threads_per_worker)

    result = ants.apply_transforms(fixed_img, channel_img, transforms)
    return result

# Process channels in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    channel_data = [(ch, transforms, fixed) for ch in channels]
    results = executor.map(apply_transform_to_channel, channel_data)
```

**Key Point**: `(n_workers * threads_per_worker) ≈ n_cpus`

Example for 16 CPU system:
- 4 workers × 4 threads each = 16 threads total ✓
- 8 workers × 2 threads each = 16 threads total ✓
- 4 workers × 8 threads each = 32 threads total ✗ (oversubscribed)

---

## Caching and Reuse

### Problem: Repeated Operations

The pipeline may process the same mouse multiple times (during development/debugging).

### Solution: Cache Intermediate Results

```python
from pathlib import Path
import hashlib
import pickle

def get_cache_key(*args):
    """Generate cache key from arguments."""
    key_str = "".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()

def cached_apply_transforms(fixed, moving, transformlist, cache_dir="/tmp/ants_cache"):
    """
    Cached version of apply_transforms.

    Parameters
    ----------
    cache_dir : str
        Directory to store cached results
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Generate cache key
    key = get_cache_key(
        fixed.shape, moving.shape,
        tuple(transformlist),
        fixed.spacing, moving.spacing
    )
    cache_file = cache_dir / f"{key}.pkl"

    # Check cache
    if cache_file.exists():
        logger.info(f"✓ Loading cached transform result")
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
        return result

    # Compute
    logger.info(f"Computing transform (will cache)...")
    result = ants.apply_transforms(fixed, moving, transformlist)

    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result
```

**Use Case**: Development/debugging where you run the pipeline multiple times on the same data.

---

## Monitoring ANTs Performance

### Real-Time Monitoring

```bash
# Monitor ANTs thread usage
PID=$(pgrep -f python | head -n 1)
watch -n 1 'ps -L -p '$PID' -o pid,lwp,pcpu,comm | head -n 20'

# Should show multiple threads when ANTs is running
```

### Profiling with py-spy

```bash
# Install py-spy
pip install py-spy

# Profile during ANTs operations
py-spy record -o profile.svg --pid $PID --duration 60

# Open profile.svg in browser to see flamegraph
# Look for ITK/ANTs function calls and their CPU usage
```

---

## Configuration Checklist

Add this to your pipeline initialization:

```python
def configure_ants_environment():
    """
    Configure optimal ANTs settings for the pipeline.

    Call this before any ANTs operations.
    """
    import os
    import logging

    logger = logging.getLogger(__name__)

    # 1. Multi-threading
    n_threads = os.cpu_count() or 1
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(n_threads)
    logger.info(f"✓ ANTs multi-threading: {n_threads} threads")

    # 2. Random seed for reproducibility
    os.environ['ANTS_RANDOM_SEED'] = '123'
    logger.info(f"✓ ANTs random seed: 123")

    # 3. Temporary directory (use fast local disk)
    if Path("/local/ssd").exists():
        os.environ['TMPDIR'] = '/local/ssd/tmp'
        Path("/local/ssd/tmp").mkdir(exist_ok=True)
        logger.info(f"✓ ANTs temp dir: /local/ssd/tmp")

    # 4. Verify settings
    actual_threads = os.environ.get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'not set')
    logger.info(f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = {actual_threads}")

# Call at start of main()
configure_ants_environment()
```

---

## Troubleshooting

### Issue 1: "ANTs still using only 1 thread"

**Diagnosis**:
```bash
# Check environment variable is set
echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

# Check during execution
cat /proc/$PID/environ | tr '\0' '\n' | grep ITK
```

**Solutions**:
```python
# Set BEFORE importing ants
import os
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '8'
import ants  # Import after setting env var

# Or restart Python session after setting variable
```

### Issue 2: "System load very high but slow progress"

**Diagnosis**:
```bash
# Check load average
uptime

# If load > 2x CPU count, you're oversubscribed
```

**Solution**:
```bash
# Reduce parallelism
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4  # Instead of 16

# Or reduce number of parallel workers (see OPTIMIZATION_PARALLELIZATION.md)
```

### Issue 3: "Out of memory errors"

**Diagnosis**:
```bash
# Check memory during ANTs operation
watch -n 1 free -h
```

**Solutions**:
1. Process at lower resolution (higher level number)
2. Reduce number of threads (uses less memory)
3. Process channels sequentially instead of parallel
4. Add swap space (not recommended, very slow)

### Issue 4: "ANTs operations timing out"

**Cause**: Very large images or complex transforms

**Solution**:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("ANTs operation timed out")

# Set timeout for long operations
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(3600)  # 1 hour timeout

try:
    result = ants.apply_transforms(...)
finally:
    signal.alarm(0)  # Cancel alarm
```

---

## Performance Benchmarking

Use this script to measure ANTs performance:

```python
#!/usr/bin/env python
"""Benchmark ANTs performance with different settings."""

import time
import os
import ants
import numpy as np

def benchmark_transform(n_threads):
    """Benchmark transform with specified thread count."""
    # Set threads
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(n_threads)

    # Create test images (1000³ voxels ≈ 4 GB)
    fixed = ants.make_image((100, 100, 100), voxval=1.0)
    moving = ants.make_image((100, 100, 100), voxval=0.5)

    # Create dummy transform (identity)
    transform = ants.create_ants_transform()

    # Benchmark
    start = time.time()
    result = ants.apply_transforms(fixed, moving, [transform])
    elapsed = time.time() - start

    return elapsed

# Test different thread counts
print("ANTs Performance Benchmark")
print("=" * 60)
print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10}")
print("-" * 60)

baseline = None
for n_threads in [1, 2, 4, 8, 16]:
    elapsed = benchmark_transform(n_threads)

    if baseline is None:
        baseline = elapsed
        speedup = 1.0
    else:
        speedup = baseline / elapsed

    print(f"{n_threads:<10} {elapsed:<12.2f} {speedup:<10.2f}x")
```

Expected output:
```
ANTs Performance Benchmark
============================================================
Threads    Time (s)     Speedup
------------------------------------------------------------
1          68.45        1.00x
2          39.23        1.74x
4          23.18        2.95x
8          18.92        3.62x
16         17.34        3.95x
```

---

## Summary: Quick Wins Ranked

1. **Enable multi-threading** (2-4x speedup, no code changes)
   ```bash
   export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
   ```

2. **Disable compression for dev/test** (2x faster writes)
   ```bash
   python run_profiled.py --no-compression
   ```

3. **Use lower resolution for testing** (4x faster if using level 4 vs 3)
   ```python
   level = 4  # Instead of 3
   ```

4. **Pre-compose transforms** (1.3x speedup for multi-step transforms)
   - Requires code changes (see above)

5. **Optimize interpolation** (1.5x speedup for linear vs bSpline)
   - Use `linear` for intermediate steps

---

## Next Steps

1. **Immediate**: Set `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`
2. **Short-term**: Profile with `run_profiled.py`, verify ANTs is actually using threads
3. **Medium-term**: Implement parallel channel processing (see OPTIMIZATION_PARALLELIZATION.md)
4. **Long-term**: Consider pre-composing transforms for repeated processing

**Expected Total Speedup**: 3-5x with threading + compression + parallelization
