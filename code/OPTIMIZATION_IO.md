# IO Optimization Guide

How to optimize disk and network IO performance for the SmartSPIM → IBL converter.

## Problem Identification

The pipeline is IO-bound if:
- CPU utilization is low (<40%) during processing
- `pidstat -d` shows high read/write rates
- Operations with high read/write bytes take a long time

```bash
# Check if IO-bound
./io_monitor.sh

# Look for:
# - High Read KB/s or Write KB/s
# - Low CPU% at same time
```

---

## Quick Diagnosis

### 1. Check Storage Type

```bash
# Where is your data?
df -T /data
df -T /results

# Filesystem types:
#   ext4, xfs, btrfs → Local disk (fast)
#   nfs, nfs4        → Network File System (slow)
#   fuse, fuse.s3fs  → S3 FUSE mount (very slow)
#   cifs, smb        → Windows share (slow)
```

**If network storage detected**: This is your bottleneck. See solutions below.

### 2. Measure IO Performance

```bash
# Test read speed from /data
dd if=/data/some_large_file of=/dev/null bs=1M count=1000 2>&1 | grep copied

# Test write speed to /results
dd if=/dev/zero of=/results/testfile bs=1M count=1000 oflag=direct 2>&1 | grep copied
rm /results/testfile

# Expected speeds:
#   Local SSD:     300-2000 MB/s
#   Local HDD:     100-200 MB/s
#   Fast network:  100-500 MB/s
#   S3 FUSE:       10-100 MB/s
#   Slow network:  <50 MB/s
```

### 3. Check Network Latency (if network storage)

```bash
# Ping the storage server (replace with actual hostname)
ping -c 10 s3.amazonaws.com
ping -c 10 your-nfs-server

# Latency impact:
#   <1ms    → Local network (good)
#   1-10ms  → Same datacenter (acceptable)
#   10-50ms → Different datacenter (slow)
#   >50ms   → Cross-region (very slow)
```

---

## Optimization Strategies

### Strategy 1: Local Caching (Highest Impact for Network Storage)

**Problem**: Reading 100 GB from S3 at 50 MB/s = 2000 seconds (33 minutes)

**Solution**: Copy to local SSD first

```bash
#!/bin/bash
# copy_to_local.sh

# Copy input data to local SSD
LOCAL_DATA="/local/ssd/data"
mkdir -p $LOCAL_DATA

# For S3
echo "Copying from S3..."
aws s3 sync s3://your-bucket/smartspim-data $LOCAL_DATA/smartspim-data --no-progress

# For NFS
echo "Copying from NFS..."
rsync -avP --info=progress2 /data/smartspim-data/ $LOCAL_DATA/smartspim-data/

# Update paths in your workflow
export DATA_ROOT=$LOCAL_DATA
```

**Expected Speedup**: 5-10x for network storage

**Trade-off**: Initial copy takes time, but amortized over multiple runs or large processing

#### Selective Caching

If space is limited, cache only what you need:

```python
def cache_zarr_locally(remote_zarr_path, local_cache_dir):
    """
    Cache a Zarr array to local disk if not already cached.

    Returns path to local copy.
    """
    from pathlib import Path
    import shutil

    remote_path = Path(remote_zarr_path)
    local_path = Path(local_cache_dir) / remote_path.name

    if not local_path.exists():
        logger.info(f"Caching {remote_path.name} to local disk...")
        shutil.copytree(remote_path, local_path)
        logger.info(f"✓ Cached to {local_path}")

    return str(local_path)

# Usage in pipeline
if is_network_path(zarr_path):
    zarr_path = cache_zarr_locally(zarr_path, "/tmp/zarr_cache")
```

---

### Strategy 2: Optimize Zarr Reading

#### Check Zarr Chunk Sizes

```python
import zarr

# Check chunk configuration
z = zarr.open(zarr_path, mode='r')
print(f"Shape: {z.shape}")
print(f"Chunks: {z.chunks}")
print(f"Dtype: {z.dtype}")
print(f"Compression: {z.compressor}")

# Optimal chunks for 3D volumes:
#   Too small (e.g., 64³): Many reads, high overhead
#   Too large (e.g., 2048³): Large per-read memory, inefficient partial reads
#   Good range: 256³ to 512³ for 3D volumes
```

#### Multi-Threaded Zarr Reading (if supported)

```python
from dask import array as da

def load_zarr_with_dask(zarr_path, level=3):
    """
    Load Zarr with Dask for parallel reading.
    """
    import dask.array as da
    import zarr

    z = zarr.open(zarr_path, mode='r')
    # Assuming multiscale Zarr
    darr = da.from_zarr(z[level])

    # Configure parallel reading
    with da.config.set(scheduler='threads', num_workers=4):
        # Read into memory
        arr = darr.compute()

    return arr
```

**Expected Speedup**: 1.5-2x on multi-threaded storage

---

### Strategy 3: Optimize Writing

#### Issue 1: NRRD Compression is Slow

**Current Code** (line 526-532):
```python
def _compress_nrrd(input_path: Path, output_path: Path) -> None:
    img = sitk.ReadImage(str(input_path))
    temp_output_path = output_path.with_suffix(".temp.nrrd")
    sitk.WriteImage(img, str(temp_output_path), useCompression=True)
    temp_output_path.replace(output_path)
```

**Measurement**:
```python
import time
start = time.time()
sitk.WriteImage(img, path, useCompression=True)
compressed_time = time.time() - start

start = time.time()
sitk.WriteImage(img, path, useCompression=False)
uncompressed_time = time.time() - start

print(f"Compressed: {compressed_time:.1f}s")
print(f"Uncompressed: {uncompressed_time:.1f}s")
print(f"Overhead: {compressed_time - uncompressed_time:.1f}s")
```

**Solution**: Make compression optional

```python
def _write_nrrd(image: sitk.Image, output_path: Path, compress: bool = True) -> None:
    """
    Write NRRD file with optional compression.

    Parameters
    ----------
    compress : bool
        If True, use gzip compression (slower write, smaller file).
        If False, no compression (faster write, larger file).
    """
    sitk.WriteImage(image, str(output_path), useCompression=compress)

# Add CLI flag
parser.add_argument("--no-compression", action="store_true",
                    help="Skip NRRD compression for faster writes")

# Use in pipeline
compress = not args.no_compression
_write_nrrd(img, path, compress=compress)
```

**Trade-off**:
- Uncompressed: ~2-3x faster writes, ~5-10x larger files
- Compressed: Slower writes, smaller files (better for storage)

**Recommendation**: Use `--no-compression` during development/testing, compression for final runs.

#### Issue 2: Writing to Network Storage

If `/results` is on network storage:

```bash
# Option 1: Write to local disk, then copy
LOCAL_RESULTS="/local/ssd/results"
mkdir -p $LOCAL_RESULTS

# Run pipeline with local results
python run_profiled.py --results-root $LOCAL_RESULTS

# Copy to final destination after completion
rsync -avP $LOCAL_RESULTS/ /results/

# Option 2: Use faster copy method
aws s3 cp --recursive $LOCAL_RESULTS/ s3://your-bucket/results/
```

---

### Strategy 4: Reduce Redundant IO

#### Skip Re-Processing Existing Outputs

```python
def _write_registration_channel_outputs_cached(
    asset_info: AssetInfo,
    outputs: OutputDirs,
    **kwargs,
) -> ANTsImage:
    """
    Version with output caching to skip re-processing.
    """
    ccf_output = outputs.histology_ccf / "histology_registration.nrrd"
    img_output = outputs.histology_img / "histology_registration.nii.gz"

    # Check if outputs already exist
    if ccf_output.exists() and img_output.exists():
        logger.info("✓ Registration outputs exist, skipping...")
        # Load and return the image-space version
        raw_img = ants.image_read(str(img_output))
        return raw_img

    # Otherwise, process as normal
    return _write_registration_channel_outputs(asset_info, outputs, **kwargs)

# Usage
raw_img = _write_registration_channel_outputs_cached(asset_info, out, ...)
```

Add CLI flag:

```python
parser.add_argument("--skip-existing", action="store_true",
                    help="Skip processing if outputs already exist")
```

---

### Strategy 5: Batch Processing

For multiple mice/sessions, process them in one pipeline run to amortize setup costs:

```python
# Instead of:
#   python script.py --mouse 1
#   python script.py --mouse 2
#   python script.py --mouse 3

# Use a batch manifest:
#   python script.py --batch-manifest mice_list.csv

# Where mice_list.csv contains:
#   mouse_id,neuroglancer_file,manifest_csv
#   mouse1,path1.json,manifest1.csv
#   mouse2,path2.json,manifest2.csv
#   mouse3,path3.json,manifest3.csv
```

This avoids:
- Re-loading CCF atlas for each mouse
- Re-initializing libraries
- Separate Python startup times

**Expected Speedup**: 1.2-1.5x for batch processing

---

## Network Storage Specific Optimizations

### S3 FUSE Optimization

If using s3fs or similar:

```bash
# Mount with better caching
s3fs your-bucket /data \
    -o use_cache=/tmp/s3_cache \
    -o max_stat_cache_size=100000 \
    -o stat_cache_expire=900 \
    -o multipart_size=52 \
    -o parallel_count=10 \
    -o multireq_max=10

# Increase cache size
s3fs ... -o use_cache=/local/ssd/s3_cache  # Use SSD for cache
```

### NFS Optimization

```bash
# Re-mount NFS with better options
sudo mount -o remount,rsize=1048576,wsize=1048576,timeo=14,intr /data

# rsize/wsize=1048576 → 1 MB read/write buffer (larger = fewer round trips)
# timeo=14 → Longer timeout (more patient with slow network)
# intr → Allow interrupting hung operations
```

### AWS EBS Optimization

If running on AWS with EBS volumes:

```bash
# Check current volume type
aws ec2 describe-volumes --volume-ids vol-xxxxx

# Upgrade to gp3 or io2 for better performance
aws ec2 modify-volume --volume-id vol-xxxxx --volume-type gp3 --iops 16000 --throughput 1000

# Or use instance store (ephemeral but very fast)
# Mount instance store at /local/ssd
sudo mkfs.ext4 /dev/nvme1n1
sudo mount /dev/nvme1n1 /local/ssd
```

---

## Monitoring IO Performance

### During Processing

```bash
# Real-time IO monitoring
./io_monitor.sh

# Look for:
# 1. Read KB/s - Should be consistent (not bursty)
# 2. Write KB/s - Should be high when writing outputs
# 3. CPU% - Should be >60% when not IO-bound
```

### After Processing

```bash
# Analyze timing data
python analyze_performance.py /results/performance_timing.csv

# Look for operations with:
# - High read_bytes or write_bytes
# - Long duration relative to bandwidth
# - Low MB/s throughput (<50 MB/s indicates bottleneck)
```

---

## Decision Tree

```
Is your pipeline slow?
 └─> Check CPU% with `./io_monitor.sh`
      ├─> CPU high (>70%) → See OPTIMIZATION_ANTS.md
      └─> CPU low (<40%)
           └─> Check IO rates with `pidstat -d`
                ├─> IO rates low (<10 MB/s) → Something else is wrong
                │                              (network latency? Python GIL?)
                └─> IO rates high (>50 MB/s)
                     └─> Check storage type with `df -T /data`
                          ├─> Local disk (ext4/xfs) → Disk may be slow,
                          │                           try compression=False
                          └─> Network (nfs/fuse) → COPY TO LOCAL SSD!
                                                     (Strategy 1 above)
```

---

## Benchmarking Template

Use this to measure IO improvements:

```python
#!/usr/bin/env python
"""Benchmark IO performance for pipeline."""

import time
from pathlib import Path
from aind_zarr_utils.zarr import zarr_to_ants
import ants

def benchmark_read(zarr_path):
    """Benchmark Zarr reading."""
    start = time.time()
    img = zarr_to_ants(zarr_path, metadata, level=3)
    elapsed = time.time() - start
    size_mb = img.numpy().nbytes / (1024 * 1024)
    throughput = size_mb / elapsed
    print(f"Read:  {elapsed:.1f}s | {size_mb:.0f} MB | {throughput:.1f} MB/s")
    return img

def benchmark_write_compressed(img, output_path):
    """Benchmark compressed write."""
    start = time.time()
    ants.image_write(img, str(output_path))
    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / (1024 * 1024)
    throughput = size_mb / elapsed
    print(f"Write (compressed): {elapsed:.1f}s | {size_mb:.0f} MB | {throughput:.1f} MB/s")

def benchmark_write_uncompressed(img, output_path):
    """Benchmark uncompressed write."""
    import SimpleITK as sitk
    sitk_img = ants.from_numpy(img.numpy())
    start = time.time()
    sitk.WriteImage(sitk_img, str(output_path), useCompression=False)
    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / (1024 * 1024)
    throughput = size_mb / elapsed
    print(f"Write (uncompressed): {elapsed:.1f}s | {size_mb:.0f} MB | {throughput:.1f} MB/s")

# Run benchmarks
print("Benchmarking IO performance...")
print("=" * 60)

zarr_path = "/data/your/zarr/array"
img = benchmark_read(zarr_path)

benchmark_write_compressed(img, Path("/results/test_compressed.nrrd"))
benchmark_write_uncompressed(img, Path("/results/test_uncompressed.nrrd"))

# Clean up
Path("/results/test_compressed.nrrd").unlink()
Path("/results/test_uncompressed.nrrd").unlink()
```

Expected output:
```
Read:  12.4s | 3145 MB | 253.6 MB/s
Write (compressed): 23.1s | 892 MB | 38.6 MB/s
Write (uncompressed): 8.2s | 3145 MB | 383.5 MB/s
```

---

## Summary Checklist

- [ ] Identify storage type: `df -T /data`
- [ ] Measure IO speeds: `dd` tests
- [ ] If network storage: Copy to local SSD (Strategy 1)
- [ ] If slow writes: Disable compression with `--no-compression`
- [ ] If slow reads: Check Zarr chunk sizes
- [ ] If many mice: Use batch processing
- [ ] Monitor with: `./io_monitor.sh`
- [ ] Analyze with: `python analyze_performance.py`

**Expected Cumulative Speedup**: 2-5x for network storage, 1.5-2x for local storage

---

## Next Steps

1. Run IO benchmark script to establish baseline
2. Implement Strategy 1 (local caching) if network storage
3. Add `--no-compression` flag and test
4. Re-run benchmarks and compare
5. Move to `OPTIMIZATION_PARALLELIZATION.md` for further gains
