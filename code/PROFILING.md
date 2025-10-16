# Performance Profiling Quick Reference

Quick commands and workflows for diagnosing performance issues in `extract_ephys_and_histology.py`.

## Quick Start (3-Minute Setup)

### 1. Run Pipeline with Profiling

```bash
cd /home/galen.lynch/Documents/Code/ibl-app-and-conversion/aind-ibl-capsule-converter/code

# Option A: With instrumentation (recommended)
python run_profiled.py --neuroglancer path/to/file.json --manifest path/to/manifest.csv 2>&1 | tee /results/pipeline.log

# Option B: Original script with basic timing
time python extract_ephys_and_histology.py --neuroglancer path/to/file.json --manifest path/to/manifest.csv
```

### 2. Monitor in Real-Time (Separate Terminal)

```bash
# Setup monitoring dashboard
./setup_monitoring.sh

# Or manually monitor IO
./io_monitor.sh  # Auto-detects pipeline PID
```

### 3. Analyze Results

```bash
# After pipeline completes
python analyze_performance.py /results/performance_timing.csv --output-dir /results

# View recommendations
cat /results/optimization_recommendations.txt
```

---

## Finding the Bottleneck

### Is it IO-bound?

```bash
# Get pipeline PID
PID=$(pgrep -f "extract_ephys_and_histology|run_profiled" | head -n 1)

# Monitor IO rates
pidstat -d -p $PID 2

# Output will show:
#   kB_rd/s  - Read KB/sec (high = reading from disk/network)
#   kB_wr/s  - Write KB/sec (high = writing to disk)

# If IO rates are high (>50 MB/s sustained), you're IO-bound
```

**High Read KB/s** → Zarr loading is slow (see IO Optimization section)
**High Write KB/s** → NRRD/NIfTI writing is slow (disable compression?)

### Is it CPU-bound?

```bash
# Check CPU usage
ps -p $PID -o %cpu,%mem,etime

# If CPU% is high (>90%) consistently, you're CPU-bound
# Likely cause: ANTs transforms
```

**Solution**: Enable ANTs multi-threading (see ANTs Optimization section)

### Is it waiting?

```bash
# If both CPU and IO are LOW, check what the process is doing
sudo strace -p $PID -e trace=read,write,open,close,connect 2>&1 | head -n 100

# Look for:
#   - Repeated connect() calls → Network latency (S3?)
#   - Many small read() calls → Inefficient IO pattern
#   - Long pauses → Waiting on something
```

---

## Detailed Diagnostic Commands

### Process-Level Monitoring

```bash
# Real-time process stats (updates every 2 seconds)
watch -n 2 'ps -p $PID -o pid,pcpu,pmem,vsz,rss,etime,comm'

# Detailed IO stats
pidstat -d -r -u -p $PID 2

# Cumulative IO counters
cat /proc/$PID/io
```

### System-Level Monitoring

```bash
# Disk IO activity
iostat -x 2

# Network IO (requires iftop)
sudo iftop -t -s 2

# Overall system view
htop  # Or: top
```

### Filesystem Performance

```bash
# Check if data is on network mount
df -T /data
mount | grep /data

# Expected filesystem types:
#   ext4, xfs, btrfs → Local disk (fast)
#   nfs, fuse, cifs → Network mount (slow)

# Test read speed
dd if=/data/some_large_file of=/dev/null bs=1M count=1000

# Test write speed
dd if=/dev/zero of=/results/testfile bs=1M count=1000; rm /results/testfile
```

### Python-Level Profiling

```bash
# CPU profiling with py-spy (no code changes needed!)
pip install py-spy

# Real-time flamegraph
py-spy top --pid $PID

# Generate flamegraph SVG (let it run for 30+ seconds)
py-spy record --pid $PID --output flamegraph.svg --duration 60

# Memory profiling
py-spy dump --pid $PID
```

### Zarr Performance

```bash
# Check if Zarr is local or remote
ls -lh /data/path/to/your.zarr/

# Time to list Zarr (should be <1 second if local)
time ls -R /data/path/to/your.zarr/ | wc -l

# Check chunk sizes (small chunks = many reads = slow)
python -c "import zarr; print(zarr.open('/data/path/to/your.zarr/').chunks)"

# Check Zarr filesystem
df -h /data/path/to/your.zarr/
```

### ANTs Performance

```bash
# Check current threading setting
echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

# Check number of CPU cores
nproc

# Monitor if ANTs is using multiple cores
top -p $PID -H  # Shows threads (-H flag)

# If you see only 1 thread at 100%, ANTs is single-threaded
```

---

## Code-Server Specific Tips

### Terminal Multiplexing

```bash
# Install tmux if not available
sudo apt-get install tmux

# Create monitoring session
./setup_monitoring.sh

# Controls:
#   Ctrl+B then arrow keys → Switch panes
#   Ctrl+B then [          → Scroll mode (q to exit)
#   Ctrl+B then z          → Zoom current pane
#   Ctrl+B then d          → Detach
#   tmux attach            → Reattach
```

### Long-Running Tasks

```bash
# Run pipeline in background with nohup
nohup python run_profiled.py > /results/pipeline.log 2>&1 &
echo $! > /tmp/pipeline.pid

# Monitor progress
tail -f /results/pipeline.log | grep TIMING

# Check if still running
kill -0 $(cat /tmp/pipeline.pid) && echo "Still running" || echo "Finished"
```

### Remote Monitoring

If you're disconnecting from code-server:

```bash
# Use screen instead of tmux (more resilient)
screen -S pipeline
python run_profiled.py
# Ctrl+A then d to detach

# Reattach later
screen -r pipeline
```

---

## Interpreting Performance Results

### Example Output

```
[TIMING] 4.write_registration_channel: 45.2s | Read: 4.2 GB | Write: 1.8 GB
[TIMING] 5.process_additional_channels: 215.3s | Read: 8.9 GB | Write: 5.2 GB | num_channels: 3
[TIMING] ants.apply_transforms: 67.8s | Read: 0 B | Write: 0 B
[TIMING] zarr.zarr_to_ants: 12.4s | Read: 3.1 GB | Write: 0 B | channel: 488
```

### What This Tells You

1. **write_registration_channel (45s, 4.2 GB read)**
   - Reading 4.2 GB takes 45s = ~93 MB/s
   - **If local SSD**: This is slow (expect 300-500 MB/s) → Check disk IO
   - **If network (S3/NFS)**: This is normal → Consider local caching

2. **process_additional_channels (215s total, 3 channels)**
   - ~72 seconds per channel
   - Channels are processed serially → **Parallelization opportunity**
   - Reading 8.9 GB in 215s = ~41 MB/s → Likely network-limited

3. **ants.apply_transforms (68s, no IO)**
   - Pure CPU operation
   - If this repeats 3x (once per channel): 204 seconds total on ANTs
   - **If CPU% was low during this**: ANTs is single-threaded → Enable threading
   - **If CPU% was high**: ANTs is using CPU efficiently

4. **zarr_to_ants (12s, 3.1 GB read)**
   - 3.1 GB in 12s = ~260 MB/s → Good performance
   - If this is fast but overall pipeline is slow → Look elsewhere

### Performance Targets

| Operation | Good | Acceptable | Slow |
|-----------|------|------------|------|
| Local SSD Read | >300 MB/s | 100-300 MB/s | <100 MB/s |
| Network Read (S3) | >100 MB/s | 50-100 MB/s | <50 MB/s |
| Local SSD Write | >200 MB/s | 50-200 MB/s | <50 MB/s |
| CPU Usage (ANTs) | >80% | 40-80% | <40% |

---

## Quick Fixes Checklist

### Before Running Pipeline

- [ ] Check if input data is on network mount → Copy to local SSD if possible
- [ ] Set `export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)`
- [ ] Check available disk space in /results (need >50 GB for typical run)
- [ ] Verify manifest CSV is correct (avoid processing unnecessary probes)

### During Pipeline Execution

- [ ] Monitor with `./setup_monitoring.sh` or `./io_monitor.sh`
- [ ] Check CPU% stays high (>60%) - if low, something is wrong
- [ ] Check IO rates match your storage type (SSD vs network)
- [ ] Look for "[TIMING]" messages in log for progress indication

### After Pipeline Completion

- [ ] Run `python analyze_performance.py /results/performance_timing.csv`
- [ ] Read recommendations in `/results/optimization_recommendations.txt`
- [ ] Check if any operation consumed >30% of total time
- [ ] Look for operations with count >10 (candidates for batch optimization)

---

## Common Issues and Solutions

### "Pipeline is slow but CPU/memory are low"

**Likely Cause**: IO bottleneck (network storage)

**Diagnosis**:
```bash
./io_monitor.sh
# Look at Read/Write KB/s columns
```

**Solution**: See `OPTIMIZATION_IO.md`

### "ANTs transforms take forever"

**Likely Cause**: Single-threaded execution

**Diagnosis**:
```bash
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
# Re-run pipeline
```

**Solution**: See `OPTIMIZATION_ANTS.md`

### "Multiple channels/probes processed slowly"

**Likely Cause**: Sequential processing

**Solution**: See `OPTIMIZATION_PARALLELIZATION.md`

### "Zarr loading is slow"

**Likely Cause**: Network latency or small chunk sizes

**Diagnosis**:
```bash
# Check filesystem type
df -T /data/path/to/zarr

# If NFS/FUSE, data is remote
```

**Solution**:
```bash
# Copy to local SSD before processing
aws s3 sync s3://bucket/data /local/ssd/data
# Or: rsync -avP remote:/data /local/data
```

---

## Advanced Profiling

### System Call Tracing

```bash
# Trace all syscalls (VERBOSE)
sudo strace -p $PID -e trace=all -o strace.log

# Trace only file operations
sudo strace -p $PID -e trace=open,read,write,close -o strace_io.log

# Trace network operations
sudo strace -p $PID -e trace=socket,connect,send,recv -o strace_net.log

# Get timing summary
sudo strace -c -p $PID
# Ctrl+C after 30 seconds to see summary
```

### Line-by-Line Profiling

```bash
# Install line_profiler
pip install line_profiler

# Add @profile decorator to functions in your code
# Run with kernprof
kernprof -l -v extract_ephys_and_histology.py
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Run with memory tracking
python -m memory_profiler extract_ephys_and_histology.py

# Or use tracemalloc (built-in)
python -X tracemalloc=5 extract_ephys_and_histology.py
```

---

## Next Steps

1. **Identify the bottleneck** using the tools above
2. **Read the optimization guide** for that bottleneck type:
   - `OPTIMIZATION_IO.md` - If IO-bound
   - `OPTIMIZATION_ANTS.md` - If ANTs/CPU-bound
   - `OPTIMIZATION_PARALLELIZATION.md` - If processing multiple items serially
3. **Implement optimization** and re-measure
4. **Document results** for future reference

---

## Getting Help

If you're stuck:

1. Run the full profiled pipeline: `python run_profiled.py`
2. Capture the timing summary from the log
3. Run analysis: `python analyze_performance.py /results/performance_timing.csv`
4. Share the `performance_report.txt` and `optimization_recommendations.txt`

The automated analysis will identify the most promising optimization opportunities.
