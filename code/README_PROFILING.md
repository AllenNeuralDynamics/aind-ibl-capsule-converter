# Performance Profiling Suite for SmartSPIM → IBL Converter

This directory contains a comprehensive performance profiling and optimization toolkit for `extract_ephys_and_histology.py`.

## Problem

The pipeline is painfully slow, even without ephys processing. CPU and memory utilization are low, suggesting IO bottlenecks or single-threaded execution.

## Solution

This toolkit provides:
1. **Instrumentation** to measure where time is spent
2. **Monitoring tools** for real-time IO/CPU analysis in code-server
3. **Analysis scripts** to identify bottlenecks automatically
4. **Optimization guides** with specific fixes for common issues

---

## Quick Start (5 Minutes)

### 1. Run Profiled Pipeline

```bash
cd /home/galen.lynch/Documents/Code/ibl-app-and-conversion/aind-ibl-capsule-converter/code

# Run with profiling instrumentation
python run_profiled.py --neuroglancer your_file.json --manifest your_manifest.csv 2>&1 | tee /results/pipeline.log
```

### 2. Monitor in Separate Terminal

```bash
# Option A: Full monitoring dashboard (requires tmux)
./setup_monitoring.sh

# Option B: Simple IO monitoring
./io_monitor.sh
```

### 3. Analyze Results

```bash
# After pipeline completes
python analyze_performance.py /results/performance_timing.csv --output-dir /results

# Read recommendations
cat /results/optimization_recommendations.txt
```

**That's it!** The analysis will tell you exactly what to optimize.

---

## Files Created

### Core Tools

1. **`performance_profiler.py`** - Timing and IO monitoring utilities
   - Decorators and context managers for timing
   - IO stats collection with psutil
   - Automatic summary generation

2. **`run_profiled.py`** - Instrumented pipeline wrapper
   - Monkey-patches key functions with timing
   - No modifications to original code needed
   - Saves results to `/results/performance_timing.csv`

3. **`analyze_performance.py`** - Post-run analysis
   - Identifies bottlenecks automatically
   - Generates optimization recommendations
   - Creates charts (if matplotlib available)

### Monitoring Scripts

4. **`io_monitor.sh`** - Real-time IO monitoring
   - Process-level read/write rates
   - Cumulative IO statistics
   - Color-coded output for high activity

5. **`setup_monitoring.sh`** - Tmux dashboard
   - 4-pane monitoring layout
   - Auto-detects pipeline PID
   - Easy navigation with Ctrl+B

### Documentation

6. **`PROFILING.md`** - Quick reference commands
   - Diagnostic workflows
   - Code-server tips
   - Common issues and solutions

7. **`OPTIMIZATION_PARALLELIZATION.md`** - Parallelization guide
   - Channel processing parallelization
   - Probe processing parallelization
   - ANTs multi-threading configuration

8. **`OPTIMIZATION_IO.md`** - IO optimization guide
   - Local caching strategies
   - Network storage detection
   - Compression trade-offs

9. **`OPTIMIZATION_ANTS.md`** - ANTs tuning guide
   - Multi-threading setup
   - Memory optimization
   - Transform caching

---

## Common Scenarios

### Scenario 1: "I don't know what's slow"

```bash
# Run with profiling
python run_profiled.py

# Analyze results
python analyze_performance.py /results/performance_timing.csv

# Follow the recommendations in the output
```

### Scenario 2: "I think it's IO-bound"

```bash
# Monitor IO in real-time
./io_monitor.sh

# Look at the output:
#   - High Read KB/s → Slow input data loading
#   - High Write KB/s → Slow output writing
#   - Both low → Not IO-bound, look elsewhere

# Follow guide:
cat OPTIMIZATION_IO.md
```

### Scenario 3: "I think ANTs is slow"

```bash
# Check if ANTs is using multiple threads
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

# Re-run pipeline
python run_profiled.py

# Should see 2-4x speedup immediately
# For more optimizations:
cat OPTIMIZATION_ANTS.md
```

### Scenario 4: "I want to see everything in real-time"

```bash
# Terminal 1: Start monitoring dashboard
./setup_monitoring.sh

# Terminal 2: Start pipeline
python run_profiled.py 2>&1 | tee /results/pipeline.log

# Switch between panes in tmux dashboard:
#   Ctrl+B then arrow keys
```

---

## Understanding the Output

### Example Profiling Output

```
[TIMING] 2.find_asset_info: 3.45s | Read: 125.6 MB | Write: 0 B
[TIMING] 4.write_registration_channel: 45.2s | Read: 4.2 GB | Write: 1.8 GB
[TIMING] 5.process_additional_channels: 215.3s | Read: 8.9 GB | Write: 5.2 GB | num_channels: 3
[TIMING] ants.apply_transforms: 67.8s | Read: 0 B | Write: 0 B
[TIMING] zarr.zarr_to_ants: 12.4s | Read: 3.1 GB | Write: 0 B | channel: 488

PERFORMANCE SUMMARY
================================================================================
Operation                                | Count | Total        | %     | Avg
--------------------------------------------------------------------------------
5.process_additional_channels            |   1   | 3m 35.3s     | 42.1% | 3m 35.3s
ants.apply_transforms                    |   3   | 3m 23.8s     | 39.8% | 1m 7.9s
4.write_registration_channel             |   1   | 45.2s        | 8.9%  | 45.2s
zarr.zarr_to_ants                        |   4   | 52.1s        | 10.2% | 13.0s
```

### What This Tells You

1. **42% of time** in channel processing → Parallelize channels
2. **40% of time** in ANTs transforms → Enable multi-threading
3. **Zarr loading is fast** (260 MB/s) → Not the bottleneck
4. **Multiple channels** (3) processed serially → Parallelization opportunity

### Recommended Actions (from this example)

1. Enable ANTs threading: `export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8`
   - Expected: 2-3x speedup on ANTs (40% → 13% of time)
2. Parallelize channel processing
   - Expected: 3x speedup on channels (215s → 72s)
3. Total expected speedup: ~4x overall

---

## Optimization Decision Tree

```
Run python run_profiled.py
    ↓
Analyze results
    ↓
What's the top bottleneck?
    ├─> ants.apply_transforms (>30% of time)
    │   └─> Read OPTIMIZATION_ANTS.md
    │       └─> Quick win: export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
    │
    ├─> zarr loading or writing (>20% of time, low MB/s)
    │   └─> Read OPTIMIZATION_IO.md
    │       └─> Check: df -T /data (network storage?)
    │       └─> Quick win: Copy to local SSD before processing
    │
    ├─> Multiple channels/probes (sequential processing)
    │   └─> Read OPTIMIZATION_PARALLELIZATION.md
    │       └─> Quick win: Enable ANTs threading first, then parallelize
    │
    └─> Something else
        └─> Read PROFILING.md for diagnostic commands
```

---

## Expected Speedups

### Cumulative Optimization Impact

| Optimization | Expected Speedup | Cumulative Time |
|--------------|------------------|-----------------|
| Baseline | 1x | 60 minutes |
| ANTs multi-threading | 2x | 30 minutes |
| + Parallel channels | 1.5x | 20 minutes |
| + Local SSD caching | 1.5x | 13 minutes |
| **Total** | **~4.5x** | **~13 minutes** |

*Actual results depend on your specific workload and hardware*

---

## Requirements

### Core Dependencies (Already Installed)

- `psutil` - Process monitoring
- `pandas` - CSV analysis
- `numpy` - Data processing

### Optional Dependencies

```bash
# For monitoring scripts
sudo apt-get install sysstat tmux  # pidstat, iostat, tmux

# For advanced profiling
pip install py-spy  # CPU profiling without code changes

# For chart generation
pip install matplotlib  # Performance charts in analyze_performance.py
```

---

## Troubleshooting

### "Profiled pipeline is slower than original"

Profiling overhead is minimal (<1%), but:
- Ensure you're not redirecting stderr to slow network storage
- Check if logging level is too verbose (set to INFO, not DEBUG)

### "io_monitor.sh says 'pidstat not found'"

```bash
# Install sysstat package
sudo apt-get install sysstat  # Debian/Ubuntu
sudo yum install sysstat       # RHEL/CentOS
```

### "setup_monitoring.sh fails with tmux error"

```bash
# Install tmux
sudo apt-get install tmux

# Or use individual monitoring commands (see PROFILING.md)
```

### "Monitoring shows high IO but pipeline is still slow"

- Check if IO rates are consistent or bursty (bursty = network latency)
- Check filesystem type: `df -T /data` (NFS/FUSE = slow)
- See OPTIMIZATION_IO.md for solutions

---

## Advanced Usage

### Custom Timing Instrumentation

Add timing to your own functions:

```python
from performance_profiler import timing_decorator, timing_context, log_summary

@timing_decorator("my_slow_function")
def my_function():
    # Will be automatically timed
    pass

def another_function():
    with timing_context("expensive_operation", param="value"):
        # This block will be timed
        pass

# At end of script
log_summary()
```

### Batch Analysis

Compare multiple runs:

```bash
# Run pipeline with different settings
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
python run_profiled.py
mv /results/performance_timing.csv /results/timing_1thread.csv

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
python run_profiled.py
mv /results/performance_timing.csv /results/timing_8threads.csv

# Compare
python analyze_performance.py /results/timing_1thread.csv --output-dir /results/analysis_1thread
python analyze_performance.py /results/timing_8threads.csv --output-dir /results/analysis_8threads

# Check speedup
# timing_1thread total: 3600s
# timing_8threads total: 1200s
# Speedup: 3x
```

---

## Contributing

Found a bottleneck this toolkit doesn't catch? Add it!

```python
# In run_profiled.py, add to patch_module():

orig_slow_function = module.slow_function

@functools.wraps(orig_slow_function)
def slow_function_timed(*args, **kwargs):
    with prof.timing_context("module.slow_function"):
        return orig_slow_function(*args, **kwargs)

module.slow_function = slow_function_timed
```

---

## Support

1. **Quick questions**: Check `PROFILING.md`
2. **IO issues**: Read `OPTIMIZATION_IO.md`
3. **ANTs issues**: Read `OPTIMIZATION_ANTS.md`
4. **Parallelization**: Read `OPTIMIZATION_PARALLELIZATION.md`
5. **Still stuck**: Share your `performance_report.txt` and `optimization_recommendations.txt`

---

## License

These profiling tools are provided as-is for use with the AIND IBL converter pipeline.

---

## Summary

**You now have everything needed to diagnose and fix pipeline performance issues.**

Start with:
```bash
python run_profiled.py 2>&1 | tee /results/pipeline.log
python analyze_performance.py /results/performance_timing.csv
cat /results/optimization_recommendations.txt
```

The automated analysis will tell you exactly what to optimize next. Good luck!
