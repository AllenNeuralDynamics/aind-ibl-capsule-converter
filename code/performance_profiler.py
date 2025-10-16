"""
Performance profiling utilities for extract_ephys_and_histology.py

Provides timing decorators, context managers, and IO monitoring utilities
to identify bottlenecks in the SmartSPIM → IBL conversion pipeline.

Usage:
    # As a decorator
    @timing_decorator("function_name")
    def my_function():
        pass

    # As a context manager
    with timing_context("operation_name"):
        # code block
        pass

    # Manually
    start = time_start("operation")
    # ... do work ...
    time_end(start, "operation")
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import psutil

logger = logging.getLogger(__name__)

# Global timing results storage
_timing_results: dict[str, list[float]] = {}
_io_stats: dict[str, dict[str, Any]] = {}


def reset_stats():
    """Reset all timing and IO statistics."""
    global _timing_results, _io_stats
    _timing_results = {}
    _io_stats = {}


def get_io_stats() -> dict[str, int]:
    """
    Get current process IO statistics.

    Returns
    -------
    dict
        read_bytes: Total bytes read
        write_bytes: Total bytes written
        read_count: Number of read operations
        write_count: Number of write operations
    """
    try:
        io_counters = psutil.Process().io_counters()
        return {
            "read_bytes": io_counters.read_bytes,
            "write_bytes": io_counters.write_bytes,
            "read_count": io_counters.read_count,
            "write_count": io_counters.write_count,
        }
    except (AttributeError, NotImplementedError):
        # IO counters not available on this platform
        return {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
        }


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def time_start(operation: str) -> tuple[float, dict[str, int]]:
    """
    Start timing an operation.

    Parameters
    ----------
    operation : str
        Name of the operation being timed

    Returns
    -------
    tuple
        (start_time, io_stats_before)
    """
    io_before = get_io_stats()
    start_time = time.perf_counter()
    logger.debug(f"[TIMING] Starting: {operation}")
    return start_time, io_before


def time_end(
    start_data: tuple[float, dict[str, int]],
    operation: str,
    extra_info: dict[str, Any] | None = None,
):
    """
    End timing an operation and log results.

    Parameters
    ----------
    start_data : tuple
        Return value from time_start()
    operation : str
        Name of the operation (should match time_start)
    extra_info : dict, optional
        Additional information to log (e.g., file sizes, item counts)
    """
    start_time, io_before = start_data
    elapsed = time.perf_counter() - start_time
    io_after = get_io_stats()

    # Calculate IO deltas
    io_delta = {
        "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
        "write_bytes": io_after["write_bytes"] - io_before["write_bytes"],
        "read_count": io_after["read_count"] - io_before["read_count"],
        "write_count": io_after["write_count"] - io_before["write_count"],
    }

    # Store results
    if operation not in _timing_results:
        _timing_results[operation] = []
    _timing_results[operation].append(elapsed)

    if operation not in _io_stats:
        _io_stats[operation] = {"read_bytes": 0, "write_bytes": 0}
    _io_stats[operation]["read_bytes"] += io_delta["read_bytes"]
    _io_stats[operation]["write_bytes"] += io_delta["write_bytes"]

    # Log results
    msg_parts = [
        f"[TIMING] {operation}: {format_duration(elapsed)}",
        f"Read: {format_bytes(io_delta['read_bytes'])}",
        f"Write: {format_bytes(io_delta['write_bytes'])}",
    ]

    if extra_info:
        for key, value in extra_info.items():
            msg_parts.append(f"{key}: {value}")

    logger.info(" | ".join(msg_parts))


@contextmanager
def timing_context(operation: str, **extra_info):
    """
    Context manager for timing a code block.

    Parameters
    ----------
    operation : str
        Name of the operation being timed
    **extra_info
        Additional key-value pairs to log

    Example
    -------
    >>> with timing_context("load_zarr", path="/data/foo.zarr"):
    ...     data = load_zarr(path)
    """
    start_data = time_start(operation)
    try:
        yield
    finally:
        time_end(start_data, operation, extra_info or None)


def timing_decorator(operation: str | None = None):
    """
    Decorator for timing function execution.

    Parameters
    ----------
    operation : str, optional
        Name for the operation. If None, uses function name.

    Example
    -------
    >>> @timing_decorator("my_slow_function")
    ... def process_data(x):
    ...     return x * 2
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_data = time_start(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                time_end(start_data, op_name)

        return wrapper
    return decorator


def log_summary():
    """
    Log a summary of all timing and IO statistics.

    Should be called at the end of the pipeline.
    """
    if not _timing_results:
        logger.info("[TIMING] No timing data collected")
        return

    logger.info("=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)

    # Sort by total time (sum of all calls)
    sorted_ops = sorted(
        _timing_results.items(),
        key=lambda x: sum(x[1]),
        reverse=True,
    )

    total_time = sum(sum(times) for times in _timing_results.values())

    for operation, times in sorted_ops:
        count = len(times)
        total = sum(times)
        avg = total / count
        pct = (total / total_time * 100) if total_time > 0 else 0

        io_info = _io_stats.get(operation, {})
        read_bytes = io_info.get("read_bytes", 0)
        write_bytes = io_info.get("write_bytes", 0)

        logger.info(
            f"{operation:40s} | "
            f"Count: {count:3d} | "
            f"Total: {format_duration(total):12s} ({pct:5.1f}%) | "
            f"Avg: {format_duration(avg):10s} | "
            f"R: {format_bytes(read_bytes):10s} | "
            f"W: {format_bytes(write_bytes):10s}"
        )

    logger.info("-" * 80)
    logger.info(f"TOTAL TIME: {format_duration(total_time)}")

    total_read = sum(s.get("read_bytes", 0) for s in _io_stats.values())
    total_write = sum(s.get("write_bytes", 0) for s in _io_stats.values())
    logger.info(f"TOTAL READ: {format_bytes(total_read)}")
    logger.info(f"TOTAL WRITE: {format_bytes(total_write)}")
    logger.info("=" * 80)


def save_timing_csv(output_path: Path):
    """
    Save timing results to CSV for further analysis.

    Parameters
    ----------
    output_path : Path
        Path to output CSV file
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "operation",
            "count",
            "total_seconds",
            "avg_seconds",
            "min_seconds",
            "max_seconds",
            "read_bytes",
            "write_bytes",
        ])

        for operation, times in _timing_results.items():
            io_info = _io_stats.get(operation, {})
            writer.writerow([
                operation,
                len(times),
                sum(times),
                sum(times) / len(times),
                min(times),
                max(times),
                io_info.get("read_bytes", 0),
                io_info.get("write_bytes", 0),
            ])

    logger.info(f"[TIMING] Saved timing data to {output_path}")


# Convenience aliases
timer = timing_context
timed = timing_decorator
