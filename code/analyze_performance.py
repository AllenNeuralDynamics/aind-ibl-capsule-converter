#!/usr/bin/env python
"""
Performance analysis script for converter pipeline.

Reads timing data from performance_timing.csv and generates:
- Text summary report
- Bar chart of operation times
- Pie chart of time distribution
- Recommendations for optimization

Usage:
    python analyze_performance.py performance_timing.csv
    python analyze_performance.py /results/performance_timing.csv

Outputs:
    - performance_report.txt
    - performance_chart.png (if matplotlib available)
    - optimization_recommendations.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

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


def load_timing_data(csv_path: Path) -> list[dict[str, Any]]:
    """Load timing data from CSV file."""
    data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            data.append({
                "operation": row["operation"],
                "count": int(row["count"]),
                "total_seconds": float(row["total_seconds"]),
                "avg_seconds": float(row["avg_seconds"]),
                "min_seconds": float(row["min_seconds"]),
                "max_seconds": float(row["max_seconds"]),
                "read_bytes": int(row["read_bytes"]),
                "write_bytes": int(row["write_bytes"]),
            })
    return data


def analyze_bottlenecks(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Identify performance bottlenecks."""
    total_time = sum(row["total_seconds"] for row in data)
    total_read = sum(row["read_bytes"] for row in data)
    total_write = sum(row["write_bytes"] for row in data)

    # Sort by total time
    sorted_data = sorted(data, key=lambda x: x["total_seconds"], reverse=True)

    # Calculate percentages
    for row in sorted_data:
        row["pct_time"] = (row["total_seconds"] / total_time * 100) if total_time > 0 else 0
        row["read_mb_s"] = (row["read_bytes"] / 1024 / 1024 / row["total_seconds"]) if row["total_seconds"] > 0 else 0
        row["write_mb_s"] = (row["write_bytes"] / 1024 / 1024 / row["total_seconds"]) if row["total_seconds"] > 0 else 0

    # Identify categories
    slow_ops = [row for row in sorted_data if row["total_seconds"] > 60]  # >1 minute
    io_heavy_ops = [row for row in sorted_data if (row["read_bytes"] + row["write_bytes"]) > 1024*1024*100]  # >100MB
    repetitive_ops = [row for row in sorted_data if row["count"] > 10]

    return {
        "total_time": total_time,
        "total_read": total_read,
        "total_write": total_write,
        "sorted_data": sorted_data,
        "slow_ops": slow_ops,
        "io_heavy_ops": io_heavy_ops,
        "repetitive_ops": repetitive_ops,
    }


def generate_recommendations(analysis: dict[str, Any]) -> list[str]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []
    sorted_data = analysis["sorted_data"]
    total_time = analysis["total_time"]

    # Top bottleneck
    if sorted_data:
        top_op = sorted_data[0]
        if top_op["pct_time"] > 30:
            recommendations.append(
                f"⚠️  MAJOR BOTTLENECK: '{top_op['operation']}' consumes "
                f"{top_op['pct_time']:.1f}% of total time. "
                f"This is the #1 priority for optimization."
            )

    # IO-bound operations
    io_ops = [row for row in sorted_data if row["read_mb_s"] > 10 or row["write_mb_s"] > 10]
    if io_ops:
        recommendations.append(
            f"📊 IO-Heavy Operations: {len(io_ops)} operations with significant IO. "
            f"Consider: (1) Local SSD caching for input data, (2) Parallel processing, "
            f"(3) Reduce output compression if bottleneck is writing."
        )

    # ANTs operations
    ants_ops = [row for row in sorted_data if "ants" in row["operation"].lower()]
    if ants_ops:
        total_ants_time = sum(row["total_seconds"] for row in ants_ops)
        ants_pct = (total_ants_time / total_time * 100) if total_time > 0 else 0
        if ants_pct > 20:
            recommendations.append(
                f"🧮 ANTs Transforms: Consuming {ants_pct:.1f}% of total time. "
                f"Set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8 (or CPU count) to enable multi-threading."
            )

    # Zarr operations
    zarr_ops = [row for row in sorted_data if "zarr" in row["operation"].lower()]
    if zarr_ops:
        total_zarr_time = sum(row["total_seconds"] for row in zarr_ops)
        total_zarr_read = sum(row["read_bytes"] for row in zarr_ops)
        zarr_pct = (total_zarr_time / total_time * 100) if total_time > 0 else 0
        if zarr_pct > 15:
            recommendations.append(
                f"📦 Zarr Loading: Consuming {zarr_pct:.1f}% of time, reading {format_bytes(total_zarr_read)}. "
                f"Check if data is on network storage (S3 FUSE). "
                f"Consider: aws s3 sync to local SSD before processing."
            )

    # Channel processing
    channel_ops = [row for row in sorted_data if "channel" in row["operation"].lower()]
    if channel_ops and len(channel_ops) > 1:
        recommendations.append(
            f"🔄 Sequential Channel Processing: {len(channel_ops)} channels processed serially. "
            f"Implement parallel processing with concurrent.futures.ProcessPoolExecutor."
        )

    # Probe processing
    probe_ops = [row for row in sorted_data if "probe" in row["operation"].lower()]
    if probe_ops and sum(row["count"] for row in probe_ops) > 3:
        total_probe_count = sum(row["count"] for row in probe_ops)
        recommendations.append(
            f"🔬 Sequential Probe Processing: {total_probe_count} probes processed serially. "
            f"Each probe is independent - parallelize for {total_probe_count}x speedup potential."
        )

    # Low CPU utilization indicators
    cpu_bound_ops = [row for row in sorted_data if
                      "ants" not in row["operation"].lower() and
                      row["read_mb_s"] < 1 and row["write_mb_s"] < 1 and
                      row["total_seconds"] > 30]
    if cpu_bound_ops:
        recommendations.append(
            f"⏱️  Slow operations with low IO: Check if single-threaded or waiting on network. "
            f"Run with 'py-spy' to profile CPU usage."
        )

    if not recommendations:
        recommendations.append("✅ No major bottlenecks identified. Pipeline appears well-optimized.")

    return recommendations


def write_text_report(analysis: dict[str, Any], recommendations: list[str], output_path: Path):
    """Write text report to file."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Pipeline Time:  {format_duration(analysis['total_time'])}\n")
        f.write(f"Total Data Read:      {format_bytes(analysis['total_read'])}\n")
        f.write(f"Total Data Written:   {format_bytes(analysis['total_write'])}\n\n")

        f.write("=" * 80 + "\n")
        f.write("TOP 10 TIME-CONSUMING OPERATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Operation':<40} {'Count':>6} {'Total':>12} {'%':>6} {'Avg':>12} {'R MB/s':>10} {'W MB/s':>10}\n")
        f.write("-" * 80 + "\n")

        for row in analysis["sorted_data"][:10]:
            f.write(
                f"{row['operation']:<40} "
                f"{row['count']:>6} "
                f"{format_duration(row['total_seconds']):>12} "
                f"{row['pct_time']:>5.1f}% "
                f"{format_duration(row['avg_seconds']):>12} "
                f"{row['read_mb_s']:>9.1f} "
                f"{row['write_mb_s']:>9.1f}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("OPTIMIZATION RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED OPERATION STATS\n")
        f.write("=" * 80 + "\n\n")

        for row in analysis["sorted_data"]:
            f.write(f"\n{row['operation']}:\n")
            f.write(f"  Count:        {row['count']}\n")
            f.write(f"  Total:        {format_duration(row['total_seconds'])} ({row['pct_time']:.1f}%)\n")
            f.write(f"  Average:      {format_duration(row['avg_seconds'])}\n")
            f.write(f"  Min/Max:      {format_duration(row['min_seconds'])} / {format_duration(row['max_seconds'])}\n")
            f.write(f"  Read:         {format_bytes(row['read_bytes'])} ({row['read_mb_s']:.1f} MB/s)\n")
            f.write(f"  Write:        {format_bytes(row['write_bytes'])} ({row['write_mb_s']:.1f} MB/s)\n")


def try_generate_charts(analysis: dict[str, Any], output_path: Path):
    """Generate charts if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not available - skipping chart generation")
        return

    sorted_data = analysis["sorted_data"][:10]  # Top 10

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart of operation times
    operations = [row["operation"] for row in sorted_data]
    times = [row["total_seconds"] for row in sorted_data]
    colors = plt.cm.RdYlGn_r([row["pct_time"]/100 for row in sorted_data])

    ax1.barh(operations, times, color=colors)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Top 10 Operations by Total Time")
    ax1.invert_yaxis()

    # Pie chart of time distribution
    labels_with_pct = [f"{row['operation']}\n({row['pct_time']:.1f}%)" for row in sorted_data]
    ax2.pie(times, labels=labels_with_pct, autopct='', startangle=90)
    ax2.set_title("Time Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze pipeline performance data")
    parser.add_argument("csv_file", help="Path to performance_timing.csv")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading timing data from {csv_path}...")
    data = load_timing_data(csv_path)
    print(f"Loaded {len(data)} operations")

    print("Analyzing bottlenecks...")
    analysis = analyze_bottlenecks(data)

    print("Generating recommendations...")
    recommendations = generate_recommendations(analysis)

    # Write text report
    report_path = output_dir / "performance_report.txt"
    print(f"Writing report to {report_path}...")
    write_text_report(analysis, recommendations, report_path)

    # Write recommendations separately
    rec_path = output_dir / "optimization_recommendations.txt"
    with open(rec_path, "w") as f:
        f.write("OPTIMIZATION RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n\n")

    # Try to generate charts
    chart_path = output_dir / "performance_chart.png"
    try_generate_charts(analysis, chart_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time: {format_duration(analysis['total_time'])}")
    print(f"Top bottleneck: {analysis['sorted_data'][0]['operation']} "
          f"({analysis['sorted_data'][0]['pct_time']:.1f}%)")
    print(f"\nReports generated:")
    print(f"  - {report_path}")
    print(f"  - {rec_path}")
    if (output_dir / "performance_chart.png").exists():
        print(f"  - {chart_path}")

    print("\nTop 3 Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
