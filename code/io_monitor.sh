#!/bin/bash
#
# IO Performance Monitoring Script for extract_ephys_and_histology.py
#
# Usage:
#   ./io_monitor.sh <PID>           # Monitor specific process
#   ./io_monitor.sh                 # Auto-detect Python pipeline process
#
# Requirements:
#   - pidstat (from sysstat package)
#   - Optional: iostat, iotop for more detailed monitoring
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are available
check_tools() {
    local missing_tools=()

    if ! command -v pidstat &> /dev/null; then
        missing_tools+=("pidstat (install: apt-get install sysstat or yum install sysstat)")
    fi

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi

    # Check for optional tools
    if ! command -v iostat &> /dev/null; then
        log_warn "iostat not found (optional): Install sysstat for disk-level stats"
    fi

    if ! command -v iotop &> /dev/null; then
        log_warn "iotop not found (optional): Install iotop for per-process disk I/O"
    fi
}

# Find the pipeline process PID
find_pipeline_pid() {
    # Look for extract_ephys_and_histology or run_profiled
    local pids=$(pgrep -f "extract_ephys_and_histology\|run_profiled" || true)

    if [ -z "$pids" ]; then
        log_error "No pipeline process found"
        log_info "Start the pipeline first, then run this monitor"
        exit 1
    fi

    # If multiple PIDs, take the first one
    echo "$pids" | head -n 1
}

# Monitor specific process IO
monitor_process_io() {
    local pid=$1
    local interval=2

    log_info "Monitoring process $pid every ${interval}s"
    log_info "Process command: $(ps -p $pid -o comm=)"
    log_info "Press Ctrl+C to stop"
    echo ""

    # Print header
    printf "${BLUE}%-10s${NC} | ${BLUE}%-12s${NC} | ${BLUE}%-12s${NC} | ${BLUE}%-10s${NC} | ${BLUE}%-10s${NC} | ${BLUE}%-8s${NC} | ${BLUE}%-8s${NC}\n" \
        "TIME" "READ KB/s" "WRITE KB/s" "READ MB" "WRITE MB" "CPU%" "MEM%"
    printf "%s\n" "$(printf '%.0s-' {1..90})"

    # Running totals
    local total_read_kb=0
    local total_write_kb=0

    while true; do
        # Check if process still exists
        if ! kill -0 $pid 2>/dev/null; then
            log_warn "Process $pid has terminated"
            break
        fi

        # Get IO stats using pidstat
        local stats=$(pidstat -d -p $pid 1 1 | tail -n 1)

        if [ -z "$stats" ]; then
            sleep $interval
            continue
        fi

        # Parse pidstat output
        # Format: Time  UID  PID  kB_rd/s  kB_wr/s  kB_ccwr/s  iodelay  Command
        local timestamp=$(date +%H:%M:%S)
        local read_kb_s=$(echo "$stats" | awk '{print $4}')
        local write_kb_s=$(echo "$stats" | awk '{print $5}')

        # Get cumulative IO from /proc
        if [ -f "/proc/$pid/io" ]; then
            local read_bytes=$(grep "^read_bytes:" /proc/$pid/io | awk '{print $2}')
            local write_bytes=$(grep "^write_bytes:" /proc/$pid/io | awk '{print $2}')
            local read_mb=$((read_bytes / 1024 / 1024))
            local write_mb=$((write_bytes / 1024 / 1024))
        else
            read_mb="N/A"
            write_mb="N/A"
        fi

        # Get CPU and memory
        local cpu_mem=$(ps -p $pid -o %cpu,%mem --no-headers)
        local cpu=$(echo "$cpu_mem" | awk '{print $1}')
        local mem=$(echo "$cpu_mem" | awk '{print $2}')

        # Color code based on activity
        local read_color=""
        local write_color=""

        if (( $(echo "$read_kb_s > 10000" | bc -l) )); then
            read_color="$RED"  # High read activity (>10 MB/s)
        elif (( $(echo "$read_kb_s > 1000" | bc -l) )); then
            read_color="$YELLOW"  # Moderate read activity (>1 MB/s)
        fi

        if (( $(echo "$write_kb_s > 10000" | bc -l) )); then
            write_color="$RED"  # High write activity (>10 MB/s)
        elif (( $(echo "$write_kb_s > 1000" | bc -l) )); then
            write_color="$YELLOW"  # Moderate write activity (>1 MB/s)
        fi

        # Print stats
        printf "%-10s | ${read_color}%12.1f${NC} | ${write_color}%12.1f${NC} | %10s | %10s | %8s | %8s\n" \
            "$timestamp" "$read_kb_s" "$write_kb_s" "$read_mb" "$write_mb" "$cpu" "$mem"

        sleep $interval
    done

    echo ""
    log_info "Monitoring stopped"
}

# Display disk-level IO stats
show_disk_stats() {
    if command -v iostat &> /dev/null; then
        log_info "Disk-level IO statistics (iostat -x 1 3):"
        echo ""
        iostat -x 1 3 | grep -E "^Device|^sd|^nvme"
        echo ""
    else
        log_warn "iostat not available for disk-level stats"
    fi
}

# Check if running on network filesystem
check_filesystem_type() {
    local data_path="/data"
    local results_path="/results"

    log_info "Filesystem information:"

    for path in "$data_path" "$results_path"; do
        if [ -d "$path" ]; then
            local fs_type=$(df -T "$path" | tail -n 1 | awk '{print $2}')
            local mount_point=$(df "$path" | tail -n 1 | awk '{print $6}')

            echo "  $path -> $mount_point ($fs_type)"

            if [[ "$fs_type" == "nfs"* ]] || [[ "$fs_type" == "fuse"* ]] || [[ "$fs_type" == "cifs" ]]; then
                log_warn "    ⚠️  Network filesystem detected - expect slower IO"
            fi
        fi
    done
    echo ""
}

# Main script
main() {
    log_info "==================================================================="
    log_info "     IO Performance Monitor for SmartSPIM → IBL Converter"
    log_info "==================================================================="
    echo ""

    check_tools

    # Get PID from argument or auto-detect
    local pid=""
    if [ $# -eq 1 ]; then
        pid=$1
        # Validate PID
        if ! kill -0 $pid 2>/dev/null; then
            log_error "Process $pid does not exist or is not accessible"
            exit 1
        fi
    else
        pid=$(find_pipeline_pid)
    fi

    log_info "Target PID: $pid"
    echo ""

    # Show filesystem info
    check_filesystem_type

    # Show initial disk stats
    show_disk_stats

    # Start monitoring
    monitor_process_io $pid
}

# Run main function
main "$@"
