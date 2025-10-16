#!/bin/bash
#
# Setup monitoring dashboard for pipeline performance analysis
#
# Creates a tmux session with 4 panes showing:
#   1. Pipeline log output
#   2. IO statistics (per-process)
#   3. CPU/Memory usage
#   4. Disk IO activity
#
# Usage:
#   ./setup_monitoring.sh               # Auto-detect pipeline PID
#   ./setup_monitoring.sh <PID>         # Monitor specific PID
#   ./setup_monitoring.sh --attach      # Attach to existing session
#

set -e

SESSION_NAME="pipeline-perf"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    log_error "tmux is not installed"
    log_info "Install with: apt-get install tmux (or yum install tmux)"
    exit 1
fi

# Handle attach flag
if [ "$1" == "--attach" ]; then
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        log_info "Attaching to existing session '$SESSION_NAME'"
        tmux attach-session -t $SESSION_NAME
    else
        log_error "No session named '$SESSION_NAME' found"
        exit 1
    fi
    exit 0
fi

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    log_error "Session '$SESSION_NAME' already exists"
    log_info "Attach with: ./setup_monitoring.sh --attach"
    log_info "Or kill with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Find or get PID
if [ $# -eq 1 ]; then
    PID=$1
else
    log_info "Auto-detecting pipeline process..."
    PID=$(pgrep -f "extract_ephys_and_histology|run_profiled" | head -n 1 || echo "")

    if [ -z "$PID" ]; then
        log_error "No pipeline process found"
        log_info "Start the pipeline in another terminal first:"
        log_info "  python run_profiled.py"
        log_info "Then run this script again"
        exit 1
    fi
fi

# Validate PID
if ! kill -0 $PID 2>/dev/null; then
    log_error "Process $PID does not exist or is not accessible"
    exit 1
fi

log_info "Monitoring PID: $PID"
log_info "Creating tmux session '$SESSION_NAME'..."

# Determine log file location
LOG_FILE="/results/pipeline.log"
if [ ! -f "$LOG_FILE" ]; then
    # Try to find in current directory
    if [ -f "pipeline.log" ]; then
        LOG_FILE="pipeline.log"
    else
        LOG_FILE="/dev/null"  # Fallback
    fi
fi

# Create tmux session with 4 panes
# Layout:
#  +-------------------+-------------------+
#  |   Pane 0          |   Pane 1          |
#  |   Pipeline Log    |   Process IO      |
#  +-------------------+-------------------+
#  |   Pane 2          |   Pane 3          |
#  |   CPU/Memory      |   Disk IO         |
#  +-------------------+-------------------+

# Create session with first window
tmux new-session -d -s $SESSION_NAME -n "monitoring"

# Pane 0: Pipeline log (top-left)
tmux send-keys -t $SESSION_NAME:0.0 "clear" C-m
if [ -f "$LOG_FILE" ]; then
    tmux send-keys -t $SESSION_NAME:0.0 "tail -f $LOG_FILE | grep --line-buffered TIMING" C-m
else
    tmux send-keys -t $SESSION_NAME:0.0 "echo 'Waiting for pipeline log...'; echo 'Start pipeline with: python run_profiled.py 2>&1 | tee /results/pipeline.log'; sleep infinity" C-m
fi

# Split horizontally (creates pane 1: top-right)
tmux split-window -h -t $SESSION_NAME:0

# Pane 1: Process IO stats (top-right)
tmux send-keys -t $SESSION_NAME:0.1 "clear" C-m
if [ -f "./io_monitor.sh" ]; then
    tmux send-keys -t $SESSION_NAME:0.1 "./io_monitor.sh $PID" C-m
elif command -v pidstat &> /dev/null; then
    tmux send-keys -t $SESSION_NAME:0.1 "watch -n 2 'pidstat -d -r -u -p $PID 1 1 | tail -n 5'" C-m
else
    tmux send-keys -t $SESSION_NAME:0.1 "watch -n 2 'ps -p $PID -o pid,pcpu,pmem,vsz,rss,etime,comm'" C-m
fi

# Select pane 0 and split vertically (creates pane 2: bottom-left)
tmux select-pane -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0

# Pane 2: CPU/Memory (bottom-left)
tmux send-keys -t $SESSION_NAME:0.2 "clear" C-m
if command -v htop &> /dev/null; then
    tmux send-keys -t $SESSION_NAME:0.2 "htop -p $PID" C-m
else
    tmux send-keys -t $SESSION_NAME:0.2 "top -p $PID" C-m
fi

# Select pane 1 and split vertically (creates pane 3: bottom-right)
tmux select-pane -t $SESSION_NAME:0.1
tmux split-window -v -t $SESSION_NAME:0

# Pane 3: Disk IO (bottom-right)
tmux send-keys -t $SESSION_NAME:0.3 "clear" C-m
if command -v iostat &> /dev/null; then
    tmux send-keys -t $SESSION_NAME:0.3 "watch -n 2 'iostat -x 1 1 | grep -E \"Device|sd|nvme\" | head -n 10'" C-m
elif command -v vmstat &> /dev/null; then
    tmux send-keys -t $SESSION_NAME:0.3 "vmstat 2" C-m
else
    tmux send-keys -t $SESSION_NAME:0.3 "watch -n 2 'df -h'" C-m
fi

# Set pane titles
tmux select-pane -t $SESSION_NAME:0.0 -T "Pipeline Log (TIMING)"
tmux select-pane -t $SESSION_NAME:0.1 -T "Process IO Stats"
tmux select-pane -t $SESSION_NAME:0.2 -T "CPU/Memory (htop)"
tmux select-pane -t $SESSION_NAME:0.3 -T "Disk IO (iostat)"

# Focus on pipeline log pane
tmux select-pane -t $SESSION_NAME:0.0

# Enable mouse support for easier navigation
tmux set-option -t $SESSION_NAME mouse on

log_info "Tmux session created successfully!"
echo ""
log_info "==================================================================="
log_info "  Monitoring Dashboard Controls:"
log_info "==================================================================="
log_info "  Switch panes:         Ctrl+B then arrow keys"
log_info "  Scroll in pane:       Ctrl+B then [ (q to exit scroll mode)"
log_info "  Zoom pane:            Ctrl+B then z (toggle fullscreen)"
log_info "  Detach:               Ctrl+B then d"
log_info "  Reattach:             ./setup_monitoring.sh --attach"
log_info "  Kill session:         tmux kill-session -t $SESSION_NAME"
log_info "==================================================================="
echo ""

# Attach to the session
log_info "Attaching to monitoring session..."
sleep 1
tmux attach-session -t $SESSION_NAME
