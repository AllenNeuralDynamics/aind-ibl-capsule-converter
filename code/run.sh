#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# --- config knobs (override via env) ------------------------------------------
PYTHON="${PYTHON:-python3}"
TREE_DEPTH="${TREE_DEPTH:-2}"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
ATTACHED_LIST="${ATTACHED_LIST:-$RESULTS_DIR/attached_data.txt}"
ENTRYPOINT="${ENTRYPOINT:-/root/capsule/code/main.py}"

# --- sanity checks ------------------------------------------------------------
command -v "$PYTHON" >/dev/null || { echo "Error: $PYTHON not found" >&2; exit 127; }
mkdir -p "$RESULTS_DIR"

# --- optional: snapshot the /data tree for debugging -------------------------
if command -v tree >/dev/null; then
  # Don't fail the run just because /data is empty or unreadable
  { tree -L "$TREE_DEPTH" --dirsfirst "$DATA_DIR" >"$ATTACHED_LIST"; } || true
else
  # Fallback: a simple listing if `tree` is not present
  { printf "tree not found; falling back to find\n" >"$ATTACHED_LIST"
    find "$DATA_DIR" -maxdepth "$TREE_DEPTH" -printf "%y %p\n" 2>/dev/null >>"$ATTACHED_LIST" || true; }
fi

# --- final handoff: make Python the process (better signals/exit codes) ------
exec "$PYTHON" "$ENTRYPOINT" "$@"
