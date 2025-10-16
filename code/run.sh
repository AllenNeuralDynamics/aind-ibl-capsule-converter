#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# --- config knobs (override via env) ------------------------------------------
PYTHON="${PYTHON:-python3}"
PACKAGE_NAME="${PACKAGE_NAME:-aind-ephys-ibl-gui-conversion}"
PACKAGE_REF="${PACKAGE_REF:-main}"   # branch/tag/sha to install when updating
TREE_DEPTH="${TREE_DEPTH:-2}"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
ATTACHED_LIST="${ATTACHED_LIST:-$RESULTS_DIR/attached_data.txt}"
ENTRYPOINT="${ENTRYPOINT:-/root/capsule/code/extract_ephys_and_histology.py}"

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

# --- parse args: detect and strip --update_packages_from_source=1 ------------
DO_UPDATE=0
PASSTHRU_ARGS=()
for arg in "$@"; do
  case "$arg" in
      # bare flag or empty value → treat as "on"
    --update_packages_from_source|--update_packages_from_source=)
      DO_UPDATE=1
      ;;
    # any value -> strip from passthru; only truthy values enable
    --update_packages_from_source=*)
      val=${arg#*=}
      shopt -s nocasematch
      if [[ "$val" == 1 || "$val" == true || "$val" == yes ]]; then
        DO_UPDATE=1
      else
        DO_UPDATE=0
      fi
      shopt -u nocasematch
      ;;
    # everything else passes through
    *)
      PASSTHRU_ARGS+=("$arg")
      ;;
  esac
done

# --- optionally update/install package from source ---------------------------
if [[ "$DO_UPDATE" -eq 1 ]]; then
  echo "Updating package from source: $PACKAGE_NAME@$PACKAGE_REF"

  # Ensure pip can build/install
  export PIP_ROOT_USER_ACTION=ignore
  "$PYTHON" -m pip install -U -q pip

  REPO_URL="https://github.com/AllenNeuralDynamics/${PACKAGE_NAME}"
  if [[ -d "$PACKAGE_NAME/.git" ]]; then
    # Repo exists: fetch and checkout desired ref
    git -C "$PACKAGE_NAME" fetch --tags --force --prune --depth=1 origin "$PACKAGE_REF"
    git -C "$PACKAGE_NAME" checkout -qf FETCH_HEAD
  else
    # Fresh shallow clone of the ref
    git clone --depth=1 --branch "$PACKAGE_REF" --no-single-branch \
      --filter=blob:none "$REPO_URL" "$PACKAGE_NAME"
  fi

  # Install editable
  pushd "$PACKAGE_NAME" >/dev/null
  "$PYTHON" -m pip install -q -e .
  commit_hash="$(git rev-parse HEAD)"
  echo "Installed $PACKAGE_NAME at commit: $commit_hash"
  popd >/dev/null
fi

# --- final handoff: make Python the process (better signals/exit codes) ------
exec "$PYTHON" "$ENTRYPOINT" "${PASSTHRU_ARGS[@]}"
