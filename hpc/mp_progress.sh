#!/bin/bash
###############################################################################
# MOONPIERCER — Pipeline progress monitor (shell wrapper)
#
# Usage:
#   bash hpc/mp_progress.sh [results-dir] [--watch SEC]
#
# Examples:
#   bash hpc/mp_progress.sh                               # auto-detect
#   bash hpc/mp_progress.sh results/moonpiercer_full_run
#   bash hpc/mp_progress.sh results/moonpiercer_full_run --watch 30
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Source environment (for PYTHON_BIN)
if [ -f "${SCRIPT_DIR}/setup_env.sh" ]; then
    # Clear positional params to avoid setup_env.sh exec mode
    SAVED_ARGS=("$@")
    set --
    . "${SCRIPT_DIR}/setup_env.sh" 2>/dev/null || true
    set -- "${SAVED_ARGS[@]}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Auto-detect results directory if not provided
RESULTS_DIR=""
EXTRA_ARGS=()

for arg in "$@"; do
    if [ -z "$RESULTS_DIR" ] && [ -d "$arg" ]; then
        RESULTS_DIR="$arg"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [ -z "$RESULTS_DIR" ]; then
    # Try to find a results directory automatically
    for candidate in \
        "${PROJECT_DIR}/results/moonpiercer_full_run" \
        "${PROJECT_DIR}/results/moonpiercer_test" \
        "${PROJECT_DIR}/results/moonpiercer_run"; do
        if [ -d "$candidate" ]; then
            RESULTS_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: No results directory found. Provide the path as an argument." >&2
    echo "Usage: bash hpc/mp_progress.sh [results-dir] [--watch SEC]" >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/progress.py" "$RESULTS_DIR" "${EXTRA_ARGS[@]}"
