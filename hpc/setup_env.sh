#!/bin/bash
###############################################################################
# MOONPIERCER — Environment bootstrap
#
# Two usage modes:
#
#   1. Sourced by orchestrators:   . hpc/setup_env.sh
#      Sets PYTHON_BIN and returns to caller.
#
#   2. Executed as a child-job script:   sbatch ... hpc/setup_env.sh CMD ARGS
#      Sets up environment, then exec's CMD ARGS.
#      When sbatch runs a script file (not --wrap), the shell is properly
#      initialised and `module load` works — matching pbh_lunar_crater_analysis.
#
# After sourcing / before exec, PYTHON_BIN holds the working interpreter.
#
# Environment variable overrides (all optional):
#   MOONPIERCER_PYTHON_BIN      — skip all detection; use this interpreter
#   MOONPIERCER_PROJECT_DIR     — project root (for PYTHONPATH)
###############################################################################

# ─── 1. Load Python from the module system ────────────────────────────────

if type module >/dev/null 2>&1; then
    module load python 2>/dev/null || true
fi

# ─── 2. Resolve PYTHON_BIN ────────────────────────────────────────────────

if [ -n "${MOONPIERCER_PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$(command -v "${MOONPIERCER_PYTHON_BIN}")"
else
    PYTHON_BIN=""
    for _c in python3 python; do
        if command -v "${_c}" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "${_c}")"
            break
        fi
    done
    unset _c
fi

if [ -z "${PYTHON_BIN}" ]; then
    echo "[setup_env] ERROR: no python3 or python found on PATH." >&2
    echo "       Make sure 'module load python' provides a Python interpreter." >&2
    exit 127
fi

echo "[setup_env] Ready: Python=${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))" >&2

# ─── 3. Runner mode: if called with arguments, exec them ─────────────────

if [ $# -gt 0 ]; then
    set -euo pipefail
    exec "$@"
fi
