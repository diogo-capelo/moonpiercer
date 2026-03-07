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
    exit 127
fi

# ─── 3. Ensure required packages are importable ──────────────────────────
# If module load python didn't provide numpy etc., install them into
# ~/.local/ (user site-packages).  This only runs once — subsequent jobs
# find the packages already installed.

if ! "${PYTHON_BIN}" -c "import numpy" >/dev/null 2>&1; then
    echo "[setup_env] numpy not found; installing required packages to ~/.local/ ..." >&2

    # Make sure pip is available (system Python on Debian may lack it).
    if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
        echo "[setup_env] pip not available; bootstrapping via get-pip.py ..." >&2
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/_mp_get_pip.py
        "${PYTHON_BIN}" /tmp/_mp_get_pip.py --user --break-system-packages --quiet
        rm -f /tmp/_mp_get_pip.py
    fi

    # Pin to versions whose manylinux2014 wheels work on any x86_64 CPU.
    # numpy >=2 and scipy >=1.14 ship manylinux_2_28 wheels that require
    # SSE4.2+ and crash with "Illegal instruction" on older nodes.
    "${PYTHON_BIN}" -m pip install --user --break-system-packages \
        "numpy<2" "scipy<1.14" pandas matplotlib Pillow requests PyYAML 2>&1

    # Install moonpiercer in editable mode if pyproject.toml exists.
    _project_dir="${MOONPIERCER_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
    if [ -f "${_project_dir}/pyproject.toml" ]; then
        "${PYTHON_BIN}" -m pip install --user --break-system-packages \
            -e "${_project_dir}" 2>&1
    fi
    unset _project_dir

    # Verify it worked.
    if ! "${PYTHON_BIN}" -c "import numpy" >/dev/null 2>&1; then
        echo "[setup_env] ERROR: numpy still not importable after install." >&2
        exit 1
    fi

    echo "[setup_env] Packages installed successfully." >&2
fi

echo "[setup_env] Ready: Python=${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))" >&2

# ─── 4. Runner mode: if called with arguments, exec them ─────────────────

if [ $# -gt 0 ]; then
    set -euo pipefail
    exec "$@"
fi
