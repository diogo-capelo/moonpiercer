#!/bin/bash
#SBATCH --job-name=mp_test
#SBATCH --output=results/slurm_logs/mp_test_%j.out
#SBATCH --error=results/slurm_logs/mp_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

###############################################################################
# MOONPIERCER — Quick Test Pipeline
#
# Minimal run (20 chips, 100 null-model trials) to verify the full three-stage
# workflow completes. Produces enough output for the analysis notebook to
# generate plots, even if coverage is sparse.
#
# Usage:
#   sbatch hpc/test_moonpiercer.sbatch
#
# Override parameters via environment variables:
#   MOONPIERCER_MAX_CHIPS      — chips in manifest (default: 20)
#   MOONPIERCER_GRID_STEP      — sweep grid step in degrees (default: 10.0)
#   MOONPIERCER_MAX_WORKERS    — manifest builder threads (default: 4)
#   MOONPIERCER_RANDOM_TRIALS  — null model MC trials (default: 100)
#   MOONPIERCER_CACHE_DIR      — WMS disk cache directory (default: ./cache)
#
# See hpc/setup_env.sh for environment overrides.
###############################################################################

set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SCRIPT_DIR}/hpc/manifest.py" ]; then
    PROJECT_DIR="${SCRIPT_DIR}"
    HPC_DIR="${SCRIPT_DIR}/hpc"
elif [ -f "${SCRIPT_DIR}/manifest.py" ]; then
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    HPC_DIR="${SCRIPT_DIR}"
else
    echo "ERROR: Could not locate hpc/manifest.py from submit directory: ${SCRIPT_DIR}" >&2
    exit 1
fi

export MOONPIERCER_PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ─── Configuration ──────────────────────────────────────────────────────────

MAX_CHIPS="${MOONPIERCER_MAX_CHIPS:-20}"
GRID_STEP="${MOONPIERCER_GRID_STEP:-10.0}"
MAX_WORKERS="${MOONPIERCER_MAX_WORKERS:-4}"
RANDOM_TRIALS="${MOONPIERCER_RANDOM_TRIALS:-100}"

RESULTS_DIR="${SCRIPT_DIR}/results/moonpiercer_run"
MANIFEST_PATH="${RESULTS_DIR}/manifest.csv"
CHIP_RESULTS_DIR="${RESULTS_DIR}/chips"
GLOBAL_OUTPUT_DIR="${RESULTS_DIR}/global"
LOG_DIR="${SCRIPT_DIR}/results/slurm_logs"
CACHE_DIR="${MOONPIERCER_CACHE_DIR:-${SCRIPT_DIR}/cache}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHIP_RESULTS_DIR" "$GLOBAL_OUTPUT_DIR" "$CACHE_DIR"

# ─── Environment bootstrap ────────────────────────────────────────────────

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "================================================================"
echo "MOONPIERCER Quick Test Pipeline"
echo "================================================================"
echo "Project dir:    $PROJECT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max chips:      $MAX_CHIPS"
echo "Grid step:      $GRID_STEP deg"
echo "Max workers:    $MAX_WORKERS"
echo "Random trials:  $RANDOM_TRIALS"
echo "Cache dir:      $CACHE_DIR"
echo "Python bin:     $PYTHON_BIN"
echo "================================================================"

# ─── Stage 1: Build Manifest ────────────────────────────────────────────────
# Child jobs run setup_env.sh as a script file (not --wrap) so that sbatch
# initialises the shell properly and `module load python` works.

echo ""
echo "[Stage 1] Building chip manifest (test: ${MAX_CHIPS} chips, ${GRID_STEP} deg grid)..."
echo ""

MANIFEST_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --job-name=mp_t_manifest \
    --output="${LOG_DIR}/manifest_%j.out" \
    --error="${LOG_DIR}/manifest_%j.err" \
    --time=01:00:00 \
    --mem=2G \
    --cpus-per-task="$MAX_WORKERS" \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/manifest.py" \
        --manifest-path "${MANIFEST_PATH}" \
        --max-chips "${MAX_CHIPS}" \
        --sweep-grid-step-deg "${GRID_STEP}" \
        --max-workers "${MAX_WORKERS}" \
        --max-requests-per-second 2.0 \
        --lon-min-deg -180.0 \
        --lon-max-deg 180.0 \
        --min-lat-deg -60.0 \
        --max-lat-deg 60.0 \
        --cache-dir "${CACHE_DIR}"
)

echo "  Manifest job submitted: $MANIFEST_JOB_ID"

# ─── Stage 2: Chip Processing (single array, <= MAX_CHIPS tasks) ───────────

echo ""
echo "[Stage 2] Submitting chip processing array (${MAX_CHIPS} tasks)..."
echo ""

ARRAY_END=$((MAX_CHIPS - 1))

CHIP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterok:${MANIFEST_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_t_chip \
    --output="${LOG_DIR}/chip_%A_%a.out" \
    --error="${LOG_DIR}/chip_%A_%a.err" \
    --time=01:00:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --array="0-${ARRAY_END}" \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/chip_worker.py" \
        --manifest-path "${MANIFEST_PATH}" \
        --output-dir "${CHIP_RESULTS_DIR}" \
        --chip-size-px 1024 \
        --chip-span-m 1200.0 \
        --cache-dir "${CACHE_DIR}" \
        --save-annotated
)

echo "  Chip array job submitted: $CHIP_JOB_ID (indices 0-${ARRAY_END})"

# ─── Stage 3: Global Aggregation ────────────────────────────────────────────

echo ""
echo "[Stage 3] Submitting global aggregation..."
echo ""

GLOBAL_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterok:${CHIP_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_t_global \
    --output="${LOG_DIR}/global_%j.out" \
    --error="${LOG_DIR}/global_%j.err" \
    --time=02:00:00 \
    --mem=8G \
    --cpus-per-task=4 \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
        --chip-results-dir "${CHIP_RESULTS_DIR}" \
        --output-dir "${GLOBAL_OUTPUT_DIR}" \
        --random-trials "${RANDOM_TRIALS}" \
        --random-seed 42 \
        --fdr-alpha 0.05 \
        --top-pairs 20 \
        --save-pair-images
)

echo "  Global aggregation job submitted: $GLOBAL_JOB_ID"

# ─── Cleanup ────────────────────────────────────────────────────────────────

CLEANUP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterany:${GLOBAL_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_t_cleanup \
    --output=/dev/null \
    --error=/dev/null \
    --time=00:05:00 \
    --mem=256M \
    --wrap="find ${LOG_DIR} -name '*.err' -empty -delete && echo 'Cleaned empty .err files'"
)

echo ""
echo "================================================================"
echo "Test pipeline submitted:"
echo "  Stage 1 (manifest):    $MANIFEST_JOB_ID"
echo "  Stage 2 (chips):       $CHIP_JOB_ID"
echo "  Stage 3 (aggregation): $GLOBAL_JOB_ID"
echo "  Cleanup:               $CLEANUP_JOB_ID"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "================================================================"
