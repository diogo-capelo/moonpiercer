#!/bin/bash
#SBATCH --job-name=mp_full
#SBATCH --output=results/slurm_logs/mp_full_%j.out
#SBATCH --error=results/slurm_logs/mp_full_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4

###############################################################################
# MOONPIERCER — Full-Coverage Chord PBH Lunar Crater Search Pipeline
#
# Maximum-coverage production run (15,000 chips, 1.5° grid, 2,000 MC trials).
# Designed for a total wall-time budget of ~70 hours.
#
# Three-stage SLURM pipeline:
#   Stage 1: Build global chip manifest (single node, threaded)       ~4 h
#   Stage 2: Process chips (single array, multi-chip per task)        ~7-12 h
#   Stage 3: Global aggregation + null model (single node)            ~9-20 h
#
# Usage:
#   sbatch hpc/full_run_moonpiercer.sbatch
#
# Override parameters via environment variables:
#   MOONPIERCER_MAX_CHIPS      — max chips in manifest (default: 15000)
#   MOONPIERCER_GRID_STEP      — sweep grid step in degrees (default: 1.5)
#   MOONPIERCER_MAX_WORKERS    — manifest build thread count (default: 8)
#   MOONPIERCER_RANDOM_TRIALS  — null model MC trials (default: 2000)
#   MOONPIERCER_ARRAY_SIZE     — max SLURM array size (default: 999)
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

MAX_CHIPS="${MOONPIERCER_MAX_CHIPS:-15000}"
GRID_STEP="${MOONPIERCER_GRID_STEP:-1.5}"
MAX_WORKERS="${MOONPIERCER_MAX_WORKERS:-8}"
RANDOM_TRIALS="${MOONPIERCER_RANDOM_TRIALS:-2000}"
ARRAY_SIZE="${MOONPIERCER_ARRAY_SIZE:-999}"

RESULTS_DIR="${SCRIPT_DIR}/results/moonpiercer_full_run"
MANIFEST_PATH="${RESULTS_DIR}/manifest.csv"
CHIP_RESULTS_DIR="${RESULTS_DIR}/chips"
GLOBAL_OUTPUT_DIR="${RESULTS_DIR}/global"
LOG_DIR="${SCRIPT_DIR}/results/slurm_logs"
CHIP_LOG_DIR="${LOG_DIR}/chip_logs"
CACHE_DIR="${MOONPIERCER_CACHE_DIR:-${SCRIPT_DIR}/cache}"

mkdir -p "$LOG_DIR" "$CHIP_LOG_DIR" "$RESULTS_DIR" "$CHIP_RESULTS_DIR" "$GLOBAL_OUTPUT_DIR" "$CACHE_DIR"

# ─── Environment bootstrap ────────────────────────────────────────────────

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "================================================================"
echo "MOONPIERCER Full-Coverage Chord Pipeline"
echo "================================================================"
echo "Project dir:    $PROJECT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max chips:      $MAX_CHIPS"
echo "Grid step:      $GRID_STEP deg"
echo "Max workers:    $MAX_WORKERS"
echo "Random trials:  $RANDOM_TRIALS"
echo "Array size:     $ARRAY_SIZE"
echo "Cache dir:      $CACHE_DIR"
echo "Python bin:     $PYTHON_BIN"
echo "================================================================"

# ─── Stage 1: Build Manifest ────────────────────────────────────────────────
# Child jobs run setup_env.sh as a script file (not --wrap) so that sbatch
# initialises the shell properly and `module load python` works.

echo ""
echo "[Stage 1] Building global chip manifest (${GRID_STEP}° grid, up to ${MAX_CHIPS} chips)..."
echo ""

MANIFEST_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --job-name=mp_f_manifest \
    --output="${LOG_DIR}/manifest_%j.out" \
    --error="${LOG_DIR}/manifest_%j.err" \
    --time=24:00:00 \
    --mem=8G \
    --cpus-per-task="$MAX_WORKERS" \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/manifest.py" \
        --manifest-path "${MANIFEST_PATH}" \
        --max-chips "${MAX_CHIPS}" \
        --sweep-grid-step-deg "${GRID_STEP}" \
        --max-workers "${MAX_WORKERS}" \
        --max-requests-per-second 2.0 \
        --lon-min-deg -180.0 \
        --lon-max-deg 180.0 \
        --min-lat-deg -85.0 \
        --max-lat-deg 85.0 \
        --cache-dir "${CACHE_DIR}"
)

echo "  Manifest job submitted: $MANIFEST_JOB_ID"

# ─── Stage 2: Chip Processing (Array Job) ───────────────────────────────────

echo ""
echo "[Stage 2] Submitting chip processing array job..."
echo ""

# Distribute MAX_CHIPS across at most ARRAY_SIZE array tasks.
# Each task processes ceil(MAX_CHIPS / ARRAY_SIZE) chips sequentially.
CHIPS_PER_TASK=$(( (MAX_CHIPS + ARRAY_SIZE - 1) / ARRAY_SIZE ))
N_TASKS=$(( (MAX_CHIPS + CHIPS_PER_TASK - 1) / CHIPS_PER_TASK ))
ARRAY_END=$((N_TASKS - 1))

echo "  Chips per task: $CHIPS_PER_TASK"
echo "  Number of array tasks: $N_TASKS (array 0-${ARRAY_END})"

CHIP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterok:${MANIFEST_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_f_chip \
    --output="${CHIP_LOG_DIR}/chip_%A_%a.out" \
    --error="${CHIP_LOG_DIR}/chip_%A_%a.err" \
    --time=10:00:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --array="0-${ARRAY_END}" \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/chip_worker.py" \
        --manifest-path "${MANIFEST_PATH}" \
        --output-dir "${CHIP_RESULTS_DIR}" \
        --chip-size-px 1024 \
        --chip-span-m 1200.0 \
        --cache-dir "${CACHE_DIR}" \
        --chips-per-task "${CHIPS_PER_TASK}" \
        --total-chips "${MAX_CHIPS}"
)

echo "  Chip array job submitted: $CHIP_JOB_ID (${N_TASKS} tasks × ${CHIPS_PER_TASK} chips/task)"

# ─── Stage 3: Global Aggregation ────────────────────────────────────────────

echo ""
echo "[Stage 3] Submitting global aggregation job..."
echo ""

GLOBAL_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterok:${CHIP_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_f_global \
    --output="${LOG_DIR}/global_%j.out" \
    --error="${LOG_DIR}/global_%j.err" \
    --time=72:00:00 \
    --mem=32G \
    --cpus-per-task=8 \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
        --chip-results-dir "${CHIP_RESULTS_DIR}" \
        --output-dir "${GLOBAL_OUTPUT_DIR}" \
        --random-trials "${RANDOM_TRIALS}" \
        --random-seed 42 \
        --fdr-alpha 0.05 \
        --top-pairs 50 \
        --null-model-workers 8 \
        --save-pair-images
)

echo "  Global aggregation job submitted: $GLOBAL_JOB_ID"

# ─── Cleanup ────────────────────────────────────────────────────────────────

CLEANUP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterany:${GLOBAL_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_f_cleanup \
    --output=/dev/null \
    --error=/dev/null \
    --time=00:15:00 \
    --mem=256M \
    --wrap="find ${LOG_DIR} -name '*.err' -empty -delete && find ${CHIP_LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; echo 'Cleaned empty .err files'"
)

echo ""
echo "================================================================"
echo "Full-coverage pipeline submitted:"
echo "  Stage 1 (manifest):    $MANIFEST_JOB_ID"
echo "  Stage 2 (chips):       $CHIP_JOB_ID"
echo "  Stage 3 (aggregation): $GLOBAL_JOB_ID"
echo "  Cleanup:               $CLEANUP_JOB_ID"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "================================================================"
