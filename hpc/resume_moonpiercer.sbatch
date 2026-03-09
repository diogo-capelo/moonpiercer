#!/bin/bash
#SBATCH --job-name=mp_resume
#SBATCH --output=results/slurm_logs/mp_resume_%j.out
#SBATCH --error=results/slurm_logs/mp_resume_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2

###############################################################################
# MOONPIERCER — Resume pipeline from an existing manifest
#
# Skips Stage 1 (manifest build) and runs Stage 2 (chips) + Stage 3
# (aggregation) + cleanup.  Results are written into the same folder that
# contains the manifest.
#
# Usage:
#   sbatch hpc/resume_moonpiercer.sbatch <results-dir>
#
# Example:
#   sbatch hpc/resume_moonpiercer.sbatch results/moonpiercer_full_run
#
# The actual chip count is read from the manifest CSV, so no MAX_CHIPS
# override is needed.  Other environment overrides still apply:
#   MOONPIERCER_RANDOM_TRIALS  — null model MC trials (default: 2000)
#   MOONPIERCER_ARRAY_SIZE     — max SLURM array size (default: 999)
#   MOONPIERCER_CACHE_DIR      — WMS disk cache directory (default: ./cache)
###############################################################################

set -euo pipefail

# ─── Arguments ───────────────────────────────────────────────────────────────

RESULTS_DIR="${1:?Usage: sbatch hpc/resume_moonpiercer.sbatch <results-dir>}"

# ─── Paths ───────────────────────────────────────────────────────────────────

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

# Resolve RESULTS_DIR to absolute path
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

MANIFEST_PATH="${RESULTS_DIR}/manifest.csv"
CHIP_RESULTS_DIR="${RESULTS_DIR}/chips"
GLOBAL_OUTPUT_DIR="${RESULTS_DIR}/global"
LOG_DIR="${SCRIPT_DIR}/results/slurm_logs"
CHIP_LOG_DIR="${LOG_DIR}/chip_logs"
CACHE_DIR="${MOONPIERCER_CACHE_DIR:-${SCRIPT_DIR}/cache}"

# ─── Verify manifest ────────────────────────────────────────────────────────

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: No manifest found at ${MANIFEST_PATH}" >&2
    echo "       Run full_run_moonpiercer.sbatch first, or check the path." >&2
    exit 1
fi

# Read actual chip count from the manifest (lines minus header)
TOTAL_CHIPS=$(( $(wc -l < "$MANIFEST_PATH") - 1 ))
if [ "$TOTAL_CHIPS" -le 0 ]; then
    echo "ERROR: Manifest appears empty (${TOTAL_CHIPS} data rows)." >&2
    exit 1
fi

# ─── Configuration ──────────────────────────────────────────────────────────

RANDOM_TRIALS="${MOONPIERCER_RANDOM_TRIALS:-2000}"
ARRAY_SIZE="${MOONPIERCER_ARRAY_SIZE:-999}"

mkdir -p "$LOG_DIR" "$CHIP_LOG_DIR" "$CHIP_RESULTS_DIR" "$GLOBAL_OUTPUT_DIR" "$CACHE_DIR"

# ─── Environment bootstrap ──────────────────────────────────────────────────

# Clear positional parameters so that setup_env.sh (which checks $# to
# distinguish "sourced" from "executed as script") does not try to exec $1.
set --

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "================================================================"
echo "MOONPIERCER Resume Pipeline (from existing manifest)"
echo "================================================================"
echo "Project dir:    $PROJECT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Manifest:       $MANIFEST_PATH"
echo "Total chips:    $TOTAL_CHIPS"
echo "Random trials:  $RANDOM_TRIALS"
echo "Array size:     $ARRAY_SIZE"
echo "Cache dir:      $CACHE_DIR"
echo "Python bin:     $PYTHON_BIN"
echo "================================================================"

# ─── Stage 2: Chip Processing (Array Job) ───────────────────────────────────

echo ""
echo "[Stage 2] Submitting chip processing array job..."
echo ""

CHIPS_PER_TASK=$(( (TOTAL_CHIPS + ARRAY_SIZE - 1) / ARRAY_SIZE ))
N_TASKS=$(( (TOTAL_CHIPS + CHIPS_PER_TASK - 1) / CHIPS_PER_TASK ))
ARRAY_END=$((N_TASKS - 1))

echo "  Chips per task: $CHIPS_PER_TASK"
echo "  Number of array tasks: $N_TASKS (array 0-${ARRAY_END})"

CHIP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --job-name=mp_r_chip \
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
        --total-chips "${TOTAL_CHIPS}"
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
    --job-name=mp_r_global \
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
    --job-name=mp_r_cleanup \
    --output=/dev/null \
    --error=/dev/null \
    --time=00:15:00 \
    --mem=256M \
    --wrap="find ${LOG_DIR} -name '*.err' -empty -delete && find ${CHIP_LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; echo 'Cleaned empty .err files'"
)

echo ""
echo "================================================================"
echo "Resume pipeline submitted:"
echo "  Stage 2 (chips):       $CHIP_JOB_ID"
echo "  Stage 3 (aggregation): $GLOBAL_JOB_ID"
echo "  Cleanup:               $CLEANUP_JOB_ID"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "================================================================"
