#!/bin/bash
#SBATCH --job-name=moonpiercer
#SBATCH --output=results/slurm_logs/moonpiercer_%j.out
#SBATCH --error=results/slurm_logs/moonpiercer_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

###############################################################################
# MOONPIERCER — Full Chord PBH Lunar Crater Search Pipeline
#
# Three-stage SLURM pipeline:
#   Stage 1: Build global chip manifest (single node, threaded)
#   Stage 2: Process chips (array jobs, batched to max 999 per array)
#   Stage 3: Global aggregation (single node)
#
# Usage:
#   sbatch hpc/run_moonpiercer.sbatch
#
# Override parameters via environment variables:
#   MOONPIERCER_MAX_CHIPS      — max chips in manifest (default: 2000)
#   MOONPIERCER_GRID_STEP      — sweep grid step in degrees (default: 2.0)
#   MOONPIERCER_MAX_WORKERS    — manifest build thread count (default: 6)
#   MOONPIERCER_RANDOM_TRIALS  — null model MC trials (default: 2000)
#   MOONPIERCER_ARRAY_SIZE     — max SLURM array size (default: 999)
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

MAX_CHIPS="${MOONPIERCER_MAX_CHIPS:-2000}"
GRID_STEP="${MOONPIERCER_GRID_STEP:-2.0}"
MAX_WORKERS="${MOONPIERCER_MAX_WORKERS:-6}"
RANDOM_TRIALS="${MOONPIERCER_RANDOM_TRIALS:-2000}"
ARRAY_SIZE="${MOONPIERCER_ARRAY_SIZE:-999}"

RESULTS_DIR="${SCRIPT_DIR}/results/moonpiercer_run"
MANIFEST_PATH="${RESULTS_DIR}/manifest.csv"
CHIP_RESULTS_DIR="${RESULTS_DIR}/chips"
GLOBAL_OUTPUT_DIR="${RESULTS_DIR}/global"
LOG_DIR="${SCRIPT_DIR}/results/slurm_logs"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$CHIP_RESULTS_DIR" "$GLOBAL_OUTPUT_DIR"

# ─── Environment bootstrap ────────────────────────────────────────────────

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "================================================================"
echo "MOONPIERCER Full Chord Pipeline"
echo "================================================================"
echo "Project dir:    $PROJECT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "Max chips:      $MAX_CHIPS"
echo "Grid step:      $GRID_STEP deg"
echo "Max workers:    $MAX_WORKERS"
echo "Random trials:  $RANDOM_TRIALS"
echo "Array size:     $ARRAY_SIZE"
echo "Python bin:     $PYTHON_BIN"
echo "================================================================"

# ─── Stage 1: Build Manifest ────────────────────────────────────────────────
# Child jobs run setup_env.sh as a script file (not --wrap) so that sbatch
# initialises the shell properly and `module load python` works.

echo ""
echo "[Stage 1] Building global chip manifest..."
echo ""

MANIFEST_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --job-name=mp_manifest \
    --output="${LOG_DIR}/manifest_%j.out" \
    --error="${LOG_DIR}/manifest_%j.err" \
    --time=06:00:00 \
    --mem=4G \
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
        --max-lat-deg 85.0
)

echo "  Manifest job submitted: $MANIFEST_JOB_ID"

# ─── Stage 2: Chip Processing (Array Jobs) ──────────────────────────────────

echo ""
echo "[Stage 2] Submitting chip processing array jobs..."
echo ""

# Split into batches of at most ARRAY_SIZE tasks.  All batches depend on the
# manifest job and can run concurrently across the cluster.

REMAINING=$MAX_CHIPS
OFFSET=0
CHIP_JOB_IDS=""

while [ "$REMAINING" -gt 0 ]; do
    BATCH_SIZE=$((REMAINING < ARRAY_SIZE ? REMAINING : ARRAY_SIZE))
    ARRAY_END=$((OFFSET + BATCH_SIZE - 1))

    CHIP_JOB_ID=$(sbatch \
        --parsable \
        --export=ALL \
        --dependency=afterok:${MANIFEST_JOB_ID} \
        --kill-on-invalid-dep=yes \
        --job-name=mp_chip \
        --output="${LOG_DIR}/chip_%A_%a.out" \
        --error="${LOG_DIR}/chip_%A_%a.err" \
        --time=01:00:00 \
        --mem=4G \
        --cpus-per-task=1 \
        --array="${OFFSET}-${ARRAY_END}" \
        "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/chip_worker.py" \
            --manifest-path "${MANIFEST_PATH}" \
            --output-dir "${CHIP_RESULTS_DIR}" \
            --chip-size-px 1024 \
            --chip-span-m 1200.0
    )

    echo "  Chip array job submitted: $CHIP_JOB_ID (indices ${OFFSET}-${ARRAY_END})"

    if [ -n "$CHIP_JOB_IDS" ]; then
        CHIP_JOB_IDS="${CHIP_JOB_IDS}:${CHIP_JOB_ID}"
    else
        CHIP_JOB_IDS="$CHIP_JOB_ID"
    fi

    OFFSET=$((OFFSET + BATCH_SIZE))
    REMAINING=$((REMAINING - BATCH_SIZE))
done

# ─── Stage 3: Global Aggregation ────────────────────────────────────────────

echo ""
echo "[Stage 3] Submitting global aggregation job..."
echo ""

GLOBAL_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterok:${CHIP_JOB_IDS} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_global \
    --output="${LOG_DIR}/global_%j.out" \
    --error="${LOG_DIR}/global_%j.err" \
    --time=12:00:00 \
    --mem=16G \
    --cpus-per-task=4 \
    "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
        --chip-results-dir "${CHIP_RESULTS_DIR}" \
        --output-dir "${GLOBAL_OUTPUT_DIR}" \
        --random-trials "${RANDOM_TRIALS}" \
        --random-seed 42 \
        --fdr-alpha 0.05 \
        --top-pairs 50 \
        --save-pair-images
)

echo "  Global aggregation job submitted: $GLOBAL_JOB_ID"

# ─── Cleanup ────────────────────────────────────────────────────────────────

CLEANUP_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL \
    --dependency=afterany:${GLOBAL_JOB_ID} \
    --kill-on-invalid-dep=yes \
    --job-name=mp_cleanup \
    --output=/dev/null \
    --error=/dev/null \
    --time=00:05:00 \
    --mem=256M \
    --wrap="find ${LOG_DIR} -name '*.err' -empty -delete && echo 'Cleaned empty .err files'"
)

echo ""
echo "================================================================"
echo "Pipeline submitted:"
echo "  Stage 1 (manifest):    $MANIFEST_JOB_ID"
echo "  Stage 2 (chips):       $CHIP_JOB_IDS"
echo "  Stage 3 (aggregation): $GLOBAL_JOB_ID"
echo "  Cleanup:               $CLEANUP_JOB_ID"
echo "================================================================"
