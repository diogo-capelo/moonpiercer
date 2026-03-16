#!/bin/bash
#SBATCH --job-name=mp_resume
#SBATCH --output=results/slurm_logs/mp_resume_%j.out
#SBATCH --error=results/slurm_logs/mp_resume_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=2

###############################################################################
# MOONPIERCER — Resume pipeline from an existing results directory
#
# Determines what has already completed (chips, global prep, null chunks,
# final aggregation) and submits only the missing stages.
#
# Usage:
#   sbatch hpc/mp_resume.sbatch <results-dir>
#
# Example:
#   sbatch hpc/mp_resume.sbatch results/moonpiercer_full_run
#
# Environment overrides (optional):
#   MOONPIERCER_RANDOM_TRIALS         — null model MC trials (default: 2000)
#   MOONPIERCER_ARRAY_SIZE            — max SLURM array size (default: 999)
#   MOONPIERCER_PROGRESS_INTERVAL_SEC — progress log interval (default: 300)
#   MOONPIERCER_RESUME                — set to 0 to force recompute (default: 1)
#   MOONPIERCER_CACHE_DIR             — WMS disk cache directory (default: ./cache)
###############################################################################

set -euo pipefail

# ─── Arguments ───────────────────────────────────────────────────────────────

RESULTS_DIR="${1:?Usage: sbatch hpc/mp_resume.sbatch <results-dir>}"

# Resolve RESULTS_DIR to absolute path
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR=""
HPC_DIR=""

# Prefer resolving project root from results dir (…/results/<run>)
RESULTS_PARENT="$(dirname "$RESULTS_DIR")"
if [ "$(basename "$RESULTS_PARENT")" = "results" ]; then
    CANDIDATE_PROJECT="$(dirname "$RESULTS_PARENT")"
    if [ -f "${CANDIDATE_PROJECT}/hpc/manifest.py" ]; then
        PROJECT_DIR="${CANDIDATE_PROJECT}"
        HPC_DIR="${CANDIDATE_PROJECT}/hpc"
    fi
fi

# Fallback to submit directory discovery
if [ -z "$PROJECT_DIR" ]; then
    if [ -f "${SCRIPT_DIR}/hpc/manifest.py" ]; then
        PROJECT_DIR="${SCRIPT_DIR}"
        HPC_DIR="${SCRIPT_DIR}/hpc"
    elif [ -f "${SCRIPT_DIR}/manifest.py" ]; then
        PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
        HPC_DIR="${SCRIPT_DIR}"
    else
        echo "ERROR: Could not locate hpc/manifest.py from results dir or submit directory." >&2
        echo "       Results dir: ${RESULTS_DIR}" >&2
        echo "       Submit dir:  ${SCRIPT_DIR}" >&2
        exit 1
    fi
fi

export MOONPIERCER_PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

MANIFEST_PATH="${RESULTS_DIR}/manifest.csv"
CHIP_RESULTS_DIR="${RESULTS_DIR}/chips"
GLOBAL_OUTPUT_DIR="${RESULTS_DIR}/global"
CHECKPOINT_DIR="${GLOBAL_OUTPUT_DIR}/checkpoints"
LOG_DIR="${SCRIPT_DIR}/results/slurm_logs"
CHIP_LOG_DIR="${LOG_DIR}/chip_logs"
NULL_LOG_DIR="${LOG_DIR}/null_logs"
STAGE_LOG_DIR="${LOG_DIR}/stage_logs"
CACHE_DIR="${MOONPIERCER_CACHE_DIR:-${SCRIPT_DIR}/cache}"

# ─── Verify manifest ────────────────────────────────────────────────────────

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERROR: No manifest found at ${MANIFEST_PATH}" >&2
    echo "       Run mp_run.sbatch first, or check the path." >&2
    exit 1
fi

# Read actual chip count from the manifest (lines minus header)
TOTAL_CHIPS=$(( $(wc -l < "$MANIFEST_PATH") - 1 ))
if [ "$TOTAL_CHIPS" -le 0 ]; then
    echo "ERROR: Manifest appears empty (${TOTAL_CHIPS} data rows)." >&2
    exit 1
fi

# ─── Configuration ──────────────────────────────────────────────────────────

ARRAY_SIZE="${MOONPIERCER_ARRAY_SIZE:-999}"
PROGRESS_INTERVAL_SEC="${MOONPIERCER_PROGRESS_INTERVAL_SEC:-300}"

RANDOM_TRIALS="${MOONPIERCER_RANDOM_TRIALS:-2000}"
RANDOM_SEED="${MOONPIERCER_RANDOM_SEED:-42}"
FDR_ALPHA="${MOONPIERCER_FDR_ALPHA:-0.05}"
TOP_PAIRS="${MOONPIERCER_TOP_PAIRS:-50}"

RESUME_FLAG="--resume"
if [ "${MOONPIERCER_RESUME:-1}" -eq 0 ]; then
    RESUME_FLAG=""
fi

mkdir -p "$LOG_DIR" "$CHIP_LOG_DIR" "$NULL_LOG_DIR" "$STAGE_LOG_DIR" "$CHIP_RESULTS_DIR" "$GLOBAL_OUTPUT_DIR" "$CHECKPOINT_DIR" "$CACHE_DIR"

# ─── Environment bootstrap ──────────────────────────────────────────────────

# Clear positional parameters so that setup_env.sh (which checks $# to
# distinguish "sourced" from "executed as script") does not try to exec $1.
set --

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

# ─── Load checkpoint metadata (if present) ──────────────────────────────────

META_PATH="${CHECKPOINT_DIR}/checkpoint_meta.json"
if [ -f "$META_PATH" ]; then
    while IFS='=' read -r key value; do
        case "$key" in
            random_trials) RANDOM_TRIALS="$value" ;;
            random_seed) RANDOM_SEED="$value" ;;
            fdr_alpha) FDR_ALPHA="$value" ;;
            top_pairs) TOP_PAIRS="$value" ;;
            null_chunk_count) NULL_CHUNK_COUNT="$value" ;;
        esac
    done < <(
        "${PYTHON_BIN}" - "$META_PATH" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
for k in ("random_trials", "random_seed", "fdr_alpha", "top_pairs", "null_chunk_count"):
    v = meta.get(k, "")
    print(f"{k}={v}")
PY
    )
fi

if [ -z "${NULL_CHUNK_COUNT:-}" ]; then
    NULL_CHUNK_COUNT="${ARRAY_SIZE}"
    if [ "${NULL_CHUNK_COUNT}" -lt 1 ]; then
        NULL_CHUNK_COUNT=1
    fi
    if [ "${NULL_CHUNK_COUNT}" -gt "${RANDOM_TRIALS}" ]; then
        NULL_CHUNK_COUNT="${RANDOM_TRIALS}"
    fi
fi

# ─── Status detection ───────────────────────────────────────────────────────

CRATERS_PATH="${CHECKPOINT_DIR}/craters.pkl"
TOP_PAIRS_PATH="${CHECKPOINT_DIR}/top_pairs.csv"
FINAL_SUMMARY="${GLOBAL_OUTPUT_DIR}/global_summary.json"

MISSING_CHIPS_FILE="${RESULTS_DIR}/missing_chips.txt"
MISSING_NULL_FILE="${RESULTS_DIR}/missing_null_chunks.txt"

> "$MISSING_CHIPS_FILE"
for ((i=0; i<TOTAL_CHIPS; i++)); do
    printf -v chip_dir "%s/chip_%04d" "$CHIP_RESULTS_DIR" "$i"
    if [ ! -f "${chip_dir}/craters.csv" ]; then
        echo "$i" >> "$MISSING_CHIPS_FILE"
    fi
done
MISSING_CHIPS_COUNT=$(wc -l < "$MISSING_CHIPS_FILE" | tr -d ' ')

CHIPS_COMPLETE=1
if [ "$MISSING_CHIPS_COUNT" -gt 0 ]; then
    CHIPS_COMPLETE=0
fi

PREP_DONE=0
if [ -f "$CRATERS_PATH" ] && [ -f "$TOP_PAIRS_PATH" ] && [ -f "$META_PATH" ]; then
    PREP_DONE=1
fi

> "$MISSING_NULL_FILE"
MISSING_NULL_COUNT=0
NULL_DONE=0
if [ "$PREP_DONE" -eq 1 ]; then
    for ((i=0; i<NULL_CHUNK_COUNT; i++)); do
        printf -v part_path "%s/null_scores_part_%05d.npy" "$CHECKPOINT_DIR" "$i"
        if [ ! -f "$part_path" ]; then
            echo "$i" >> "$MISSING_NULL_FILE"
        fi
    done
    MISSING_NULL_COUNT=$(wc -l < "$MISSING_NULL_FILE" | tr -d ' ')
    if [ "$MISSING_NULL_COUNT" -eq 0 ]; then
        NULL_DONE=1
    fi
else
    MISSING_NULL_COUNT="$NULL_CHUNK_COUNT"
fi

FINAL_DONE=0
if [ -f "$FINAL_SUMMARY" ]; then
    FINAL_DONE=1
fi

FORCE_REBUILD=0
if [ "$CHIPS_COMPLETE" -eq 0 ]; then
    FORCE_REBUILD=1
    PREP_DONE=0
    NULL_DONE=0
    FINAL_DONE=0
elif [ "$PREP_DONE" -eq 0 ]; then
    FORCE_REBUILD=1
    NULL_DONE=0
    FINAL_DONE=0
elif [ "$NULL_DONE" -eq 0 ]; then
    FINAL_DONE=0
fi

if [ "$FORCE_REBUILD" -eq 1 ]; then
    RESUME_FLAG=""
fi

# ─── Previous-run diagnostics ──────────────────────────────────────────────

CHIPS_OK=0
CHIPS_EMPTY=0
CHIPS_ERROR_COUNT=0
if [ -d "$CHIP_RESULTS_DIR" ]; then
    for ((i=0; i<TOTAL_CHIPS; i++)); do
        printf -v chip_dir "%s/chip_%04d" "$CHIP_RESULTS_DIR" "$i"
        meta_file="${chip_dir}/metadata.json"
        if [ -f "$meta_file" ]; then
            chip_status=$("${PYTHON_BIN}" -c "
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    print(m.get('status','unknown'), m.get('n_craters',0))
except: print('unreadable 0')
" "$meta_file" 2>/dev/null || echo "unreadable 0")
            read -r cs cn <<< "$chip_status"
            if [ "$cs" = "ok" ] && [ "$cn" -gt 0 ] 2>/dev/null; then
                CHIPS_OK=$((CHIPS_OK + 1))
            elif [ "$cs" = "ok" ] || [ "$cs" = "no_data" ] || [ "$cs" = "blank_chip" ] || \
                 [ "$cs" = "noise_floor" ] || [ "$cs" = "invalid_bbox" ] || [ "$cs" = "chip_download_failed" ]; then
                CHIPS_EMPTY=$((CHIPS_EMPTY + 1))
            else
                CHIPS_ERROR_COUNT=$((CHIPS_ERROR_COUNT + 1))
            fi
        fi
    done
fi

# ─── Banner ─────────────────────────────────────────────────────────────────

echo "================================================================"
echo "MOONPIERCER Resume Pipeline"
echo "================================================================"
echo "Project dir:      $PROJECT_DIR"
echo "Results dir:      $RESULTS_DIR"
echo "Manifest:         $MANIFEST_PATH"
echo "Total chips:      $TOTAL_CHIPS"
echo "Random trials:    $RANDOM_TRIALS"
echo "Null chunks:      $NULL_CHUNK_COUNT"
echo "Progress (sec):   $PROGRESS_INTERVAL_SEC"
echo "Cache dir:        $CACHE_DIR"
echo "Python bin:       $PYTHON_BIN"
echo "Null memory:      64G (sequential, 1 CPU per chunk)"
echo "================================================================"
echo ""
echo "── Previous Run Accounting ──────────────────────────────────────"
COMPLETED_CHIPS=$((TOTAL_CHIPS - MISSING_CHIPS_COUNT))
echo "  Chip processing:       ${COMPLETED_CHIPS}/${TOTAL_CHIPS} completed"
echo "    With craters:          $CHIPS_OK"
echo "    Empty/no-data:         $CHIPS_EMPTY"
echo "    Errors:                $CHIPS_ERROR_COUNT"
echo "    Missing entirely:      $MISSING_CHIPS_COUNT"
echo ""
echo "  Prep stage:            $([ $PREP_DONE -eq 1 ] && echo 'DONE' || echo 'NEEDED')"
NULL_COMPLETED=$((NULL_CHUNK_COUNT - MISSING_NULL_COUNT))
echo "  Null model:            ${NULL_COMPLETED}/${NULL_CHUNK_COUNT} chunks"
if [ "$MISSING_NULL_COUNT" -gt 0 ] && [ "$MISSING_NULL_COUNT" -le 15 ]; then
    echo "    Missing chunks:      $(cat "$MISSING_NULL_FILE" | tr '\n' ', ' | sed 's/,$//')"
fi
echo "  Final aggregation:     $([ $FINAL_DONE -eq 1 ] && echo 'DONE' || echo 'NEEDED')"
echo ""
echo "  What will be re-submitted:"
[ "$CHIPS_COMPLETE" -eq 0 ] && echo "    - ${MISSING_CHIPS_COUNT} missing chips"
[ "$PREP_DONE" -eq 0 ] && echo "    - Prep stage (full rebuild)"
if [ "$NULL_DONE" -eq 0 ]; then
    if [ "$PREP_DONE" -eq 0 ]; then
        echo "    - All ${NULL_CHUNK_COUNT} null chunks (after prep)"
    elif [ "$MISSING_NULL_COUNT" -gt 0 ]; then
        echo "    - ${MISSING_NULL_COUNT} missing null chunks"
    fi
fi
[ "$FINAL_DONE" -eq 0 ] && echo "    - Final aggregation"
echo "────────────────────────────────────────────────────────────────"

if [ "$CHIPS_COMPLETE" -eq 1 ] && [ "$PREP_DONE" -eq 1 ] && [ "$NULL_DONE" -eq 1 ] && [ "$FINAL_DONE" -eq 1 ]; then
    echo ""
    echo "All stages complete. Nothing to resume."
    exit 0
fi

# ─── Stage 2: Missing chips ────────────────────────────────────────────────

CHIP_JOB_ID=""
if [ "$CHIPS_COMPLETE" -eq 0 ]; then
    echo ""
    echo "[Stage 2] Submitting missing chip tasks (${MISSING_CHIPS_COUNT} chips)..."
    echo ""

    CHIPS_PER_TASK=$(( (MISSING_CHIPS_COUNT + ARRAY_SIZE - 1) / ARRAY_SIZE ))
    if [ "$CHIPS_PER_TASK" -lt 1 ]; then
        CHIPS_PER_TASK=1
    fi
    CHIP_TASKS=$(( (MISSING_CHIPS_COUNT + CHIPS_PER_TASK - 1) / CHIPS_PER_TASK ))
    ARRAY_END=$((CHIP_TASKS - 1))

    CHIP_WRAPPER="${RESULTS_DIR}/resume_missing_chips.sh"
    cat > "$CHIP_WRAPPER" <<'EOF'
#!/bin/bash
set -euo pipefail

: "${MISSING_CHIPS_FILE:?Missing MISSING_CHIPS_FILE}"
: "${CHIPS_PER_TASK:?Missing CHIPS_PER_TASK}"
: "${HPC_DIR:?Missing HPC_DIR}"
: "${MANIFEST_PATH:?Missing MANIFEST_PATH}"
: "${CHIP_RESULTS_DIR:?Missing CHIP_RESULTS_DIR}"
: "${CACHE_DIR:?Missing CACHE_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * CHIPS_PER_TASK))
END=$((START + CHIPS_PER_TASK - 1))

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

i=0
while IFS= read -r chip_idx; do
    if [ "$i" -lt "$START" ]; then
        i=$((i + 1))
        continue
    fi
    if [ "$i" -gt "$END" ]; then
        break
    fi
    echo "[resume] Processing missing chip_index=$chip_idx (list offset $i)"
    "${PYTHON_BIN}" "${HPC_DIR}/chip_worker.py" \
        --manifest-path "${MANIFEST_PATH}" \
        --output-dir "${CHIP_RESULTS_DIR}" \
        --chip-size-px 1024 \
        --chip-span-m 1200.0 \
        --cache-dir "${CACHE_DIR}" \
        --chip-index "${chip_idx}"
    i=$((i + 1))
done < "${MISSING_CHIPS_FILE}"
EOF
    chmod +x "$CHIP_WRAPPER"

    CHIP_JOB_ID=$(sbatch \
        --parsable \
        --export=ALL,MISSING_CHIPS_FILE="${MISSING_CHIPS_FILE}",CHIPS_PER_TASK="${CHIPS_PER_TASK}",HPC_DIR="${HPC_DIR}",MANIFEST_PATH="${MANIFEST_PATH}",CHIP_RESULTS_DIR="${CHIP_RESULTS_DIR}",CACHE_DIR="${CACHE_DIR}" \
        --job-name=mp_resume_chip \
        --output="${CHIP_LOG_DIR}/chip_resume_%A_%a.out" \
        --error="${CHIP_LOG_DIR}/chip_resume_%A_%a.err" \
        --time=10:00:00 \
        --mem=4G \
        --cpus-per-task=1 \
        --array="0-${ARRAY_END}" \
        "$CHIP_WRAPPER"
    )

    echo "  Missing chip array submitted: $CHIP_JOB_ID (${CHIP_TASKS} tasks × ${CHIPS_PER_TASK} chips/task)"
fi

# ─── Stage 3a: Prep ─────────────────────────────────────────────────────────

PREP_JOB_ID=""
if [ "$PREP_DONE" -eq 0 ]; then
    echo ""
    echo "[Stage 3a] Submitting global aggregation prep..."
    echo ""

    DEP_OPTS=()
    if [ -n "$CHIP_JOB_ID" ]; then
        DEP_OPTS=(--dependency=afterok:${CHIP_JOB_ID} --kill-on-invalid-dep=yes)
    fi

    PREP_JOB_ID=$(sbatch \
        --parsable \
        --export=ALL \
        "${DEP_OPTS[@]}" \
        --job-name=mp_resume_prep \
        --output="${STAGE_LOG_DIR}/global_prep_%j.out" \
        --error="${STAGE_LOG_DIR}/global_prep_%j.err" \
        --time=24:00:00 \
        --mem=32G \
        --cpus-per-task=8 \
        "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
            --mode prep \
            --chip-results-dir "${CHIP_RESULTS_DIR}" \
            --output-dir "${GLOBAL_OUTPUT_DIR}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --random-trials "${RANDOM_TRIALS}" \
            --random-seed "${RANDOM_SEED}" \
            --fdr-alpha "${FDR_ALPHA}" \
            --top-pairs "${TOP_PAIRS}" \
            --null-chunk-count "${NULL_CHUNK_COUNT}" \
            --progress-interval-sec "${PROGRESS_INTERVAL_SEC}" \
            ${RESUME_FLAG}
    )

    echo "  Prep job submitted: $PREP_JOB_ID"
fi

# ─── Stage 3b: Null model ───────────────────────────────────────────────────

NULL_JOB_ID=""
if [ "$NULL_DONE" -eq 0 ]; then
    echo ""
    echo "[Stage 3b] Submitting null-model work..."
    echo ""

    DEP_OPTS=()
    if [ -n "$PREP_JOB_ID" ]; then
        DEP_OPTS=(--dependency=afterok:${PREP_JOB_ID} --kill-on-invalid-dep=yes)
    fi

    if [ "$PREP_DONE" -eq 0 ]; then
        NULL_ARRAY_END=$((NULL_CHUNK_COUNT - 1))
        NULL_JOB_ID=$(sbatch \
            --parsable \
            --export=ALL \
            "${DEP_OPTS[@]}" \
            --job-name=mp_resume_null \
            --output="${NULL_LOG_DIR}/null_%A_%a.out" \
            --error="${NULL_LOG_DIR}/null_%A_%a.err" \
            --time=24:00:00 \
            --mem=64G \
            --cpus-per-task=1 \
            --array="0-${NULL_ARRAY_END}" \
            "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
                --mode null \
                --chip-results-dir "${CHIP_RESULTS_DIR}" \
                --output-dir "${GLOBAL_OUTPUT_DIR}" \
                --checkpoint-dir "${CHECKPOINT_DIR}" \
                --random-trials "${RANDOM_TRIALS}" \
                --random-seed "${RANDOM_SEED}" \
                --fdr-alpha "${FDR_ALPHA}" \
                --top-pairs "${TOP_PAIRS}" \
                --null-chunk-count "${NULL_CHUNK_COUNT}" \
                --progress-interval-sec "${PROGRESS_INTERVAL_SEC}" \
                ${RESUME_FLAG}
        )
        echo "  Null-model array submitted: $NULL_JOB_ID (${NULL_CHUNK_COUNT} chunks)"
    else
        if [ "$MISSING_NULL_COUNT" -gt 0 ]; then
            NULL_WRAPPER="${RESULTS_DIR}/resume_missing_null_chunks.sh"
            cat > "$NULL_WRAPPER" <<'EOF'
#!/bin/bash
set -euo pipefail

: "${MISSING_NULL_FILE:?Missing MISSING_NULL_FILE}"
: "${HPC_DIR:?Missing HPC_DIR}"
: "${CHIP_RESULTS_DIR:?Missing CHIP_RESULTS_DIR}"
: "${GLOBAL_OUTPUT_DIR:?Missing GLOBAL_OUTPUT_DIR}"
: "${CHECKPOINT_DIR:?Missing CHECKPOINT_DIR}"
: "${RANDOM_TRIALS:?Missing RANDOM_TRIALS}"
: "${RANDOM_SEED:?Missing RANDOM_SEED}"
: "${FDR_ALPHA:?Missing FDR_ALPHA}"
: "${TOP_PAIRS:?Missing TOP_PAIRS}"
: "${NULL_CHUNK_COUNT:?Missing NULL_CHUNK_COUNT}"
: "${PROGRESS_INTERVAL_SEC:?Missing PROGRESS_INTERVAL_SEC}"

RESUME_FLAG="${RESUME_FLAG:-}"

# shellcheck disable=SC1091
. "${HPC_DIR}/setup_env.sh"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
chunk_idx="$(sed -n "$((TASK_ID + 1))p" "${MISSING_NULL_FILE}")"
if [ -z "$chunk_idx" ]; then
    echo "ERROR: missing chunk index for task ${TASK_ID}" >&2
    exit 1
fi

echo "[resume] Processing missing null chunk ${chunk_idx}"
"${PYTHON_BIN}" "${HPC_DIR}/global_aggregation.py" \
    --mode null \
    --chip-results-dir "${CHIP_RESULTS_DIR}" \
    --output-dir "${GLOBAL_OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --random-trials "${RANDOM_TRIALS}" \
    --random-seed "${RANDOM_SEED}" \
    --fdr-alpha "${FDR_ALPHA}" \
    --top-pairs "${TOP_PAIRS}" \
    --null-chunk-count "${NULL_CHUNK_COUNT}" \
    --null-chunk-index "${chunk_idx}" \
    --progress-interval-sec "${PROGRESS_INTERVAL_SEC}" \
    ${RESUME_FLAG}
EOF
            chmod +x "$NULL_WRAPPER"

            NULL_ARRAY_END=$((MISSING_NULL_COUNT - 1))
            NULL_JOB_ID=$(sbatch \
                --parsable \
                --export=ALL,MISSING_NULL_FILE="${MISSING_NULL_FILE}",HPC_DIR="${HPC_DIR}",CHIP_RESULTS_DIR="${CHIP_RESULTS_DIR}",GLOBAL_OUTPUT_DIR="${GLOBAL_OUTPUT_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR}",RANDOM_TRIALS="${RANDOM_TRIALS}",RANDOM_SEED="${RANDOM_SEED}",FDR_ALPHA="${FDR_ALPHA}",TOP_PAIRS="${TOP_PAIRS}",NULL_CHUNK_COUNT="${NULL_CHUNK_COUNT}",PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC}",RESUME_FLAG="${RESUME_FLAG}" \
                "${DEP_OPTS[@]}" \
                --job-name=mp_resume_null \
                --output="${NULL_LOG_DIR}/null_%A_%a.out" \
                --error="${NULL_LOG_DIR}/null_%A_%a.err" \
                --time=24:00:00 \
                --mem=64G \
                --cpus-per-task=1 \
                --array="0-${NULL_ARRAY_END}" \
                "$NULL_WRAPPER"
            )

            echo "  Missing null array submitted: $NULL_JOB_ID (${MISSING_NULL_COUNT} chunks)"
        fi
    fi
fi

# ─── Stage 3c: Final aggregation ────────────────────────────────────────────

FINAL_JOB_ID=""
if [ "$FINAL_DONE" -eq 0 ]; then
    echo ""
    echo "[Stage 3c] Submitting final aggregation..."
    echo ""

    DEP_OPTS=()
    if [ -n "$NULL_JOB_ID" ]; then
        DEP_OPTS=(--dependency=afterok:${NULL_JOB_ID} --kill-on-invalid-dep=yes)
    elif [ -n "$PREP_JOB_ID" ]; then
        DEP_OPTS=(--dependency=afterok:${PREP_JOB_ID} --kill-on-invalid-dep=yes)
    fi

    FINAL_JOB_ID=$(sbatch \
        --parsable \
        --export=ALL \
        "${DEP_OPTS[@]}" \
        --job-name=mp_resume_final \
        --output="${STAGE_LOG_DIR}/global_final_%j.out" \
        --error="${STAGE_LOG_DIR}/global_final_%j.err" \
        --time=12:00:00 \
        --mem=16G \
        --cpus-per-task=4 \
        "${HPC_DIR}/setup_env.sh" python3 "${HPC_DIR}/global_aggregation.py" \
            --mode final \
            --chip-results-dir "${CHIP_RESULTS_DIR}" \
            --output-dir "${GLOBAL_OUTPUT_DIR}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --random-trials "${RANDOM_TRIALS}" \
            --random-seed "${RANDOM_SEED}" \
            --fdr-alpha "${FDR_ALPHA}" \
            --top-pairs "${TOP_PAIRS}" \
            --null-chunk-count "${NULL_CHUNK_COUNT}" \
            --save-pair-images \
            ${RESUME_FLAG}
    )

    echo "  Final aggregation job submitted: $FINAL_JOB_ID"
fi

# ─── Cleanup ────────────────────────────────────────────────────────────────

if [ -n "$FINAL_JOB_ID" ]; then
    CLEANUP_JOB_ID=$(sbatch \
        --parsable \
        --export=ALL \
        --dependency=afterany:${FINAL_JOB_ID} \
        --kill-on-invalid-dep=yes \
        --job-name=mp_resume_cleanup \
        --output=/dev/null \
        --error=/dev/null \
        --time=00:15:00 \
        --mem=256M \
        --wrap="find ${LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; find ${CHIP_LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; find ${NULL_LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; find ${STAGE_LOG_DIR} -name '*.err' -empty -delete 2>/dev/null; echo 'Cleaned empty .err files'"
    )
fi

echo ""
echo "================================================================"
echo "Resume pipeline submitted:"
echo "  Missing chips:         ${CHIP_JOB_ID:-none}"
echo "  Prep:                  ${PREP_JOB_ID:-none}"
echo "  Null model:            ${NULL_JOB_ID:-none}"
echo "  Final aggregation:     ${FINAL_JOB_ID:-none}"
if [ -n "${CLEANUP_JOB_ID:-}" ]; then
    echo "  Cleanup:               ${CLEANUP_JOB_ID}"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  python hpc/progress.py ${RESULTS_DIR}"
echo "  python hpc/progress.py ${RESULTS_DIR} --watch 60"
echo "================================================================"
