#!/usr/bin/env bash
# ==============================================================================
# run_pipeline.sh
# General-purpose runner for the LOS fusion-architecture benchmark on any
# Linux/Mac machine WITHOUT a SLURM scheduler. Runs every stage sequentially
# with a plain `python`. (For an HPC cluster with SLURM, use
# run_final_sweep.slurm instead, which parallelises across GPU nodes.)
#
# Pipeline per dataset:
#   build -> impute -> preprocess -> train (6 models x N tasks x N seeds)
#     -> combine into summary CSVs.
#
# Requirements (install both into the active environment):
#   pip install -r ../requirements-dataprep.txt      # CPU data prep
#   pip install -r ../requirements-train.txt         # CUDA torch + mamba_ssm
#
# Usage examples:
#   bash run_pipeline.sh                              # everything, defaults
#   bash run_pipeline.sh --datasets mimic --tasks multiclass
#   bash run_pipeline.sh --skip-build --seeds "42 43 44"
#   bash run_pipeline.sh --python /path/to/venv/bin/python
#   bash run_pipeline.sh --help
# ==============================================================================
set -euo pipefail

# ------------------------------------------------------------------------------
# Defaults (override via flags below)
# ------------------------------------------------------------------------------
PYTHON="python3"
DATASETS="mimic eicu"                 # which datasets to run
TASKS="binary ordinal multiclass"     # which tasks to run
COHORT_FILTER="original"              # MIMIC-III cohort: original | strict
IMPUTER="median"                      # knn | mean | median
EICU_SAMPLE_PATIENTS="0"              # 0 = full eICU cohort; N>0 = subsample
EICU_DATA_DIR=""                      # override eICU raw dir (else <root>/eicu_data)
SEEDS="42 43 44 45 46"                # seeds -> mean/std in the summaries
SKIP_BUILD="false"                    # true = reuse existing tensors, skip data prep

print_help() {
    grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Flags:
  --python PATH            Python executable to use (default: python3)
  --datasets "mimic eicu"  Space-separated subset of datasets
  --tasks "binary ..."     Space-separated subset of tasks
  --cohort-filter MODE     MIMIC-III cohort filter: original | strict
  --imputer MODE           knn | mean | median
  --eicu-sample N          eICU patient cap (0 = full cohort)
  --eicu-data-dir PATH     eICU raw-data directory
  --seeds "42 43 ..."      Seeds to train (summary reports mean +/- std)
  --skip-build             Skip build/impute/preprocess (reuse existing tensors)
  --help                   Show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)         PYTHON="$2"; shift 2 ;;
        --datasets)       DATASETS="$2"; shift 2 ;;
        --tasks)          TASKS="$2"; shift 2 ;;
        --cohort-filter)  COHORT_FILTER="$2"; shift 2 ;;
        --imputer)        IMPUTER="$2"; shift 2 ;;
        --eicu-sample)    EICU_SAMPLE_PATIENTS="$2"; shift 2 ;;
        --eicu-data-dir)  EICU_DATA_DIR="$2"; shift 2 ;;
        --seeds)          SEEDS="$2"; shift 2 ;;
        --skip-build)     SKIP_BUILD="true"; shift ;;
        --help|-h)        print_help; exit 0 ;;
        *) echo "[ERROR] Unknown flag: $1 (use --help)"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMIC_DIR="${SCRIPT_DIR}/MIMIC_III"
EICU_DIR="${SCRIPT_DIR}/EICU"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MIMIC_RESULTS="${PROJECT_ROOT}/results_LOS_dynamic"
EICU_RESULTS="${PROJECT_ROOT}/results_eicu"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/pipeline_${TS}"
mkdir -p "${LOG_DIR}"

# 6 architectures: (subdir, mimic-stem/eicu-stem is shared)
ARCH_SUBDIRS=(
    "BiLSTM_with_static_ANN_fusion"
    "Hybrid_BiLSTM_BiMAMBA_Static_ANN_Fusion"
    "OME_BiLSTM_Transformer_Mamba_Static_ANN"
    "PatchTST_style_Transformer_Static_ANN_Fusion"
    "Pure_Mamba_Static_ANN_Fusion"
    "Transformer_Static_ANN_Fusion"
)
ARCH_STEMS=(lstm_fusion hybrid_lstm_mamba ome_fusion patchtst_fusion pure_mamba transformer_fusion)

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_step() {  # <logfile-tag> <command...>
    local tag="$1"; shift
    log "START ${tag}"
    if ! "$@" > "${LOG_DIR}/${tag}.log" 2>&1; then
        log "FAILED ${tag} -- see ${LOG_DIR}/${tag}.log"
        return 1
    fi
    log "DONE  ${tag}"
}

# ------------------------------------------------------------------------------
# Stage 0: data pipelines
# ------------------------------------------------------------------------------
run_mimic_dataprep() {
    run_step "mimic_build"      ${PYTHON} "${MIMIC_DIR}/V12.1_build_script.py" --cohort_filter "${COHORT_FILTER}"
    run_step "mimic_impute"     ${PYTHON} "${MIMIC_DIR}/V12.1_impute_script.py" --imputer "${IMPUTER}"
    run_step "mimic_preprocess" ${PYTHON} "${MIMIC_DIR}/V12.1_preprocess_script.py"
}

run_eicu_dataprep() {
    local extra=()
    [[ -n "${EICU_DATA_DIR}" ]] && extra=(--data-dir "${EICU_DATA_DIR}")
    run_step "eicu_build"      ${PYTHON} "${EICU_DIR}/V2.1_build_eicu_los_dataset_corrected_leakage.py" --sample-patients "${EICU_SAMPLE_PATIENTS}" "${extra[@]}"
    run_step "eicu_impute"     ${PYTHON} "${EICU_DIR}/V1_impute_eicu_los_dataset.py"
    run_step "eicu_preprocess" ${PYTHON} "${EICU_DIR}/V1_preprocess_eicu_LOS_data.py"
}

# ------------------------------------------------------------------------------
# Stage 1: training (sequential over model x task x seed)
# ------------------------------------------------------------------------------
train_dataset() {  # <mimic|eicu>
    local ds="$1" dir stem_prefix
    if [[ "${ds}" == "mimic" ]]; then dir="${MIMIC_DIR}"; stem_prefix="mimic3"; else dir="${EICU_DIR}"; stem_prefix="eicu"; fi

    for task in ${TASKS}; do
        for i in "${!ARCH_SUBDIRS[@]}"; do
            local subdir="${ARCH_SUBDIRS[$i]}" stem="${ARCH_STEMS[$i]}"
            # Title-case the task for the folder name (binary -> Binary, etc.)
            local task_dir="$(tr '[:lower:]' '[:upper:]' <<< ${task:0:1})${task:1}"
            local script="${dir}/Training/${task_dir}/${subdir}/V1_train_${stem_prefix}_${stem}_${task}_auc.py"
            if [[ ! -f "${script}" ]]; then
                log "SKIP  missing ${script}"; continue
            fi
            for seed in ${SEEDS}; do
                run_step "${ds}_${task}_${stem}_seed${seed}" \
                    ${PYTHON} "${script}" --workers 4 --seed "${seed}" \
                    || log "  (continuing after failure of ${ds}_${task}_${stem}_seed${seed})"
            done
        done
    done
}

# ------------------------------------------------------------------------------
# Stage 2: combine into summary CSVs
# ------------------------------------------------------------------------------
n_seeds() { echo ${SEEDS} | wc -w; }

combine_dataset() {  # <mimic|eicu>
    local ds="$1" dir results
    if [[ "${ds}" == "mimic" ]]; then dir="${MIMIC_DIR}"; results="${MIMIC_RESULTS}"; else dir="${EICU_DIR}"; results="${EICU_RESULTS}"; fi
    for task in ${TASKS}; do
        local cmb="${dir}/Training/$(tr '[:lower:]' '[:upper:]' <<< ${task:0:1})${task:1}/combine_results.py"
        [[ -f "${cmb}" ]] || { log "SKIP combine (missing ${cmb})"; continue; }
        run_step "${ds}_combine_${task}" ${PYTHON} "${cmb}" --results_dir "${results}" --n_seeds "$(n_seeds)" \
            || log "  (combine ${ds} ${task} failed -- check its log)"
    done
}

# ==============================================================================
# Main
# ==============================================================================
log "Pipeline start | datasets='${DATASETS}' tasks='${TASKS}' seeds='${SEEDS}' skip_build=${SKIP_BUILD}"
log "Python: $(${PYTHON} --version 2>&1) | logs: ${LOG_DIR}"

for ds in ${DATASETS}; do
    if [[ "${SKIP_BUILD}" == "false" ]]; then
        log "=== ${ds}: data preparation ==="
        [[ "${ds}" == "mimic" ]] && run_mimic_dataprep || true
        [[ "${ds}" == "eicu" ]]  && run_eicu_dataprep  || true
    else
        log "=== ${ds}: skipping data prep (--skip-build) ==="
    fi
    log "=== ${ds}: training ==="
    train_dataset "${ds}"
    log "=== ${ds}: combining results ==="
    combine_dataset "${ds}"
done

log "Pipeline complete. Summary CSVs:"
for f in "${MIMIC_RESULTS}"/{binary,ordinal,multiclass}_summary.csv \
         "${EICU_RESULTS}"/eicu_{binary,ordinal,multiclass}_summary.csv; do
    [[ -f "${f}" ]] && echo "  [OK]      ${f}" || echo "  [missing] ${f}"
done
