#!/bin/bash
# ==============================================================================
# run_final_sweep.slurm
# Master driver for the final comprehensive sweep:
#   - MIMIC-III: Binary / Ordinal / Multiclass  (original cohort filters)
#   - eICU:      Binary / Ordinal / Multiclass
# Produces 6 summary CSVs (one per dataset × task).
#
# Pipeline:
#   Stage 0a: MIMIC-III build (--cohort_filter original) → impute → preprocess
#   Stage 0b: eICU build → impute → preprocess          (parallel with 0a)
#   Stage 1:  36 GPU training jobs (6 models × 3 tasks × 2 datasets)
#             — all parallel once their dataset's preprocess finishes
#   Stage 2:  6 combine jobs (afterany — partial failures still produce CSV)
#   Stage 3:  Final report job (prints all 6 CSV paths + pass/fail status)
#
# Robustness:
#   - Script existence checked before ANY job is submitted (fails fast and
#     clearly rather than producing DependencyNeverSatisfied chains)
#   - Data-pipeline jobs chained with afterok (no point training on bad data)
#   - All combine/report jobs use afterany (always run, even if some models fail)
#   - Each combine job passes --results_dir explicitly (no cwd assumptions)
#   - A per-task EXTRA_ARGS array lets individual models override hparams
#     without touching the rest (all empty = true apples-to-apples comparison)
#
# Usage:  bash run_final_sweep.slurm
# ==============================================================================

set -euo pipefail

# ==============================================================================
# USER CONFIG — edit this block for your account and cluster
# ==============================================================================
ACCOUNT=""            # e.g. "my_pi_lab" — leave empty if not required
PARTITION_CPU="main"
PARTITION_GPU="gpu"
GPUS_PER_JOB="1"
IMPUTER="median"       # knn (slow) | mean | median
COHORT_FILTER="original"   # MIMIC-III cohort filter: original | strict
EICU_SAMPLE_PATIENTS="0"   # eICU build patient cap: 0 = FULL cohort; N>0 = quick subsample

# eICU raw data location on Wahab (the directory containing eICU CSV files).
# Override here if your copy is not at <project_root>/eicu_data
EICU_DATA_DIR=""       # leave empty to use the __file__-relative default

# ==============================================================================
# Paths — auto-resolved, no editing needed
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMIC_DIR="${SCRIPT_DIR}/MIMIC_III"
EICU_DIR="${SCRIPT_DIR}/EICU"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MIMIC_RESULTS_BIN="${PROJECT_ROOT}/results_LOS_dynamic"
MIMIC_RESULTS_ORD="${PROJECT_ROOT}/results_LOS_dynamic"
MIMIC_RESULTS_MC="${PROJECT_ROOT}/results_timeseries"
EICU_RESULTS="${PROJECT_ROOT}/results_eicu"

LOG_DIR="${SCRIPT_DIR}/logs_sweep"
mkdir -p "${LOG_DIR}/output" "${LOG_DIR}/error"

# ==============================================================================
# Environments (Wahab crun-based — same as run_all_timeseries.slurm)
# ==============================================================================
DATAPREP_ENV="$HOME/envs/MIMIC-dataprep"
TRAIN_ENV="$HOME/envs/MIMIC-train"

ENV_SETUP="module load container_env python3/2024.2-py310; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} VECLIB_MAXIMUM_THREADS=\${SLURM_CPUS_PER_TASK:-1};"
PYRUN_DATAPREP="crun -p ${DATAPREP_ENV} python"
PYRUN_TRAIN="crun -p ${TRAIN_ENV} python"

ACCOUNT_ARGS=()
[[ -n "${ACCOUNT}" ]] && ACCOUNT_ARGS=(--account="${ACCOUNT}")

# ==============================================================================
# Architecture arrays — 6 models, same subdirectory names for all tasks
# ==============================================================================
declare -a TAGS=(bilstm hybrid ome patchtst puremamba transformer)

declare -a SUBDIRS=(
    "BiLSTM_with_static_ANN_fusion"
    "Hybrid_BiLSTM_BiMAMBA_Static_ANN_Fusion"
    "OME_BiLSTM_Transformer_Mamba_Static_ANN"
    "PatchTST_style_Transformer_Static_ANN_Fusion"
    "Pure_Mamba_Static_ANN_Fusion"
    "Transformer_Static_ANN_Fusion"
)

declare -a MIMIC_BIN_SCRIPTS=(
    "V1_train_mimic3_lstm_fusion_binary_auc.py"
    "V1_train_mimic3_hybrid_lstm_mamba_binary_auc.py"
    "V1_train_mimic3_ome_fusion_binary_auc.py"
    "V1_train_mimic3_patchtst_fusion_binary_auc.py"
    "V1_train_mimic3_pure_mamba_binary_auc.py"
    "V1_train_mimic3_transformer_fusion_binary_auc.py"
)
declare -a MIMIC_ORD_SCRIPTS=(
    "V1_train_mimic3_lstm_fusion_ordinal_auc.py"
    "V1_train_mimic3_hybrid_lstm_mamba_ordinal_auc.py"
    "V1_train_mimic3_ome_fusion_ordinal_auc.py"
    "V1_train_mimic3_patchtst_fusion_ordinal_auc.py"
    "V1_train_mimic3_pure_mamba_ordinal_auc.py"
    "V1_train_mimic3_transformer_fusion_ordinal_auc.py"
)
declare -a MIMIC_MC_SCRIPTS=(
    "V1_train_mimic3_lstm_fusion_multiclass_auc.py"
    "V1_train_mimic3_hybrid_lstm_mamba_multiclass_auc.py"
    "V1_train_mimic3_ome_fusion_multiclass_auc.py"
    "V1_train_mimic3_patchtst_fusion_multiclass_auc.py"
    "V1_train_mimic3_pure_mamba_multiclass_auc.py"
    "V1_train_mimic3_transformer_fusion_multiclass_auc.py"
)
declare -a EICU_BIN_SCRIPTS=(
    "V1_train_eicu_lstm_fusion_binary_auc.py"
    "V1_train_eicu_hybrid_lstm_mamba_binary_auc.py"
    "V1_train_eicu_ome_fusion_binary_auc.py"
    "V1_train_eicu_patchtst_fusion_binary_auc.py"
    "V1_train_eicu_pure_mamba_binary_auc.py"
    "V1_train_eicu_transformer_fusion_binary_auc.py"
)
declare -a EICU_ORD_SCRIPTS=(
    "V1_train_eicu_lstm_fusion_ordinal_auc.py"
    "V1_train_eicu_hybrid_lstm_mamba_ordinal_auc.py"
    "V1_train_eicu_ome_fusion_ordinal_auc.py"
    "V1_train_eicu_patchtst_fusion_ordinal_auc.py"
    "V1_train_eicu_pure_mamba_ordinal_auc.py"
    "V1_train_eicu_transformer_fusion_ordinal_auc.py"
)
declare -a EICU_MC_SCRIPTS=(
    "V1_train_eicu_lstm_fusion_multiclass_auc.py"
    "V1_train_eicu_hybrid_lstm_mamba_multiclass_auc.py"
    "V1_train_eicu_ome_fusion_multiclass_auc.py"
    "V1_train_eicu_patchtst_fusion_multiclass_auc.py"
    "V1_train_eicu_pure_mamba_multiclass_auc.py"
    "V1_train_eicu_transformer_fusion_multiclass_auc.py"
)

# Per-model extra args: all empty = true apples-to-apples comparison.
# Add overrides here if a specific model needs them, e.g.:
#   EXTRA_ARGS[4]="--lr 3e-4 --grad_clip 0.5"
declare -a EXTRA_ARGS=("" "" "" "" "" "")

# ==============================================================================
# Pre-flight: verify ALL scripts exist before submitting a single job
# ==============================================================================
echo "[CHECK] Verifying all training scripts exist..."
MISSING=0
for i in "${!TAGS[@]}"; do
    for task_dir in \
        "${MIMIC_DIR}/Training/Binary/${SUBDIRS[$i]}/${MIMIC_BIN_SCRIPTS[$i]}" \
        "${MIMIC_DIR}/Training/Ordinal/${SUBDIRS[$i]}/${MIMIC_ORD_SCRIPTS[$i]}" \
        "${MIMIC_DIR}/Training/Multiclass/${SUBDIRS[$i]}/${MIMIC_MC_SCRIPTS[$i]}" \
        "${EICU_DIR}/Training/Binary/${SUBDIRS[$i]}/${EICU_BIN_SCRIPTS[$i]}" \
        "${EICU_DIR}/Training/Ordinal/${SUBDIRS[$i]}/${EICU_ORD_SCRIPTS[$i]}" \
        "${EICU_DIR}/Training/Multiclass/${SUBDIRS[$i]}/${EICU_MC_SCRIPTS[$i]}"
    do
        if [[ ! -f "${task_dir}" ]]; then
            echo "  [MISSING] ${task_dir}" >&2
            MISSING=$((MISSING + 1))
        fi
    done
done

for combine_script in \
    "${MIMIC_DIR}/Training/Binary/combine_results.py" \
    "${MIMIC_DIR}/Training/Ordinal/combine_results.py" \
    "${MIMIC_DIR}/Training/Multiclass/combine_results.py" \
    "${EICU_DIR}/Training/Binary/combine_results.py" \
    "${EICU_DIR}/Training/Ordinal/combine_results.py" \
    "${EICU_DIR}/Training/Multiclass/combine_results.py"
do
    [[ ! -f "${combine_script}" ]] && { echo "  [MISSING] ${combine_script}" >&2; MISSING=$((MISSING + 1)); }
done

if [[ ${MISSING} -gt 0 ]]; then
    echo "[ERROR] ${MISSING} required script(s) missing. Fix before rerunning." >&2
    exit 1
fi
echo "[CHECK] All scripts found. Submitting..."

# ==============================================================================
# Submission helpers
# ==============================================================================

# submit_data_job <job_name> <time> <mem> <cpus> <dep_arg> <wrap_cmd>
submit_data_job () {
    local name=$1 time=$2 mem=$3 cpus=$4 dep_arg=$5 wrap=$6
    local dep_flags=()
    [[ -n "${dep_arg}" ]] && dep_flags=(--dependency="${dep_arg}")
    sbatch --parsable \
           "${ACCOUNT_ARGS[@]}" \
           --partition="${PARTITION_CPU}" \
           "${dep_flags[@]}" \
           --job-name="${name}" \
           --output="${LOG_DIR}/output/${name}_%j.log" \
           --error="${LOG_DIR}/error/${name}_%j.log" \
           --cpus-per-task="${cpus}" \
           --mem="${mem}" \
           --time="${time}" \
           --wrap="${wrap}"
}

# submit_train_job <job_name> <script_path> <dep_id> <extra_args>
submit_train_job () {
    local name=$1 script=$2 dep=$3 extra=$4
    sbatch --parsable \
           "${ACCOUNT_ARGS[@]}" \
           --partition="${PARTITION_GPU}" \
           --gres=gpu:"${GPUS_PER_JOB}" \
           --dependency=afterok:"${dep}" \
           --job-name="${name}" \
           --output="${LOG_DIR}/output/${name}_%j.log" \
           --error="${LOG_DIR}/error/${name}_%j.log" \
           --cpus-per-task=4 \
           --mem=32G \
           --time=06:00:00 \
           --wrap="${ENV_SETUP} ${PYRUN_TRAIN} '${script}' --workers 4 ${extra}"
}

# submit_combine_job <job_name> <dep_list_colon> <combine_script> <results_dir> <out_csv>
submit_combine_job () {
    local name=$1 dep=$2 script=$3 results_dir=$4 out_csv=$5
    sbatch --parsable \
           "${ACCOUNT_ARGS[@]}" \
           --partition="${PARTITION_CPU}" \
           --dependency=afterany:"${dep}" \
           --job-name="${name}" \
           --output="${LOG_DIR}/output/${name}_%j.log" \
           --error="${LOG_DIR}/error/${name}_%j.log" \
           --cpus-per-task=1 \
           --mem=4G \
           --time=00:15:00 \
           --wrap="${ENV_SETUP} ${PYRUN_DATAPREP} '${script}' --results_dir '${results_dir}' --out_csv '${results_dir}/${out_csv}'"
}

# ==============================================================================
# Stage 0a: MIMIC-III data pipeline
# ==============================================================================
echo ""
echo "[STAGE 0a] Submitting MIMIC-III data pipeline (cohort_filter=${COHORT_FILTER})..."

MIMIC_BUILD=$(submit_data_job "sw_mb_build" "08:00:00" "64G" "4" "" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${MIMIC_DIR}/V12.1_build_script.py' --cohort_filter ${COHORT_FILTER}")

MIMIC_IMPUTE=$(submit_data_job "sw_mb_impute" "04:00:00" "32G" "8" "afterok:${MIMIC_BUILD}" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${MIMIC_DIR}/V12.1_impute_script.py' --imputer ${IMPUTER}")

MIMIC_PREP=$(submit_data_job "sw_mb_prep" "01:30:00" "32G" "4" "afterok:${MIMIC_IMPUTE}" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${MIMIC_DIR}/V12.1_preprocess_script.py'")

echo "  build=${MIMIC_BUILD} -> impute=${MIMIC_IMPUTE} -> preprocess=${MIMIC_PREP}"

# ==============================================================================
# Stage 0b: eICU data pipeline (runs in parallel with 0a)
# ==============================================================================
echo ""
echo "[STAGE 0b] Submitting eICU data pipeline..."

EICU_DATA_ARG=""
[[ -n "${EICU_DATA_DIR}" ]] && EICU_DATA_ARG="--data-dir '${EICU_DATA_DIR}'"

EICU_BUILD=$(submit_data_job "sw_ei_build" "06:00:00" "64G" "4" "" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${EICU_DIR}/V2.1_build_eicu_los_dataset_corrected_leakage.py' --sample-patients ${EICU_SAMPLE_PATIENTS} ${EICU_DATA_ARG}")

EICU_IMPUTE=$(submit_data_job "sw_ei_impute" "04:00:00" "32G" "8" "afterok:${EICU_BUILD}" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${EICU_DIR}/V1_impute_eicu_los_dataset.py'")

EICU_PREP=$(submit_data_job "sw_ei_prep" "01:30:00" "32G" "4" "afterok:${EICU_IMPUTE}" \
    "${ENV_SETUP} ${PYRUN_DATAPREP} '${EICU_DIR}/V1_preprocess_eicu_LOS_data.py'")

echo "  build=${EICU_BUILD} -> impute=${EICU_IMPUTE} -> preprocess=${EICU_PREP}"

# ==============================================================================
# Stage 1: Training — 6 models × 3 tasks × 2 datasets = 36 GPU jobs
# ==============================================================================
echo ""
echo "[STAGE 1] Submitting 36 training jobs..."

MIMIC_BIN_IDS=(); MIMIC_ORD_IDS=(); MIMIC_MC_IDS=()
EICU_BIN_IDS=();  EICU_ORD_IDS=();  EICU_MC_IDS=()

for i in "${!TAGS[@]}"; do
    tag="${TAGS[$i]}"
    subdir="${SUBDIRS[$i]}"
    extra="${EXTRA_ARGS[$i]}"

    # MIMIC-III Binary
    jid=$(submit_train_job "sw_mb_bin_${tag}" \
          "${MIMIC_DIR}/Training/Binary/${subdir}/${MIMIC_BIN_SCRIPTS[$i]}" \
          "${MIMIC_PREP}" "${extra}")
    MIMIC_BIN_IDS+=("${jid}")

    # MIMIC-III Ordinal
    jid=$(submit_train_job "sw_mb_ord_${tag}" \
          "${MIMIC_DIR}/Training/Ordinal/${subdir}/${MIMIC_ORD_SCRIPTS[$i]}" \
          "${MIMIC_PREP}" "${extra}")
    MIMIC_ORD_IDS+=("${jid}")

    # MIMIC-III Multiclass
    jid=$(submit_train_job "sw_mb_mc_${tag}" \
          "${MIMIC_DIR}/Training/Multiclass/${subdir}/${MIMIC_MC_SCRIPTS[$i]}" \
          "${MIMIC_PREP}" "${extra}")
    MIMIC_MC_IDS+=("${jid}")

    # eICU Binary
    jid=$(submit_train_job "sw_ei_bin_${tag}" \
          "${EICU_DIR}/Training/Binary/${subdir}/${EICU_BIN_SCRIPTS[$i]}" \
          "${EICU_PREP}" "${extra}")
    EICU_BIN_IDS+=("${jid}")

    # eICU Ordinal
    jid=$(submit_train_job "sw_ei_ord_${tag}" \
          "${EICU_DIR}/Training/Ordinal/${subdir}/${EICU_ORD_SCRIPTS[$i]}" \
          "${EICU_PREP}" "${extra}")
    EICU_ORD_IDS+=("${jid}")

    # eICU Multiclass
    jid=$(submit_train_job "sw_ei_mc_${tag}" \
          "${EICU_DIR}/Training/Multiclass/${subdir}/${EICU_MC_SCRIPTS[$i]}" \
          "${EICU_PREP}" "${extra}")
    EICU_MC_IDS+=("${jid}")

    echo "  ${tag}: mb_bin=${MIMIC_BIN_IDS[-1]} mb_ord=${MIMIC_ORD_IDS[-1]} mb_mc=${MIMIC_MC_IDS[-1]}" \
         "ei_bin=${EICU_BIN_IDS[-1]} ei_ord=${EICU_ORD_IDS[-1]} ei_mc=${EICU_MC_IDS[-1]}"
done

# ==============================================================================
# Stage 2: 6 combine jobs (afterany — always run even if some models failed)
# ==============================================================================
echo ""
echo "[STAGE 2] Submitting 6 combine jobs..."

dep_mb_bin=$(IFS=:; echo "${MIMIC_BIN_IDS[*]}")
dep_mb_ord=$(IFS=:; echo "${MIMIC_ORD_IDS[*]}")
dep_mb_mc=$(IFS=:;  echo "${MIMIC_MC_IDS[*]}")
dep_ei_bin=$(IFS=:; echo "${EICU_BIN_IDS[*]}")
dep_ei_ord=$(IFS=:; echo "${EICU_ORD_IDS[*]}")
dep_ei_mc=$(IFS=:;  echo "${EICU_MC_IDS[*]}")

CMB_MB_BIN=$(submit_combine_job "sw_cmb_mb_bin" "${dep_mb_bin}" \
    "${MIMIC_DIR}/Training/Binary/combine_results.py" \
    "${MIMIC_RESULTS_BIN}" "binary_summary.csv")

CMB_MB_ORD=$(submit_combine_job "sw_cmb_mb_ord" "${dep_mb_ord}" \
    "${MIMIC_DIR}/Training/Ordinal/combine_results.py" \
    "${MIMIC_RESULTS_ORD}" "ordinal_summary.csv")

CMB_MB_MC=$(submit_combine_job "sw_cmb_mb_mc" "${dep_mb_mc}" \
    "${MIMIC_DIR}/Training/Multiclass/combine_results.py" \
    "${MIMIC_RESULTS_MC}" "multiclass_summary.csv")

CMB_EI_BIN=$(submit_combine_job "sw_cmb_ei_bin" "${dep_ei_bin}" \
    "${EICU_DIR}/Training/Binary/combine_results.py" \
    "${EICU_RESULTS}" "eicu_binary_summary.csv")

CMB_EI_ORD=$(submit_combine_job "sw_cmb_ei_ord" "${dep_ei_ord}" \
    "${EICU_DIR}/Training/Ordinal/combine_results.py" \
    "${EICU_RESULTS}" "eicu_ordinal_summary.csv")

CMB_EI_MC=$(submit_combine_job "sw_cmb_ei_mc" "${dep_ei_mc}" \
    "${EICU_DIR}/Training/Multiclass/combine_results.py" \
    "${EICU_RESULTS}" "eicu_multiclass_summary.csv")

echo "  MIMIC bin=${CMB_MB_BIN} ord=${CMB_MB_ORD} mc=${CMB_MB_MC}"
echo "  eICU  bin=${CMB_EI_BIN} ord=${CMB_EI_ORD} mc=${CMB_EI_MC}"

# ==============================================================================
# Stage 3: Final report — prints all 6 CSV paths + existence status
# ==============================================================================
ALL_CMB="${CMB_MB_BIN}:${CMB_MB_ORD}:${CMB_MB_MC}:${CMB_EI_BIN}:${CMB_EI_ORD}:${CMB_EI_MC}"

REPORT_JOB=$(sbatch --parsable \
       "${ACCOUNT_ARGS[@]}" \
       --partition="${PARTITION_CPU}" \
       --dependency=afterany:"${ALL_CMB}" \
       --job-name="sw_report" \
       --output="${LOG_DIR}/output/sw_report_%j.log" \
       --error="${LOG_DIR}/error/sw_report_%j.log" \
       --cpus-per-task=1 --mem=1G --time=00:05:00 \
       --wrap="echo '=== Final sweep results ===' && \
for csv in \
  '${MIMIC_RESULTS_BIN}/binary_summary.csv' \
  '${MIMIC_RESULTS_ORD}/ordinal_summary.csv' \
  '${MIMIC_RESULTS_MC}/multiclass_summary.csv' \
  '${EICU_RESULTS}/eicu_binary_summary.csv' \
  '${EICU_RESULTS}/eicu_ordinal_summary.csv' \
  '${EICU_RESULTS}/eicu_multiclass_summary.csv'; do \
  if [ -f \"\${csv}\" ]; then \
    echo \"  [OK]     \${csv}  (\$(wc -l < \"\${csv}\") rows)\"; \
  else \
    echo \"  [MISSING] \${csv}\"; \
  fi; \
done")

echo ""
echo "[INFO] Report job: ${REPORT_JOB}"
echo ""
echo "========================================================"
echo " Submission complete.  Track with:"
echo "   squeue -u \$USER -o '%.10i %20j %.8T %.10M %.9P'"
echo ""
echo " Expected CSVs when done:"
echo "   ${MIMIC_RESULTS_BIN}/binary_summary.csv"
echo "   ${MIMIC_RESULTS_ORD}/ordinal_summary.csv"
echo "   ${MIMIC_RESULTS_MC}/multiclass_summary.csv"
echo "   ${EICU_RESULTS}/eicu_binary_summary.csv"
echo "   ${EICU_RESULTS}/eicu_ordinal_summary.csv"
echo "   ${EICU_RESULTS}/eicu_multiclass_summary.csv"
echo "========================================================"
