#!/bin/bash
# ==============================================================================
# Driver for the MIMIC-III plain 3-class (multiclass, NOT ordinal) LOS sweep:
# trains all 6 BIBM architectures (BiLSTM, Hybrid BiLSTM+Mamba, OME-Fusion,
# PatchTST, Pure-Mamba, Transformer) on the 3-way softmax/CrossEntropy
# formulation under Training/Multiclass/, then combines all 6 models' test
# metrics into one summary CSV (micro + weighted scores, per the request).
#
# Mirrors clean_LOS/run_all.slurm's structure/conventions (ACCOUNT/PARTITION
# block, ENV_SETUP thread pinning, per-job sbatch submission, combine step
# depending on afterany so a partial sweep still gets a summary) -- but this
# sweep needs a GPU partition (mamba_ssm requires CUDA) and a single shared
# training env instead of clean_LOS's CPU-only DATAPREP_ENV/ML_ENV split.
#
# Assumes the data pipeline has already produced
# <PROJECT_ROOT>/processed_data/structured_LOS_dynamic/lstm_data_v12.1_seq/
# (V12.1_build_script.py -> V12.1_impute_script.py -> V12.1_preprocess_script.py).
#
# Usage:   bash run_all_multiclass.slurm
# (Launcher script: it calls `sbatch` itself for every job, rather than being
#  a single #SBATCH job.)
# ==============================================================================
# ODU HPC (Wahab) settings -- EDIT THIS BLOCK FOR YOUR ACCOUNT
# Reference: https://wiki.hpc.odu.edu/en/slurm (verify partition names against
# `sinfo` / `sacctmgr show user $USER` before your first real run -- Wahab's
# GPU partition name/GPU type may differ from the "gpu" placeholder below).
# ==============================================================================
set -euo pipefail

ACCOUNT=""                # e.g. ACCOUNT="my_pi_lab" -- leave empty if not required
PARTITION_GPU="gpu"        # GPU partition for all 6 training jobs (mamba_ssm needs CUDA)
GPUS_PER_JOB="1"

# Wahab wraps conda in a container -- there is no `conda activate`. ONE shared
# env here (unlike clean_LOS's DATAPREP_ENV/ML_ENV split) since every script
# in this sweep needs the same torch + mamba_ssm + scikit-learn stack:
#   module load container_env python3
#   mkdir -p ~/envs
#   crun -c -p ~/envs/MIMIC-train
#   crun -p ~/envs/MIMIC-train pip install torch scikit-learn numpy mamba-ssm
TRAIN_ENV="$HOME/envs/MIMIC-train"

ACCOUNT_ARGS=()
if [[ -n "${ACCOUNT}" ]]; then
  ACCOUNT_ARGS=(--account="${ACCOUNT}")
fi

# Same BLAS/thread-pool pinning rationale as clean_LOS/run_all.slurm: a
# numpy/sklearn op inside a 4-CPU job can otherwise spawn 40+ threads on a
# shared node and cause contention (manifests as heap corruption / SIGSEGV,
# not a clean Python error).
ENV_SETUP="module load container_env python3; export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} VECLIB_MAXIMUM_THREADS=\${SLURM_CPUS_PER_TASK:-1};"
PYRUN="crun -p ${TRAIN_ENV} python"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}/output" "${LOG_DIR}/error"

# Project root (5 levels up from this script, since this script lives
# directly in Multiclass/ rather than inside an architecture subfolder --
# the V1_train_*.py scripts are one level deeper and use 6 levels, see each
# one's _PROJECT_ROOT computation) -- combine_results.py reads from
# <PROJECT_ROOT>/results_LOS_dynamic by default, matching every
# V1_train_*.py script's OUT_DIR.
RESULTS_DIR="${SCRIPT_DIR}/../../../../../results_LOS_dynamic"

# (job_tag, subdirectory, script filename) for each of the 6 architectures.
declare -a JOB_TAGS=(bilstm hybrid ome patchtst puremamba transformer)
declare -a SUBDIRS=(
  "BiLSTM_with_static_ANN_fusion"
  "Hybrid_BiLSTM_BiMAMBA_Static_ANN_Fusion"
  "OME_BiLSTM_Transformer_Mamba_Static_ANN"
  "PatchTST_style_Transformer_Static_ANN_Fusion"
  "Pure_Mamba_Static_ANN_Fusion"
  "Transformer_Static_ANN_Fusion"
)
declare -a SCRIPTS=(
  "V1_train_mimic3_lstm_fusion_multiclass_auc.py"
  "V1_train_mimic3_hybrid_lstm_mamba_multiclass_auc.py"
  "V1_train_mimic3_ome_fusion_multiclass_auc.py"
  "V1_train_mimic3_patchtst_fusion_multiclass_auc.py"
  "V1_train_mimic3_pure_mamba_multiclass_auc.py"
  "V1_train_mimic3_transformer_fusion_multiclass_auc.py"
)

submit_train_job () {
  # args: job_tag script_path
  local job_tag=$1 script_path=$2
  local job_name="mc_${job_tag}"

  sbatch --parsable \
         "${ACCOUNT_ARGS[@]}" \
         --partition="${PARTITION_GPU}" \
         --gres=gpu:"${GPUS_PER_JOB}" \
         --job-name="${job_name}" \
         --output="${LOG_DIR}/output/${job_name}_%j.log" \
         --error="${LOG_DIR}/error/${job_name}_%j.log" \
         --cpus-per-task=4 \
         --mem=32G \
         --time=04:00:00 \
         --wrap="${ENV_SETUP} ${PYRUN} '${script_path}' --workers 4"
}

echo "[INFO] Submitting 6 multiclass training jobs (Training/Multiclass/*)..."
ALL_JOB_IDS=()

for i in "${!JOB_TAGS[@]}"; do
  tag="${JOB_TAGS[$i]}"
  script_path="${SCRIPT_DIR}/${SUBDIRS[$i]}/${SCRIPTS[$i]}"

  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Missing script: ${script_path}" >&2
    exit 1
  fi

  job_id=$(submit_train_job "${tag}" "${script_path}")
  echo "  submitted ${tag} -> job ${job_id}"
  ALL_JOB_IDS+=("${job_id}")
done

# ------------------------------------------------------------------------------
# Combine all 6 models' metrics into one CSV (micro + weighted scores) once
# every training job has finished -- afterany (not afterok) so a partial
# sweep (e.g. one model crashed) still produces a summary of whichever models
# did succeed.
# ------------------------------------------------------------------------------
DEP=$(IFS=:; echo "${ALL_JOB_IDS[*]}")
sbatch --parsable \
       "${ACCOUNT_ARGS[@]}" \
       --partition=main \
       --dependency=afterany:${DEP} \
       --job-name="mc_combine_results" \
       --output="${LOG_DIR}/output/mc_combine_results_%j.log" \
       --error="${LOG_DIR}/error/mc_combine_results_%j.log" \
       --cpus-per-task=1 \
       --mem=4G \
       --time=00:10:00 \
       --wrap="${ENV_SETUP} ${PYRUN} '${SCRIPT_DIR}/combine_results.py' --results_dir '${RESULTS_DIR}'"

echo "Submitted ${#ALL_JOB_IDS[@]} training jobs + 1 combine job."
echo "Final summary will be written to ${RESULTS_DIR}/multiclass_summary.csv"
