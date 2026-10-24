# Benchmarking Deep Temporal Architectures for ICU Length of Stay Prediction on MIMIC-III and eICU

**Authors:**

- **Sibtanu Raha** — Department of Computer Science, Virginia Commonwealth University, Richmond, VA, USA (rahas2@vcu.edu)
- **Shakib Dipto** — Department of Computer Science, Old Dominion University, Norfolk, VA, USA (sdipt001@odu.edu)
- **Pratip Rana** — Department of Computer Science, Old Dominion University, Norfolk, VA, USA (prana@odu.edu)
- **Preetam Ghosh** — Department of Computer Science, Virginia Commonwealth University, Richmond, VA, USA (pghosh@vcu.edu)

# Benchmarking Fusion Architectures for ICU Length-of-Stay Prediction

Time-series deep-learning benchmark for **ICU Length-of-Stay (LOS) prediction** on
**MIMIC-III** and **eICU-CRD**, comparing six static+dynamic fusion architectures
across three task formulations.

- **Datasets:** MIMIC-III, eICU-CRD
- **Tasks:** Binary (LOS > 7 days), Ordinal (3-class cumulative-logit), Multiclass (3-class softmax)
- **Classes (3-class):** `< 3 days`, `3–7 days`, `≥ 7 days`
- **Inputs:** first-24h dynamic features (hourly vitals/labs/etc.) + static features
- **Reporting:** mean ± std over 5 seeds

## Architectures (6)

| Model | Dynamic encoder |
|-------|-----------------|
| BiLSTM + static ANN fusion | Bidirectional LSTM |
| Hybrid BiLSTM + BiMamba + static ANN | BiLSTM ⊕ bidirectional Mamba |
| OME-Fusion | attention-fused BiLSTM + Transformer + Mamba |
| PatchTST-style Transformer | patch Transformer |
| Pure BiMamba | bidirectional Mamba (state-space) |
| Transformer | CLS-token Transformer |

All six share a standardized classifier head, optimizer, scheduler, and
hyperparameters (`BIBM_versions/MIMIC_III/Training/model_common.py`) for a fair
comparison. The two Mamba-based encoders normalize their dynamic input
(`LayerNorm`) for numerical stability at the shared learning rate.

## Repository layout

```
BIBM_versions/
  MIMIC_III/
    V12.1_build_script.py          # raw MIMIC-III CSVs -> wide 24h feature table
    V12.1_impute_script.py         # leakage-safe imputation (train-only fit)
    V12.1_preprocess_script.py     # -> model-ready .npy tensors
    Training/{Binary,Ordinal,Multiclass}/<Arch>/V1_train_mimic3_*.py
    Training/{Binary,Ordinal,Multiclass}/combine_results.py
    Training/model_common.py       # shared head/optimizer/metrics/hparams
  EICU/
    V2.1_build_eicu_los_dataset_corrected_leakage.py
    V1_impute_eicu_los_dataset.py
    V1_preprocess_eicu_LOS_data.py
    Training/{Binary,Ordinal,Multiclass}/<Arch>/V1_train_eicu_*.py
    Training/{Binary,Ordinal,Multiclass}/combine_results.py
  run_pipeline.sh                  # general-server runner (no SLURM)
  run_final_sweep.slurm            # HPC/SLURM driver (parallel, job arrays)
requirements-dataprep.txt          # CPU data-prep deps
requirements-train.txt             # CUDA torch + mamba_ssm deps
```

## Data layout

Raw data and all outputs live **outside** the repo, one level above it (never
committed — see `.gitignore`):

```
<project_root>/
  sibtanu-time-series-research/    # this repo
  original_data/                   # raw MIMIC-III CSVs (ICUSTAYS, ADMISSIONS, ...)
  eicu_data/                       # raw eICU CSVs (patient.csv.gz, vitalPeriodic.csv.gz, ...)
  processed_data/                  # MIMIC-III intermediate + tensors
  results_LOS_dynamic/             # MIMIC-III model outputs + summary CSVs
  results_eicu/                    # eICU model outputs + summary CSVs
```

## Setup

Two dependency sets. The data-prep stage is CPU-only; training needs CUDA PyTorch
and `mamba_ssm` (used by 3 of the 6 architectures).

```bash
pip install -r requirements-dataprep.txt
pip install -r requirements-train.txt     # see file header for mamba_ssm / causal-conv1d wheels
```

## Running

### General server (no SLURM)

```bash
cd BIBM_versions
bash run_pipeline.sh                       # all datasets/tasks, 5 seeds
bash run_pipeline.sh --help                # all options
bash run_pipeline.sh --datasets mimic --tasks multiclass --seeds "42 43 44"
bash run_pipeline.sh --skip-build          # reuse existing tensors, retrain only
```

Stages run sequentially; per-stage logs land in `BIBM_versions/logs/pipeline_<timestamp>/`.

### HPC cluster (SLURM)

```bash
cd BIBM_versions
# edit the config block (account/partition/SEEDS/SKIP_BUILD) at the top, then:
bash run_final_sweep.slurm
```

Submits the data pipelines, then 6 models × 3 tasks × 2 datasets as per-seed job
arrays, then the combine + report jobs. Set `SKIP_BUILD="true"` to reuse tensors.

## Outputs

Six summary CSVs, each with per-model `mean` and `std` across seeds:

| Dataset | Location | Files |
|---------|----------|-------|
| MIMIC-III | `results_LOS_dynamic/` | `binary_summary.csv`, `ordinal_summary.csv`, `multiclass_summary.csv` |
| eICU | `results_eicu/` | `eicu_binary_summary.csv`, `eicu_ordinal_summary.csv`, `eicu_multiclass_summary.csv` |

## Notes

- **Leakage safety:** imputers/scalers are fit on the train split only; splits are
  patient-level (a patient's admissions never straddle train/valid/test); only
  first-24h data is used; whole-stay diagnosis/procedure codes are excluded.
- **MIMIC cohort filter:** `--cohort-filter strict` applies LOS≥1d,
  single-qualifying-stay, and age≥18 filters; `original` (default here) keeps the
  broader cohort for comparability with prior eICU-style pipelines.
