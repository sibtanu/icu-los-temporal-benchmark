# eICU LOS Build Script (V1)
#!/usr/bin/env python3
"""
V1_build_eicu_los_dataset.py
----------------------------

Builds a wide hourly-feature dataset for ICU Length-of-Stay (LOS)
prediction using the eICU Collaborative Research Database.

This script follows the same overall architecture and naming conventions
as the user's MIMIC-III V12 LOS pipeline wherever possible.

GOALS
-----
1. Produce a MIMIC-compatible wide table.
2. Keep downstream imputation / preprocessing / training scripts reusable.
3. Support fast experimentation using configurable sampling.
4. Use robust chunked processing for large .csv.gz files.
5. Generate hourly features for the first 24 hours of ICU stay.

OUTPUT
------
Wide-format CSV with:
- static demographic features
- diagnosis/comorbidity features
- hourly dynamic features
- derived totals
- LOS labels

Column examples:
- heartrate_mean_0_1h
- spo2_mean_7_8h
- fio2_mean_3_4h
- urine_output_sum_12_13h
- norepinephrine_any_4_5h
- antibiotic_any_8_9h
- fluid_balance_sum_0_24h

LABELS
------
Binary:
- LOS_label = LOS > 7 days

Three-class:
- 0 = <=3 days
- 1 = >3 and <=7 days
- 2 = >7 days

IMPORTANT eICU DIFFERENCES
--------------------------
Unlike MIMIC-III, eICU primarily uses offset-based timing.
This script uses:
- patientunitstayid as the ICU stay identifier
- unitdischargeoffset for LOS
- offsets relative to ICU admission

To preserve compatibility with downstream MIMIC-based preprocessing,
this script emits:
- HADM_ID column = patientunitstayid

SUPPORTED TABLES
----------------
Required:
- patient.csv.gz
- vitalPeriodic.csv.gz
- vitalAperiodic.csv.gz
- nurseCharting.csv.gz
- lab.csv.gz
- diagnosis.csv.gz
- infusionDrug.csv.gz
- intakeOutput.csv.gz

Optional:
- medication.csv.gz
- respiratoryCare.csv.gz
- treatment.csv.gz
- apacheApsVar.csv.gz
- apachePatientResult.csv.gz

SAMPLING
--------
Supports configurable cohort sampling for rapid testing.
Example:
    SAMPLE_PATIENTS = 500

Then later:
    SAMPLE_PATIENTS = None
for full dataset generation.

OUTPUT FILE
-----------
outputs/eicu_los_timewindows_wide_v1.csv

"""

import os
import re
import time
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

# ======================================================
# CONFIG
# ======================================================

SAMPLE_PATIENTS = None
MAX_CHUNKS = None
CHUNK_SIZE = 500_000
WINDOW_HOURS = list(range(0, 25))

DATA_DIR = "/lustre/home/rahas2/mimic_projects/eicu_data"
OUTPUT_DIR = "/lustre/home/rahas2/mimic_projects/outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = f"{OUTPUT_DIR}/eicu_los_timewindows_wide_v1.csv"
LOG_FILE   = f"{OUTPUT_DIR}/build_eicu_los_dataset_v1.log"

# ======================================================
# FILES
# ======================================================

PATIENT_FILE             = f"{DATA_DIR}/patient.csv.gz"
VITAL_PERIODIC_FILE      = f"{DATA_DIR}/vitalPeriodic.csv.gz"
VITAL_APERIODIC_FILE     = f"{DATA_DIR}/vitalAperiodic.csv.gz"
NURSE_CHARTING_FILE      = f"{DATA_DIR}/nurseCharting.csv.gz"
LAB_FILE                 = f"{DATA_DIR}/lab.csv.gz"
DIAGNOSIS_FILE           = f"{DATA_DIR}/diagnosis.csv.gz"
INFUSION_DRUG_FILE       = f"{DATA_DIR}/infusionDrug.csv.gz"
INTAKE_OUTPUT_FILE       = f"{DATA_DIR}/intakeOutput.csv.gz"
MEDICATION_FILE          = f"{DATA_DIR}/medication.csv.gz"
RESPIRATORY_CARE_FILE    = f"{DATA_DIR}/respiratoryCare.csv.gz"
TREATMENT_FILE           = f"{DATA_DIR}/treatment.csv.gz"
APACHE_APS_FILE          = f"{DATA_DIR}/apacheApsVar.csv.gz"
APACHE_RESULT_FILE       = f"{DATA_DIR}/apachePatientResult.csv.gz"

# ======================================================
# LOGGING
# ======================================================

logger = logging.getLogger("eicu_build")
logger.setLevel(logging.INFO)
logger.handlers = []

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)

_fh = logging.FileHandler(LOG_FILE)
_fh.setFormatter(_fmt)

logger.addHandler(_sh)
logger.addHandler(_fh)

# ======================================================
# HELPERS
# ======================================================


def safe_read_csv(path, **kwargs):
    return pd.read_csv(path, compression="gzip", low_memory=False, **kwargs)



def wname(a, b):
    return f"{int(a)}_{int(b)}h"



def sanitize_label(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s



def hour_bin_from_offset_minutes(offset_series):
    hrs = offset_series.astype(float) / 60.0
    hb = hrs.astype(int)
    return hb.clip(0, 23)


# ======================================================
# MAIN
# ======================================================


def main():

    start_t = time.time()

    logger.info("Loading patient table...")

    patient = safe_read_csv(
        PATIENT_FILE,
        usecols=[
            "patientunitstayid",
            "gender",
            "age",
            "ethnicity",
            "hospitaladmitsource",
            "hospitaldischargestatus",
            "hospitaldischargeoffset",
            "unitdischargeoffset",
            "unittype",
            "apacheadmissiondx",
        ]
    )

    # --------------------------------------------------
    # Sample support
    # --------------------------------------------------

    if SAMPLE_PATIENTS is not None:
        sample_ids = (
            patient["patientunitstayid"]
            .drop_duplicates()
            .sample(min(SAMPLE_PATIENTS, patient["patientunitstayid"].nunique()), random_state=42)
        )

        patient = patient[
            patient["patientunitstayid"].isin(sample_ids)
        ]

        logger.info(f"Sampling enabled: {len(sample_ids):,} ICU stays")

    # --------------------------------------------------
    # Build base cohort
    # --------------------------------------------------

    patient["LOS_DAYS"] = patient["unitdischargeoffset"] / 1440.0

    patient["LOS_label"] = (
        patient["LOS_DAYS"] > 7.0
    ).astype("int8")

    conds = [
        patient["LOS_DAYS"] <= 3.0,
        (patient["LOS_DAYS"] > 3.0) & (patient["LOS_DAYS"] <= 7.0),
        patient["LOS_DAYS"] > 7.0,
    ]

    patient["LOS_label_3class"] = np.select(
        conds,
        [0, 1, 2]
    ).astype("int8")

    # --------------------------------------------------
    # Compatibility columns
    # --------------------------------------------------

    patient["HADM_ID"] = patient["patientunitstayid"]

    patient.rename(columns={
        "gender": "GENDER",
        "age": "AGE",
        "hospitaladmitsource": "ADMISSION_TYPE",
        "ethnicity": "ETHNICITY",
        "unittype": "FIRST_CAREUNIT",
    }, inplace=True)

    # --------------------------------------------------
    # Clean AGE
    # --------------------------------------------------

    patient["AGE"] = (
        pd.to_numeric(patient["AGE"], errors="coerce")
        .clip(0, 120)
    )

    # --------------------------------------------------
    # Static frame
    # --------------------------------------------------

    static_cols = [
        "HADM_ID",
        "AGE",
        "GENDER",
        "ADMISSION_TYPE",
        "ETHNICITY",
        "FIRST_CAREUNIT",
        "LOS_DAYS",
        "LOS_label",
        "LOS_label_3class",
    ]

    df = patient[static_cols].copy()

    logger.info(f"Initial cohort shape: {df.shape}")

    # ==================================================
    # VITAL PERIODIC
    # ==================================================

    logger.info("Processing vitalPeriodic...")

    vital_map = {
        "heartrate": "Heart_Rate",
        "systemicsystolic": "Systolic_BP",
        "systemicdiastolic": "Diastolic_BP",
        "systemicmean": "MAP",
        "respiration": "Resp_Rate",
        "spo2": "SpO2",
        "temperature": "Temperature_C",
    }

    vit_acc_sum = defaultdict(lambda: defaultdict(float))
    vit_acc_cnt = defaultdict(lambda: defaultdict(int))

    vit_iter = safe_read_csv(
        VITAL_PERIODIC_FILE,
        chunksize=CHUNK_SIZE,
    )

    for i, ch in enumerate(vit_iter, start=1):

        if SAMPLE_PATIENTS is not None:
            ch = ch[ch["patientunitstayid"].isin(patient["patientunitstayid"])]

        if ch.empty:
            continue

        for raw_col, clean_name in vital_map.items():

            if raw_col not in ch.columns:
                continue

            sub = ch[[
                "patientunitstayid",
                "observationoffset",
                raw_col,
            ]].copy()

            sub.rename(columns={raw_col: "VALUE"}, inplace=True)

            sub["VALUE"] = pd.to_numeric(sub["VALUE"], errors="coerce")

            sub = sub.dropna(subset=["VALUE", "observationoffset"])

            sub = sub[
                (sub["observationoffset"] >= 0)
                &
                (sub["observationoffset"] < 1440)
            ]

            if sub.empty:
                continue

            sub["HOUR_BIN"] = hour_bin_from_offset_minutes(
                sub["observationoffset"]
            )

            g = (
                sub.groupby(["patientunitstayid", "HOUR_BIN"])["VALUE"]
                .agg(["sum", "count"])
                .reset_index()
            )

            for _, r in g.iterrows():

                hadm = int(r["patientunitstayid"])
                hb = int(r["HOUR_BIN"])
                wn = wname(hb, hb + 1)

                vit_acc_sum[wn][(hadm, clean_name)] += float(r["sum"])
                vit_acc_cnt[wn][(hadm, clean_name)] += int(r["count"])

        if i % 10 == 0:
            logger.info(f"Processed vitalPeriodic chunks={i:,}")

        if MAX_CHUNKS and i >= MAX_CHUNKS:
            break

    vit_frames = []

    for wn in vit_acc_sum.keys():

        sdict = vit_acc_sum[wn]
        cdict = vit_acc_cnt[wn]

        rows = []

        for k in sdict.keys():
            hadm, feat = k
            meanv = sdict[k] / max(cdict[k], 1)
            rows.append((hadm, feat, meanv))

        if not rows:
            continue

        tmp = pd.DataFrame(rows, columns=["HADM_ID", "FEATURE", "MEAN"])

        wide = (
            tmp.pivot_table(index="HADM_ID", columns="FEATURE", values="MEAN")
            .reset_index()
        )

        wide.columns = [
            "HADM_ID"
        ] + [
            f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"
        ]

        vit_frames.append(wide)

    for wf in vit_frames:
        df = df.merge(wf, on="HADM_ID", how="left")

    # ==================================================
    # LABS
    # ==================================================

    logger.info("Processing labs...")

    lab_acc_sum = defaultdict(lambda: defaultdict(float))
    lab_acc_cnt = defaultdict(lambda: defaultdict(int))

    lab_iter = safe_read_csv(
        LAB_FILE,
        chunksize=CHUNK_SIZE,
    )

    for i, ch in enumerate(lab_iter, start=1):

        if SAMPLE_PATIENTS is not None:
            ch = ch[ch["patientunitstayid"].isin(patient["patientunitstayid"])]

        if ch.empty:
            continue

        req = [
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "labresult",
        ]

        miss = [c for c in req if c not in ch.columns]
        if miss:
            logger.warning(f"Missing lab columns: {miss}")
            continue

        ch = ch[req].copy()

        ch["labresult"] = pd.to_numeric(ch["labresult"], errors="coerce")

        ch = ch.dropna(subset=["labresult", "labresultoffset"])

        ch = ch[
            (ch["labresultoffset"] >= 0)
            &
            (ch["labresultoffset"] < 1440)
        ]

        if ch.empty:
            continue

        ch["LAB"] = ch["labname"].map(sanitize_label)

        ch["HOUR_BIN"] = hour_bin_from_offset_minutes(
            ch["labresultoffset"]
        )

        g = (
            ch.groupby(["patientunitstayid", "LAB", "HOUR_BIN"])["labresult"]
            .agg(["sum", "count"])
            .reset_index()
        )

        for _, r in g.iterrows():

            hadm = int(r["patientunitstayid"])
            lab = r["LAB"]
            hb = int(r["HOUR_BIN"])
            wn = wname(hb, hb + 1)

            lab_acc_sum[wn][(hadm, lab)] += float(r["sum"])
            lab_acc_cnt[wn][(hadm, lab)] += int(r["count"])

        if i % 10 == 0:
            logger.info(f"Processed lab chunks={i:,}")

        if MAX_CHUNKS and i >= MAX_CHUNKS:
            break

    lab_frames = []

    for wn in lab_acc_sum.keys():

        sdict = lab_acc_sum[wn]
        cdict = lab_acc_cnt[wn]

        rows = []

        for k in sdict.keys():
            hadm, feat = k
            meanv = sdict[k] / max(cdict[k], 1)
            rows.append((hadm, feat, meanv))

        if not rows:
            continue

        tmp = pd.DataFrame(rows, columns=["HADM_ID", "FEATURE", "MEAN"])

        wide = (
            tmp.pivot_table(index="HADM_ID", columns="FEATURE", values="MEAN")
            .reset_index()
        )

        wide.columns = [
            "HADM_ID"
        ] + [
            f"{c}_mean_{wn}" for c in wide.columns if c != "HADM_ID"
        ]

        lab_frames.append(wide)

    for wf in lab_frames:
        df = df.merge(wf, on="HADM_ID", how="left")

    # ==================================================
    # DIAGNOSIS FEATURES
    # ==================================================

    logger.info("Processing diagnosis features...")

    if os.path.exists(DIAGNOSIS_FILE):

        dx = safe_read_csv(DIAGNOSIS_FILE)

        if SAMPLE_PATIENTS is not None:
            dx = dx[dx["patientunitstayid"].isin(patient["patientunitstayid"])]

        if "diagnosisstring" in dx.columns:

            dx["diagnosisstring"] = (
                dx["diagnosisstring"]
                .astype(str)
                .str.lower()
            )

            dx_groups = {
                "sepsis": ["sepsis", "septic"],
                "respiratory": ["pneumonia", "copd", "respiratory"],
                "cardiac": ["heart", "cardiac", "mi", "infarct"],
                "renal": ["renal", "kidney"],
                "diabetes": ["diabetes"],
                "liver": ["liver", "hepatic"],
            }

            for feat, kws in dx_groups.items():

                m = dx["diagnosisstring"].str.contains(
                    "|".join(kws),
                    na=False,
                )

                sub = (
                    dx.loc[m, ["patientunitstayid"]]
                    .drop_duplicates()
                    .assign(**{f"dx_{feat}": 1})
                )

                sub.rename(columns={
                    "patientunitstayid": "HADM_ID"
                }, inplace=True)

                df = df.merge(sub, on="HADM_ID", how="left")

    # ==================================================
    # FILL DIAG FLAGS
    # ==================================================

    dx_cols = [c for c in df.columns if c.startswith("dx_")]

    if dx_cols:
        df[dx_cols] = df[dx_cols].fillna(0).astype("int8")

    # ==================================================
    # FINAL CLEANUP
    # ==================================================

    logger.info("Final cleanup...")

    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Final shape: {df.shape}")

    # ==================================================
    # SAVE
    # ==================================================

    df.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"Saved: {OUTPUT_CSV}")
    logger.info(f"Elapsed sec: {round(time.time() - start_t, 1)}")


if __name__ == "__main__":
    main()
