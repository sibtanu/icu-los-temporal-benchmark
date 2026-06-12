#!/usr/bin/env python3
"""
V1_impute_eicu_los_dataset.py

Robust eICU imputation for large HPRC runs.

Policy:
1. Load wide eICU LOS CSV.
2. Drop hourly columns with >99% missingness.
3. Detect all hourly columns ending in _<start>_<end>h.
4. Group by feature base name.
5. Convert hourly features to numeric.
6. LOCF across 0–24h.
7. Fill remaining leading NaNs with column median.
8. Fill fully empty numeric columns with 0.
9. Fill static numeric NaNs with median.
10. Fill static categorical NaNs with UNKNOWN.
11. Validate zero NaNs remain.
"""

import os
import re
import logging
import numpy as np
import pandas as pd

INPUT_CSV = "/lustre/home/rahas2/mimic_projects/outputs/eicu_los_timewindows_wide_v1.csv"
OUTPUT_CSV = "/lustre/home/rahas2/mimic_projects/outputs/eicu_los_timewindows_wide_v1_imputed.csv"

SPARSE_DROP_THRESHOLD = 0.99
HOUR_RE = re.compile(r"_(\d+)_(\d+)h$")

LABEL_COLS = {"LOS_label", "LOS_label_3class", "LOS_DAYS"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("eicu_impute")


def is_hourly(c):
    return bool(HOUR_RE.search(c))


def base_name(c):
    return HOUR_RE.sub("", c)


def hour_start(c):
    m = HOUR_RE.search(c)
    return int(m.group(1)) if m else -1


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(INPUT_CSV)

    logger.info(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    logger.info(f"Loaded shape: {df.shape}")

    hourly_cols = [c for c in df.columns if is_hourly(c)]
    logger.info(f"Hourly cols before sparse drop: {len(hourly_cols):,}")

    sparse = df[hourly_cols].isna().mean()
    drop_cols = sparse[sparse > SPARSE_DROP_THRESHOLD].index.tolist()

    if drop_cols:
        logger.info(f"Dropping hourly cols >{SPARSE_DROP_THRESHOLD:.0%} missing: {len(drop_cols):,}")
        df.drop(columns=drop_cols, inplace=True)

    hourly_cols = [c for c in df.columns if is_hourly(c)]
    logger.info(f"Hourly cols after sparse drop: {len(hourly_cols):,}")

    groups = {}
    for c in hourly_cols:
        groups.setdefault(base_name(c), []).append(c)

    for b in groups:
        groups[b].sort(key=hour_start)

    logger.info(f"Hourly feature groups: {len(groups):,}")

    for i, cols in enumerate(groups.values(), start=1):
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df[cols] = df[cols].ffill(axis=1)

        if i % 100 == 0:
            logger.info(f"LOCF groups processed: {i:,}")

    logger.info("Filling remaining hourly NaNs with column medians...")
    med = df[hourly_cols].median(axis=0, skipna=True)
    med = med.fillna(0.0)
    df[hourly_cols] = df[hourly_cols].fillna(med)

    logger.info("Filling non-hourly numeric NaNs...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in LABEL_COLS and c not in hourly_cols]

    for c in numeric_cols:
        if df[c].isna().any():
            val = df[c].median()
            df[c] = df[c].fillna(0.0 if pd.isna(val) else val)

    logger.info("Filling categorical/static NaNs...")
    cat_cols = [c for c in df.columns if c not in hourly_cols and c not in numeric_cols and c not in LABEL_COLS]

    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    logger.info("Final validation...")
    total_nans = int(df.isna().sum().sum())

    if total_nans != 0:
        bad = df.columns[df.isna().any()].tolist()[:30]
        raise RuntimeError(f"NaNs remain: {total_nans}. First bad cols: {bad}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"Saved: {OUTPUT_CSV}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()