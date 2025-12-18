"""
V12_impute_script.py
-------------------
Impute the wide hourly dataset from build for LSTM training.

Imputation policy (per variable):
1) LOCF (Last Observation Carried Forward) across hourly columns for the same feature (row-wise, 0→24h).
2) Remaining NaNs (typically leading hours) → global mean of that column.
   If a column is entirely NaN → fill with 0.0 (and log).

Handles all time-variant features:
- Vitals, Labs, Outputs, Inputs, Vasopressors, Procedures, etc.
All columns follow the same naming convention: <feature>_<agg>_<start>_<end>h

Input : outputs/mimic_los_timewindows_wide_v12.csv
Output: outputs/mimic_los_timewindows_wide_v12_imputed.csv
"""

import os
import re
import logging
import argparse
import pandas as pd
import numpy as np

# ---------------- Logging ----------------
logger = logging.getLogger("v12_impute")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_sh)

# ---------------- Config / CLI ----------------
DEF_BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DEF_IN   = f"{DEF_BASE}/mimic_los_timewindows_wide_v12.csv"
DEF_OUT  = f"{DEF_BASE}/mimic_los_timewindows_wide_v12_imputed.csv"

# Canonical static and label columns
STATIC_COLS_CANON = ["HADM_ID", "AGE", "GENDER", "ADMISSION_TYPE"]
LABEL_COLS_CANON  = ["LOS_label", "LOS_label_3class", "LOS_DAYS"]

# Regex to detect hourly columns …_<start>_<end>h (e.g., Heart_Rate_mean_3_4h)
HOUR_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)h$")

# ---------------- Helpers ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEF_IN,  help="Path to V12 wide CSV")
    ap.add_argument("--output", default=DEF_OUT, help="Path to save imputed CSV")
    return ap.parse_args()

def is_hourly_col(col: str) -> bool:
    """Return True if column matches the hourly suffix pattern (_0_1h etc.)."""
    return bool(HOUR_SUFFIX_RE.search(col))

def feature_base(col: str) -> str:
    """Return the feature base name without the trailing _<a>_<b>h."""
    m = HOUR_SUFFIX_RE.search(col)
    return col[:m.start()] if m else col

def hour_start(col: str) -> int:
    """Extract the start hour as int from a column name."""
    m = HOUR_SUFFIX_RE.search(col)
    return int(m.group(1)) if m else -1

# ---------------- Main ----------------
def main():
    args = parse_args()
    in_path, out_path = args.input, args.output

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logger.info(f"Loading: {in_path}")
    df = pd.read_csv(in_path)
    n_rows, n_cols = df.shape
    logger.info(f"Loaded shape: rows={n_rows:,} cols={n_cols:,}")

    # Identify static, labels, and feature columns
    static_cols = [c for c in STATIC_COLS_CANON if c in df.columns]
    label_cols  = [c for c in LABEL_COLS_CANON  if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in static_cols + label_cols
        and is_hourly_col(c)
        and not c.endswith("_0_24h")
    ]


    if not feature_cols:
        logger.warning("No hourly feature columns detected. Nothing to impute.")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved (unchanged): {out_path}")
        return

    # Group hourly columns by feature base name
    groups = {}
    for c in feature_cols:
        base = feature_base(c)
        groups.setdefault(base, []).append(c)

    # Sort each group's columns by start hour (ensures 0→24h order)
    for base, cols in groups.items():
        cols.sort(key=hour_start)
        groups[base] = cols

    # ---------------- Step 1: LOCF across hours ----------------
    logger.info("Applying LOCF (Last Observation Carried Forward) across hours...")
    for base, cols in groups.items():
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
        df[cols] = df[cols].ffill(axis=1)

    # ---------------- Step 2: Fill remaining NaNs ----------------
    logger.info("Filling remaining NaNs with column means (and 0 for entirely NaN cols)...")
    col_means = df[feature_cols].mean(axis=0, skipna=True)
    entirely_nan = col_means[col_means.isna()].index.tolist()
    if entirely_nan:
        logger.warning(f"{len(entirely_nan)} columns entirely NaN after LOCF; filling with 0.0")
        col_means.loc[entirely_nan] = 0.0

    # Fill NaNs with precomputed means
    df[feature_cols] = df[feature_cols].fillna(col_means)

    # Optional: clip binary-like columns to {0,1}
    # (for pressor/procedure flags; prevents 0.5 mean-fill from leaking)
    binary_like = [c for c in feature_cols if df[c].dropna().isin([0, 1]).all()]
    if binary_like:
        df[binary_like] = (df[binary_like] > 0.5).astype(int)

    # ---------------- Validation ----------------
    remaining = int(df[feature_cols].isna().sum().sum())
    if remaining != 0:
        raise RuntimeError(f"Imputation failed: {remaining} NaN values remain in feature columns.")
    else:
        logger.info("Validation passed: no NaNs remain in feature columns.")

    # ---------------- Reorder Columns ----------------
    feature_cols_ordered = []
    for base in sorted(groups.keys()):
        feature_cols_ordered.extend(groups[base])

    other = [c for c in df.columns if c not in static_cols + feature_cols_ordered + label_cols]
    ordered_cols = static_cols + feature_cols_ordered + other + label_cols
    df = df[ordered_cols]

    # ---------------- Save ----------------
    df.to_csv(out_path, index=False)
    logger.info(f"Saved imputed dataset: {out_path}  rows={len(df):,} cols={df.shape[1]:,}")

if __name__ == "__main__":
    main()
