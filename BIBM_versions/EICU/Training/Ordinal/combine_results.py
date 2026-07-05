#!/usr/bin/env python3
"""
combine_results.py
-------------------
Collects metrics from multiple seed runs for each of the 6 eICU ordinal
model output directories and writes a summary CSV with mean +- std across seeds.

Run after all seed training jobs have finished:
    python combine_results.py --results_dir <results_eicu> [--n_seeds 5]
"""

import os
import csv
import glob
import json
import math
import argparse


def collect_seed_jsons(folder: str, n_seeds: int):
    files = glob.glob(os.path.join(folder, "metrics_*.json"))
    if not files:
        return []
    return sorted(files, key=os.path.getmtime)[-n_seeds:]


def _mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def aggregate_metrics(seed_metrics: list, numeric_cols: list, identity_cols: list) -> dict:
    first = seed_metrics[0]
    agg = {col: first.get(col, "") for col in identity_cols}
    agg["n_seeds"] = len(seed_metrics)
    for col in numeric_cols:
        vals = [float(m[col]) for m in seed_metrics if isinstance(m.get(col), (int, float))]
        mv, sv = _mean_std(vals)
        agg[f"{col}_mean"] = "" if math.isnan(mv) else round(mv, 6)
        agg[f"{col}_std"] = "" if math.isnan(sv) else round(sv, 6)
    return agg

MODEL_OUT_DIRS = [
    "eicu_lstm_fusion_ordinal_auc_models_v1",
    "eicu_hybrid_lstm_mamba_ordinal_auc_models_v1",
    "eicu_ome_fusion_ordinal_auc_models_v1",
    "eicu_patchtst_fusion_ordinal_auc_models_v1",
    "eicu_pure_mamba_ordinal_auc_models_v1",
    "eicu_transformer_fusion_ordinal_auc_models_v1",
]

NUMERIC_COLS = [
    "test_acc",
    "test_f1_macro",
    "test_f1_weighted",
    "test_auroc_macro",
    "test_auprc_macro",
    "test_auroc_gt3",
    "test_auroc_gt7",
    "test_auprc_gt3",
    "test_auprc_gt7",
    "test_auroc_weighted",
    "test_auprc_weighted",
    "best_epoch",
    "elapsed_sec",
]

IDENTITY_COLS = ["model", "dataset", "label"]
CSV_COLUMNS = (
    IDENTITY_COLS
    + ["n_seeds"]
    + [f"{c}_mean" for c in NUMERIC_COLS]
    + [f"{c}_std" for c in NUMERIC_COLS]
    + ["metrics_files"]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "..", "results_eicu",
        ),
    )
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Number of most-recent seed runs to aggregate per model.")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_csv = args.out_csv or os.path.join(results_dir, "eicu_ordinal_summary.csv")

    rows, missing = [], []

    for model_dir_name in MODEL_OUT_DIRS:
        folder = os.path.join(results_dir, model_dir_name)
        json_files = collect_seed_jsons(folder, args.n_seeds)
        if not json_files:
            missing.append(model_dir_name)
            continue
        seed_metrics = [json.load(open(jf)) for jf in json_files]
        row = aggregate_metrics(seed_metrics, NUMERIC_COLS, IDENTITY_COLS)
        row["metrics_files"] = "|".join(os.path.basename(jf) for jf in json_files)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No metrics_*.json found in {results_dir}.")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        for row in rows:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

    n_s = rows[0]["n_seeds"] if rows else 0
    print(f"[INFO] Wrote {len(rows)} models ({n_s} seeds each) to {out_csv}")
    if missing:
        print(f"[WARN] No metrics found for: {missing}")
    print("\nSummary (f1_macro mean +- std):")
    for row in rows:
        m, s = row.get("test_f1_macro_mean", ""), row.get("test_f1_macro_std", "")
        val = f"{m:.4f} +- {s:.4f}" if isinstance(m, float) else "(no data)"
        print(f"  {row['model']:60s} | f1_macro = {val}")


if __name__ == "__main__":
    main()