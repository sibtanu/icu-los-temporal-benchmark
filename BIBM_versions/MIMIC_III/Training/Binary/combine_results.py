#!/usr/bin/env python3
"""
combine_results.py
-------------------
Gathers the latest metrics_*.json from each of the 6 binary model
output directories and writes a single summary CSV.

Run after all 6 V1_train_mimic3_*_binary_auc.py jobs have finished:
    python combine_results.py --results_dir <results_LOS_dynamic>
"""

import os
import csv
import glob
import json
import argparse

MODEL_OUT_DIRS = [
    "mimic3_lstm_fusion_binary_auc_models_v1",
    "mimic3_hybrid_lstm_mamba_binary_auc_models_v1",
    "mimic3_ome_fusion_binary_auc_models_v1",
    "mimic3_patchtst_fusion_binary_auc_models_v1",
    "mimic3_pure_mamba_binary_auc_models_v1",
    "mimic3_transformer_fusion_binary_auc_models_v1",
]

CSV_COLUMNS = [
    "model",
    "dataset",
    "label",
    "test_acc",
    "test_f1_macro",
    "test_f1_weighted",
    "test_auroc",
    "test_auprc",
    "best_epoch",
    "elapsed_sec",
    "metrics_path",
]


def latest_metrics_json(folder: str):
    files = glob.glob(os.path.join(folder, "metrics_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..",
            "results_LOS_dynamic",
        ),
        help="Parent directory containing each model's *_binary_auc_models_v1 output folder.",
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Where to write the summary CSV (defaults to <results_dir>/binary_summary.csv).",
    )
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    out_csv = args.out_csv or os.path.join(results_dir, "binary_summary.csv")

    rows = []
    missing = []

    for model_dir_name in MODEL_OUT_DIRS:
        folder = os.path.join(results_dir, model_dir_name)
        metrics_path = latest_metrics_json(folder)

        if metrics_path is None:
            missing.append(model_dir_name)
            continue

        with open(metrics_path, "r") as f:
            m = json.load(f)

        row = {col: m.get(col, "") for col in CSV_COLUMNS}
        row["metrics_path"] = metrics_path
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No metrics_*.json found under any of the expected output folders in {results_dir}. "
            "Did the training jobs finish?"
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Wrote {len(rows)} model rows to {out_csv}")
    if missing:
        print(f"[WARN] No metrics found for: {missing}")

    print("\nSummary (acc / f1_macro / auroc):")
    for row in rows:
        print(
            f"  {row['model']:60s} | "
            f"acc={row['test_acc']:.4f} | "
            f"f1_macro={row['test_f1_macro']:.4f} | "
            f"auroc={row['test_auroc']:.4f}"
            if isinstance(row["test_acc"], float)
            else f"  {row['model']:60s} | (non-numeric metrics, check {row['metrics_path']})"
        )


if __name__ == "__main__":
    main()
