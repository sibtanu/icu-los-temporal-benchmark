#!/usr/bin/env python3
"""
V1_train_eicu_transformer_static_only_ordinal.py

Static-only ablation for eICU 3-class ordinal LOS prediction.

Purpose:
    Ablation for the fusion model:
        Dynamic-only vs Static-only vs Dynamic+Static Fusion

Reads:
  /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1/X_static_*.npy
  /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1/y_3c_*.npy

Output:
  /lustre/home/rahas2/mimic_projects/outputs/eicu_transformer_static_only_ordinal_models_v1

Run:
  python V1_train_eicu_transformer_static_only_ordinal.py --workers 0 --batch 128
"""

import os
import json
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_transformer_static_only_ordinal_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StaticOnlyOrdinalDataset(Dataset):
    def __init__(self, X_static, y_3c):
        self.Xs = X_static
        self.y = y_3c.astype(np.int64)

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, idx):
        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xs, y


def make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch=256, workers=0):
    kwargs = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(StaticOnlyOrdinalDataset(Xtr, ytr), shuffle=True, **kwargs)
    valid_loader = DataLoader(StaticOnlyOrdinalDataset(Xva, yva), shuffle=False, **kwargs)
    test_loader = DataLoader(StaticOnlyOrdinalDataset(Xte, yte), shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.long()
    return torch.stack([(y > 0).float(), (y > 1).float()], dim=1)


def ordinal_probs_to_class(prob_gt3, prob_gt7, thr1, thr2):
    pred = np.ones_like(prob_gt3, dtype=np.int64)
    pred[prob_gt3 < thr1] = 0
    pred[prob_gt7 >= thr2] = 2
    return pred


def tune_ordinal_thresholds(y_true, prob_gt3, prob_gt7):
    best = {"f1": -1.0, "acc": 0.0, "thr1": 0.5, "thr2": 0.5}

    for t1 in np.arange(0.30, 0.86, 0.02):
        for t2 in np.arange(0.30, 0.91, 0.02):
            pred = ordinal_probs_to_class(prob_gt3, prob_gt7, t1, t2)
            f1 = f1_score(y_true, pred, average="macro", zero_division=0)
            if f1 > best["f1"]:
                best = {
                    "f1": float(f1),
                    "acc": float(accuracy_score(y_true, pred)),
                    "thr1": float(t1),
                    "thr2": float(t2),
                }

    return best


class OrdinalBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, y_class):
        y_ord = ordinal_targets(y_class).to(logits.device)
        return F.binary_cross_entropy_with_logits(
            logits,
            y_ord,
            pos_weight=self.pos_weight,
            reduction="mean",
        )


class StaticOnlyOrdinalMLP(nn.Module):
    """
    Static-only ordinal model.

    Input:
        X_static: (B, F_static)

    Output:
        two ordinal logits:
            LOS > 3 days
            LOS > 7 days
    """

    def __init__(
        self,
        static_dim: int,
        hidden: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(static_dim),

            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x_static):
        return self.net(x_static)


def train_epoch(model, loader, device, criterion, optimizer, scheduler=None, scaler=None, amp=False, grad_clip=1.0):
    model.train()
    loss_sum = 0.0
    n = 0

    for Xs, y in loader:
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xs)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(Xs)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, amp=False):
    model.eval()
    loss_sum = 0.0
    n = 0
    probs_all = []
    y_all = []

    for Xs, y in loader:
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xs)
                loss = criterion(logits, y)
        else:
            logits = model(Xs)
            loss = criterion(logits, y)

        probs = torch.sigmoid(logits)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

        probs_all.append(probs.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())

    return loss_sum / max(n, 1), np.concatenate(probs_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING eICU STATIC-ONLY ORDINAL 3-CLASS")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    X_static_train = np.load(f"{DATA_DIR}/X_static_train.npy")
    X_static_valid = np.load(f"{DATA_DIR}/X_static_valid.npy")
    X_static_test = np.load(f"{DATA_DIR}/X_static_test.npy")

    y_train = np.load(f"{DATA_DIR}/y_3c_train.npy")
    y_valid = np.load(f"{DATA_DIR}/y_3c_valid.npy")
    y_test = np.load(f"{DATA_DIR}/y_3c_test.npy")

    print(f"Loaded static: train={X_static_train.shape}, valid={X_static_valid.shape}, test={X_static_test.shape}")

    class_counts = np.bincount(y_train, minlength=3)
    print("Train class counts:", class_counts)

    y_ord_train = np.stack([(y_train > 0).astype(np.float32), (y_train > 1).astype(np.float32)], axis=1)

    pos = y_ord_train.sum(axis=0)
    neg = y_ord_train.shape[0] - pos

    pos_weight = neg / np.maximum(pos, 1.0)
    pos_weight = np.minimum(pos_weight, np.array([6.0, 12.0], dtype=np.float32))

    print("Using ordinal BCE loss with validation threshold tuning.")
    print("Ordinal pos_weight:", pos_weight.tolist())

    criterion = OrdinalBCELoss(pos_weight=pos_weight).to(device)

    tr_loader, va_loader, te_loader = make_loaders(
        X_static_train, y_train,
        X_static_valid, y_valid,
        X_static_test, y_test,
        batch=args.batch,
        workers=args.workers,
    )

    static_dim = X_static_train.shape[1]

    model = StaticOnlyOrdinalMLP(
        static_dim=static_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * max(len(tr_loader), 1)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_thresholds = (0.5, 0.5)
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, tr_loader, device, criterion, optimizer,
            scheduler=scheduler, scaler=scaler,
            amp=(args.amp and device.type == "cuda"),
            grad_clip=args.grad_clip,
        )

        val_loss, val_probs, val_true = eval_epoch(
            model, va_loader, device, criterion,
            amp=(args.amp and device.type == "cuda"),
        )

        threshold_info = tune_ordinal_thresholds(val_true, val_probs[:, 0], val_probs[:, 1])

        val_f1 = threshold_info["f1"]
        val_acc = threshold_info["acc"]

        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"F1 {val_f1:.4f} | Acc {val_acc:.4f} | "
            f"t=({threshold_info['thr1']:.2f}, {threshold_info['thr2']:.2f})"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            best_thresholds = (float(threshold_info["thr1"]), float(threshold_info["thr2"]))
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)

    test_loss, test_probs, test_true = eval_epoch(
        model, te_loader, device, criterion,
        amp=(args.amp and device.type == "cuda"),
    )

    test_pred = ordinal_probs_to_class(
        test_probs[:, 0],
        test_probs[:, 1],
        best_thresholds[0],
        best_thresholds[1],
    )

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)

    report = classification_report(test_true, test_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"eicu_v1_static_only_ordinal_3class_"
        f"H{args.hidden}_Fstat{static_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"static_only_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "static_dim": int(static_dim),
            "hidden": int(args.hidden),
            "dropout": float(args.dropout),
            "num_classes_original": 3,
            "num_ordinal_logits": 2,
            "label": "3class_ordinal",
            "ablation": "static_only",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_thresholds": list(best_thresholds),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Static-only ANN ordinal",
        "label": "3class_ordinal",
        "ablation": "static_only",
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_thresholds": list(best_thresholds),
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_static": list(X_static_train.shape),
        "valid_shape_static": list(X_static_valid.shape),
        "test_shape_static": list(X_static_test.shape),
        "input_static_dim": int(static_dim),
        "checkpoint_metric": "validation_macro_f1",
        "ordinal_targets": ["LOS_gt_3_days", "LOS_gt_7_days"],
        "output_dir": OUT_DIR,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {test_acc:.4f} | F1 macro: {test_f1_macro:.4f} | "
        f"Test loss: {test_loss:.4f} | Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f} | Thresholds: {best_thresholds}"
    )


if __name__ == "__main__":
    main()
