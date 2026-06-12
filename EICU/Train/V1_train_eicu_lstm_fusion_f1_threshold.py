#!/usr/bin/env python3
"""
V1_train_eicu_lstm_fusion_f1_threshold.py
----------------------------

Full eICU training script using the SAME architecture family as the MIMIC-III
V12.8 training pipeline.

Architecture:
- Dynamic branch: BiLSTM over hourly features, shape (N, 24, F_dyn)
- Static branch: ANN / MLP over static features, shape (N, F_static)
- Fusion: dynamic representation + static representation
- Classifier head: gated fusion + dense layers

Inputs expected from:
    V1_preprocess_eicu_LSTM.py

Uses:
    X_dyn_train.npy
    X_dyn_valid.npy
    X_dyn_test.npy
    X_static_train.npy
    X_static_valid.npy
    X_static_test.npy
    y_bin_train.npy / valid / test
    y_3c_train.npy / valid / test

This script can be used for BOTH:
- sample run
- full eICU run

No architecture changes are needed between sample and full dataset.
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

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_lstm_models_v1"

os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# SEED
# ======================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# DATASET
# ======================================================

class DynStaticDataset(Dataset):
    def __init__(self, X_dyn, X_static, y):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, xs, y


def make_loaders(
    X_dyn_train,
    X_static_train,
    y_train,
    X_dyn_valid,
    X_static_valid,
    y_valid,
    X_dyn_test,
    X_static_test,
    y_test,
    batch=256,
    workers=2,
):
    train_loader = DataLoader(
        DynStaticDataset(X_dyn_train, X_static_train, y_train),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    valid_loader = DataLoader(
        DynStaticDataset(X_dyn_valid, X_static_valid, y_valid),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    test_loader = DataLoader(
        DynStaticDataset(X_dyn_test, X_static_test, y_test),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    return train_loader, valid_loader, test_loader


# ======================================================
# LOSS
# ======================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer(
                "alpha",
                torch.tensor(alpha, dtype=torch.float32),
            )
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.long()

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * focal_factor * log_pt
        else:
            loss = -focal_factor * log_pt

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


# ======================================================
# MODEL: SAME FAMILY AS MIMIC V12.8
# ======================================================

class FusionLSTMPlus(nn.Module):
    """
    BiLSTM + ANN/static MLP fusion model.

    Dynamic branch:
    - BiLSTM over hourly time series
    - Uses last hidden state + mean-pooled output

    Static branch:
    - 2-layer ANN/MLP with LayerNorm

    Fusion:
    - concatenate dynamic + static embeddings
    - LayerNorm
    - gated fusion
    - dense classifier head
    """

    def __init__(
        self,
        input_dyn_dim: int,
        input_static_dim: int,
        hidden: int = 384,
        layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        static_hidden: int = 128,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dyn_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        dyn_out_dim = hidden * 2
        fused_dyn_dim = dyn_out_dim * 2

        self.static_mlp = nn.Sequential(
            nn.Linear(input_static_dim, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
        )

        fusion_in_dim = fused_dyn_dim + static_hidden

        self.fusion_ln = nn.LayerNorm(fusion_in_dim)
        self.fusion_gate = nn.Linear(fusion_in_dim, fusion_in_dim)

        self.head = nn.Sequential(
            nn.Linear(fusion_in_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_dyn, x_static):
        out, (hn, _) = self.lstm(x_dyn)

        h_forward = hn[-2]
        h_backward = hn[-1]

        h_last = torch.cat([h_forward, h_backward], dim=1)
        h_mean = out.mean(dim=1)

        h_dyn = torch.cat([h_last, h_mean], dim=1)

        h_static = self.static_mlp(x_static)

        fused = torch.cat([h_dyn, h_static], dim=1)
        fused = self.fusion_ln(fused)

        gate = torch.sigmoid(self.fusion_gate(fused))
        fused = fused * gate

        return self.head(fused)


# ======================================================
# TRAIN / EVAL
# ======================================================

def train_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scaler=None,
    grad_clip: float | None = 1.0,
    amp: bool = False,
):
    model.train()

    loss_sum = 0.0
    n = 0

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd, Xs)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(Xd, Xs)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")

            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, amp: bool = False):
    """
    Evaluation helper that returns loss, argmax predictions, true labels,
    and class probabilities.

    For binary classification, probabilities are needed for validation-time
    threshold tuning. For 3-class classification, argmax predictions are used.
    """
    model.eval()

    loss_sum = 0.0
    n = 0

    preds = []
    targets = []
    probs_all = []

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd, Xs)
                loss = criterion(logits, y)
        else:
            logits = model(Xd, Xs)
            loss = criterion(logits, y)

        bs = y.size(0)

        loss_sum += loss.item() * bs
        n += bs

        probs = torch.softmax(logits, dim=1)

        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(y.cpu().numpy())
        probs_all.append(probs.cpu().numpy())

    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    probs_all = np.concatenate(probs_all) if probs_all else np.array([])

    return loss_sum / max(n, 1), preds, targets, probs_all


def tune_binary_threshold(y_true, prob_pos):
    """
    Select the probability threshold that maximizes validation macro F1.

    This is important for eICU because the positive LOS>7d class is much rarer
    than in MIMIC-III, so default argmax/0.5 threshold can underperform.
    """
    best_t = 0.5
    best_f1 = -1.0

    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob_pos >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", choices=["binary", "3class"], default="binary")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    X_dyn_train = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    X_dyn_valid = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    X_dyn_test = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    X_static_train = np.load(f"{DATA_DIR}/X_static_train.npy")
    X_static_valid = np.load(f"{DATA_DIR}/X_static_valid.npy")
    X_static_test = np.load(f"{DATA_DIR}/X_static_test.npy")

    if args.label == "binary":
        y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
        y_valid = np.load(f"{DATA_DIR}/y_bin_valid.npy")
        y_test = np.load(f"{DATA_DIR}/y_bin_test.npy")
        num_classes = 2

    else:
        y_train = np.load(f"{DATA_DIR}/y_3c_train.npy")
        y_valid = np.load(f"{DATA_DIR}/y_3c_valid.npy")
        y_test = np.load(f"{DATA_DIR}/y_3c_test.npy")
        num_classes = 3

    print(f"Loaded dynamic train: {X_dyn_train.shape}")
    print(f"Loaded static train: {X_static_train.shape}")
    print(f"Classes: {num_classes}")

    tr_loader, va_loader, te_loader = make_loaders(
        X_dyn_train,
        X_static_train,
        y_train,
        X_dyn_valid,
        X_static_valid,
        y_valid,
        X_dyn_test,
        X_static_test,
        y_test,
        batch=args.batch,
        workers=args.workers,
    )

    class_counts = np.bincount(y_train, minlength=num_classes)

    print("Train class counts:", class_counts)

    if args.label == "binary":
        weights = class_counts.sum() / (class_counts + 1e-9)
        weights = weights / weights.sum()
        ce_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=ce_weights)

        print("Using weighted CrossEntropy.")
        print("Weights:", weights)

    else:
        alpha = [0.25, 0.5, 0.25]
        criterion = FocalLoss(
            gamma=1.5,
            alpha=alpha,
            reduction="mean",
        ).to(device)

        print("Using FocalLoss.")
        print("Alpha:", alpha)

    input_dyn_dim = X_dyn_train.shape[2]
    input_static_dim = X_static_train.shape[1]

    model = FusionLSTMPlus(
        input_dyn_dim=input_dyn_dim,
        input_static_dim=input_static_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=0.3,
        num_classes=num_classes,
        static_hidden=128,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler now follows validation macro F1, because macro F1 is the metric
    # we care about for imbalanced LOS classification.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_threshold = 0.5 if args.label == "binary" else None
    best_state = None
    best_epoch = -1

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            tr_loader,
            device,
            criterion,
            optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp=(args.amp and device.type == "cuda"),
        )

        val_loss, val_pred_argmax, val_true, val_probs = eval_epoch(
            model,
            va_loader,
            device,
            criterion,
            amp=(args.amp and device.type == "cuda"),
        )

        if args.label == "binary":
            epoch_threshold, val_f1 = tune_binary_threshold(
                val_true,
                val_probs[:, 1],
            )
            val_pred = (val_probs[:, 1] >= epoch_threshold).astype(int)
        else:
            epoch_threshold = None
            val_pred = val_pred_argmax
            val_f1 = f1_score(
                val_true,
                val_pred,
                average="macro",
                zero_division=0,
            )

        scheduler.step(val_f1)

        if args.label == "binary":
            print(
                f"Epoch {epoch:02d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"F1 {val_f1:.4f} | "
                f"thr {epoch_threshold:.2f}"
            )
        else:
            print(
                f"Epoch {epoch:02d} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"F1 {val_f1:.4f}"
            )

        # Save the checkpoint that maximizes validation macro F1, not val loss.
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            best_threshold = epoch_threshold if args.label == "binary" else None
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch

        elif epoch - best_epoch >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    test_loss, test_pred_argmax, test_true, test_probs = eval_epoch(
        model,
        te_loader,
        device,
        criterion,
        amp=(args.amp and device.type == "cuda"),
    )

    if args.label == "binary":
        test_pred = (test_probs[:, 1] >= best_threshold).astype(int)
    else:
        test_pred = test_pred_argmax

    test_acc = accuracy_score(test_true, test_pred)

    test_f1_macro = f1_score(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        test_true,
        test_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"eicu_v1_{args.label}_"
        f"H{args.hidden}_L{args.layers}_"
        f"Fdyn{input_dyn_dim}_Fstat{input_static_dim}_"
        f"f1ckpt"
    )

    model_path = os.path.join(OUT_DIR, f"lstm_fusion_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(input_dyn_dim),
            "input_static_dim": int(input_static_dim),
            "hidden": int(args.hidden),
            "layers": int(args.layers),
            "num_classes": int(num_classes),
            "label": args.label,
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_threshold": float(best_threshold) if best_threshold is not None else None,
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "label": args.label,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_threshold) if best_threshold is not None else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(X_dyn_train.shape),
        "valid_shape_dyn": list(X_dyn_valid.shape),
        "test_shape_dyn": list(X_dyn_test.shape),
        "train_shape_static": list(X_static_train.shape),
        "input_dyn_dim": int(input_dyn_dim),
        "input_static_dim": int(input_static_dim),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1" if args.label == "binary" else None,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

    if args.label == "binary":
        print(
            f"Test Acc: {test_acc:.4f} | "
            f"F1 macro: {test_f1_macro:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Best epoch: {best_epoch} | "
            f"Best val F1: {best_val_f1:.4f} | "
            f"Threshold: {best_threshold:.2f}"
        )
    else:
        print(
            f"Test Acc: {test_acc:.4f} | "
            f"F1 macro: {test_f1_macro:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Best epoch: {best_epoch} | "
            f"Best val F1: {best_val_f1:.4f}"
        )


if __name__ == "__main__":
    main()
