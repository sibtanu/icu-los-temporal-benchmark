#!/usr/bin/env python3
"""
V1_train_eicu_hybrid_lstm_mamba.py
----------------------------------

Full eICU training script using the same Hybrid BiLSTM + BiMamba architecture
family used in the MIMIC-III V13.2 experiment, adapted for the eICU pipeline.

Reads the SAME eICU preprocessed files used by V1_train_eicu_lstm_fusion.py:

    /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1/
        X_dyn_train.npy
        X_dyn_valid.npy
        X_dyn_test.npy
        X_static_train.npy
        X_static_valid.npy
        X_static_test.npy
        y_bin_train.npy / valid / test
        y_3c_train.npy / valid / test

Architecture: 
- Dynamic branch 1: BiLSTM over hourly sequence, shape (N, 24, F_dyn)
- Dynamic branch 2: BiMamba over same hourly sequence
- Static branch: ANN/MLP over static features, shape (N, F_static)
- Fusion head: concat dynamic + static representations -> dense classifier

Important eICU improvements over the original MIMIC V13.2 script:
- checkpoints by validation macro F1, not validation loss
- binary mode tunes positive-class probability threshold on validation macro F1
- applies tuned threshold to test set
- keeps focal loss for 3-class
- supports AMP
- hard-fails if mamba_ssm is missing

Run examples:
    python V1_train_eicu_hybrid_lstm_mamba.py --label binary
    python V1_train_eicu_hybrid_lstm_mamba.py --label 3class
    python V1_train_eicu_hybrid_lstm_mamba.py --label 3class --amp
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

from mamba_ssm.modules.mamba_simple import Mamba  # hard fail if unavailable


# ======================================================
# CONFIG
# ======================================================

BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/eicu_lstm_data_v1"
OUT_DIR = f"{BASE}/eicu_hybrid_lstm_mamba_models_v1"

os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# SEED
# ======================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ======================================================
# DATASET
# ======================================================

class DynStaticDataset(Dataset):
    def __init__(self, X_dyn, X_static, y, copy_mmap: bool = True):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y
        self.copy_mmap = copy_mmap

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        if self.copy_mmap:
            xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        else:
            xd = torch.from_numpy(self.Xd[idx]).float()

        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, xs, y


def make_loaders(
    Xd_tr,
    Xs_tr,
    y_tr,
    Xd_va,
    Xs_va,
    y_va,
    Xd_te,
    Xs_te,
    y_te,
    batch: int,
    workers: int,
    copy_mmap: bool = True,
):
    kwargs = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(
        DynStaticDataset(Xd_tr, Xs_tr, y_tr, copy_mmap=copy_mmap),
        shuffle=True,
        **kwargs,
    )

    valid_loader = DataLoader(
        DynStaticDataset(Xd_va, Xs_va, y_va, copy_mmap=copy_mmap),
        shuffle=False,
        **kwargs,
    )

    test_loader = DataLoader(
        DynStaticDataset(Xd_te, Xs_te, y_te, copy_mmap=copy_mmap),
        shuffle=False,
        **kwargs,
    )

    return train_loader, valid_loader, test_loader


# ======================================================
# LOSS
# ======================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, alpha=None, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        targets = targets.long()
        logp = F.log_softmax(logits, dim=1)
        pt_log = logp.gather(1, targets[:, None]).squeeze(1)
        pt = pt_log.exp().clamp(self.eps, 1.0 - self.eps)

        focal = (1.0 - pt).pow(self.gamma)
        loss = -focal * pt_log

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ======================================================
# MODEL
# ======================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class BiMamba(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        layers: int = 2,
        resid_scale: float = 0.5,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(d_model)
        self.resid_scale = float(resid_scale)

    def encode(self, x):
        h = self.proj(x)
        for block in self.blocks:
            h = h + self.resid_scale * block(h)
            h = self.norm(h)
        return h

    def forward(self, x):
        fwd = self.encode(x)
        bwd = self.encode(torch.flip(x, dims=[1]))

        return torch.cat(
            [
                fwd[:, -1],
                bwd[:, -1],
                fwd.mean(dim=1),
                bwd.mean(dim=1),
            ],
            dim=1,
        )


class HybridDynEncoder(nn.Module):
    def __init__(
        self,
        input_dyn_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        mamba_dim: int = 128,
        mamba_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm_hidden = lstm_hidden
        self.mamba_dim = mamba_dim

        self.lstm = nn.LSTM(
            input_size=input_dyn_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.mamba = BiMamba(
            input_dim=input_dyn_dim,
            d_model=mamba_dim,
            layers=mamba_layers,
            resid_scale=0.5,
        )

    @property
    def output_dim(self):
        # LSTM: hn[-2], hn[-1], out.mean = H + H + 2H = 4H
        # Mamba: f_last + b_last + f_mean + b_mean = 4M
        return 4 * self.lstm_hidden + 4 * self.mamba_dim

    def forward(self, x):
        out, (hn, _) = self.lstm(x)

        h_lstm = torch.cat(
            [
                hn[-2],
                hn[-1],
                out.mean(dim=1),
            ],
            dim=1,
        )

        h_mamba = self.mamba(x)
        h = torch.cat([h_lstm, h_mamba], dim=1)
        return F.layer_norm(h, (h.size(1),))


class HybridLSTMMambaModel(nn.Module):
    def __init__(
        self,
        input_dyn_dim: int,
        input_static_dim: int,
        num_classes: int,
        lstm_hidden: int = 256,
        mamba_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.dyn = HybridDynEncoder(
            input_dyn_dim=input_dyn_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=2,
            mamba_dim=mamba_dim,
            mamba_layers=2,
            dropout=0.3,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(input_static_dim, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
        )

        fusion_dim = self.dyn.output_dim + 128

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.fusion_gate = nn.Linear(fusion_dim, fusion_dim)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_dyn, x_static):
        h_dyn = self.dyn(x_dyn)
        h_static = self.static_mlp(x_static)

        h = torch.cat([h_dyn, h_static], dim=1)
        h = self.fusion_norm(h)

        gate = torch.sigmoid(self.fusion_gate(h))
        h = h * gate

        return self.head(h)


# ======================================================
# TRAIN / EVAL
# ======================================================

def train_epoch(model, loader, device, criterion, optimizer, scaler=None, grad_clip=1.0, amp=False):
    model.train()
    loss_sum = 0.0
    n = 0

    for xd, xs, y in loader:
        xd = xd.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(xd, xs)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xd, xs)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = y.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

    return loss_sum / max(n, 1)


@torch.no_grad()
def eval_epoch_with_probs(model, loader, device, criterion, amp=False):
    model.eval()
    loss_sum = 0.0
    n = 0

    preds = []
    targets = []
    probs_all = []

    for xd, xs, y in loader:
        xd = xd.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(xd, xs)
                loss = criterion(logits, y)
        else:
            logits = model(xd, xs)
            loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)

        bs = y.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        probs_all.append(probs.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    return (
        loss_sum / max(n, 1),
        np.concatenate(preds),
        np.concatenate(targets),
        np.concatenate(probs_all),
    )


def tune_binary_threshold(y_true, prob_pos):
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--mamba_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    seed_everything(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run this on a GPU node.")

    device = torch.device("cuda")

    print("USING eICU HYBRID LSTM + MAMBA")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Device: {device}")

    Xd_tr = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    Xd_va = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    Xd_te = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    Xs_tr = np.load(f"{DATA_DIR}/X_static_train.npy")
    Xs_va = np.load(f"{DATA_DIR}/X_static_valid.npy")
    Xs_te = np.load(f"{DATA_DIR}/X_static_test.npy")

    if args.label == "binary":
        y_tr = np.load(f"{DATA_DIR}/y_bin_train.npy")
        y_va = np.load(f"{DATA_DIR}/y_bin_valid.npy")
        y_te = np.load(f"{DATA_DIR}/y_bin_test.npy")
        num_classes = 2
    else:
        y_tr = np.load(f"{DATA_DIR}/y_3c_train.npy")
        y_va = np.load(f"{DATA_DIR}/y_3c_valid.npy")
        y_te = np.load(f"{DATA_DIR}/y_3c_test.npy")
        num_classes = 3

    print(f"Loaded dyn: train={Xd_tr.shape}, valid={Xd_va.shape}, test={Xd_te.shape}")
    print(f"Loaded static: train={Xs_tr.shape}, valid={Xs_va.shape}, test={Xs_te.shape}")
    print(f"Classes: {num_classes}")

    train_loader, valid_loader, test_loader = make_loaders(
        Xd_tr,
        Xs_tr,
        y_tr,
        Xd_va,
        Xs_va,
        y_va,
        Xd_te,
        Xs_te,
        y_te,
        batch=args.batch,
        workers=args.workers,
        copy_mmap=True,
    )

    class_counts = np.bincount(y_tr, minlength=num_classes)
    print("Train class counts:", class_counts)

    if args.label == "binary":
        weights = class_counts.sum() / (class_counts + 1e-9)
        weights = weights / weights.sum()
        ce_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        print("Using weighted CrossEntropy for binary.")
        print("Weights:", weights)
    else:
        criterion = FocalLoss(gamma=1.5, alpha=[0.25, 0.5, 0.25]).to(device)
        print("Using FocalLoss for 3-class.")

    model = HybridLSTMMambaModel(
        input_dyn_dim=Xd_tr.shape[2],
        input_static_dim=Xs_tr.shape[1],
        num_classes=num_classes,
        lstm_hidden=args.lstm_hidden,
        mamba_dim=args.mamba_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_f1 = -1.0
    best_val_loss = None
    best_state = None
    best_epoch = -1
    best_threshold = 0.5 if args.label == "binary" else None

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp=args.amp,
        )

        val_loss, val_pred_argmax, val_true, val_probs = eval_epoch_with_probs(
            model,
            valid_loader,
            device,
            criterion,
            amp=args.amp,
        )

        if args.label == "binary":
            threshold, val_f1 = tune_binary_threshold(val_true, val_probs[:, 1])
        else:
            threshold = None
            val_f1 = f1_score(val_true, val_pred_argmax, average="macro", zero_division=0)

        scheduler.step(val_f1)

        if threshold is not None:
            print(
                f"Epoch {epoch:02d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | F1 {val_f1:.4f} | thr {threshold:.2f}"
            )
        else:
            print(
                f"Epoch {epoch:02d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | F1 {val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_threshold = threshold if threshold is not None else best_threshold
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        elif best_epoch != -1 and (epoch - best_epoch) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint produced.")

    model.load_state_dict(best_state)

    test_loss, test_pred_argmax, test_true, test_probs = eval_epoch_with_probs(
        model,
        test_loader,
        device,
        criterion,
        amp=args.amp,
    )

    if args.label == "binary":
        test_pred = (test_probs[:, 1] >= best_threshold).astype(int)
    else:
        test_pred = test_pred_argmax

    test_acc = accuracy_score(test_true, test_pred)
    test_f1_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)

    report = classification_report(
        test_true,
        test_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(test_true, test_pred).tolist()

    stamp = (
        f"eicu_v1_hybrid_lstm_mamba_{args.label}_"
        f"H{args.lstm_hidden}_M{args.mamba_dim}_"
        f"Fdyn{Xd_tr.shape[2]}_Fstat{Xs_tr.shape[1]}"
    )

    model_path = os.path.join(OUT_DIR, f"model_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "num_classes": int(num_classes),
            "label": args.label,
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_threshold": float(best_threshold) if best_threshold is not None else None,
            "lstm_hidden": int(args.lstm_hidden),
            "mamba_dim": int(args.mamba_dim),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Hybrid BiLSTM + BiMamba + static ANN fusion",
        "label": args.label,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "classification_report": report,
        "confusion_matrix": cm,
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_threshold": float(best_threshold) if best_threshold is not None else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
        "train_shape_dyn": list(Xd_tr.shape),
        "valid_shape_dyn": list(Xd_va.shape),
        "test_shape_dyn": list(Xd_te.shape),
        "train_shape_static": list(Xs_tr.shape),
        "input_dyn_dim": int(Xd_tr.shape[2]),
        "input_static_dim": int(Xs_tr.shape[1]),
        "checkpoint_metric": "validation_macro_f1",
        "binary_threshold_tuned_on": "validation_macro_f1" if args.label == "binary" else None,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {test_acc:.4f} | "
        f"F1 macro: {test_f1_macro:.4f} | "
        f"Test loss: {test_loss:.4f} | "
        f"Best epoch: {best_epoch} | "
        f"Best val F1: {best_val_f1:.4f}"
    )
    if args.label == "binary":
        print(f"Best threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
