#!/usr/bin/env python3
"""
V1_train_eicu_transformer_dynamic_only_ordinal.py

Dynamic-only ablation for eICU 3-class ordinal LOS prediction.

Reads:
  /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1/X_dyn_*.npy
  /lustre/home/rahas2/mimic_projects/outputs/eicu_lstm_data_v1/y_3c_*.npy

Output:
  /lustre/home/rahas2/mimic_projects/outputs/eicu_transformer_dynamic_only_ordinal_models_v1

Run:
  python V1_train_eicu_transformer_dynamic_only_ordinal.py --workers 0 --batch 128
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
OUT_DIR = f"{BASE}/eicu_transformer_dynamic_only_ordinal_models_v1"
os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DynamicOnlyOrdinalDataset(Dataset):
    def __init__(self, X_dyn, y_3c):
        self.Xd = X_dyn
        self.y = y_3c.astype(np.int64)

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32, copy=True))
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, y


def make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch=256, workers=0):
    kwargs = dict(
        batch_size=batch,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    train_loader = DataLoader(DynamicOnlyOrdinalDataset(Xtr, ytr), shuffle=True, **kwargs)
    valid_loader = DataLoader(DynamicOnlyOrdinalDataset(Xva, yva), shuffle=False, **kwargs)
    test_loader = DataLoader(DynamicOnlyOrdinalDataset(Xte, yte), shuffle=False, **kwargs)

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


class TransformerDynamicOnlyOrdinal(nn.Module):
    def __init__(
        self,
        dyn_dim: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 3,
        ff_dim: int = 384,
        dropout: float = 0.15,
        max_timesteps: int = 24,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.max_timesteps = max_timesteps

        self.dyn_norm = nn.LayerNorm(dyn_dim)

        self.dyn_proj = nn.Sequential(
            nn.Linear(dyn_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.pos_embed = nn.Embedding(max_timesteps, d_model)
        self.time_decay = nn.Parameter(torch.tensor(0.03))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pooled_dim = d_model * 2

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x_dyn):
        _, T, _ = x_dyn.shape

        if T > self.max_timesteps:
            raise ValueError(f"Unexpected T={T}, max_timesteps={self.max_timesteps}")

        x_dyn = self.dyn_norm(x_dyn)
        tokens = self.dyn_proj(x_dyn)

        time_idx = torch.arange(T, device=x_dyn.device, dtype=torch.float32)
        decay = torch.exp(-torch.clamp(self.time_decay, 0.0, 1.0) * time_idx)
        tokens = tokens * decay.view(1, T, 1)

        pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0)
        tokens = tokens + self.pos_embed(pos_ids)

        encoded = self.encoder(tokens)

        h_last = encoded[:, -1, :]
        h_mean = encoded.mean(dim=1)
        pooled = torch.cat([h_last, h_mean], dim=1)

        return self.head(pooled)


def train_epoch(model, loader, device, criterion, optimizer, scheduler=None, scaler=None, amp=False, grad_clip=1.0):
    model.train()
    loss_sum = 0.0
    n = 0

    for Xd, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(Xd)
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

    for Xd, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(Xd)
                loss = criterion(logits, y)
        else:
            logits = model(Xd)
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
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("USING eICU TRANSFORMER DYNAMIC-ONLY ORDINAL 3-CLASS")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    X_dyn_train = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    X_dyn_valid = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    X_dyn_test = np.load(f"{DATA_DIR}/X_dyn_test.npy", mmap_mode="r")

    y_train = np.load(f"{DATA_DIR}/y_3c_train.npy")
    y_valid = np.load(f"{DATA_DIR}/y_3c_valid.npy")
    y_test = np.load(f"{DATA_DIR}/y_3c_test.npy")

    print(f"Loaded dyn: train={X_dyn_train.shape}, valid={X_dyn_valid.shape}, test={X_dyn_test.shape}")

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
        X_dyn_train, y_train,
        X_dyn_valid, y_valid,
        X_dyn_test, y_test,
        batch=args.batch,
        workers=args.workers,
    )

    dyn_dim = X_dyn_train.shape[2]

    model = TransformerDynamicOnlyOrdinal(
        dyn_dim=dyn_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_timesteps=X_dyn_train.shape[1],
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
        f"eicu_v1_transformer_dynamic_only_ordinal_3class_"
        f"D{args.d_model}_H{args.nhead}_L{args.layers}_FF{args.ff_dim}_Fdyn{dyn_dim}"
    )

    model_path = os.path.join(OUT_DIR, f"transformer_dynamic_only_ordinal_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dyn_dim": int(dyn_dim),
            "d_model": int(args.d_model),
            "nhead": int(args.nhead),
            "layers": int(args.layers),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "num_classes_original": 3,
            "num_ordinal_logits": 2,
            "label": "3class_ordinal",
            "ablation": "dynamic_only",
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "best_val_loss": float(best_val_loss),
            "best_thresholds": list(best_thresholds),
        },
        model_path,
    )

    metrics = {
        "dataset": "eICU",
        "model": "Transformer dynamic-only ordinal",
        "label": "3class_ordinal",
        "ablation": "dynamic_only",
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
        "train_shape_dyn": list(X_dyn_train.shape),
        "valid_shape_dyn": list(X_dyn_valid.shape),
        "test_shape_dyn": list(X_dyn_test.shape),
        "input_dyn_dim": int(dyn_dim),
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
