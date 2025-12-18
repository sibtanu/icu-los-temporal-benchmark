#!/usr/bin/env python3
"""
V12.8_train_lstm_model.py
-------------------------
Improved LSTM+MLP fusion model on:
- Dynamic features:  X_dyn_* (N, 24, F_dyn)
- Static features:   X_static_* (N, F_static)

Goals vs V12.3 / V12.4:
- Stay in LSTM family (which fits 24x376 well)
- Stronger dynamic encoder:
    * BiLSTM
    * Last hidden + mean-pooled outputs (trend + final state)
- Stronger static encoder:
    * 2-layer MLP with LayerNorm
- Better fusion:
    * LayerNorm + gated fusion + deeper head with GELU
- Same data pipeline (no changes to build/preprocess scripts)

Run:
  python V12.8_train_lstm_model.py --label binary
  python V12.8_train_lstm_model.py --label 3class
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

# ==== CONFIG ====
BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR  = f"{BASE}/lstm_models_v12.8"
os.makedirs(OUT_DIR, exist_ok=True)


# ==== SEED ====
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==== DATASET ====
class DynStaticDataset(Dataset):
    def __init__(self, X_dyn, X_static, y):
        self.Xd = X_dyn
        self.Xs = X_static
        self.y = y

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, idx):
        xd = torch.from_numpy(np.array(self.Xd[idx], dtype=np.float32))      # (T, F_dyn)
        xs = torch.from_numpy(np.array(self.Xs[idx], dtype=np.float32))      # (F_static,)
        y  = torch.tensor(self.y[idx], dtype=torch.long)
        return xd, xs, y


def make_loaders(
    X_dyn_train, X_static_train, y_train,
    X_dyn_valid, X_static_valid, y_valid,
    X_dyn_test,  X_static_test,  y_test,
    batch=256, workers=2,
):
    tr = DataLoader(
        DynStaticDataset(X_dyn_train, X_static_train, y_train),
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )
    va = DataLoader(
        DynStaticDataset(X_dyn_valid, X_static_valid, y_valid),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )
    te = DataLoader(
        DynStaticDataset(X_dyn_test, X_static_test, y_test),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )
    return tr, va, te


# ==== FOCAL LOSS (for 3-class, like 12.4) ====
class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    gamma: focusing parameter.
    alpha: class weights tensor of shape [C] or list, applied per target class.
    """
    def __init__(self, gamma: float = 1.5, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
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
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ==== MODEL ====
class FusionLSTMPlus(nn.Module):
    """
    Improved dynamic+static fusion model:

    Dynamic branch:
      - BiLSTM over (B, T, F_dyn)
      - Use both last hidden state and mean-pooled outputs
      -> concat([h_last_fwd; h_last_bwd; mean_pool])  = (B, 4*hidden)

    Static branch:
      - 2-layer MLP with LayerNorm

    Fusion:
      - LayerNorm on concat([dyn, static])
      - Gated fusion + 2-layer head with GELU
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
        self.hidden = hidden

        # Dynamic encoder: BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_dyn_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )

        dyn_out_dim = hidden * 2  # from BiLSTM
        # We'll use last hidden (2*hidden) + mean-pooled (2*hidden) -> 4*hidden
        fused_dyn_dim = dyn_out_dim * 2

        # Static encoder: 2-layer MLP with LayerNorm
        self.static_mlp = nn.Sequential(
            nn.Linear(input_static_dim, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(static_hidden),
        )

        # Fusion input dim
        fusion_in_dim = fused_dyn_dim + static_hidden

        # Fusion normalization + gated block
        self.fusion_ln = nn.LayerNorm(fusion_in_dim)
        self.fusion_gate = nn.Linear(fusion_in_dim, fusion_in_dim)

        # Head: deeper, GELU, LayerNorm inside
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
        # x_dyn:    (B, T, F_dyn)
        # x_static: (B, F_static)
        out, (hn, _) = self.lstm(x_dyn)  # out: (B, T, 2H), hn: (num_layers*2, B, H)

        # Last forward/backward hidden
        h_forward = hn[-2]   # (B, H)
        h_backward = hn[-1]  # (B, H)
        h_last = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)

        # Mean-pooling over time
        h_mean = out.mean(dim=1)  # (B, 2H)

        # Dynamic representation: last + mean
        h_dyn = torch.cat([h_last, h_mean], dim=1)  # (B, 4H)

        # Static representation
        h_static = self.static_mlp(x_static)  # (B, static_hidden)

        # Fusion
        fused = torch.cat([h_dyn, h_static], dim=1)  # (B, fusion_in_dim)
        fused = self.fusion_ln(fused)

        # Gating
        gate = torch.sigmoid(self.fusion_gate(fused))
        fused = fused * gate

        # Head
        logits = self.head(fused)
        return logits


# ==== TRAIN / EVAL ====
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
        y  = y.to(device, non_blocking=True)

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
    model.eval()
    loss_sum = 0.0
    n = 0
    preds = []
    targs = []

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

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

        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targs.append(y.cpu().numpy())

    preds = np.concatenate(preds) if preds else np.array([])
    targs = np.concatenate(targs) if targs else np.array([])

    return (loss_sum / max(n, 1)), preds, targs


# ==== MAIN ====
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

    # Load data
    X_dyn_train = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    X_dyn_valid = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    X_dyn_test  = np.load(f"{DATA_DIR}/X_dyn_test.npy",  mmap_mode="r")

    X_static_train = np.load(f"{DATA_DIR}/X_static_train.npy")
    X_static_valid = np.load(f"{DATA_DIR}/X_static_valid.npy")
    X_static_test  = np.load(f"{DATA_DIR}/X_static_test.npy")

    if args.label == "binary":
        y_train = np.load(f"{DATA_DIR}/y_bin_train.npy")
        y_valid = np.load(f"{DATA_DIR}/y_bin_valid.npy")
        y_test  = np.load(f"{DATA_DIR}/y_bin_test.npy")
        num_classes = 2
    else:
        y_train = np.load(f"{DATA_DIR}/y_3c_train.npy")
        y_valid = np.load(f"{DATA_DIR}/y_3c_valid.npy")
        y_test  = np.load(f"{DATA_DIR}/y_3c_test.npy")
        num_classes = 3

    print(
        f"Loaded dyn: train={X_dyn_train.shape}, valid={X_dyn_valid.shape}, test={X_dyn_test.shape}"
    )
    print(
        f"Loaded static: train={X_static_train.shape}, valid={X_static_valid.shape}, test={X_static_test.shape}"
    )
    print(f"Classes: {num_classes}")

    tr, va, te = make_loaders(
        X_dyn_train, X_static_train, y_train,
        X_dyn_valid, X_static_valid, y_valid,
        X_dyn_test,  X_static_test,  y_test,
        batch=args.batch,
        workers=args.workers,
    )

    # Class stats
    class_counts = np.bincount(y_train, minlength=num_classes)
    print("Train class counts:", class_counts)

    # Loss
    if args.label == "binary":
        # Keep the weighting scheme that already worked well in 12.3
        weights = class_counts.sum() / (class_counts + 1e-9)
        weights = weights / weights.sum()
        ce_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        print("Using weighted CrossEntropy (binary). Weights:", weights)
    else:
        # Focal loss for 3-class, emphasize class-1
        alpha = [0.25, 0.5, 0.25]
        criterion = FocalLoss(gamma=1.5, alpha=alpha, reduction="mean").to(device)
        print("Using FocalLoss (3-class). alpha=", alpha, "gamma=1.5")

    # Model
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

    # Scheduler: ReduceLROnPlateau on val loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    best_ep = -1
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model,
            tr,
            device,
            criterion,
            optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
            amp=(args.amp and device.type == "cuda"),
        )

        va_loss, va_pred, va_true = eval_epoch(
            model,
            va,
            device,
            criterion,
            amp=(args.amp and device.type == "cuda"),
        )
        va_f1 = f1_score(va_true, va_pred, average="macro")

        scheduler.step(va_loss)

        print(
            f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | F1 {va_f1:.3f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = model.state_dict()
            best_ep = ep
        elif ep - best_ep >= args.patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    te_loss, te_pred, te_true = eval_epoch(
        model,
        te,
        device,
        criterion,
        amp=(args.amp and device.type == "cuda"),
    )
    te_acc = accuracy_score(te_true, te_pred)
    te_f1  = f1_score(te_true, te_pred, average="macro")
    rep    = classification_report(te_true, te_pred, digits=4, output_dict=True)
    cm     = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"v12.8_{args.label}_H{args.hidden}_L{args.layers}"
    model_path   = os.path.join(OUT_DIR, f"lstm_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(input_dyn_dim),
            "input_static_dim": int(input_static_dim),
            "hidden": args.hidden,
            "layers": args.layers,
            "num_classes": num_classes,
        },
        model_path,
    )

    metrics = {
        "label": args.label,
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1),
        "class_report": rep,
        "confusion_matrix": cm,
        "best_epoch": best_ep,
        "elapsed_sec": round(time.time() - t0, 1),
        "class_counts_train": class_counts.tolist(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved model:   {model_path}")
    print(f"✅ Saved metrics: {metrics_path}")
    print(
        f"Test Acc: {te_acc:.3f} | F1(macro): {te_f1:.3f} | Test loss: {te_loss:.4f} | Best epoch: {best_ep}"
    )


if __name__ == "__main__":
    main()
