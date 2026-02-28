#!/usr/bin/env python3
"""
V12.7_train_script.py
----------------------
Transformer classifier with:
- Static MLP -> CLS token
- Dynamic feature MLP per timestep (376 -> d_model with nonlinearity)
- Temporal decay on dynamic tokens
- Correct class weights

Run:
  python V12.7_train_script.py --label binary
  python V12.7_train_script.py --label 3class
"""

import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ==== CONFIG ====
BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR  = f"{BASE}/lstm_models_v12.7"
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
    batch=256, workers=2
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


# ==== MODEL ====
class LOSTransformerV127(nn.Module):
    """
    Transformer with:
      - Dynamic feature MLP per timestep (376 -> d_model)
      - Static MLP -> CLS token
      - Temporal decay on dynamic tokens
    """

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_timesteps: int = 24,
    ):
        super().__init__()
        self.max_timesteps = max_timesteps
        self.d_model = d_model

        # Normalize raw inputs
        self.dyn_norm = nn.LayerNorm(dyn_dim)
        self.static_norm = nn.LayerNorm(static_dim)

        # Dynamic feature MLP per timestep (key change vs 12.5/12.6)
        self.dyn_mlp = nn.Sequential(
            nn.Linear(dyn_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Static MLP -> CLS token
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Positional embeddings for (CLS + 24 hours)
        self.pos_embed = nn.Embedding(max_timesteps + 1, d_model)

        # Temporal decay parameter
        self.time_decay = nn.Parameter(torch.tensor(0.05))

        # Pre-norm before encoder
        self.pre_ln = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head on CLS token
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x_dyn, x_static):
        """
        x_dyn:    (B, T, dyn_dim)
        x_static: (B, static_dim)
        """
        B, T, D = x_dyn.shape
        assert T <= self.max_timesteps, "Unexpected number of timesteps."

        # Normalize
        x_dyn = self.dyn_norm(x_dyn)
        x_static = self.static_norm(x_static)

        # Dynamic feature MLP per timestep
        dyn_tokens = self.dyn_mlp(x_dyn)    # (B, T, d_model)

        # Temporal decay across time
        time_idx = torch.arange(T, device=x_dyn.device)
        decay = torch.exp(-self.time_decay * time_idx)   # (T,)
        dyn_tokens = dyn_tokens * decay.view(1, T, 1)

        # Static -> CLS token
        cls_token = self.static_mlp(x_static).unsqueeze(1)   # (B, 1, d_model)

        # Concatenate CLS + dynamic tokens
        tokens = torch.cat([cls_token, dyn_tokens], dim=1)   # (B, 1+T, d_model)

        # Add positions
        seq_len = tokens.size(1)
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)  # (1, L)
        tokens = tokens + self.pos_embed(pos_ids)

        # Pre-norm + encoder
        tokens = self.pre_ln(tokens)
        encoded = self.encoder(tokens)      # (B, L, d_model)

        # CLS head
        cls_out = encoded[:, 0, :]          # (B, d_model)
        logits = self.head(cls_out)         # (B, C)
        return logits


# ==== TRAIN / EVAL ====
def train_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    scheduler=None,
    grad_clip: float = 1.0,
):
    model.train()
    loss_sum = 0.0
    n = 0

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(Xd, Xs)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

        if scheduler is not None:
            scheduler.step()

    avg_loss = loss_sum / max(n, 1)
    return avg_loss


@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    n = 0
    preds = []
    targs = []

    for Xd, Xs, y in loader:
        Xd = Xd.to(device, non_blocking=True)
        Xs = Xs.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        logits = model(Xd, Xs)
        loss = criterion(logits, y)

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n += bs

        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targs.append(y.cpu().numpy())

    avg_loss = loss_sum / max(n, 1)
    preds = np.concatenate(preds) if preds else np.array([])
    targs = np.concatenate(targs) if targs else np.array([])
    return avg_loss, preds, targs


# ==== MAIN ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["binary", "3class"], default="binary")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # --- Load data ---
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

    tr, va, te = make_loaders(
        X_dyn_train, X_static_train, y_train,
        X_dyn_valid, X_static_valid, y_valid,
        X_dyn_test,  X_static_test,  y_test,
        batch=args.batch,
        workers=args.workers,
    )

    # --- Class weights (fixed) ---
    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-9)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    dyn_dim = X_dyn_train.shape[2]
    static_dim = X_static_train.shape[1]

    model = LOSTransformerV127(
        dyn_dim=dyn_dim,
        static_dim=static_dim,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_timesteps=X_dyn_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Warmup scheduler (5% steps)
    total_steps = args.epochs * max(len(tr), 1)
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --- Training loop with early stopping ---
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
            scheduler=scheduler,
            grad_clip=args.grad_clip,
        )
        va_loss, va_pred, va_true = eval_epoch(
            model, va, device, criterion
        )
        va_f1 = f1_score(va_true, va_pred, average="macro")
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | F1 {va_f1:.3f}")

        if np.isfinite(va_loss) and va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = model.state_dict()
            best_ep = ep
        elif ep - best_ep >= args.patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final test ---
    te_loss, te_pred, te_true = eval_epoch(
        model, te, device, criterion
    )
    te_acc = accuracy_score(te_true, te_pred)
    te_f1 = f1_score(te_true, te_pred, average="macro")
    rep = classification_report(te_true, te_pred, digits=4, output_dict=True)
    cm = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"v12_7_{args.label}_TRF{args.d_model}_L{args.layers}_H{args.ff_dim}"
    model_path = os.path.join(OUT_DIR, f"model_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dyn_dim": dyn_dim,
            "static_dim": static_dim,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
            "num_classes": num_classes,
        },
        model_path,
    )

    metrics = {
        "label": args.label,
        "test_loss": float(te_loss) if np.isfinite(te_loss) else None,
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

Results:

{
  "label": "binary",
  "test_loss": 0.5051750191578173,
  "test_acc": 0.7845852082612207,
  "test_f1_macro": 0.6987678931440212,
  "class_report": {
    "0": {
      "precision": 0.9532788252961789,
      "recall": 0.7826027397260275,
      "f1-score": 0.8595501391709922,
      "support": 7300.0
    },
    "1": {
      "precision": 0.40650710545998503,
      "recall": 0.7951719092904169,
      "f1-score": 0.5379856471170502,
      "support": 1367.0
    },
    "accuracy": 0.7845852082612207,
    "macro avg": {
      "precision": 0.679892965378082,
      "recall": 0.7888873245082222,
      "f1-score": 0.6987678931440212,
      "support": 8667.0
    },
    "weighted avg": {
      "precision": 0.867039418233057,
      "recall": 0.7845852082612207,
      "f1-score": 0.8088314751998673,
      "support": 8667.0
    }
  },
  "confusion_matrix": [
    [
      5713,
      1587
    ],
    [
      280,
      1087
    ]
  ],
  "best_epoch": 7,
  "elapsed_sec": 811.0,
  "class_counts_train": [
    34065,
    6378
  ]
}