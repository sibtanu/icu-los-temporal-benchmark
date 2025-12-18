#!/usr/bin/env python3
"""
V13.2_train_hybrid_lstm_mamba.py
--------------------------------
HYBRID dynamic encoder: BiLSTM + BiMamba (parallel)

Matches V12.8 behavior:
- Early stopping with patience
- Best checkpoint restore
- Save model + metrics JSON

Hard fail if mamba-ssm missing.
"""

import os, json, time, random, argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from mamba_ssm.modules.mamba_simple import Mamba  # hard-fail if missing

# ==== CONFIG ====
BASE = "/lustre/home/rahas2/mimic_projects/outputs"
DATA_DIR = f"{BASE}/lstm_data_v12.1_seq"
OUT_DIR  = f"{BASE}/lstm_models_v13.2"
os.makedirs(OUT_DIR, exist_ok=True)

# ==== SEED ====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==== DATASET ====
class DynStaticDataset(Dataset):
    def __init__(self, Xd, Xs, y, copy_mmap: bool = True):
        self.Xd, self.Xs, self.y = Xd, Xs, y
        self.copy_mmap = copy_mmap

    def __len__(self):
        return self.Xd.shape[0]

    def __getitem__(self, i):
        # Xd is memmap read-only -> copy avoids PyTorch non-writable warning/undefined behavior.
        if self.copy_mmap:
            xd = torch.from_numpy(np.array(self.Xd[i], dtype=np.float32, copy=True))
        else:
            xd = torch.from_numpy(self.Xd[i]).float()

        xs = torch.from_numpy(np.array(self.Xs[i], dtype=np.float32, copy=False))
        y  = torch.tensor(self.y[i], dtype=torch.long)
        return xd, xs, y

def make_loaders(Xd_tr, Xs_tr, y_tr, Xd_va, Xs_va, y_va, Xd_te, Xs_te, y_te, batch, workers, copy_mmap=True):
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0))
    return (
        DataLoader(DynStaticDataset(Xd_tr, Xs_tr, y_tr, copy_mmap=copy_mmap), shuffle=True,  **kw),
        DataLoader(DynStaticDataset(Xd_va, Xs_va, y_va, copy_mmap=copy_mmap), shuffle=False, **kw),
        DataLoader(DynStaticDataset(Xd_te, Xs_te, y_te, copy_mmap=copy_mmap), shuffle=False, **kw),
    )

# ==== LOSS ====
class FocalLoss(nn.Module):
    """
    Stable multi-class focal loss.
    Clamp pt away from 0/1 to avoid NaNs/Infs.
    """
    def __init__(self, gamma=1.5, alpha=None, eps=1e-6, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=1)
        pt_log = logp.gather(1, y[:, None]).squeeze(1)
        pt = pt_log.exp().clamp(self.eps, 1.0 - self.eps)

        focal = (1.0 - pt).pow(self.gamma)
        loss = -focal * pt_log

        if self.alpha is not None:
            loss = loss * self.alpha[y]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# ==== MODEL ====
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class BiMamba(nn.Module):
    def __init__(self, f_in, d_model=128, layers=2, resid_scale=0.5):
        super().__init__()
        self.proj = nn.Linear(f_in, d_model)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(d_model)
        self.resid_scale = float(resid_scale)

    def encode(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = h + self.resid_scale * b(h)
            h = self.norm(h)
        return h

    def forward(self, x):
        f = self.encode(x)
        b = self.encode(torch.flip(x, dims=[1]))
        return torch.cat([f[:, -1], b[:, -1], f.mean(1), b.mean(1)], dim=1)  # (B, 4*d_model)

class HybridDynEncoder(nn.Module):
    def __init__(self, f_dyn, lstm_hidden=256, mamba_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=f_dyn,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.mamba = BiMamba(f_dyn, d_model=mamba_dim, layers=2, resid_scale=0.5)

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        # (B,H),(B,H),(B,2H) -> (B,4H)
        h_lstm = torch.cat([hn[-2], hn[-1], out.mean(1)], dim=1)
        h_mamba = self.mamba(x)  # (B, 4*M)
        h = torch.cat([h_lstm, h_mamba], dim=1)
        return F.layer_norm(h, (h.size(1),))

class HybridModel(nn.Module):
    def __init__(self, f_dyn, f_stat, n_classes):
        super().__init__()
        self.dyn = HybridDynEncoder(f_dyn)
        self.stat = nn.Sequential(
            nn.Linear(f_stat, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128),
        )
        self.head = nn.Sequential(
            nn.Linear(4*256 + 4*128 + 128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, xd, xs):
        h = torch.cat([self.dyn(xd), self.stat(xs)], dim=1)
        return self.head(h)

# ==== TRAIN / EVAL ====
def run_epoch(model, loader, device, criterion, opt=None, grad_clip=1.0, amp=False, scaler=None):
    train = opt is not None
    model.train() if train else model.eval()

    loss_sum, n = 0.0, 0
    preds, targs = [], []

    for xd, xs, y in loader:
        xd = xd.to(device, non_blocking=True)
        xs = xs.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        if train:
            opt.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(xd, xs)
                loss = criterion(out, y)
        else:
            out = model(xd, xs)
            loss = criterion(out, y)

        if train:
            if not torch.isfinite(loss):
                raise RuntimeError("Loss became NaN/Inf. Aborting.")

            if amp and device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

        bs = y.size(0)
        loss_sum += float(loss.detach().cpu().item()) * bs
        n += bs

        preds.append(out.argmax(1).detach().cpu().numpy())
        targs.append(y.detach().cpu().numpy())

    return loss_sum / max(n, 1), np.concatenate(preds), np.concatenate(targs)

# ==== MAIN ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["binary", "3class"], default="binary")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)   # SAME IDEA AS V12.8
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--copy_mmap", action="store_true", help="Copy memmap slices (recommended)")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    seed_everything(42)
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    print("USING HYBRID LSTM + MAMBA ✔")

    # Load data
    Xd_tr = np.load(f"{DATA_DIR}/X_dyn_train.npy", mmap_mode="r")
    Xd_va = np.load(f"{DATA_DIR}/X_dyn_valid.npy", mmap_mode="r")
    Xd_te = np.load(f"{DATA_DIR}/X_dyn_test.npy",  mmap_mode="r")
    Xs_tr = np.load(f"{DATA_DIR}/X_static_train.npy")
    Xs_va = np.load(f"{DATA_DIR}/X_static_valid.npy")
    Xs_te = np.load(f"{DATA_DIR}/X_static_test.npy")

    if args.label == "binary":
        y_tr = np.load(f"{DATA_DIR}/y_bin_train.npy")
        y_va = np.load(f"{DATA_DIR}/y_bin_valid.npy")
        y_te = np.load(f"{DATA_DIR}/y_bin_test.npy")
        ncls = 2

        cnt = np.bincount(y_tr, minlength=ncls)
        w = torch.tensor(cnt.sum() / (cnt + 1e-9), dtype=torch.float32, device=device)
        w = w / w.sum()
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        y_tr = np.load(f"{DATA_DIR}/y_3c_train.npy")
        y_va = np.load(f"{DATA_DIR}/y_3c_valid.npy")
        y_te = np.load(f"{DATA_DIR}/y_3c_test.npy")
        ncls = 3

        criterion = FocalLoss(gamma=1.5, alpha=[0.25, 0.5, 0.25]).to(device)

    print(f"Loaded dyn: train={Xd_tr.shape}, valid={Xd_va.shape}, test={Xd_te.shape}")
    print(f"Loaded static: train={Xs_tr.shape}, valid={Xs_va.shape}, test={Xs_te.shape}")
    print(f"Classes: {ncls}")

    tr, va, te = make_loaders(
        Xd_tr, Xs_tr, y_tr,
        Xd_va, Xs_va, y_va,
        Xd_te, Xs_te, y_te,
        args.batch, args.workers,
        copy_mmap=(True if args.copy_mmap else True),  # default safe
    )

    model = HybridModel(Xd_tr.shape[2], Xs_tr.shape[1], ncls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val = float("inf")
    best_state = None
    best_ep = -1
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(
            model, tr, device, criterion, opt=opt,
            grad_clip=args.grad_clip, amp=args.amp, scaler=scaler
        )
        va_loss, va_pred, va_true = run_epoch(
            model, va, device, criterion, opt=None,
            grad_clip=args.grad_clip, amp=args.amp, scaler=None
        )
        va_f1 = f1_score(va_true, va_pred, average="macro")

        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f} | F1 {va_f1:.3f}")

        if np.isfinite(va_loss) and va_loss < best_val:
            best_val = float(va_loss)
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif best_ep != -1 and (ep - best_ep) >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Never achieved a finite validation loss.")

    model.load_state_dict(best_state)  # loads CPU tensors into GPU params (Torch copies)
    te_loss, te_pred, te_true = run_epoch(model, te, device, criterion, opt=None, amp=args.amp)

    te_acc = accuracy_score(te_true, te_pred)
    te_f1  = f1_score(te_true, te_pred, average="macro")
    rep    = classification_report(te_true, te_pred, digits=4, output_dict=True)
    cm     = confusion_matrix(te_true, te_pred).tolist()

    stamp = f"v13.2_hybrid_{args.label}_ep{best_ep}"
    model_path   = os.path.join(OUT_DIR, f"hybrid_{stamp}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{stamp}.json")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dyn_dim": int(Xd_tr.shape[2]),
            "input_static_dim": int(Xs_tr.shape[1]),
            "num_classes": int(ncls),
            "best_epoch": int(best_ep),
            "label": args.label,
        },
        model_path,
    )

    metrics = {
        "label": args.label,
        "best_epoch": int(best_ep),
        "val_best_loss": float(best_val),
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_macro": float(te_f1),
        "confusion_matrix": cm,
        "class_report": rep,
        "elapsed_sec": round(time.time() - t0, 1),
        "train_shape_dyn": list(Xd_tr.shape),
        "train_shape_static": list(Xs_tr.shape),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved model:   {model_path}")
    print(f"✅ Saved metrics: {metrics_path}")
    print(f"Test Acc: {te_acc:.3f} | F1(macro): {te_f1:.3f} | Test loss: {te_loss:.4f} | Best epoch: {best_ep}")

if __name__ == "__main__":
    main()
