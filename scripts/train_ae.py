# scripts/train_ae.py
# ─────────────────────────────────────────────────────────────
# Train TCN Autoencoder on ETTh1 (or any ETT dataset).
#
# Usage :
#   python -m scripts.train_ae
#   or
#   python scripts/train_ae.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tcn_ae       import TCNAutoEncoder
from src.training.ae_trainers.ae_trainer import AETrainer


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    """
    Sliding window dataset for univariate time series.

    Args:
        series : np.ndarray of shape (T, 1)
        window : int  window length
        stride : int  stride between windows
    """

    def __init__(self, series: np.ndarray,
                 window: int, stride: int = 1):
        self.series = series.astype(np.float32)
        self.window = window
        T           = len(series)
        self.idxs   = np.arange(0, T - window + 1, stride)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        x = self.series[s:s + self.window]   # (L, 1)
        return (torch.from_numpy(x),)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        return SimpleNamespace(**json.load(f))


def load_series(cfg: SimpleNamespace):
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ot = df[[cfg.target_col]].values.astype(np.float32)
    T  = len(ot)

    t_train = int(cfg.train_ratio * T)
    t_val   = int((cfg.train_ratio + cfg.val_ratio) * T)

    scaler      = StandardScaler()
    ot_train_s  = scaler.fit_transform(ot[:t_train]).astype(np.float32)
    ot_val_s    = scaler.transform(ot[t_train:t_val]).astype(np.float32)
    ot_test_s   = scaler.transform(ot[t_val:]).astype(np.float32)

    print(f"[Data] {cfg.dataset_name} | "
          f"train={len(ot_train_s)} val={len(ot_val_s)} test={len(ot_test_s)}")
    return ot_train_s, ot_val_s, ot_test_s, scaler


def plot_reconstruction(model, x_sample, cfg, device, n=4):
    """
    Plot original vs reconstructed series for n samples.
    """
    model.eval()
    os.makedirs(cfg.figures_dir, exist_ok=True)

    with torch.no_grad():
        x   = x_sample[:n].to(device).float()
        rec, _ = model(x)

    x   = x.cpu().numpy()
    rec = rec.cpu().numpy()

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x[i, :, 0],   label="Original",       color="steelblue")
        ax.plot(rec[i, :, 0], label="Reconstructed",
                color="coral", linestyle="--")
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Time step")

    plt.suptitle(f"AE Reconstruction — {cfg.dataset_name}", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(cfg.figures_dir,
                             f"ae_recon_{cfg.dataset_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Reconstruction saved → {save_path}")


def plot_loss_curve(history, cfg):
    """Plot train/val loss curves."""
    os.makedirs(cfg.figures_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="Train loss", color="steelblue")
    ax.plot(history["val"],   label="Val loss",   color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"AE Training — {cfg.dataset_name}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(cfg.figures_dir,
                             f"ae_loss_{cfg.dataset_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Loss curve saved → {save_path}")


def latent_sanity_check(model, loader, device, n_batches=5):
    """
    Verify latent space is bounded in [-1, 1].
    """
    model.eval()
    zs = []
    with torch.no_grad():
        for i, (x,) in enumerate(loader):
            x = x.to(device).float()
            z = model.encode(x)
            zs.append(z.cpu())
            if i + 1 >= n_batches:
                break

    Z    = torch.cat(zs, dim=0)
    zmin = float(Z.min().item())
    zmax = float(Z.max().item())
    zmean= float(Z.abs().mean().item())

    print(f"\n[Latent sanity] z min={zmin:.4f}  max={zmax:.4f}  "
          f"|mean|={zmean:.4f}")
    print(f"  → bounded in [-1,1] : "
          f"{'✓' if zmin >= -1.01 and zmax <= 1.01 else '✗'}")
    return Z


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    config_path = "assets/configs/models/etth1_dataset/ae/tcn_etth1.json"
    cfg         = load_config(config_path)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir,    exist_ok=True)
    os.makedirs(cfg.figures_dir,    exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    ot_train_s, ot_val_s, ot_test_s, scaler = load_series(cfg)

    ds_train = WindowDataset(ot_train_s, window=cfg.seq_len, stride=1)
    ds_val   = WindowDataset(ot_val_s,   window=cfg.seq_len, stride=1)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                          shuffle=True,  drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size,
                          shuffle=False, drop_last=False)

    print(f"[Data] windows train={len(ds_train)} val={len(ds_val)}")

    # ── Train ────────────────────────────────────────────────
    trainer       = AETrainer(cfg)
    model, history = trainer.train(dl_train, dl_val)

    device = trainer.device

    # ── Plots ────────────────────────────────────────────────
    plot_loss_curve(history, cfg)

    # reconstruction on val samples
    val_batch = next(iter(dl_val))[0]
    plot_reconstruction(model, val_batch, cfg, device, n=4)

    # ── Latent sanity ────────────────────────────────────────
    Z = latent_sanity_check(model, dl_val, device)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dataset      : {cfg.dataset_name}")
    print(f"Architecture : TCN AE")
    print(f"  seq_len    = {cfg.seq_len}")
    print(f"  hidden_dim = {cfg.hidden_dim}")
    print(f"  latent_dim = {cfg.latent_dim}")
    print(f"  n_blocks   = {cfg.n_blocks}")
    print(f"  receptive  = {model.receptive_field()} / {cfg.seq_len}")
    print(f"  params     = {model.count_parameters():,}")
    print(f"\nBest val loss : {min(history['val']):.6f}")
    print(f"Checkpoint    : {os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)}")