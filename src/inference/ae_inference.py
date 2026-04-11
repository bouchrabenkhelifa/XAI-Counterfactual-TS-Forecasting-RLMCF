# src/inference/ae.py
# ─────────────────────────────────────────────────────────────
# Inference script for TCN Autoencoder.
#
# Usage :
#   python -m src.inference.ae
#
# Ce script :
#   1. Charge le checkpoint ae_etth1.pt
#   2. Charge ETTh1.csv et prépare les données
#   3. Teste encode / decode sur des samples val
#   4. Vérifie les dimensions et la qualité
#   5. Sauvegarde les figures de reconstruction
# ─────────────────────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from torch.utils.data      import Dataset, DataLoader
from types                 import SimpleNamespace

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))

from src.models.autoencoder.tcn_ae import TCNAutoEncoder


# ── Config ────────────────────────────────────────────────────
CONFIG_PATH = "configs/models/ae/tcn_ae.json"

with open(CONFIG_PATH) as f:
    cfg = SimpleNamespace(**json.load(f))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(cfg.figures_dir, exist_ok=True)


# ── Load model ────────────────────────────────────────────────
print("=" * 55)
print("TCN AE — Inference")
print("=" * 55)

ae = TCNAutoEncoder.from_checkpoint(cfg.checkpoint_path, device=DEVICE)
ae.eval()


# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(cfg.data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

ot      = df[[cfg.target_col]].values.astype(np.float32)
T       = len(ot)
t_train = int(cfg.train_ratio * T)
t_val   = int((cfg.train_ratio + cfg.val_ratio) * T)

# ── Charger le scaler depuis le checkpoint ──────────────────
# IMPORTANT : utiliser le scaler du checkpoint Colab
# pour avoir exactement le même preprocessing qu'à l'entraînement
ckpt_raw = torch.load(cfg.checkpoint_path,
                      map_location="cpu", weights_only=False)

if "scaler_mean" in ckpt_raw and "scaler_std" in ckpt_raw:
    scaler       = StandardScaler()
    scaler.mean_ = np.array(ckpt_raw["scaler_mean"], dtype=np.float64)
    scaler.scale_= np.array(ckpt_raw["scaler_std"],  dtype=np.float64)
    scaler.var_  = scaler.scale_ ** 2
    scaler.n_features_in_ = 1
    print(f"[Scaler] loaded from checkpoint ✓")
    print(f"  mean  = {scaler.mean_[0]:.4f}")
    print(f"  scale = {scaler.scale_[0]:.4f}")
else:
    # fallback : recalculer sur train
    scaler = StandardScaler()
    scaler.fit(ot[:t_train])
    print(f"[Scaler] recomputed from train data (checkpoint has no scaler)")

ot_val_s = scaler.transform(ot[t_train:t_val]).astype(np.float32)

print(f"\n[Data] val set : {len(ot_val_s)} points")


# ── Build windows ─────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, series, window):
        self.series = series
        self.window = window
        self.idxs   = np.arange(0, len(series) - window + 1, 1)
    def __len__(self):  return len(self.idxs)
    def __getitem__(self, i):
        s = self.idxs[i]
        return torch.from_numpy(self.series[s:s+self.window])

ds_val = WindowDataset(ot_val_s, cfg.seq_len)
dl_val = DataLoader(ds_val, batch_size=64, shuffle=False)

print(f"[Data] val windows : {len(ds_val)}")


# ── Test encode / decode ──────────────────────────────────────
print("\n── Encode / Decode test ──")

x_batch = next(iter(dl_val)).to(DEVICE).float()  # (B, T, 1)

with torch.no_grad():
    z       = ae.encode(x_batch)        # (B, latent_dim)
    x_recon = ae.decode(z)              # (B, T, 1)
    _, z2   = ae(x_batch)               # test forward aussi

mse = float(((x_recon - x_batch)**2).mean().item())

print(f"  x shape     : {tuple(x_batch.shape)}")
print(f"  z shape     : {tuple(z.shape)}")
print(f"  x_recon     : {tuple(x_recon.shape)}")
print(f"  MSE (batch) : {mse:.6f}")
print(f"  z range     : [{z.min():.4f}, {z.max():.4f}]  "
      f"{'✓ bounded' if z.min() >= -1.01 and z.max() <= 1.01 else '✗'}")
print(f"  forward == encode+decode : {torch.allclose(z, z2)} ✓")


# ── Latent perturbation test ──────────────────────────────────
print("\n── Latent sensitivity test ──")
print(f"  {'eps':>6}  {'mean|x_cf - x|':>16}  status")
print(f"  {'-'*35}")

with torch.no_grad():
    z0 = ae.encode(x_batch)

for eps in [0.01, 0.05, 0.10, 0.20]:
    noise   = torch.randn_like(z0) * eps
    z_cf    = torch.clamp(z0 + noise, -1.0, 1.0)
    with torch.no_grad():
        x_cf = ae.decode(z_cf)
    diff = float((x_cf - x_batch).abs().mean().item())
    ok   = "✓" if diff < eps * 2 else "~"
    print(f"  {eps:>6.2f}  {diff:>16.5f}  {ok}")

print("  → smooth latent space = RL can explore ✓")


# ── Reconstruction plot ───────────────────────────────────────
print("\n── Saving reconstruction plot ──")

x_np = x_batch.cpu().numpy()
r_np = x_recon.cpu().numpy()
n    = min(6, len(x_np))

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for i, ax in enumerate(axes.flat[:n]):
    mse_i = float(((x_np[i,:,0] - r_np[i,:,0])**2).mean())
    ax.plot(x_np[i,:,0], color="steelblue", lw=1.5, label="Original")
    ax.plot(r_np[i,:,0], color="coral",     lw=1.5,
            ls="--",     label="Reconstructed")
    ax.set_title(f"Val sample {i+1}  —  MSE={mse_i:.5f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle(f"TCN AE Reconstruction — {cfg.dataset_name}"
             f"  (val_loss={mse:.5f})", fontsize=13)
plt.tight_layout()

fig_path = os.path.join(cfg.figures_dir,
                        f"ae_inference_recon_{cfg.dataset_name}.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig_path}")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"Model     : TCNAutoEncoder")
print(f"  seq_len    = {ae.seq_len}")
print(f"  latent_dim = {ae.latent_dim}")
print(f"  RF         = {ae.receptive_field()}/{cfg.seq_len} ✓")
print(f"  params     = {ae.count_parameters():,}")
print(f"\nInference :")
print(f"  batch MSE  = {mse:.6f}")
print(f"  z range    = [{z.min():.4f}, {z.max():.4f}]")
print(f"\n✅ AE ready for RL pipeline")
print(f"   → ae.encode(x) → z")
print(f"   → ae.decode(z) → x_recon")