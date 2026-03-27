
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

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.models.tcn_ae import TCNAutoEncoder
from src.training.TCN_AE_trainer import AETrainerDerivativeLoss


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int, stride: int = 1):
        self.series = series.astype(np.float32)
        self.window = window
        T           = len(series)
        self.idxs   = np.arange(0, T - window + 1, stride)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        return (torch.from_numpy(self.series[s:s + self.window]),)


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

    scaler     = StandardScaler()
    ot_train_s = scaler.fit_transform(ot[:t_train]).astype(np.float32)
    ot_val_s   = scaler.transform(ot[t_train:t_val]).astype(np.float32)
    ot_test_s  = scaler.transform(ot[t_val:]).astype(np.float32)

    print(f"[Data] ETTh1 | "
          f"train={len(ot_train_s)} "
          f"val={len(ot_val_s)} "
          f"test={len(ot_test_s)}")
    return ot_train_s, ot_val_s, ot_test_s, scaler


def latent_sanity_check(model, loader, device, n_batches=5):
    model.eval()
    zs = []
    with torch.no_grad():
        for i, (x,) in enumerate(loader):
            z = model.encode(x.to(device).float())
            zs.append(z.cpu())
            if i + 1 >= n_batches:
                break
    Z = torch.cat(zs, dim=0)
    print(f"\n[Latent sanity] z min={Z.min():.4f} "
          f"max={Z.max():.4f} "
          f"|mean|={Z.abs().mean():.4f}")
    print(f"  bounded [-1,1] : "
          f"{'✓' if Z.min() >= -1.01 and Z.max() <= 1.01 else '✗'}")
    return Z


def compare_with_mse_baseline(model, val_loader, device, n=4):
    """
    Visual comparison : show how derivative loss reduces smoothing
    vs pure MSE (qualitative, no baseline model needed —
    we just visualize how close Δx matches).
    """
    model.eval()
    x_batch = next(iter(val_loader))[0][:n].to(device).float()

    with torch.no_grad():
        x_recon, _ = model(x_batch)

    x     = x_batch.cpu().numpy()
    x_rec = x_recon.cpu().numpy()

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1: axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x[i, :, 0],     color="steelblue", lw=1.5,
                label="Original")
        ax.plot(x_rec[i, :, 0], color="coral",     lw=1.5,
                ls="--", label="Reconstructed (deriv loss)")

        mse_i = float(((x[i] - x_rec[i]) ** 2).mean())
        d1_err = float(np.abs(
            np.diff(x[i, :, 0]) - np.diff(x_rec[i, :, 0])).mean())

        ax.set_title(
            f"Sample {i+1} | MSE={mse_i:.5f} | "
            f"mean|Δx-Δx̂|={d1_err:.5f}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Derivative Loss AE — ETTh1 val | "
        "Does it preserve local patterns?",
        fontsize=13)
    plt.tight_layout()

    os.makedirs("assets/figures/ae/deriv_loss", exist_ok=True)
    path = "assets/figures/ae/deriv_loss/ae_deriv_loss_qualitative.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Qualitative] → {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    CONFIG = "assets/configs/models/ae/TCN_AE_nv.json"
    cfg    = load_config(CONFIG)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir,    exist_ok=True)
    os.makedirs(cfg.results_dir,    exist_ok=True)

    # ── Data ─────────────────────────────────────────────────
    ot_train_s, ot_val_s, ot_test_s, scaler = load_series(cfg)

    ds_train = WindowDataset(ot_train_s, window=cfg.seq_len, stride=1)
    ds_val   = WindowDataset(ot_val_s,   window=cfg.seq_len, stride=1)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size,
                          shuffle=True,  drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size,
                          shuffle=False, drop_last=False)

    print(f"[Data] windows train={len(ds_train)} val={len(ds_val)}")

    # ── Train ────────────────────────────────────────────────
    trainer = AETrainerDerivativeLoss(cfg)
    model, history, ckpt_path = trainer.train(dl_train, dl_val)

    device = trainer.device

    # ── Figures ──────────────────────────────────────────────
    trainer.plot_loss_curves(history)
    trainer.plot_reconstruction(model, dl_val, n=6)
    compare_with_mse_baseline(model, dl_val, device, n=4)

    # ── Latent sanity ────────────────────────────────────────
    Z = latent_sanity_check(model, dl_val, device)

    # ── Save scaler in checkpoint ─────────────────────────────
    # (needed by RL pipeline)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["scaler_mean"] = scaler.mean_.tolist()
    ckpt["scaler_std"]  = scaler.scale_.tolist()
    torch.save(ckpt, ckpt_path)
    print(f"[Scaler] saved in checkpoint ✓")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("SUMMARY — AE Derivative Loss")
    print("=" * 55)
    print(f"  Loss     : {cfg.lambda_rec}*MSE "
          f"+ {cfg.lambda_d1}*L1(Δx) "
          f"+ {cfg.lambda_d2}*L1(Δ²x)")
    print(f"  latent   : {cfg.latent_dim}d | "
          f"hidden={cfg.hidden_dim} | "
          f"blocks={cfg.n_blocks}")
    print(f"  best val_total : {min(history['val_total']):.5f}")
    print(f"  best val_mse   : {min(history['val_mse']):.5f}")
    print(f"  checkpoint     : {ckpt_path}")
    print(f"  figures        : {cfg.figures_dir}/")
    print("=" * 55)
    print("\n✅ Done ! Prochaine étape :")
    print("  → Re-fit LatentPlausibility avec ce checkpoint")
    print("  → Relancer le pipeline RL")
    print("  → Comparer plausibilité IF/LOF/OC-SVM avec AE original")