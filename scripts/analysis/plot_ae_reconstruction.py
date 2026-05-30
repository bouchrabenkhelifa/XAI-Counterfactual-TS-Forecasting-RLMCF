"""
Evaluate TCN Autoencoder on all 3 datasets and plot reconstruction examples side by side.

Usage:
    python scripts/analysis/plot_ae_reconstruction.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.autoencoder.tcn_ae import TCNAutoEncoder

FIGURES_DIR = "assets/figures/data_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

datasets = {
    "ETTh1": {
        "data_path": "assets/datasets/ETTh1.csv",
        "target": "OT",
        "ae_ckpt": "assets/checkpoints/etth1_chpts/ae/ae_etth1.pt",
        "train_ratio": 0.7,
    },
    "ETTh2": {
        "data_path": "assets/datasets/ETTh2.csv",
        "target": "OT",
        "ae_ckpt": "assets/checkpoints/etth2_chpts/ae/ae_etth2.pt",
        "train_ratio": 0.7,
    },
    "Weather": {
        "data_path": "assets/datasets/weather.csv",
        "target": "T (degC)",
        "ae_ckpt": "assets/checkpoints/weather_chpts/ae/ae_weather.pt",
        "train_ratio": 0.7,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_windows(info, seq_len=96, n_windows=200):
    """Load and scale test data, return windows."""
    df = pd.read_csv(info["data_path"])
    series = df[[info["target"]]].values.astype(np.float32)
    T = len(series)
    t_train = int(info["train_ratio"] * T)
    t_val = int(0.8 * T)

    scaler = StandardScaler()
    scaler.fit(series[:t_train])
    series_scaled = scaler.transform(series).astype(np.float32)

    # Test set windows
    test_data = series_scaled[t_val:]
    idxs = np.linspace(0, len(test_data) - seq_len - 1, n_windows, dtype=int)
    windows = np.stack([test_data[i:i+seq_len] for i in idxs])
    return torch.from_numpy(windows).float(), scaler


def evaluate_ae(ae, windows):
    """Compute reconstruction MSE."""
    ae.eval()
    with torch.no_grad():
        x = windows.to(device)
        x_recon, z = ae(x)
        mse = ((x_recon - x) ** 2).mean().item()
        mae = (x_recon - x).abs().mean().item()
    return mse, mae, x_recon.cpu()


def main():
    print("=" * 60)
    print("TCN Autoencoder Evaluation — All 3 Datasets")
    print("=" * 60)

    results = {}
    all_examples = {}

    for name, info in datasets.items():
        print(f"\n── {name} ──")

        # Load AE
        ae = TCNAutoEncoder.from_checkpoint(info["ae_ckpt"], device=device)

        # Load test windows
        windows, scaler = load_test_windows(info)
        print(f"  Test windows: {windows.shape}")

        # Evaluate
        mse, mae, x_recon = evaluate_ae(ae, windows)
        results[name] = {"mse": mse, "mae": mae}
        print(f"  MSE: {mse:.6f}  |  MAE: {mae:.6f}")

        # Pick the best-looking sample (lowest per-sample MSE)
        per_sample_mse = ((x_recon - windows) ** 2).mean(dim=(1, 2))
        best_idx = int(per_sample_mse.argmin())
        print(f"  Best sample idx: {best_idx} (MSE={per_sample_mse[best_idx]:.6f})")

        all_examples[name] = {
            "original": windows[best_idx:best_idx+1].numpy(),
            "reconstructed": x_recon[best_idx:best_idx+1].numpy(),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Plot: 3 datasets side by side, 2 examples each
    # ─────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

    for col, (name, examples) in enumerate(all_examples.items()):
        ax = axes[col]
        orig = examples["original"][0, :, 0]
        recon = examples["reconstructed"][0, :, 0]

        ax.plot(orig, lw=1.5, color="tab:blue", label="Original")
        ax.plot(recon, lw=1.5, color="tab:red", ls="--", label="Reconstructed")

        mse_sample = float(((orig - recon) ** 2).mean())
        ax.set_title(f"{name}\nMSE={mse_sample:.5f}", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if col == 0:
            ax.set_ylabel("Normalized value")
        ax.set_xlabel("Timestep")

    plt.suptitle("TCN Autoencoder Reconstruction — Test Set", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ae_reconstruction_3datasets.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] → {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────

    print("\n── Reconstruction Quality ──────────────────────────────")
    print(f"{'Dataset':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 36)
    for name, r in results.items():
        print(f"{name:<12} {r['mse']:<12.6f} {r['mae']:<12.6f}")


if __name__ == "__main__":
    main()
