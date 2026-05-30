"""
Plot plausibility detector training results for PFE report.
Shows how the 3 detectors (IForest, LOF, OC-SVM) are trained on ETTh1
and how they score train vs test data (no CFs involved).

Usage:
    python scripts/analysis/plot_plausibility_detectors.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.anomaly_detector.plausibility import (
    EnsemblePlausibility, extract_features, series_to_windows
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET = "etth1"
PLAUS_CKPT = f"assets/checkpoints/{DATASET}_chpts/anomaly_detector/plausibility_{DATASET}.pkl"
DATA_PATH = "assets/datasets/ETTh1.csv"
FIGURES_DIR = f"assets/figures/{DATASET}/anomaly_detector"
WINDOW = 96

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load pre-trained ensemble
    with open(PLAUS_CKPT, "rb") as f:
        ckpt = pickle.load(f)
    ensemble = ckpt["ensemble"]
    print(f"[OK] Loaded ensemble from {PLAUS_CKPT}")

    # Load data
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA_PATH)
    ot = df[["OT"]].values.astype(np.float32)
    T = len(ot)
    t_train = int(0.7 * T)
    t_val = int(0.8 * T)

    scaler = StandardScaler()
    scaler.fit(ot[:t_train])
    ot_scaled = scaler.transform(ot).astype(np.float32)

    # Build windows from train set
    ot_train = ot_scaled[:t_train]
    idxs_train = np.arange(0, len(ot_train) - WINDOW, 10)
    W_train = np.stack([ot_train[i:i+WINDOW].reshape(WINDOW, 1) for i in idxs_train])
    x_train = torch.from_numpy(W_train).float()

    # Build windows from test set
    ot_test = ot_scaled[t_val:]
    idxs_test = np.arange(0, len(ot_test) - WINDOW, 5)
    W_test = np.stack([ot_test[i:i+WINDOW].reshape(WINDOW, 1) for i in idxs_test])
    x_test = torch.from_numpy(W_test).float()

    # Generate random noise (anomalous baseline)
    x_noise = torch.randn(200, WINDOW, 1) * 1.5

    print(f"Train windows: {len(x_train)}")
    print(f"Test windows: {len(x_test)}")
    print(f"Noise windows: {len(x_noise)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Score all
    # ─────────────────────────────────────────────────────────────────────────

    scores_train = ensemble.score_all(x_train[:300])
    scores_test = ensemble.score_all(x_test[:300])
    scores_noise = ensemble.score_all(x_noise)

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1: Score distributions for each detector
    # ─────────────────────────────────────────────────────────────────────────

    detectors = ["if", "lof", "ocsvm"]
    detector_names = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for i, (det, name) in enumerate(zip(detectors, detector_names)):
        ax = axes[i]
        s_train = scores_train[det].cpu().numpy()
        s_test = scores_test[det].cpu().numpy()
        s_noise = scores_noise[det].cpu().numpy()

        ax.hist(s_train, bins=30, alpha=0.6, color="tab:blue", label="Train", density=True)
        ax.hist(s_test, bins=30, alpha=0.6, color="tab:green", label="Test", density=True)
        ax.hist(s_noise, bins=30, alpha=0.6, color="tab:red", label="Random noise", density=True)

        ax.axvline(s_train.mean(), color="tab:blue", ls="--", lw=1.5)
        ax.axvline(s_test.mean(), color="tab:green", ls="--", lw=1.5)
        ax.axvline(s_noise.mean(), color="tab:red", ls="--", lw=1.5)

        ax.set_xlabel("Plausibility Score (0=plausible, 1=anomalous)")
        ax.set_ylabel("Density")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle("Plausibility Detectors — Score Distributions (ETTh1)", fontsize=13)
    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "plausibility_distributions.png")
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2: Ensemble boxplot (train vs test vs noise)
    # ─────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(8, 5))

    data_box = [
        scores_train["ensemble"].cpu().numpy(),
        scores_test["ensemble"].cpu().numpy(),
        scores_noise["ensemble"].cpu().numpy(),
    ]
    labels_box = ["Train set", "Test set", "Random noise"]
    colors = ["tab:blue", "tab:green", "tab:red"]

    bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Ensemble Plausibility Score (0=plausible, 1=anomalous)")
    ax.set_title("Ensemble Plausibility — Train vs Test vs Noise (ETTh1)")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Threshold 0.5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    for i, d in enumerate(data_box):
        ax.annotate(f"μ={d.mean():.3f}", xy=(i+1, d.mean()),
                    xytext=(i+1.3, d.mean()+0.05), fontsize=9, color=colors[i])

    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "plausibility_boxplot.png")
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Print summary
    # ─────────────────────────────────────────────────────────────────────────

    print("\n── Summary ──────────────────────────────────────────")
    print(f"{'Type':<20} {'IForest':>10} {'LOF':>10} {'OC-SVM':>10} {'Ensemble':>10}")
    print("-" * 60)
    for label, scores in [("Train", scores_train), ("Test", scores_test), ("Noise", scores_noise)]:
        print(f"{label:<20} {scores['if'].mean():.3f}      {scores['lof'].mean():.3f}      {scores['ocsvm'].mean():.3f}      {scores['ensemble'].mean():.3f}")


if __name__ == "__main__":
    main()
