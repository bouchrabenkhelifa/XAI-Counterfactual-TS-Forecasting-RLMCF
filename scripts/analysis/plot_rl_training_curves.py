"""
Plot clean RL training curves (reward, validity, proximity) for the PFE report.
Uses iTransformer ETTh1 checkpoint (history stored inside).

Usage:
    python scripts/analysis/plot_rl_training_curves.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = "assets/figures/data_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

CKPT_PATH = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt"


def main():
    # Load history from checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    if "history" in ckpt:
        h = ckpt["history"]
    else:
        print("No history in checkpoint. Available keys:", list(ckpt.keys()))
        return

    epochs = np.arange(1, len(h["reward_total"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Total Reward
    ax = axes[0]
    ax.plot(epochs, h["reward_total"], lw=2, color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Reward")
    ax.set_title("Training Reward")
    ax.set_xticks(range(0, len(epochs)+1, 5))
    ax.grid(alpha=0.3)

    # Panel 2: Validity Metrics (soft + hard + success rate)
    ax = axes[1]
    ax.plot(epochs, h["r_validity"], lw=2, color="tab:green", label="Validity (soft)")
    ax.plot(epochs, h["r_validity_hard"], lw=2, color="tab:orange", label="Validity (hard)")
    ax.plot(epochs, h["success_rate"], lw=2, color="tab:red", ls="--", label="Success Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validity Metrics")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, len(epochs)+1, 5))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("RL Agent Training — iTransformer, ETTh1", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rl_training_itransformer_etth1.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path}")


if __name__ == "__main__":
    main()
