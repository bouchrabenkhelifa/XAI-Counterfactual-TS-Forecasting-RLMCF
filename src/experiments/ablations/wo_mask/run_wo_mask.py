"""
Ablation : Without Temporal Mask
=================================
Entraîne et évalue le framework RL sans masque temporel.
Produit les 6 métriques + figure séries originale/CF + forecasts.

Usage:
    python src/experiments/ablations/wo_mask/run_wo_mask.py
    python src/experiments/ablations/wo_mask/run_wo_mask.py --eval_only
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_wo_mask import RLNoMaskTrainer
from src.evaluation.unified_evaluator import CounterfactualEvaluator

# ── 6 métriques principales ───────────────────────────────────────────────────
MAIN_METRICS = [
    ("validity_ratio",        "Validity Ratio ↑",        True),
    ("stepwise_auc",          "Stepwise AUC ↑",          True),
    ("proximity_l2",          "Proximity L2 ↓",          False),
    ("compactness",           "Compactness ↑",           True),
    ("temporal_consistency",  "Temporal Consistency ↑",  True),
    ("plausibility_ensemble", "Plausibility Ensemble ↓", False),
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── figures ───────────────────────────────────────────────────────────────────

def plot_metrics(summary: dict, exp_name: str):
    """Barplot des 6 métriques principales."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    keys   = [k for k, _, _ in MAIN_METRICS]
    labels = [l for _, l, _ in MAIN_METRICS]
    means  = [summary[k]["mean"] if k in summary else 0.0 for k in keys]
    stds   = [summary[k]["std"]  if k in summary else 0.0 for k in keys]
    colors = ["#4C9BE8" if higher else "#E8754C" for _, _, higher in MAIN_METRICS]

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.55)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation wo_mask — 6 Main Metrics\n{exp_name}")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(handles=[
        Patch(facecolor="#4C9BE8", alpha=0.85, label="Higher is better ↑"),
        Patch(facecolor="#E8754C", alpha=0.85, label="Lower is better ↓"),
    ], fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{exp_name}_metrics_barplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_cf_examples(examples: list, exp_name: str, reward_fn):
    """Série originale vs CF + leurs forecasts + bande α/β."""
    if not examples:
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i, ex in enumerate(examples):
        x_ot  = ex["x_ot"][:, 0]
        x_cf  = ex["x_cf"][:, 0]
        y_hat = ex["y_hat"][:, 0]
        y_cf  = ex["y_cf"][:, 0]

        BH = len(x_ot)
        t_back = np.arange(BH)
        t_fore = np.arange(BH, BH + len(y_hat))

        x_t  = torch.tensor(ex["x_ot"],  dtype=torch.float32).unsqueeze(0)
        yh_t = torch.tensor(ex["y_hat"], dtype=torch.float32).unsqueeze(0)
        alpha_t, beta_t, _ = reward_fn.compute_bounds(yh_t, x_ot=x_t)
        alpha_np = alpha_t[0].numpy()
        beta_np  = beta_t[0].numpy()

        valid_ratio = float(((y_cf >= alpha_np) & (y_cf <= beta_np)).mean())

        ax = axes[i]
        ax.plot(t_back, x_ot,  color="#2196F3", lw=1.8, label="x original")
        ax.plot(t_back, x_cf,  color="#FF5722", lw=1.8, ls="--", label="x CF (wo_mask)")
        ax.plot(t_fore, y_hat, color="#2196F3", lw=1.5, ls="-.",  label="forecast original")
        ax.plot(t_fore, y_cf,  color="#FF5722", lw=1.5, ls=":",   label="forecast CF")
        ax.fill_between(t_fore, alpha_np, beta_np,
                        alpha=0.20, color="#4CAF50", label="α/β target band")
        ax.axvline(BH, color="gray", lw=1.2, ls="--", alpha=0.7)
        ax.set_title(f"Sample {i+1} — validity={valid_ratio:.2f}", fontsize=11)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Ablation wo_mask — CF Examples\n{exp_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{exp_name}_cf_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation — Without Temporal Mask")
    parser.add_argument(
        "--config",
        default="assets/configs/models/etth1_dataset/RL_ablations/wo_mask.json",
    )
    parser.add_argument(
        "--forecast_config",
        default="assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    )
    parser.add_argument(
        "--ae_config",
        default="assets/configs/models/etth1_dataset/ae/tcn_ae.json",
    )
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    cfg_rl = load_config(args.config)
    cfg_f  = load_config(args.forecast_config)
    cfg_ae = load_config(args.ae_config)
    device = get_device(cfg_f)

    print("=" * 60)
    print(f"Ablation  : Without Temporal Mask")
    print(f"use_mask  : {getattr(cfg_rl, 'use_mask', True)}")
    print(f"Device    : {device}")
    print("=" * 60)

    trainer = RLNoMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    if not args.eval_only:
        trainer.train()
    else:
        ckpt_path = os.path.join(
            cfg_rl.checkpoint_dir_lp,
            f"{getattr(cfg_rl, 'name', 'exp')}_agent_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
            trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
            print(f"[Eval] Loaded checkpoint → {ckpt_path}")
        else:
            print(f"[Eval] WARNING: no checkpoint at {ckpt_path}, using random weights")

    # Évaluation + récupération des exemples
    summary, cf_examples = trainer.evaluate(n_batches=args.eval_batches)

    exp_name = getattr(cfg_rl, "name", "wo_mask")

    # Sauvegarder JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{exp_name}_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {json_path}")

    # Figures
    plot_metrics(summary, exp_name)
    plot_cf_examples(cf_examples, exp_name, trainer.reward_fn)

    print("\nDone.")


if __name__ == "__main__":
    main()
