"""
Ablation study radar chart — 4 configurations:
  1. RL-MCF (full)
  2. RL-MCF w/o Proximity
  3. RL-MCF w/o Mask
  4. RLP (no policy / random agent)

Trains w/o Proximity (30 epochs) if checkpoint not found,
then generates a radar chart comparing all 4.

Usage:
    python scripts/analysis/ablation_radar_4configs.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_main import RLMaskTrainer

FIGURES_DIR = "assets/figures/etth1/RL/ablation"
RESULTS_DIR = "assets/results/etth1/RL/ablation"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FORECAST_CONFIG = "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
AE_CONFIG = "assets/configs/etth1_dataset/ae/tcn_ae.json"

# ─────────────────────────────────────────────────────────────────────────────
# Existing results
# ─────────────────────────────────────────────────────────────────────────────

# Full model (from etth1_all_models_eval.json — iTransformer)
FULL_MODEL = {
    "validity_ratio": 0.976,
    "stepwise_auc": 0.982,
    "proximity_l2": 0.605,
    "compactness": 0.803,
    "temporal_consistency": 0.969,
    "plausibility_ensemble": 0.170,
}

# w/o Mask (from existing evaluation)
WO_MASK = {
    "validity_ratio": 0.851,
    "stepwise_auc": 0.855,
    "proximity_l2": 2.059,
    "compactness": 0.003,
    "temporal_consistency": 0.643,
    "plausibility_ensemble": 0.451,
}


def train_no_proximity():
    """Train RL agent without proximity reward (30 epochs)."""
    cfg_f = load_config(FORECAST_CONFIG)
    cfg_ae = load_config(AE_CONFIG)
    device = get_device(cfg_f)

    cfg_rl = SimpleNamespace(
        name="ablation_wo_proximity",
        rho=0.2, fr=0.5,
        w_validity=3.5, w_proximity=0.0, w_hard_bonus=2.5, w_smooth=0.5,
        use_validity=True, use_proximity=False, use_smooth=True,
        direction=-1.0, eta=0.15, entropy_coef=0.02,
        epochs=30, lr_actor=0.0003, lr_critic=0.0001,
        lr_decay=0.5, grad_clip=1.0,
        mask_last_k=12, mask_ramp_k=4,
        filter_quantile=0.0, use_rl=True,
        checkpoint_dir_lp=os.path.join(RESULTS_DIR, "checkpoints/wo_proximity"),
        figures_dir_lp=os.path.join(FIGURES_DIR, "wo_proximity"),
        results_dir_lp=os.path.join(RESULTS_DIR, "wo_proximity"),
        eval_train_batches=50,
    )

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    trainer.train()
    summary, _ = trainer.evaluate(n_batches=20)
    return summary


def train_no_policy():
    """Evaluate with random (untrained) agent — no RL policy."""
    cfg_f = load_config(FORECAST_CONFIG)
    cfg_ae = load_config(AE_CONFIG)
    device = get_device(cfg_f)

    cfg_rl = SimpleNamespace(
        name="ablation_no_policy",
        rho=0.2, fr=0.5,
        w_validity=3.5, w_proximity=1.5, w_hard_bonus=2.5, w_smooth=0.5,
        use_validity=True, use_proximity=True, use_smooth=True,
        direction=-1.0, eta=0.15, entropy_coef=0.02,
        epochs=1, lr_actor=0.0003, lr_critic=0.0001,
        lr_decay=0.5, grad_clip=1.0,
        mask_last_k=12, mask_ramp_k=4,
        filter_quantile=0.0, use_rl=True,
        checkpoint_dir_lp=os.path.join(RESULTS_DIR, "checkpoints/no_policy"),
        figures_dir_lp=os.path.join(FIGURES_DIR, "no_policy"),
        results_dir_lp=os.path.join(RESULTS_DIR, "no_policy"),
        eval_train_batches=50,
    )

    # Create directories
    os.makedirs(cfg_rl.checkpoint_dir_lp, exist_ok=True)
    os.makedirs(cfg_rl.figures_dir_lp, exist_ok=True)
    os.makedirs(cfg_rl.results_dir_lp, exist_ok=True)

    # Train only 1 epoch (essentially random)
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    # Don't train — evaluate with random weights
    summary, _ = trainer.evaluate(n_batches=20)
    return summary


def extract_metrics(summary):
    """Extract key metrics from evaluation summary."""
    return {
        "validity_ratio": summary["validity_ratio"]["mean"],
        "stepwise_auc": summary["stepwise_auc"]["mean"],
        "proximity_l2": summary["proximity_l2"]["mean"],
        "compactness": summary["compactness"]["mean"],
        "temporal_consistency": summary["temporal_consistency"]["mean"],
        "plausibility_ensemble": summary["plausibility_ensemble"]["mean"],
    }


def plot_radar(all_results):
    """Generate radar chart for 4 ablation configs."""
    metrics = ["validity_ratio", "stepwise_auc", "compactness", "temporal_consistency", "plausibility_ensemble", "proximity_l2"]
    labels = ["Validity", "Step AUC", "Compactness", "T. Consistency", "Plausibility\n(1-score)", "Proximity\n(1-norm)"]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {
        "RL-MCF (full)": "tab:blue",
        "w/o Proximity": "tab:orange",
        "w/o Mask": "tab:red",
        "No Policy (random)": "tab:gray",
    }

    for config_name, metrics_dict in all_results.items():
        values = []
        for m in metrics:
            v = metrics_dict.get(m, 0)
            # Invert proximity and plausibility so higher = better on radar
            if m == "proximity_l2":
                v = max(0, 1 - v / 3.0)  # 0→1, 3→0
            elif m == "plausibility_ensemble":
                v = max(0, 1 - v)  # 0→1, 1→0
            values.append(v)
        values += values[:1]

        ax.plot(angles, values, "o-", lw=2, label=config_name,
                color=colors.get(config_name, "tab:purple"))
        ax.fill(angles, values, alpha=0.1, color=colors.get(config_name, "tab:purple"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation Study — Component Contribution", fontsize=13, pad=20)
    ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.3, 0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_radar_4configs.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path}")


def main():
    print("=" * 60)
    print("ABLATION STUDY — 4 Configurations")
    print("=" * 60)

    all_results = {}

    # 1. Full model (existing)
    all_results["RL-MCF (full)"] = FULL_MODEL
    print(f"\n✓ RL-MCF (full): validity={FULL_MODEL['validity_ratio']:.3f}")

    # 2. w/o Proximity — train if needed
    wo_prox_path = os.path.join(RESULTS_DIR, "wo_proximity/ablation_wo_proximity_evaluation.json")
    if os.path.exists(wo_prox_path):
        with open(wo_prox_path) as f:
            summary = json.load(f)
        all_results["w/o Proximity"] = extract_metrics(summary)
        print(f"✓ w/o Proximity (loaded): validity={all_results['w/o Proximity']['validity_ratio']:.3f}")
    else:
        print("\n── Training w/o Proximity (30 epochs) ──")
        summary = train_no_proximity()
        all_results["w/o Proximity"] = extract_metrics(summary)
        print(f"✓ w/o Proximity: validity={all_results['w/o Proximity']['validity_ratio']:.3f}")

    # 3. w/o Mask (existing)
    all_results["w/o Mask"] = WO_MASK
    print(f"✓ w/o Mask: validity={WO_MASK['validity_ratio']:.3f}")

    # 4. No policy (random) — train if needed
    no_policy_path = os.path.join(RESULTS_DIR, "no_policy/ablation_no_policy_evaluation.json")
    if os.path.exists(no_policy_path):
        with open(no_policy_path) as f:
            summary = json.load(f)
        all_results["No Policy (random)"] = extract_metrics(summary)
        print(f"✓ No Policy (loaded): validity={all_results['No Policy (random)']['validity_ratio']:.3f}")
    else:
        print("\n── Evaluating No Policy (random agent) ──")
        summary = train_no_policy()
        all_results["No Policy (random)"] = extract_metrics(summary)
        print(f"✓ No Policy: validity={all_results['No Policy (random)']['validity_ratio']:.3f}")

    # Save all results
    results_path = os.path.join(RESULTS_DIR, "ablation_4configs.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Results] → {results_path}")

    # Print table
    print("\n── Ablation Results ─────────────────────────────────────────────────")
    print(f"{'Config':<22} {'Validity':>10} {'AUC':>10} {'Prox L2':>10} {'Compact':>10} {'T.Cons':>10} {'Plaus':>10}")
    print("-" * 82)
    for name, m in all_results.items():
        print(f"{name:<22} {m['validity_ratio']:>10.3f} {m['stepwise_auc']:>10.3f} "
              f"{m['proximity_l2']:>10.3f} {m['compactness']:>10.3f} "
              f"{m['temporal_consistency']:>10.3f} {m['plausibility_ensemble']:>10.3f}")

    # Plot radar
    plot_radar(all_results)


if __name__ == "__main__":
    main()
