"""
Grid Search + Ablation Study for PFE Report (Chapter 6)
========================================================
Runs on iTransformer + ETTh1 with reduced epochs (10) for fast screening.

Three experiments:
  1. Mini-grid search: ρ × fr × mask_last_k
  2. Ablation: sans proximité (w_proximity=0)
  3. Variation des poids de la récompense

Usage:
    python scripts/analysis/grid_search_ablation.py --experiment all
    python scripts/analysis/grid_search_ablation.py --experiment grid
    python scripts/analysis/grid_search_ablation.py --experiment ablation
    python scripts/analysis/grid_search_ablation.py --experiment weights
"""

import argparse
import json
import os
import sys
import time
import itertools
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_main import RLMaskTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

FORECAST_CONFIG = "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
AE_CONFIG = "assets/configs/etth1_dataset/ae/tcn_ae.json"
RESULTS_DIR = "assets/results/etth1/RL/grid_search"
FIGURES_DIR = "assets/figures/etth1/RL/grid_search"

# Base RL config (iTransformer best)
BASE_RL_CONFIG = {
    "name": "grid_base",
    "rho": 0.2,
    "fr": 0.5,
    "w_validity": 3.5,
    "w_proximity": 1.5,
    "w_hard_bonus": 2.5,
    "w_smooth": 0.5,
    "use_validity": True,
    "use_proximity": True,
    "use_smooth": True,
    "direction": -1.0,
    "eta": 0.15,
    "entropy_coef": 0.02,
    "epochs": 30,  # Same as real training for reliable results
    "lr_actor": 0.0003,
    "lr_critic": 0.0001,
    "lr_decay": 0.5,
    "grad_clip": 1.0,
    "mask_last_k": 12,
    "mask_ramp_k": 4,
    "filter_quantile": 0.0,
    "use_rl": True,
    "checkpoint_dir_lp": "",
    "figures_dir_lp": "",
    "results_dir_lp": "",
    "eval_train_batches": 30,
}

SCREENING_EPOCHS = 30
EVAL_BATCHES = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_cfg(overrides: dict, exp_name: str) -> SimpleNamespace:
    """Create a config namespace from base + overrides."""
    cfg = deepcopy(BASE_RL_CONFIG)
    cfg.update(overrides)
    cfg["name"] = exp_name
    cfg["epochs"] = SCREENING_EPOCHS
    cfg["checkpoint_dir_lp"] = os.path.join(RESULTS_DIR, "checkpoints", exp_name)
    cfg["figures_dir_lp"] = os.path.join(FIGURES_DIR, exp_name)
    cfg["results_dir_lp"] = os.path.join(RESULTS_DIR, exp_name)
    return SimpleNamespace(**cfg)


def run_experiment(cfg_rl, cfg_f, cfg_ae, device):
    """Run one training + eval and return summary metrics."""
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    trainer.train()
    summary, _ = trainer.evaluate(n_batches=EVAL_BATCHES)
    return summary


def extract_key_metrics(summary):
    """Extract the most important metrics for comparison."""
    return {
        "validity_ratio": summary.get("validity_ratio", {}).get("mean", 0),
        "stepwise_auc": summary.get("stepwise_auc", {}).get("mean", 0),
        "proximity_l2": summary.get("proximity_l2", {}).get("mean", 0),
        "compactness": summary.get("compactness", {}).get("mean", 0),
        "temporal_consistency": summary.get("temporal_consistency", {}).get("mean", 0),
        "plausibility_ensemble": summary.get("plausibility_ensemble", {}).get("mean", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Mini Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(cfg_f, cfg_ae, device):
    """
    Grid search over: ρ × fr × mask_last_k
    Total: 4 × 3 × 4 = 48 combinations (with 10 epochs each)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: MINI GRID SEARCH")
    print("=" * 70)

    rho_values = [0.1, 0.15, 0.2, 0.25]
    fr_values = [0.4, 0.5, 0.6]
    mask_k_values = [8, 12, 16, 24]

    # Corresponding ramp_k (roughly mask_k / 3)
    ramp_k_map = {8: 3, 12: 4, 16: 6, 24: 8}

    total = len(rho_values) * len(fr_values) * len(mask_k_values)
    print(f"Total combinations: {total}")
    print(f"ρ ∈ {rho_values}")
    print(f"fr ∈ {fr_values}")
    print(f"mask_last_k ∈ {mask_k_values}")
    print(f"Epochs per run: {SCREENING_EPOCHS}")
    print()

    results = []
    run_idx = 0

    for rho, fr, mask_k in itertools.product(rho_values, fr_values, mask_k_values):
        run_idx += 1
        exp_name = f"grid_rho{rho}_fr{fr}_k{mask_k}"
        print(f"\n[{run_idx}/{total}] {exp_name}")

        cfg_rl = make_cfg(
            {
                "rho": rho,
                "fr": fr,
                "mask_last_k": mask_k,
                "mask_ramp_k": ramp_k_map[mask_k],
            },
            exp_name,
        )

        try:
            summary = run_experiment(cfg_rl, cfg_f, cfg_ae, device)
            metrics = extract_key_metrics(summary)
            metrics.update({"rho": rho, "fr": fr, "mask_last_k": mask_k, "name": exp_name})
            results.append(metrics)
            print(f"  → validity={metrics['validity_ratio']:.3f}  prox={metrics['proximity_l2']:.3f}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({"rho": rho, "fr": fr, "mask_last_k": mask_k, "name": exp_name, "error": str(e)})

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "grid_search_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Grid search results → {out_path}")

    # Find best
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r["validity_ratio"] - 0.3 * r["proximity_l2"])
        print(f"\n🏆 Best config: ρ={best['rho']}, fr={best['fr']}, mask_k={best['mask_last_k']}")
        print(f"   validity={best['validity_ratio']:.3f}, proximity={best['proximity_l2']:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Ablation Study
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(cfg_f, cfg_ae, device):
    """
    Ablation studies:
    - Full model (baseline)
    - Sans proximité (w_proximity=0)
    - Sans hard bonus (w_hard_bonus=0)
    - Sans smooth (w_smooth=0)
    - Sans masque temporel (use_mask=False)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ABLATION STUDY")
    print("=" * 70)

    ablations = {
        "full_model": {},
        "no_proximity": {"w_proximity": 0.0, "use_proximity": False},
    }

    results = {}
    for name, overrides in ablations.items():
        print(f"\n── Ablation: {name} ──")
        cfg_rl = make_cfg(overrides, f"ablation_{name}")

        try:
            summary = run_experiment(cfg_rl, cfg_f, cfg_ae, device)
            metrics = extract_key_metrics(summary)
            results[name] = metrics
            print(f"  → validity={metrics['validity_ratio']:.3f}  prox={metrics['proximity_l2']:.3f}  compact={metrics['compactness']:.3f}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[name] = {"error": str(e)}

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Ablation results → {out_path}")

    # Print comparison table
    print("\n── Ablation Comparison ─────────────────────────────────────")
    print(f"{'Config':<18} {'Validity':>10} {'Prox L2':>10} {'Compact':>10} {'Temp.Cons':>10} {'Plaus':>10}")
    print("-" * 70)
    for name, m in results.items():
        if "error" not in m:
            print(
                f"{name:<18} {m['validity_ratio']:>10.3f} {m['proximity_l2']:>10.3f} "
                f"{m['compactness']:>10.3f} {m['temporal_consistency']:>10.3f} "
                f"{m['plausibility_ensemble']:>10.3f}"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Reward Weight Variation
# ─────────────────────────────────────────────────────────────────────────────

def run_weight_variation(cfg_f, cfg_ae, device):
    """
    Vary reward weights to show their impact:
    - w_validity ∈ {2.0, 3.0, 3.5, 4.0, 5.0}
    - w_proximity ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
    - w_hard_bonus ∈ {1.0, 2.0, 2.5, 3.0, 4.0}
    Each varied independently (others fixed at baseline).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: REWARD WEIGHT VARIATION")
    print("=" * 70)

    experiments = {}

    # 3a: Vary w_validity
    print("\n── 3a: Varying w_validity ──")
    w_val_values = [2.0, 3.5, 5.0]
    results_wval = {}
    for wv in w_val_values:
        name = f"wval_{wv}"
        print(f"  w_validity={wv}")
        cfg_rl = make_cfg({"w_validity": wv}, f"weights_{name}")
        try:
            summary = run_experiment(cfg_rl, cfg_f, cfg_ae, device)
            results_wval[wv] = extract_key_metrics(summary)
        except Exception as e:
            results_wval[wv] = {"error": str(e)}
    experiments["w_validity"] = results_wval

    # 3b: Vary w_proximity
    print("\n── 3b: Varying w_proximity ──")
    w_prox_values = [0.0, 1.5, 3.0]
    results_wprox = {}
    for wp in w_prox_values:
        name = f"wprox_{wp}"
        print(f"  w_proximity={wp}")
        cfg_rl = make_cfg({"w_proximity": wp}, f"weights_{name}")
        try:
            summary = run_experiment(cfg_rl, cfg_f, cfg_ae, device)
            results_wprox[wp] = extract_key_metrics(summary)
        except Exception as e:
            results_wprox[wp] = {"error": str(e)}
    experiments["w_proximity"] = results_wprox

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "weight_variation_results.json")
    # Convert float keys to strings for JSON
    serializable = {}
    for param, values in experiments.items():
        serializable[param] = {str(k): v for k, v in values.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✅ Weight variation results → {out_path}")

    return experiments


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_grid_search(results):
    """Plot grid search results as heatmaps."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("No valid results to plot.")
        return

    # Heatmap: ρ vs fr (averaged over mask_k)
    rho_vals = sorted(set(r["rho"] for r in valid_results))
    fr_vals = sorted(set(r["fr"] for r in valid_results))

    validity_matrix = np.zeros((len(rho_vals), len(fr_vals)))
    prox_matrix = np.zeros((len(rho_vals), len(fr_vals)))

    for i, rho in enumerate(rho_vals):
        for j, fr in enumerate(fr_vals):
            subset = [r for r in valid_results if r["rho"] == rho and r["fr"] == fr]
            if subset:
                validity_matrix[i, j] = np.mean([r["validity_ratio"] for r in subset])
                prox_matrix[i, j] = np.mean([r["proximity_l2"] for r in subset])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(validity_matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(fr_vals)))
    axes[0].set_xticklabels([f"{v}" for v in fr_vals])
    axes[0].set_yticks(range(len(rho_vals)))
    axes[0].set_yticklabels([f"{v}" for v in rho_vals])
    axes[0].set_xlabel("fr")
    axes[0].set_ylabel("ρ")
    axes[0].set_title("Validity Ratio ↑")
    for i in range(len(rho_vals)):
        for j in range(len(fr_vals)):
            axes[0].text(j, i, f"{validity_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(prox_matrix, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(fr_vals)))
    axes[1].set_xticklabels([f"{v}" for v in fr_vals])
    axes[1].set_yticks(range(len(rho_vals)))
    axes[1].set_yticklabels([f"{v}" for v in rho_vals])
    axes[1].set_xlabel("fr")
    axes[1].set_ylabel("ρ")
    axes[1].set_title("Proximity L2 ↓")
    for i in range(len(rho_vals)):
        for j in range(len(fr_vals)):
            axes[1].text(j, i, f"{prox_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle("Grid Search: ρ × fr (averaged over mask_last_k)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "grid_search_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Grid search heatmap → {path}")


def plot_ablation(results):
    """Plot ablation results as radar chart: full_model vs no_proximity."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid or len(valid) < 2:
        print("Not enough valid results for radar plot.")
        return

    metrics = ["validity_ratio", "compactness", "temporal_consistency", "plausibility_ensemble", "proximity_l2"]
    # For display: invert proximity and plausibility so higher = better on radar
    metric_labels = ["Validity", "Compactness", "Temp. Consistency", "Plausibility\n(1 - score)", "Proximity\n(1 - norm)"]

    configs = list(valid.keys())
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = {"full_model": "tab:blue", "no_proximity": "tab:red"}
    labels = {"full_model": "Full Model", "no_proximity": "Without Proximity"}

    for config in configs:
        m = valid[config]
        values = []
        for metric in metrics:
            v = m.get(metric, 0)
            # Invert proximity and plausibility so higher = better
            if metric == "proximity_l2":
                v = max(0, 1 - v / 2.0)  # normalize: 0→1, 2→0
            elif metric == "plausibility_ensemble":
                v = max(0, 1 - v)  # 0→1, 1→0
            values.append(v)
        values += values[:1]  # close

        ax.plot(angles, values, "o-", lw=2, label=labels.get(config, config),
                color=colors.get(config, "tab:gray"))
        ax.fill(angles, values, alpha=0.15, color=colors.get(config, "tab:gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: Full Model vs Without Proximity", fontsize=12, pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_radar.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation radar → {path}")


def plot_weight_variation(experiments):
    """Plot weight variation as line charts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    param_labels = {
        "w_validity": "$w_{validity}$",
        "w_proximity": "$w_{proximity}$",
        "w_hard_bonus": "$w_{bonus}$",
    }

    for idx, (param, values_dict) in enumerate(experiments.items()):
        ax = axes[idx]
        valid_items = {k: v for k, v in values_dict.items() if "error" not in v}
        if not valid_items:
            continue

        x_vals = sorted(valid_items.keys())
        validity = [valid_items[x]["validity_ratio"] for x in x_vals]
        proximity = [valid_items[x]["proximity_l2"] for x in x_vals]

        ax.plot(x_vals, validity, "o-", color="tab:blue", lw=2, label="Validity ↑")
        ax.set_xlabel(param_labels.get(param, param), fontsize=11)
        ax.set_ylabel("Validity Ratio", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_ylim(0, 1.05)

        ax2 = ax.twinx()
        ax2.plot(x_vals, proximity, "s--", color="tab:red", lw=2, label="Proximity ↓")
        ax2.set_ylabel("Proximity L2", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax.set_title(f"Impact of {param_labels.get(param, param)}")
        ax.grid(alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

    plt.suptitle("Reward Weight Sensitivity", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "weight_variation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Weight variation → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid Search + Ablation for PFE Report")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["grid", "ablation", "weights", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs per run (default: 10 for screening)",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only generate plots from existing results",
    )
    args = parser.parse_args()

    global SCREENING_EPOCHS
    SCREENING_EPOCHS = args.epochs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.plot_only:
        # Load and plot existing results
        grid_path = os.path.join(RESULTS_DIR, "grid_search_results.json")
        ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")
        weights_path = os.path.join(RESULTS_DIR, "weight_variation_results.json")

        if os.path.exists(grid_path):
            with open(grid_path) as f:
                plot_grid_search(json.load(f))
        if os.path.exists(ablation_path):
            with open(ablation_path) as f:
                plot_ablation(json.load(f))
        if os.path.exists(weights_path):
            with open(weights_path) as f:
                data = json.load(f)
                # Convert string keys back to float
                experiments = {}
                for param, values in data.items():
                    experiments[param] = {float(k): v for k, v in values.items()}
                plot_weight_variation(experiments)
        return

    # Load configs
    cfg_f = load_config(FORECAST_CONFIG)
    cfg_ae = load_config(AE_CONFIG)
    device = get_device(cfg_f)
    print(f"Device: {device}")

    t_start = time.time()

    if args.experiment in ("grid", "all"):
        grid_results = run_grid_search(cfg_f, cfg_ae, device)
        plot_grid_search(grid_results)

    if args.experiment in ("ablation", "all"):
        ablation_results = run_ablation(cfg_f, cfg_ae, device)
        plot_ablation(ablation_results)

    if args.experiment in ("weights", "all"):
        weight_results = run_weight_variation(cfg_f, cfg_ae, device)
        plot_weight_variation(weight_results)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"✅ ALL EXPERIMENTS DONE in {elapsed/60:.1f} minutes")
    print(f"Results → {RESULTS_DIR}")
    print(f"Figures → {FIGURES_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
