"""
BaseNN — ETTh1 / GRU
=====================
Baseline 1-nearest-neighbour de ForecastCF :
Récupère depuis le train set le sample dont le forecast est le plus proche
du milieu de la bande cible [alpha, beta].

Usage:
    python baselines/BaseNN/run_basenn_etth1_gru.py
"""

import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from src.utils.config import load_config
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2 as ForecasterWrapper
from src.evaluation.unified_evaluator import CounterfactualEvaluator
from src.training.RL_trainers.trainer_last import prepare_rl_data

FORECAST_CONFIG = "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json"
AE_CONFIG       = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
RL_CONFIG       = "assets/configs/models/etth1_dataset/RL_ablations/config_gru.json"
DESIRED_CHANGE  = -0.1
SEEDS           = [1, 9, 30]
N_BATCHES       = 20
OUTPUT_DIR      = "baselines/BaseNN/results"

MAIN_METRICS = [
    ("validity_ratio",        "Validity Ratio ↑",        True),
    ("stepwise_auc",          "Stepwise AUC ↑",          True),
    ("proximity_l2",          "Proximity L2 ↓",          False),
    ("compactness",           "Compactness ↑",           True),
    ("temporal_consistency",  "Temporal Consistency ↑",  True),
    ("plausibility_ensemble", "Plausibility Ensemble ↓", False),
]


# ── BaseNN ────────────────────────────────────────────────────────────────────

def run_basenn(X_test, Y_hat_test, X_train, Y_hat_train, alphas, betas):
    """
    Pour chaque sample test, trouve le sample train dont le forecast
    est le plus proche du milieu de la bande cible (Euclidean sur horizon).
    Retourne x_cf = x_train[nn_idx], y_cf = y_hat_train[nn_idx].
    """
    N = len(X_test)
    x_cf_all  = np.zeros_like(X_test)
    y_cf_all  = np.zeros((N, Y_hat_test.shape[1], 1))

    # Targets : milieu de la bande pour chaque sample test [N, H]
    targets = (alphas + betas) / 2.0   # [N, H]

    # Y_hat_train : [M, H] ou [M, H, 1]
    ytr = Y_hat_train[:, :, 0] if Y_hat_train.ndim == 3 else Y_hat_train  # [M, H]

    for i in range(N):
        t = targets[i]                          # [H]
        # Distance L2 entre forecast train et target
        dists = np.linalg.norm(ytr - t[np.newaxis, :], axis=1)  # [M]
        nn_idx = int(np.argmin(dists))
        x_cf_all[i]  = X_train[nn_idx]
        y_cf_all[i]  = Y_hat_train[nn_idx] if Y_hat_train.ndim == 3 \
                       else Y_hat_train[nn_idx:nn_idx+1].T.reshape(-1, 1)

    return x_cf_all, y_cf_all


# ── figures ───────────────────────────────────────────────────────────────────

def plot_metrics(avg_summary, out_dir):
    keys   = [k for k, _, _ in MAIN_METRICS]
    labels = [l for _, l, _ in MAIN_METRICS]
    means  = [avg_summary[k]["mean"] if k in avg_summary else 0.0 for k in keys]
    stds   = [avg_summary[k]["std"]  if k in avg_summary else 0.0 for k in keys]
    colors = ["#4C9BE8" if h else "#E8754C" for _, _, h in MAIN_METRICS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.55)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("BaseNN — ETTh1 / GRU — 6 Metrics")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(handles=[
        Patch(facecolor="#4C9BE8", alpha=0.85, label="Higher is better ↑"),
        Patch(facecolor="#E8754C", alpha=0.85, label="Lower is better ↓"),
    ], fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "basenn_etth1_gru_metrics.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved → {p}")


def plot_cf_examples(all_x, all_x_cf, all_y_hat, all_y_cf,
                     alphas, betas, out_dir, n=4):
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    if n == 1: axes = [axes]

    for i in range(n):
        x_ot  = all_x[i, :, 0]
        x_cf  = all_x_cf[i, :, 0]
        y_hat = all_y_hat[i, :, 0] if all_y_hat.ndim == 3 else all_y_hat[i]
        y_cf  = all_y_cf[i, :, 0]  if all_y_cf.ndim  == 3 else all_y_cf[i]
        a_np, b_np = alphas[i], betas[i]

        BH     = len(x_ot)
        t_back = np.arange(BH)
        t_fore = np.arange(BH, BH + len(y_hat))
        valid  = float(((y_cf >= a_np) & (y_cf <= b_np)).mean())

        ax = axes[i]
        ax.plot(t_back, x_ot,  color="#2196F3", lw=1.8, label="x original")
        ax.plot(t_back, x_cf,  color="#FF5722", lw=1.8, ls="--", label="x CF (BaseNN)")
        ax.plot(t_fore, y_hat, color="#2196F3", lw=1.5, ls="-.",  label="forecast original")
        ax.plot(t_fore, y_cf,  color="#FF5722", lw=1.5, ls=":",   label="forecast CF")
        ax.fill_between(t_fore, a_np, b_np, alpha=0.20, color="#4CAF50", label="α/β target band")
        ax.axvline(BH, color="gray", lw=1.2, ls="--", alpha=0.7)
        ax.set_title(f"Sample {i+1} — validity={valid:.2f}", fontsize=11)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    fig.suptitle("BaseNN — CF Examples (ETTh1 / GRU)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "basenn_etth1_gru_cf_examples.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cfg_f  = load_config(FORECAST_CONFIG)
    cfg_ae = load_config(AE_CONFIG)
    cfg_rl = load_config(RL_CONFIG)
    device = "cpu"

    print(f"Device: {device}")

    forecaster = ForecasterWrapper(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    train_loader, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)

    # global_sigma
    _x_stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        bx, _, _, _ = batch
        _x_stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_x_stds).mean())
    print(f"global_sigma: {global_sigma:.4f}")

    # x_train pour plausibilité
    x_train_batches = []
    for i, batch in enumerate(train_loader):
        if i >= getattr(cfg_rl, "eval_train_batches", 20): break
        bx, _, _, _ = batch
        x_train_batches.append(bx[:, :, -1:].cpu().numpy())
    x_train_eval = np.concatenate(x_train_batches, axis=0)

    evaluator = CounterfactualEvaluator(x_train=x_train_eval, fit_plausibility=True)

    # Collecter train set complet pour BaseNN
    print("Collecting train set forecasts for BaseNN...")
    all_x_train, all_y_hat_train = [], []
    for i, batch in enumerate(train_loader):
        batch_x, _, batch_x_mark, _ = batch
        batch_x      = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        with torch.no_grad():
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)
        all_x_train.append(batch_x[:, :, -1:].cpu().numpy())
        all_y_hat_train.append(y_hat.cpu().numpy())
    all_x_train     = np.concatenate(all_x_train,     axis=0)
    all_y_hat_train = np.concatenate(all_y_hat_train, axis=0)
    print(f"Train set: {len(all_x_train)} samples")

    # Collecter test set
    all_x, all_y_hat = [], []
    for i, batch in enumerate(test_loader):
        if i >= N_BATCHES: break
        batch_x, _, batch_x_mark, _ = batch
        batch_x      = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        with torch.no_grad():
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)
        all_x.append(batch_x[:, :, -1:].cpu().numpy())
        all_y_hat.append(y_hat.cpu().numpy())
    all_x     = np.concatenate(all_x,     axis=0)
    all_y_hat = np.concatenate(all_y_hat, axis=0)

    # Bornes RL
    rho, fr   = cfg_rl.rho, cfg_rl.fr
    direction = getattr(cfg_rl, "direction", -1.0)
    sigma     = np.full(all_x.shape[0], global_sigma)
    gap       = (rho * sigma)[:, np.newaxis]
    width     = (fr  * sigma)[:, np.newaxis]
    y2d       = all_y_hat[:, :, 0]
    if direction < 0:
        betas  = y2d - gap
        alphas = betas - width
    else:
        alphas = y2d + gap
        betas  = alphas + width

    # BaseNN
    t0 = time.time()
    x_cf_final, y_cf_final = run_basenn(
        all_x, all_y_hat, all_x_train, all_y_hat_train, alphas, betas
    )
    runtime = time.time() - t0
    print(f"Runtime: {runtime:.2f}s  ({runtime/len(all_x)*1000:.1f}ms/sample)")

    # Évaluer
    all_summaries = []
    for seed in SEEDS:
        np.random.seed(seed)
        summary = evaluator.evaluate(
            X_orig=all_x, X_cf=x_cf_final,
            Y_hat=all_y_hat, Y_cf=y_cf_final,
            alphas=alphas, betas=betas,
        )
        all_summaries.append(summary)

    avg_summary = {}
    for k in all_summaries[0]:
        means = [s[k]["mean"] for s in all_summaries
                 if isinstance(s.get(k, {}).get("mean"), (int, float))]
        if means:
            avg_summary[k] = {"mean": float(np.mean(means)), "std": float(np.std(means))}

    print("\n── BaseNN — Results ─────────────────────────────────")
    CounterfactualEvaluator.print_table(avg_summary, label="BaseNN")
    print(f"  Runtime: {runtime:.2f}s  ({runtime/len(all_x)*1000:.1f}ms/sample)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "basenn_etth1_gru.json")
    with open(out_path, "w") as f:
        json.dump({
            "method": "BaseNN",
            "n_samples": len(all_x),
            "runtime_seconds": runtime,
            "runtime_ms_per_sample": runtime / len(all_x) * 1000,
            "avg_metrics": avg_summary,
            "all_seeds": all_summaries,
        }, f, indent=2)
    print(f"Saved → {out_path}")

    plot_metrics(avg_summary, OUTPUT_DIR)
    plot_cf_examples(all_x, x_cf_final, all_y_hat, y_cf_final,
                     alphas, betas, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
