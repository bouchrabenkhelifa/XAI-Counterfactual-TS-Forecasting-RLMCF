"""
Eval ETTh1 — 5 modèles
=======================
Lance l'évaluation --eval_only sur les 5 checkpoints RL ETTh1
et génère un tableau récapitulatif + plots comparatifs.

Usage:
    python scripts/evals/eval_etth1_all_models.py
"""

import json
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_last import RLMaskTrainer

# Patch : remplace ForecasterWrapper par V2 dans le trainer
import src.training.RL_trainers.trainer_last as _trainer_mod
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
_trainer_mod.ForecasterWrapper = ForecasterWrapperV2

AE_CONFIG = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
EVAL_BATCHES = 20
OUTPUT_DIR = "assets/results/etth1/summary"

# 5 modèles : (label, rl_config, forecast_config)
MODELS = [
    (
        "iTransformer",
        "assets/configs/models/etth1_dataset/RL_ablations/config_v2.json",
        "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    ),
    (
        "PatchTST",
        "assets/configs/models/etth1_dataset/RL_ablations/config_patchtst.json",
        "assets/configs/models/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
    ),
    (
        "TimesNet",
        "assets/configs/models/etth1_dataset/RL_ablations/config_timesnet.json",
        "assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
    ),
    (
        "GRU",
        "assets/configs/models/etth1_dataset/RL_ablations/config_gru.json",
        "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
    ),
    (
        "DLinear",
        "assets/configs/models/etth1_dataset/RL_ablations/config_dlinear.json",
        "assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
    ),
]

# Métriques à extraire pour le tableau
TABLE_METRICS = [
    ("validity_ratio",        "Valid."),
    ("stepwise_auc",          "AUC"),
    ("proximity_l2",          "Prox."),
    ("compactness",           "Comp."),
    ("roughness_ratio",       "Rough."),
    ("temporal_consistency",  "T-Cons."),
    ("plausibility_ensemble", "Plaus."),
]


def eval_one(label, rl_config_path, forecast_config_path, ae_config_path, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    cfg_rl = load_config(rl_config_path)
    cfg_f  = load_config(forecast_config_path)
    cfg_ae = load_config(ae_config_path)

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    # Override pour iTransformer : best config identique à _plot_best_config.py
    if label == "iTransformer":
        trainer.reward_fn.rho = 0.20
        trainer.reward_fn.fr  = 0.50
        trainer.agent.eta     = 0.15
        trainer.mask_last_k   = 12

    # Charger le meilleur checkpoint
    ckpt_path = os.path.join(
        cfg_rl.checkpoint_dir_lp,
        f"{cfg_rl.name}_agent_best.pt"
    )
    # Override checkpoint pour iTransformer → RL/itransformer
    if label == "iTransformer":
        ckpt_path = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
        trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"  Loaded checkpoint → {ckpt_path}")
    else:
        print(f"  WARNING: no checkpoint at {ckpt_path}, using random weights")

    summary, _ = trainer.evaluate(n_batches=EVAL_BATCHES)
    return summary


def print_table(results: dict):
    """Affiche le tableau récapitulatif dans le terminal."""
    col_w = 8
    header = f"  {'Model':<14}" + "".join(f"{m:>{col_w}}" for _, m in TABLE_METRICS)
    sep    = "-" * (14 + col_w * len(TABLE_METRICS) + 2)

    print(f"\n{'='*60}")
    print(f"  ETTh1 — Full Evaluation (5 models)")
    print(f"{'='*60}")
    print(header)
    print(sep)

    for label, summary in results.items():
        row = f"  {label:<14}"
        for key, _ in TABLE_METRICS:
            v = summary.get(key, {}).get("mean", float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)

    print(f"{'='*60}\n")


def plot_radar(results: dict, out_dir: str):
    """Radar chart pour comparer les 5 modèles sur 6 métriques."""
    metrics_radar = [
        ("validity_ratio",       "Validity"),
        ("stepwise_auc",         "AUC"),
        ("compactness",          "Compact."),
        ("temporal_consistency", "T-Cons."),
    ]
    # Inverser proximity et plausibility (lower is better → 1 - val)
    metrics_inv = [
        ("proximity_l2",          "Prox. (inv)"),
        ("plausibility_ensemble", "Plaus. (inv)"),
    ]

    labels = [l for _, l in metrics_radar] + [l for _, l in metrics_inv]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for (label, summary), color in zip(results.items(), colors):
        vals = [summary.get(k, {}).get("mean", 0.0) for k, _ in metrics_radar]
        # Normaliser proximity et plausibility : 1 - val (clampé 0-1)
        for k, _ in metrics_inv:
            v = summary.get(k, {}).get("mean", 0.0)
            vals.append(float(np.clip(1.0 - v, 0, 1)))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("ETTh1 — RL-MCF across 5 forecasters", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "etth1_radar_5models.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


def plot_barplot(results: dict, out_dir: str):
    """Barplot groupé des 7 métriques pour les 5 modèles."""
    models  = list(results.keys())
    n_m     = len(models)
    n_met   = len(TABLE_METRICS)
    x       = np.arange(n_met)
    w       = 0.15
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, color) in enumerate(zip(models, colors)):
        summary = results[label]
        vals = [summary.get(k, {}).get("mean", 0.0) for k, _ in TABLE_METRICS]
        stds = [summary.get(k, {}).get("std",  0.0) for k, _ in TABLE_METRICS]
        ax.bar(x + i*w - (n_m-1)*w/2, vals, w,
               yerr=stds, capsize=3, color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([m for _, m in TABLE_METRICS], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("ETTh1 — RL-MCF: 7 Metrics × 5 Forecasters", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "etth1_barplot_5models.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for label, rl_cfg, f_cfg in MODELS:
        summary = eval_one(label, rl_cfg, f_cfg, AE_CONFIG, device)
        results[label] = summary

    # Afficher tableau terminal
    print_table(results)

    # Sauvegarder JSON
    out_json = os.path.join(OUTPUT_DIR, "etth1_all_models_eval.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_json}")

    # Plots
    plot_radar(results, OUTPUT_DIR)
    plot_barplot(results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
