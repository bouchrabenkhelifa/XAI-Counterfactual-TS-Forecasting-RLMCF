"""
Eval Weather — 4 modèles
=========================
Lance l'évaluation --eval_only sur les checkpoints RL Weather
et génère un tableau récapitulatif + plots comparatifs.

NOTE: Only GRU and DLinear have trained checkpoints.
      iTransformer and PatchTST checkpoints are missing from the repository.
      TimesNet was not trained for Weather dataset.

Usage:
    python scripts/evals/eval_weather_all_models.py
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
from src.training.RL_trainers.trainer_last_v2 import RLMaskTrainer

AE_CONFIG = "assets/configs/models/weather_dataset/ae/tcn_ae.json"
EVAL_BATCHES = 20
OUTPUT_DIR = "assets/results/weather/summary"

# 4 modèles : (label, rl_config, forecast_config, checkpoint_override)
# NOTE: Only GRU and DLinear have trained checkpoints in the repository
# iTransformer and PatchTST checkpoints are missing from assets/checkpoints/weather_chpts/
# TimesNet was not trained for Weather dataset
MODELS = [
    (
        "iTransformer",
        "assets/configs/models/weather_dataset/RL/config_itransformer.json",
        "assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        None,  # Checkpoint missing - will use random weights
    ),
    (
        "PatchTST",
        "assets/configs/models/weather_dataset/RL/config_patchtst.json",
        "assets/configs/models/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
        None,  # Checkpoint missing - will use random weights
    ),
    (
        "GRU",
        "assets/configs/models/weather_dataset/RL/config_gru.json",
        "assets/configs/models/weather_dataset/forecasters/gru/weather_96_96_S.json",
        "assets/checkpoints/weather_chpts/RL/RL_gru/rl_cf_gru_weather_agent_best.pt",
    ),
    (
        "DLinear",
        "assets/configs/models/weather_dataset/RL/config_dlinear.json",
        "assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
        "assets/checkpoints/weather_chpts/RL/RL_dlinear/rl_cf_dlinear_weather_agent_best.pt",
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


def eval_one(label, rl_config_path, forecast_config_path, ae_config_path, checkpoint_override, device):
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")

    cfg_rl = load_config(rl_config_path)
    cfg_f  = load_config(forecast_config_path)
    cfg_ae = load_config(ae_config_path)

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    # Charger le meilleur checkpoint
    if checkpoint_override:
        ckpt_path = checkpoint_override
    else:
        ckpt_path = os.path.join(
            cfg_rl.checkpoint_dir_lp,
            f"{cfg_rl.name}_agent_best.pt"
        )
    
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
    print(f"  Weather — Full Evaluation (4 models)")
    print(f"  NOTE: iTransformer & PatchTST use random weights (checkpoints missing)")
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
    """Radar chart pour comparer les modèles sur 6 métriques."""
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
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]  # 4 colors for 4 models

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
    ax.set_title("Weather — RL-MCF across forecasters (4 models)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "weather_radar_models.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {p}")


def plot_barplot(results: dict, out_dir: str):
    """Barplot groupé des 7 métriques pour les modèles."""
    models  = list(results.keys())
    n_m     = len(models)
    n_met   = len(TABLE_METRICS)
    x       = np.arange(n_met)
    w       = 0.18  # Wider bars for 4 models
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]  # 4 colors

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
    ax.set_title("Weather — RL-MCF: 7 Metrics × 4 Forecasters", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "weather_barplot_models.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for label, rl_cfg, f_cfg, ckpt_override in MODELS:
        summary = eval_one(label, rl_cfg, f_cfg, AE_CONFIG, ckpt_override, device)
        results[label] = summary

    # Afficher tableau terminal
    print_table(results)

    # Sauvegarder JSON
    out_json = os.path.join(OUTPUT_DIR, "weather_all_models_eval.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_json}")

    # Plots
    plot_radar(results, OUTPUT_DIR)
    plot_barplot(results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
