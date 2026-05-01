"""
Eval ETTh2 — 5 modèles
=======================
Lance l'évaluation --eval_only sur les 5 checkpoints RL ETTh2
et génère un tableau récapitulatif + plots comparatifs.

Usage:
    python scripts/evals/eval_etth2_all_models.py
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

# Patch ForecasterWrapper -> V2
import src.training.RL_trainers.trainer_last as _trainer_mod
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
_trainer_mod.ForecasterWrapper = ForecasterWrapperV2

AE_CONFIG    = "assets/configs/models/etth2_dataset/ae/tcn_ae.json"
EVAL_BATCHES = 20
OUTPUT_DIR   = "assets/results/etth2/summary"

MODELS = [
    (
        "iTransformer",
        "assets/configs/models/etth2_dataset/RL/config_itransformer.json",
        "assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    ),
    (
        "PatchTST",
        "assets/configs/models/etth2_dataset/RL/config_patchtst.json",
        "assets/configs/models/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
    ),
    (
        "TimesNet",
        "assets/configs/models/etth2_dataset/RL/config_timesnet.json",
        "assets/configs/models/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
    ),
    (
        "GRU",
        "assets/configs/models/etth2_dataset/RL/config_gru.json",
        "assets/configs/models/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
    ),
    (
        "DLinear",
        "assets/configs/models/etth2_dataset/RL/config_dlinear.json",
        "assets/configs/models/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
    ),
]

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
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
    cfg_rl = load_config(rl_config_path)
    cfg_f  = load_config(forecast_config_path)
    cfg_ae = load_config(ae_config_path)

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    ckpt_path = os.path.join(
        cfg_rl.checkpoint_dir_lp,
        f"{cfg_rl.name}_agent_best.pt"
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
        trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"  Loaded checkpoint -> {ckpt_path}")
    else:
        print(f"  WARNING: no checkpoint at {ckpt_path}, using random weights")

    summary, _ = trainer.evaluate(n_batches=EVAL_BATCHES)
    return summary


def print_table(results):
    col_w  = 8
    header = f"  {'Model':<14}" + "".join(f"{m:>{col_w}}" for _, m in TABLE_METRICS)
    sep    = "-" * (14 + col_w * len(TABLE_METRICS) + 2)
    print(f"\n{'='*60}\n  ETTh2 — Full Evaluation (5 models)\n{'='*60}")
    print(header)
    print(sep)
    for label, summary in results.items():
        row = f"  {label:<14}"
        for key, _ in TABLE_METRICS:
            v = summary.get(key, {}).get("mean", float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)
    print(f"{'='*60}\n")


def plot_radar(results, out_dir):
    metrics_radar = [
        ("validity_ratio",       "Validity"),
        ("stepwise_auc",         "AUC"),
        ("compactness",          "Compact."),
        ("temporal_consistency", "T-Cons."),
    ]
    metrics_inv = [
        ("proximity_l2",          "Prox. (inv)"),
        ("plausibility_ensemble", "Plaus. (inv)"),
    ]
    labels = [l for _, l in metrics_radar] + [l for _, l in metrics_inv]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for (label, summary), color in zip(results.items(), colors):
        vals = [summary.get(k, {}).get("mean", 0.0) for k, _ in metrics_radar]
        for k, _ in metrics_inv:
            v = summary.get(k, {}).get("mean", 0.0)
            vals.append(float(np.clip(1.0 - v, 0, 1)))
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("ETTh2 — RL-MCF across 5 forecasters", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "etth2_radar_5models.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {p}")


def plot_barplot(results, out_dir):
    models = list(results.keys())
    x      = np.arange(len(TABLE_METRICS))
    w      = 0.15
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, color) in enumerate(zip(models, colors)):
        summary = results[label]
        vals = [summary.get(k, {}).get("mean", 0.0) for k, _ in TABLE_METRICS]
        stds = [summary.get(k, {}).get("std",  0.0) for k, _ in TABLE_METRICS]
        ax.bar(x + i*w - (len(models)-1)*w/2, vals, w,
               yerr=stds, capsize=3, color=color, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([m for _, m in TABLE_METRICS], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("ETTh2 — RL-MCF: 7 Metrics × 5 Forecasters", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "etth2_barplot_5models.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved -> {p}")


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for label, rl_cfg, f_cfg in MODELS:
        summary = eval_one(label, rl_cfg, f_cfg, AE_CONFIG, device)
        results[label] = summary

    print_table(results)

    out_json = os.path.join(OUTPUT_DIR, "etth2_all_models_eval.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_json}")

    plot_radar(results, OUTPUT_DIR)
    plot_barplot(results, OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
