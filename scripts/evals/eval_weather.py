"""
Eval Weather — Combine Colab results with local plausibility
============================================================
Lit les métriques de Colab (validity, AUC, proximity, etc.)
et recalcule uniquement la plausibilité avec le détecteur pré-entraîné.

Usage:
    python scripts/evals/eval_weather.py
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
from src.training.RL_trainers.trainer_main import RLMaskTrainer, run_episode_eval

AE_CONFIG    = "assets/configs/weather_dataset/ae/tcn_ae.json"
EVAL_BATCHES = 20
OUTPUT_DIR   = "assets/results/weather/summary"

# 5 modèles : (label, colab_json_path, checkpoint_path, forecast_config)
MODELS = [
    (
        "iTransformer",
        "assets/results/weather/RL/RL_itransformer/rl_cf_itransformer_weather_evaluation.json",
        "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
        "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
    ),
    (
        "PatchTST",
        "assets/results/weather/RL/RL_patchtst/rl_cf_patchtst_weather_evaluation.json",
        "assets/checkpoints/weather_chpts/RL/RL_patchtst/rl_cf_patchtst_weather_agent_best.pt",
        "assets/configs/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
    ),
    (
        "TimesNet",
        "assets/results/weather/RL/RL_timesnet/rl_cf_timesnet_weather_evaluation.json",
        "assets/checkpoints/weather_chpts/RL/RL_timesnet/rl_cf_timesnet_weather_agent_best.pt",
        "assets/configs/weather_dataset/forecasters/timesnet/weather_96_96_S.json",
    ),
    (
        "GRU",
        "assets/results/weather/RL/gru/rl_cf_gru_weather_evaluation.json",
        "assets/checkpoints/weather_chpts/RL/RL_gru/rl_cf_gru_weather_agent_best.pt",
        "assets/configs/weather_dataset/forecasters/gru/weather_96_96_S.json",
    ),
    (
        "DLinear",
        "assets/results/weather/RL/RL_dlinear/rl_cf_dlinear_weather_evaluation.json",
        "assets/checkpoints/weather_chpts/RL/RL_dlinear/rl_cf_dlinear_weather_agent_best.pt",
        "assets/configs/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
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


def recalculate_plausibility(label, checkpoint_path, forecast_config_path, ae_config_path, device):
    """Génère les CFs et recalcule uniquement la plausibilité."""
    print(f"\n{'='*60}")
    print(f"Recalculating plausibility: {label}")
    print(f"{'='*60}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_rl_dict = ckpt["cfg_rl"]

    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    cfg_rl = ConfigObj(cfg_rl_dict)
    cfg_f  = load_config(forecast_config_path)
    cfg_ae = load_config(ae_config_path)

    print(f"  Using checkpoint config: rho={cfg_rl.rho}, fr={cfg_rl.fr}, mask_last_k={cfg_rl.mask_last_k}")

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
    print(f"  Loaded checkpoint -> {checkpoint_path}")

    trainer.agent.eval()
    all_x, all_x_cf, all_y_hat, all_y_cf = [], [], [], []

    for i, batch in enumerate(trainer.test_loader):
        if i >= EVAL_BATCHES:
            break
        ep = run_episode_eval(
            batch=batch,
            ae_arch=trainer.ae_arch,
            forecaster=trainer.forecaster,
            agent=trainer.agent,
            reward_fn=trainer.reward_fn,
            device=device,
            use_rl=True,
            mask_last_k=trainer.mask_last_k,
            mask_ramp_k=trainer.mask_ramp_k,
            filter_quantile=trainer.filter_quantile,
        )
        if ep is None:
            continue

        all_x.append(ep["x_ot"].cpu().numpy())
        all_x_cf.append(ep["x_cf"].cpu().numpy())
        all_y_hat.append(ep["y_hat"].cpu().numpy())
        all_y_cf.append(ep["y_cf"].cpu().numpy())

    if not all_x:
        raise RuntimeError("No valid CFs.")

    all_x     = np.concatenate(all_x, axis=0)
    all_x_cf  = np.concatenate(all_x_cf, axis=0)
    all_y_hat = np.concatenate(all_y_hat, axis=0)
    all_y_cf  = np.concatenate(all_y_cf, axis=0)

    all_alphas, all_betas = trainer._compute_bounds_np(all_x, all_y_hat)

    full_metrics = trainer.evaluator.evaluate(
        X_orig=all_x,
        X_cf=all_x_cf,
        Y_hat=all_y_hat,
        Y_cf=all_y_cf,
        alphas=all_alphas,
        betas=all_betas,
    )

    plaus_metrics = {
        k: v for k, v in full_metrics.items() if k.startswith("plausibility_")
    }

    print(f"  Plausibility (with pre-trained detector): {plaus_metrics['plausibility_ensemble']['mean']:.4f}")
    return plaus_metrics


def combine_metrics(label, colab_json_path, plaus_metrics):
    """Combine les métriques de Colab avec la plausibilité recalculée."""
    with open(colab_json_path, "r") as f:
        colab_metrics = json.load(f)

    combined = colab_metrics.copy()
    combined["plausibility_if"]       = plaus_metrics["plausibility_if"]
    combined["plausibility_lof"]      = plaus_metrics["plausibility_lof"]
    combined["plausibility_ocsvm"]    = plaus_metrics["plausibility_ocsvm"]
    combined["plausibility_ensemble"] = plaus_metrics["plausibility_ensemble"]
    return combined


def print_table(results: dict):
    col_w  = 8
    header = f"  {'Model':<14}" + "".join(f"{m:>{col_w}}" for _, m in TABLE_METRICS)
    sep    = "-" * (14 + col_w * len(TABLE_METRICS) + 2)

    print(f"\n{'='*60}")
    print(f"  Weather — Colab metrics + Local plausibility")
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
    metrics_radar = [
        ("validity_ratio",        "Validity"),
        ("stepwise_auc",          "AUC"),
        ("compactness",           "Compact."),
        ("temporal_consistency",  "T-Cons."),
        ("plausibility_ensemble", "Plaus. (↓)"),
    ]

    labels = [l for _, l in metrics_radar]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors  = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for (label, summary), color in zip(results.items(), colors):
        vals  = [summary.get(k, {}).get("mean", 0.0) for k, _ in metrics_radar]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Weather — RL-MCF (Colab + Local Plaus.)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    p = os.path.join(out_dir, "weather_radar_5models_colab.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {p}")


def plot_barplot(results: dict, out_dir: str):
    models = list(results.keys())
    n_m    = len(models)
    n_met  = len(TABLE_METRICS)
    x      = np.arange(n_met)
    w      = 0.15
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

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
    ax.set_title("Weather — RL-MCF: Colab metrics + Local Plausibility", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "weather_barplot_5models_colab.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved -> {p}")


def main():
    device = "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for label, colab_json, ckpt_path, f_cfg in MODELS:
        plaus_metrics = recalculate_plausibility(label, ckpt_path, f_cfg, AE_CONFIG, device)
        combined      = combine_metrics(label, colab_json, plaus_metrics)
        results[label] = combined

    print_table(results)

    out_json = os.path.join(OUTPUT_DIR, "weather_all_models_colab_plaus.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_json}")

    plot_radar(results, OUTPUT_DIR)
    plot_barplot(results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
