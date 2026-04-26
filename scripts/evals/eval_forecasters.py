"""
eval_forecasters.py
--------------------
Evalue les checkpoints GRU, DLinear, TimesNet sur le test set ETTh1.
Calcule MAE, MSE, RMSE et génère les figures.

Usage :
    python scripts/evals/eval_forecasters.py
    python scripts/evals/eval_forecasters.py --models gru dlinear
"""

import argparse
import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.data_provider.data_factory import data_provider


# ─────────────────────────────────────────────────────────────────────────────
# Config paths
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "gru":      "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
    "dlinear":  "assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
    "timesnet": "assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
}

FIGURES_DIR = "assets/figures/forecaster"
RESULTS_DIR = "assets/results/forecaster"


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(configs, device):
    mt = getattr(configs, "model_type", "GRU").lower()
    if mt == "gru":
        from src.models.Forecaster.GRU import Model
    elif mt == "dlinear":
        from src.models.Forecaster.DLinear import Model
    elif mt == "timesnet":
        from src.models.Forecaster.TimesNet import Model
    else:
        raise ValueError(f"Unknown model_type: {mt}")

    model = Model(configs).float().to(device)
    ckpt_path = os.path.join(configs.checkpoint_dir, configs.checkpoint_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"[{mt.upper()}] Loaded ✓  {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, test_loader, configs, device):
    """Run inference on test set, return preds and targets (numpy)."""
    all_preds, all_targets, all_inputs = [], [], []
    f_dim = -1 if getattr(configs, "features", "S") == "MS" else 0

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, _ in test_loader:
            batch_x      = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            preds = model(batch_x, batch_x_mark)
            if isinstance(preds, tuple):
                preds = preds[0]

            preds  = preds[:, -configs.pred_len:, f_dim:].cpu().numpy()
            target = batch_y[:, -configs.pred_len:, f_dim:].numpy()
            inp    = batch_x[:, :, f_dim:].cpu().numpy()

            all_preds.append(preds)
            all_targets.append(target)
            all_inputs.append(inp)

    preds   = np.concatenate(all_preds,   axis=0)  # (N, H, 1)
    targets = np.concatenate(all_targets, axis=0)
    inputs  = np.concatenate(all_inputs,  axis=0)
    return preds, targets, inputs


def compute_metrics(preds, targets):
    mae  = float(np.mean(np.abs(preds - targets)))
    mse  = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_examples(preds, targets, inputs, model_name,
                           figures_dir, n=4):
    """Past (grey) + real future (blue) + predicted (red dashed)."""
    os.makedirs(figures_dir, exist_ok=True)
    n = min(n, preds.shape[0])

    seq_len  = inputs.shape[1]
    pred_len = preds.shape[1]
    t_past   = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_past,   inputs[i, :, 0],  color="grey",    lw=1.2, label="Past (input)")
        ax.plot(t_future, targets[i, :, 0], color="#2196F3", lw=1.5, label="Real future")
        ax.plot(t_future, preds[i, :, 0],   color="#F44336", lw=1.5,
                linestyle="--", label="Predicted")
        ax.axvline(x=seq_len, color="black", linestyle=":", lw=1)
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Forecast Examples (Test Set)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, f"{model_name.lower()}_eval_examples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig] Examples → {path}")


def plot_error_distribution(preds, targets, model_name, figures_dir):
    """Histogram of absolute errors."""
    os.makedirs(figures_dir, exist_ok=True)
    errors = np.abs(preds - targets).flatten()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.axvline(errors.mean(), color="#F44336", lw=2, linestyle="--",
               label=f"Mean = {errors.mean():.4f}")
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — Error Distribution (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(figures_dir, f"{model_name.lower()}_error_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig] Error dist → {path}")


def plot_comparison_bar(all_metrics, figures_dir):
    """Bar chart comparing MAE / MSE / RMSE across models."""
    os.makedirs(figures_dir, exist_ok=True)
    models  = list(all_metrics.keys())
    metrics = ["MAE", "MSE", "RMSE"]
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]

    x     = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [all_metrics[m][metric] for m in models]
        bars = ax.bar(x + j * width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("Error")
    ax.set_title("Forecaster Comparison — ETTh1 Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(figures_dir, "forecaster_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig] Comparison bar → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["gru", "dlinear", "timesnet"],
        choices=["gru", "dlinear", "timesnet"],
        help="Models to evaluate",
    )
    parser.add_argument("--n_examples", type=int, default=4,
                        help="Number of forecast examples to plot")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics = {}

    for model_key in args.models:
        cfg_path = MODEL_CONFIGS[model_key]
        print(f"\n{'='*60}")
        print(f"  Evaluating : {model_key.upper()}")
        print(f"  Config     : {cfg_path}")
        print(f"{'='*60}")

        configs = load_config(cfg_path)
        device  = get_device(configs)

        # Load model
        try:
            model = load_model(configs, device)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        # Load test data
        _, test_loader = data_provider(configs, "test")

        # Evaluate
        preds, targets, inputs = evaluate(model, test_loader, configs, device)
        metrics = compute_metrics(preds, targets)
        all_metrics[model_key] = metrics

        print(f"\n  Results {model_key.upper()}:")
        for k, v in metrics.items():
            print(f"    {k:6s} : {v:.6f}")

        # Save metrics JSON
        metrics_path = os.path.join(RESULTS_DIR, f"eval_{model_key}_etth1.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"  Saved → {metrics_path}")

        # Figures per model
        plot_forecast_examples(preds, targets, inputs,
                               model_key.upper(), FIGURES_DIR, n=args.n_examples)
        plot_error_distribution(preds, targets, model_key.upper(), FIGURES_DIR)

    # Comparison bar chart (only if multiple models evaluated)
    if len(all_metrics) > 1:
        plot_comparison_bar(all_metrics, FIGURES_DIR)

    # Summary table
    if all_metrics:
        print(f"\n{'='*55}")
        print(f"  SUMMARY — ETTh1 Test Set")
        print(f"{'='*55}")
        print(f"  {'Model':<12} {'MAE':>10} {'MSE':>10} {'RMSE':>10}")
        print(f"  {'-'*42}")
        for m, met in all_metrics.items():
            print(f"  {m.upper():<12} {met['MAE']:>10.6f} {met['MSE']:>10.6f} {met['RMSE']:>10.6f}")
        print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
