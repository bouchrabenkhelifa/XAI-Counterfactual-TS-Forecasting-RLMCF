"""
Evaluate iTransformer forecaster on all 3 datasets and plot best forecast examples side by side.

Usage:
    python scripts/analysis/plot_forecaster_3datasets.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.data_provider.data_factory import data_provider

FIGURES_DIR = "assets/figures/data_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

configs = {
    "ETTh1": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    "ETTh2": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    "Weather": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
}


def evaluate_forecaster(cfg_path):
    """Load forecaster, run on test set, return MSE and best example."""
    cfg = load_config(cfg_path)
    # Ensure model_type is set for the wrapper
    if not hasattr(cfg, "model_type"):
        cfg.model_type = "iTransformer"
    device = get_device(cfg)

    forecaster = ForecasterWrapperV2(cfg, device)

    test_data, test_loader = data_provider(cfg, "test")
    scaler = test_data.scaler  # StandardScaler fitted on train

    all_mse = []
    all_inputs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= 500:
                break
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            pred = forecaster.predict(batch_x, batch_x_mark)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred[:, -cfg.pred_len:, -1:]  # last feature, pred_len

            target = batch_y[:, -cfg.pred_len:, -1:].float().to(device)
            mse = ((pred - target) ** 2).mean().item()

            all_mse.append(mse)
            all_inputs.append(batch_x[0, :, -1].cpu().numpy())
            all_preds.append(pred[0, :, 0].cpu().numpy())
            all_targets.append(target[0, :, 0].cpu().numpy())

    # Pick a sample that is BOTH good (low MSE) AND visually interesting (high variance)
    all_mse = np.array(all_mse)
    variances = np.array([np.std(inp) for inp in all_inputs])
    
    # Filter: top 30% in variance (interesting), then pick lowest MSE among those
    var_threshold = np.percentile(variances, 70)
    candidates = np.where(variances >= var_threshold)[0]
    best_idx = int(candidates[np.argmin(all_mse[candidates])])
    
    avg_mse = float(all_mse.mean())

    # Inverse transform to original scale
    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    # Compute RMSE in original scale (degrees)
    pred_orig = inv(all_preds[best_idx])
    target_orig = inv(all_targets[best_idx])
    rmse_orig = float(np.sqrt(((pred_orig - target_orig) ** 2).mean()))

    return {
        "avg_mse": avg_mse,
        "best_mse": all_mse[best_idx],
        "rmse_degrees": rmse_orig,
        "input": inv(all_inputs[best_idx]),
        "pred": pred_orig,
        "target": target_orig,
        "pred_len": cfg.pred_len,
        "seq_len": cfg.seq_len,
        "target_col": cfg.target,
    }


def main():
    print("=" * 60)
    print("iTransformer Forecaster Evaluation — All 3 Datasets")
    print("=" * 60)

    results = {}
    for name, cfg_path in configs.items():
        print(f"\n── {name} ──")
        results[name] = evaluate_forecaster(cfg_path)
        print(f"  Avg MSE: {results[name]['avg_mse']:.6f}")
        print(f"  Best sample MSE: {results[name]['best_mse']:.6f}")

    # ─────────────────────────────────────────────────────────────────────
    # Plot: 1x3 — one best forecast per dataset
    # ─────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for col, (name, r) in enumerate(results.items()):
        ax = axes[col]
        seq_len = r["seq_len"]
        pred_len = r["pred_len"]

        # Full timeline
        t_input = np.arange(seq_len)
        t_pred = np.arange(seq_len, seq_len + pred_len)

        ax.plot(t_input, r["input"], lw=1.5, color="tab:blue", label="Input (history)")
        ax.plot(t_pred, r["target"], lw=1.5, color="tab:green", label="Ground truth")
        ax.plot(t_pred, r["pred"], lw=1.5, color="tab:red", ls="--", label="Prediction")

        ax.axvline(seq_len, color="gray", ls=":", lw=1)
        ax.fill_between(t_pred, r["target"], r["pred"], alpha=0.15, color="tab:red")

        ax.set_title(f"{name}\nRMSE={r['rmse_degrees']:.2f}°C", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel(r.get("target_col", "Value"))
        ax.legend(fontsize=8)

    plt.suptitle("iTransformer Forecast — Test Set (best sample per dataset)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "forecaster_itransformer_3datasets.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] → {path}")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    print("\n── iTransformer Forecast Performance ───────────────────")
    print(f"{'Dataset':<12} {'Avg MSE':<12} {'RMSE (°C)':<12} {'Pred Len'}")
    print("-" * 48)
    for name, r in results.items():
        print(f"{name:<12} {r['avg_mse']:<12.6f} {r['rmse_degrees']:<12.2f} {r['pred_len']}")


if __name__ == "__main__":
    main()
