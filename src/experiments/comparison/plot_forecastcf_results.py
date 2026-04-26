"""
Generate figures from ForecastCF results CSV.

Usage:
    python -m src.experiments.comparison.plot_forecastcf_results \
        --csv baselines/ForecastCF/results/forecastcf_etth1.csv \
        --output_dir assets/figures/forecastcf_baseline
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart: metrics per model
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart comparing all models on the 4 ForecastCF metrics."""

    # Keep only ForecastCF method
    df_fcf = df[df["cf_model"] == "ForecastCF"].copy()
    models = df_fcf["forecast_model"].unique()

    metrics = [
        ("validity_ratio",   "Validity Ratio ↑",  True),
        ("step_validity_auc","Step AUC ↑",         True),
        ("proximity",        "Proximity L2 ↓",     False),
        ("compactness",      "Compactness ↑",      True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, (col, label, higher_better) in zip(axes, metrics):
        vals = [df_fcf[df_fcf["forecast_model"] == m][col].mean() for m in models]
        bars = ax.bar(models, vals, color=colors[:len(models)], edgecolor="white", linewidth=0.8)

        # Highlight best
        best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("ForecastCF on ETTh1 — Per-model metrics (50 samples, seed=39)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "forecastcf_etth1_metrics_bar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart: ForecastCF best vs Ours
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(df: pd.DataFrame, rl_results: dict, output_dir: str) -> None:
    """Side-by-side comparison: best ForecastCF model vs our RL method."""

    df_fcf = df[df["cf_model"] == "ForecastCF"].copy()

    # Best model = highest validity_ratio
    best_model = df_fcf.groupby("forecast_model")["validity_ratio"].mean().idxmax()
    best_row   = df_fcf[df_fcf["forecast_model"] == best_model].mean(numeric_only=True)

    metrics = [
        ("validity_ratio",   "Validity Ratio ↑",  True),
        ("step_validity_auc","Step AUC ↑",         True),
        ("proximity",        "Proximity L2 ↓",     False),
        ("compactness",      "Compactness ↑",      True),
    ]

    fcf_vals = [best_row[col] for col, _, _ in metrics]
    rl_vals  = [
        rl_results.get("validity_ratio",  0),
        rl_results.get("stepwise_auc",    0),
        rl_results.get("proximity_l2",    0),
        rl_results.get("compactness",     0),
    ]

    labels      = [label for _, label, _ in metrics]
    x           = np.arange(len(labels))
    width       = 0.35
    higher_better = [hb for _, _, hb in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, fcf_vals, width, label=f"ForecastCF+{best_model}",
                   color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, rl_vals,  width, label="Ours (RL+iTransformer)",
                   color="#DD8452", alpha=0.85, edgecolor="white")

    # Mark winners
    for i, (v1, v2, hb) in enumerate(zip(fcf_vals, rl_vals, higher_better)):
        winner = bars2[i] if (hb and v2 >= v1) or (not hb and v2 <= v1) else bars1[i]
        winner.set_edgecolor("black")
        winner.set_linewidth(2)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ETTh1 — ForecastCF vs Ours (bold border = winner)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, "forecastcf_etth1_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Figure] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="baselines/ForecastCF/results/forecastcf_etth1.csv")
    parser.add_argument("--rl_results", default="assets/results/etth1_v2/rl_cf_v2_etth1_evaluation.json")
    parser.add_argument("--output_dir", default="assets/figures/forecastcf_baseline")
    A = parser.parse_args()

    os.makedirs(A.output_dir, exist_ok=True)

    df = pd.read_csv(A.csv)
    print(f"Loaded {len(df)} rows from {A.csv}")
    print(f"Models: {df['forecast_model'].unique().tolist()}")
    print(f"CF methods: {df['cf_model'].unique().tolist()}")

    # Load RL results
    rl_results = {}
    if os.path.exists(A.rl_results):
        with open(A.rl_results) as f:
            data = json.load(f)
        # Handle both flat and nested JSON formats
        rl_results = {
            "validity_ratio": data.get("validity_ratio",  data.get("Validity Ratio  ↑", {}).get("mean", 0)),
            "stepwise_auc":   data.get("stepwise_auc",    data.get("Step AUC        ↑", {}).get("mean", 0)),
            "proximity_l2":   data.get("proximity_l2",    data.get("Proximity L2    ↓", {}).get("mean", 0)),
            "compactness":    data.get("compactness",      data.get("Compactness     ↑", {}).get("mean", 0)),
        }
        print(f"RL results: {rl_results}")

    # Generate figures
    plot_metrics_bar(df, A.output_dir)

    if rl_results:
        plot_comparison(df, rl_results, A.output_dir)

    print(f"\nAll figures saved to {A.output_dir}/")


if __name__ == "__main__":
    main()
