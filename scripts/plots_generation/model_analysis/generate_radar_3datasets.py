#!/usr/bin/env python
"""
Generate Radar Charts for 3 Datasets × 5 Architectures
========================================================
Produces one radar chart per dataset showing RL-MCF performance
across 5 forecasting architectures on 6 metrics.

Usage:
    python scripts/generate_radar_3datasets.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = {
    "ETTh1": "assets/results/etth1/summary/etth1_all_models_colab_plaus.json",
    "ETTh2": "assets/results/etth2/summary/etth2_all_models_colab_plaus.json",
    "Weather": "assets/results/weather/summary/weather_all_models_colab_plaus.json",
}

OUT_DIR = "assets/figures/global_analysis"

# ── Metrics to plot ───────────────────────────────────────────────────────────
# For metrics where lower is better, we invert them for the radar
METRICS = [
    ("validity_ratio", "Validity", False),
    ("stepwise_auc", "AUC", False),
    ("compactness", "Compactness", False),
    ("temporal_consistency", "T-Consistency", False),
    ("proximity_l2", "1 - Proximity", True),      # invert: lower is better
    ("plausibility_ensemble", "1 - Plausibility", True),  # invert: lower is better
]

MODELS = ["iTransformer", "PatchTST", "TimesNet", "GRU", "DLinear"]
COLORS = {
    "iTransformer": "#1565C0",
    "PatchTST": "#E65100",
    "TimesNet": "#2E7D32",
    "GRU": "#6A1B9A",
    "DLinear": "#C62828",
}


def load_results(path):
    with open(os.path.join(ROOT, path), "r") as f:
        return json.load(f)


def get_values(data, model_name):
    """Extract metric values for a model, inverting where needed."""
    model_data = data[model_name]
    values = []
    for metric_key, _, invert in METRICS:
        val = model_data[metric_key]["mean"]
        if invert:
            # Normalize: cap at reasonable max before inverting
            val = max(0, 1 - val / 3.0)  # divide by 3 to normalize L2 proximity
        values.append(val)
    return values


def make_radar(dataset_name, data, out_path):
    """Create a single radar chart for one dataset."""
    n_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # Labels
    labels = [m[1] for m in METRICS]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    # Y-axis
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.yaxis.grid(True, color="grey", alpha=0.3, ls="--")
    ax.xaxis.grid(True, color="grey", alpha=0.3, ls="--")

    # Plot each model
    for model in MODELS:
        values = get_values(data, model)
        values += values[:1]  # close
        ax.plot(angles, values, linewidth=2.2, label=model, color=COLORS[model])
        ax.fill(angles, values, alpha=0.08, color=COLORS[model])

    ax.set_title(f"RL-MCF — {dataset_name}", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path}")


def make_radar_combined(all_data, out_path):
    """Create a single figure with 3 radar subplots (one per dataset)."""
    n_metrics = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

    labels = [m[1] for m in METRICS]

    for idx, (dataset_name, data) in enumerate(all_data.items()):
        ax = axes[idx]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
        ax.yaxis.grid(True, color="grey", alpha=0.3, ls="--")
        ax.xaxis.grid(True, color="grey", alpha=0.3, ls="--")

        for model in MODELS:
            values = get_values(data, model)
            values += values[:1]
            ax.plot(angles, values, linewidth=2.0, label=model, color=COLORS[model])
            ax.fill(angles, values, alpha=0.06, color=COLORS[model])

        ax.set_title(dataset_name, fontsize=13, fontweight="bold", pad=15)

    # Single shared legend
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5, fontsize=11,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path}")


def main():
    os.makedirs(os.path.join(ROOT, OUT_DIR), exist_ok=True)

    # Individual radars
    for dataset_name, results_path in RESULTS.items():
        print(f"\n[*] Generating radar for {dataset_name}...")
        data = load_results(results_path)
        out_path = os.path.join(ROOT, OUT_DIR, f"radar_{dataset_name.lower()}_5models.png")
        make_radar(dataset_name, data, out_path)

    # Combined figure (3 radars side by side)
    print("\n[*] Generating combined radar figure...")
    all_data = {name: load_results(path) for name, path in RESULTS.items()}
    combined_path = os.path.join(ROOT, OUT_DIR, "radar_3datasets_combined.png")
    make_radar_combined(all_data, combined_path)

    print("\n[OK] All radar charts generated!")
    print(f"   Output directory: {OUT_DIR}/")


if __name__ == "__main__":
    main()
