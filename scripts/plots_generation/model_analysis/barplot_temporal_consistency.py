"""
Grouped bar chart: Temporal Consistency Across Methods and Datasets (iTransformer only).
Output: PNG
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")

# ─── Data (iTransformer only, from Table 1) ─────────────────────────────────────
datasets = ["ETTh1", "ETTh2", "Weather"]
methods = ["BaseNN", "BaseGrad", "ForecastCF", "RL-MCF"]

# T-Cons. values for iTransformer
values = {
    "BaseNN":      [0.5727, 0.9220, 0.6016],
    "BaseGrad":    [0.6697, 0.7418, 0.5485],
    "ForecastCF":  [0.7635, 0.8904, 0.6151],
    "RL-MCF":      [0.9689, 0.9845, 0.8303],
}

# ─── Visual mappings ─────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "RL-MCF":      "#1565C0",
    "ForecastCF":  "#E65100",
    "BaseGrad":    "#C62828",
    "BaseNN":      "#616161",
}

# ─── Figure ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(datasets))
bar_width = 0.2
n_methods = len(methods)

for i, method in enumerate(methods):
    offset = (i - (n_methods - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset,
        values[method],
        bar_width,
        label=method,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    # Value labels on top
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.012,
            f"{height:.2f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color=METHOD_COLORS[method],
        )

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Temporal Consistency \u2191", fontsize=10)
ax.set_title(
    "Temporal Consistency Across Methods and Datasets (iTransformer)",
    fontsize=13, fontweight="bold", pad=20,
)
ax.text(
    0.5, 1.02,
    "RL-MCF produces the smoothest counterfactuals thanks to temporal mask continuity",
    transform=ax.transAxes, ha="center", fontsize=9.5, color="gray",
)

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# ─── Save ────────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(ROOT, "assets", "figures", "global_analysis")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "barplot_temporal_consistency.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
