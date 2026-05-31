"""
Grouped bar chart: Compactness Across Methods and Datasets.
Output: PNG
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")

# ─── Data ────────────────────────────────────────────────────────────────────────
datasets = ["ETTh1", "ETTh2", "Weather"]
methods = ["BaseNN", "BaseGrad", "ForecastCF", "RL-MCF"]

# Mean compactness per method per dataset
values = {
    "BaseNN":      [0.0060, 0.0000, 0.0016],
    "BaseGrad":    [0.0118, 0.0107, 0.0112],
    "ForecastCF":  [0.0125, 0.0167, 0.0128],
    "RL-MCF":      [0.7417, 0.7609, 0.8037],
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
            height + 0.015,
            f"{height:.2f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold",
            color=METHOD_COLORS[method],
        )

# Baselines ceiling line
ax.axhline(y=0.05, color="#C62828", ls="--", lw=1.2, alpha=0.7, zorder=1)
ax.text(2.45, 0.065, "baselines ceiling", fontsize=8, color="#C62828",
        ha="right", fontstyle="italic")

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Compactness \u2191 (fraction of unmodified timesteps)", fontsize=10)
ax.set_title(
    "Compactness Across Methods and Datasets",
    fontsize=13, fontweight="bold", pad=20,
)
ax.text(
    0.5, 1.02,
    "RL-MCF is the only method achieving meaningful compactness (temporal mask)",
    transform=ax.transAxes, ha="center", fontsize=9.5, color="gray",
)

ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# ─── Save ────────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
output_dir = os.path.join(ROOT, "assets", "figures", "global_analysis")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "barplot_compactness.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
