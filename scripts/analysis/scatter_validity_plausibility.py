"""
Scatter plot: Validity vs Plausibility Trade-off (iTransformer only)
3rd dimension: point size = Compactness (proximity to original)
One subplot per dataset (ETTh1, ETTh2, Weather).
Output: PNG
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")

# ─── Data: iTransformer only (method, validity, plausibility, compactness) ───────
data = {
    "ETTh1": [
        ("BaseGrad",    0.4781, 0.9199, 0.55),
        ("ForecastCF",  0.7830, 0.6416, 0.45),
        ("RL-MCF",      0.9740, 0.1696, 0.80),
        ("BaseNN",      0.5781, 0.1393, 0.001),
    ],
    "ETTh2": [
        ("BaseGrad",    0.2812, 0.7677, 0.50),
        ("ForecastCF",  0.4990, 0.5540, 0.40),
        ("RL-MCF",      0.8100, 0.0564, 0.82),
        ("BaseNN",      0.9083, 0.1052, 0.001),
    ],
    "Weather": [
        ("BaseGrad",    0.6375, 0.8898, 0.48),
        ("ForecastCF",  0.9099, 0.5720, 0.42),
        ("RL-MCF",      0.8443, 0.2108, 0.75),
        ("BaseNN",      0.9094, 0.2878, 0.001),
    ],
}

# ─── Visual mappings ─────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "RL-MCF":      "#1565C0",
    "ForecastCF":  "#E65100",
    "BaseGrad":    "#C62828",
    "BaseNN":      "#616161",
}

# Size mapping: compactness [0,1] -> marker area [30, 400]
SIZE_MIN = 30
SIZE_MAX = 400


def compactness_to_size(c):
    """Map compactness value to scatter marker size."""
    return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * c


# ─── Figure ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

fig.suptitle(
    "Validity vs Plausibility Trade-off (iTransformer)",
    fontsize=14, fontweight="bold", y=0.99,
)
fig.text(
    0.5, 0.93,
    "(lower-right quadrant = best  |  point size = Compactness)",
    ha="center", fontsize=10, color="gray",
)

for ax, (ds_name, points) in zip(axes, data.items()):
    # Ideal zone shading (high validity, low plausibility)
    rect = mpatches.FancyBboxPatch(
        (0.75, 0.0), 0.25, 0.35,
        boxstyle="round,pad=0.01",
        facecolor="#C8E6C9", edgecolor="none", alpha=0.45, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(0.97, 0.33, "ideal\nzone", fontsize=8, ha="right", va="top",
            color="#2E7D32", fontstyle="italic", alpha=0.8)

    # Quadrant separators
    ax.axvline(x=0.75, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=1)
    ax.axhline(y=0.35, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=1)

    # Plot points
    for method, x, y, compact in points:
        s = compactness_to_size(compact)
        ax.scatter(
            x, y,
            c=METHOD_COLORS[method],
            marker="o",
            s=s,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            zorder=3,
        )

        # Label each point with method name
        offset_x, offset_y = 8, -10
        # Adjust offset to avoid overlap
        if method == "BaseNN" and ds_name == "ETTh1":
            offset_x, offset_y = 6, 8
        elif method == "BaseNN" and ds_name == "ETTh2":
            offset_x, offset_y = -40, 10

        ax.annotate(
            method,
            (x, y),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            fontsize=7.5,
            fontweight="bold",
            color=METHOD_COLORS[method],
            alpha=0.9,
        )

    ax.set_title(ds_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Validity \u2191", fontsize=10)
    ax.set_xlim(0.15, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Plausibility Score \u2193 (lower = more realistic)", fontsize=10)

# ─── Legend ──────────────────────────────────────────────────────────────────────
# Method legend (colors)
method_handles = [
    mpatches.Patch(color=METHOD_COLORS[m], label=m)
    for m in ["RL-MCF", "ForecastCF", "BaseGrad", "BaseNN"]
]

# Size legend (compactness)
size_handles = [
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=4, label="Compact. \u2248 0.00 (BaseNN)"),
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=8, label="Compact. \u2248 0.50"),
    mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                  markersize=13, label="Compact. \u2248 0.80 (RL-MCF)"),
]

spacer = mpatches.Patch(color="none", label="")
all_handles = method_handles + [spacer] + size_handles

fig.legend(
    handles=all_handles,
    loc="center right",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=9,
    frameon=True,
    title="Method / Compactness",
    title_fontsize=9,
)

plt.tight_layout(rect=[0, 0, 0.86, 0.90])

# ─── Save as PNG ─────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(ROOT, "assets", "figures", "analysis")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "scatter_validity_plausibility.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
