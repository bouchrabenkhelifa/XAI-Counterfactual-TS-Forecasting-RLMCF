"""
Generate an illustrative figure of the temporal ramp mask for the PFE report.
Shows:
  - Top: the mask shape (ramp + full region)
  - Bottom: effect on a real series (original vs masked perturbation)

Usage:
    python scripts/analysis/plot_temporal_mask.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

FIGURES_DIR = "assets/figures/global_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parameters
seq_len = 96
mask_last_k = 12
mask_ramp_k = 4


def build_mask(seq_len, last_k, ramp_k):
    """Build the temporal ramp mask."""
    m = np.zeros(seq_len)
    start_full = max(0, seq_len - last_k)
    m[start_full:] = 1.0
    if ramp_k > 0:
        start_ramp = max(0, start_full - ramp_k)
        ramp_len = start_full - start_ramp
        if ramp_len > 0:
            m[start_ramp:start_full] = np.linspace(0.0, 1.0, ramp_len)
    return m


def main():
    mask = build_mask(seq_len, mask_last_k, mask_ramp_k)

    # Simulate a real series and a proposed perturbation
    np.random.seed(42)
    t = np.arange(seq_len)
    x_orig = np.sin(2 * np.pi * t / 24) * 3 + 30 + np.random.randn(seq_len) * 0.3
    x_proposed = x_orig + np.random.randn(seq_len) * 1.5 - 2  # AE decode output (shifted down)

    # Apply mask: x_cf = x_orig + mask * (x_proposed - x_orig)
    delta = x_proposed - x_orig
    x_cf = x_orig + mask * delta

    # ─────────────────────────────────────────────────────────────────────
    # Figure: 2 subplots stacked
    # ─────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2]})

    # ── Top: Mask shape ──────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, 0, mask, color="tab:orange", alpha=0.3)
    ax.plot(t, mask, color="tab:orange", lw=2.5)

    # Annotate regions
    start_full = seq_len - mask_last_k
    start_ramp = start_full - mask_ramp_k

    ax.axvline(start_ramp, color="gray", ls="--", lw=1)
    ax.axvline(start_full, color="gray", ls="--", lw=1)

    # Region labels
    ax.annotate("Preserved\n(m = 0)", xy=(start_ramp / 2, 0.5),
                ha="center", fontsize=10, color="tab:blue", fontweight="bold")
    ax.annotate("Ramp\n(0 → 1)", xy=((start_ramp + start_full) / 2, 0.5),
                ha="center", fontsize=10, color="tab:orange", fontweight="bold")
    ax.annotate("Full perturbation\n(m = 1)", xy=((start_full + seq_len) / 2, 0.5),
                ha="center", fontsize=10, color="tab:red", fontweight="bold")

    # Dimension annotations
    ax.annotate("", xy=(start_ramp, -0.15), xytext=(start_full, -0.15),
                arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=1.5))
    ax.text((start_ramp + start_full) / 2, -0.25, f"ramp_k={mask_ramp_k}",
            ha="center", fontsize=9, color="tab:orange")

    ax.annotate("", xy=(start_full, -0.15), xytext=(seq_len, -0.15),
                arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.5))
    ax.text((start_full + seq_len) / 2, -0.25, f"last_k={mask_last_k}",
            ha="center", fontsize=9, color="tab:red")

    ax.set_xlim(0, seq_len)
    ax.set_ylim(-0.35, 1.1)
    ax.set_ylabel("Mask value m(t)")
    ax.set_title("Temporal Ramp Mask", fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_xticks([])

    # ── Bottom: Effect on series ─────────────────────────────────────────
    ax = axes[1]

    ax.plot(t, x_orig, color="tab:blue", lw=2, label="$x$ (original)")
    ax.plot(t, x_cf, color="tab:red", lw=2, ls="--",
            label="$x_{cf} = x + m \\odot (\\tilde{x} - x)$")

    # Shade the perturbed region
    ax.axvspan(start_ramp, seq_len, alpha=0.08, color="tab:red")
    ax.axvline(start_ramp, color="gray", ls="--", lw=1)
    ax.axvline(start_full, color="gray", ls="--", lw=1)

    # Show delta arrows on a few points in the full region
    for i in range(start_full + 2, seq_len, 3):
        ax.annotate("", xy=(i, x_cf[i]), xytext=(i, x_orig[i]),
                    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1, alpha=0.6))

    ax.set_xlim(0, seq_len)
    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Effect: Historical context preserved, recent steps perturbed", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "temporal_mask_illustration.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {path}")


if __name__ == "__main__":
    main()
