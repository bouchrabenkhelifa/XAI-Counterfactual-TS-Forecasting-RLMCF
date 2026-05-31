#!/usr/bin/env python
"""
Compare RLMCF vs ForecastCF on ETTh1 GRU.
Génère les figures de comparaison via baselines/compare_results.py.

Usage:
    python scripts/compare_rlmcf_vs_forecastcf.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# ── Chemins ───────────────────────────────────────────────────────────────────
RLMCF_JSON  = "assets/results/etth1/RL/gru/rl_cf_gru_etth1_evaluation.json"
FCF_CSV     = "baselines/ForecastCF/results/forecastcf_etth1_gru.csv"
OUT_DIR     = "assets/figures/global_analysis"
RLMCF_CSV   = "assets/results/etth1/RL/gru/rl_cf_gru_etth1_evaluation.csv"


def json_to_csv(json_path, csv_path):
    """Convertit le JSON RLMCF en CSV compatible avec compare_results.py."""
    with open(json_path) as f:
        data = json.load(f)

    row = {
        "dataset":          "etth1",
        "forecast_model":   "gru",
        "cf_model":         "RLMCF",
        "validity_ratio":   data["validity_ratio"]["mean"],
        "step_validity_auc":data["stepwise_auc"]["mean"],
        "proximity":        data["proximity_l2"]["mean"],
        "compactness":      data["compactness"]["mean"],
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ RLMCF CSV → {csv_path}")
    return df


def plot_comparison(rl_df, fcf_df, output_dir):
    """Figure de comparaison RLMCF vs ForecastCF — style publication."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = [
        ("validity_ratio",    "Validity Ratio ↑",   True),
        ("step_validity_auc", "Step AUC ↑",          True),
        ("proximity",         "Proximity L2 ↓",      False),
        ("compactness",       "Compactness ↑",        True),
    ]

    # Moyennes et std
    rl_means  = {m: float(rl_df[m].mean())  for m, _, _ in metrics if m in rl_df.columns}
    rl_stds   = {m: float(rl_df[m].std())   for m, _, _ in metrics if m in rl_df.columns}
    fcf_means = {m: float(fcf_df[m].mean()) for m, _, _ in metrics if m in fcf_df.columns}
    fcf_stds  = {m: float(fcf_df[m].std())  for m, _, _ in metrics if m in fcf_df.columns}

    print("\n── Comparison Table ─────────────────────────────────────────")
    print(f"{'Metric':<25} {'RLMCF (GRU)':<20} {'ForecastCF (GRU)':<22} {'Winner'}")
    print("-" * 80)
    for m, label, higher in metrics:
        rm = rl_means.get(m, float("nan"))
        rs = rl_stds.get(m, 0.0)
        fm = fcf_means.get(m, float("nan"))
        fs = fcf_stds.get(m, 0.0)
        winner = "RLMCF ✓" if (higher and rm > fm) or (not higher and rm < fm) else "ForecastCF"
        impr = ((rm - fm) / (abs(fm) + 1e-8)) * 100 if higher else ((fm - rm) / (abs(fm) + 1e-8)) * 100
        print(f"{label:<25} {rm:.4f} ± {rs:.4f}   {fm:.4f} ± {fs:.4f}   {winner}  ({impr:+.1f}%)")
    print("-" * 80)

    # ── Figure grouped barplot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = {"RLMCF": "#C62828", "ForecastCF": "#E65100"}
    width = 0.35

    for ax, (m, label, higher) in zip(axes, metrics):
        rm  = rl_means.get(m, 0)
        rs  = rl_stds.get(m, 0)
        fm  = fcf_means.get(m, 0)
        fs  = fcf_stds.get(m, 0)

        x = np.array([0, 1])
        bars = ax.bar(x, [rm, fm], width=0.5,
                      color=[colors["RLMCF"], colors["ForecastCF"]],
                      yerr=[rs, fs], capsize=5, alpha=0.85, edgecolor="white")

        # Marquer le gagnant
        winner_idx = 0 if (higher and rm >= fm) or (not higher and rm <= fm) else 1
        bars[winner_idx].set_edgecolor("black")
        bars[winner_idx].set_linewidth(2)

        for bar, v in zip(bars, [rm, fm]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["RLMCF", "ForecastCF"], fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(rm, fm) * 1.3 + 0.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    out = os.path.join(output_dir, "rlmcf_vs_forecastcf_barplot.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Barplot → {out}")


def main():
    print("=" * 60)
    print("RLMCF vs ForecastCF — ETTh1 GRU")
    print("=" * 60)

    # Charger RLMCF
    rl_df = json_to_csv(RLMCF_JSON, RLMCF_CSV)
    print(f"\nRLMCF results:")
    print(rl_df[["validity_ratio", "step_validity_auc", "proximity", "compactness"]].to_string(index=False))

    # Charger ForecastCF
    fcf_df = pd.read_csv(FCF_CSV)
    print(f"\nForecastCF results ({len(fcf_df)} seeds):")
    print(fcf_df[["validity_ratio", "step_validity_auc", "proximity", "compactness"]].to_string(index=False))

    # Générer la figure
    plot_comparison(rl_df, fcf_df, OUT_DIR)

    print(f"\n✅ Done — figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
