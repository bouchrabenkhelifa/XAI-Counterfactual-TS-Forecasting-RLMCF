#!/usr/bin/env python
"""
Generate dataset presentation plots for PFE report.

For each dataset (ETTh1, ETTh2, Weather):
  - Full time series overview with train/val/test split
  - Seasonal decomposition (trend, seasonal, residual)
  - Key statistics annotation

Usage:
    python scripts/analysis/plot_dataset_presentation.py

Output:
    assets/figures/global_analysis/dataset_presentation_etth1.png
    assets/figures/global_analysis/dataset_presentation_etth2.png
    assets/figures/global_analysis/dataset_presentation_weather.png
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# Output directory
OUTPUT_DIR = os.path.join(ROOT, "assets", "figures", "global_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Dataset configurations ─────────────────────────────────────────────────────
DATASETS = {
    "ETTh1": {
        "path": os.path.join(ROOT, "assets", "datasets", "ETTh1.csv"),
        "target_col": "OT",
        "description": "Electricity Transformer Temperature (Station 1)",
        "unit": "°C",
        "freq": "1h",
        "period": 24,  # 24h daily cycle
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "characteristics": [
            "Hourly sampling (1 point/hour)",
            "7 variables (6 power load features + OT)",
            "Strong daily & weekly seasonality",
            "High volatility (industrial load patterns)",
        ],
    },
    "ETTh2": {
        "path": os.path.join(ROOT, "assets", "datasets", "ETTh2.csv"),
        "target_col": "OT",
        "description": "Electricity Transformer Temperature (Station 2)",
        "unit": "°C",
        "freq": "1h",
        "period": 24,  # 24h daily cycle
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "characteristics": [
            "Hourly sampling (1 point/hour)",
            "7 variables (6 power load features + OT)",
            "Similar structure to ETTh1, different station",
            "Different volatility profile",
        ],
    },
    "Weather": {
        "path": os.path.join(ROOT, "assets", "datasets", "weather.csv"),
        "target_col": "T (degC)",
        "description": "Weather Station — Temperature",
        "unit": "°C",
        "freq": "10min",
        "period": 144,  # 144 samples/day (every 10 min)
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "characteristics": [
            "10-minute sampling (144 points/day)",
            "21 meteorological variables",
            "Strong daily seasonality",
            "Lower volatility (natural process)",
        ],
    },
}


def load_dataset(config):
    """Load and prepare dataset."""
    df = pd.read_csv(config["path"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_split_indices(n, train_ratio, val_ratio):
    """Compute train/val/test split indices."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def generate_dataset_plot(name, config):
    """Generate a full presentation plot for one dataset."""

    print(f"\n{'─'*60}")
    print(f"  Processing: {name}")
    print(f"{'─'*60}")

    # Load data
    df = load_dataset(config)
    target = df[config["target_col"]].values
    dates = df["date"].values
    n = len(df)

    # Split indices
    train_end, val_end = compute_split_indices(
        n, config["train_ratio"], config["val_ratio"]
    )

    # ─── Seasonal decomposition ──────────────────────────────────────────────
    # Use a subset for decomposition to avoid memory issues on large datasets
    max_decomp_samples = min(n, 5000)
    decomp_start = train_end - max_decomp_samples if train_end > max_decomp_samples else 0
    decomp_series = pd.Series(
        target[decomp_start:train_end],
        index=pd.date_range(
            start=df["date"].iloc[decomp_start],
            periods=train_end - decomp_start,
            freq=config["freq"],
        ),
    )

    decomposition = seasonal_decompose(
        decomp_series, model="additive", period=config["period"]
    )

    # ─── Statistics ──────────────────────────────────────────────────────────
    stats = {
        "N": n,
        "Variables": len(df.columns) - 1,
        "Mean": target.mean(),
        "Std": target.std(),
        "Min": target.min(),
        "Max": target.max(),
        "Autocorr(1)": pd.Series(target).autocorr(lag=1),
        "Train": f"{config['train_ratio']*100:.0f}%",
        "Val": f"{config['val_ratio']*100:.0f}%",
        "Test": f"{config['test_ratio']*100:.0f}%",
    }

    # ─── Create figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), constrained_layout=False)
    fig.suptitle(
        f"{name} — {config['description']}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Grid: 4 rows
    # Row 0: Full series with splits
    # Row 1: Trend
    # Row 2: Seasonal
    # Row 3: Residual
    gs = fig.add_gridspec(4, 1, hspace=0.35, top=0.93, bottom=0.06, left=0.08, right=0.75)

    # ─── Row 0: Full time series with train/val/test ─────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(dates[:train_end], target[:train_end], color="#1976D2", lw=0.6, label="Train")
    ax0.plot(dates[train_end:val_end], target[train_end:val_end], color="#FF9800", lw=0.6, label="Validation")
    ax0.plot(dates[val_end:], target[val_end:], color="#E53935", lw=0.6, label="Test")

    # Vertical split lines
    ax0.axvline(dates[train_end], color="black", ls="--", lw=1, alpha=0.7)
    ax0.axvline(dates[val_end], color="black", ls="--", lw=1, alpha=0.7)

    ax0.set_title("Time Series — Target Variable with Train/Val/Test Split", fontsize=11, fontweight="bold")
    ax0.set_ylabel(f"{config['target_col']} ({config['unit']})")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # ─── Row 1: Trend ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(decomposition.trend.dropna(), color="#4CAF50", lw=1.2)
    ax1.set_title("Trend Component", fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"{config['unit']}")
    ax1.grid(True, alpha=0.3)

    # ─── Row 2: Seasonal ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    # Show only a few periods for clarity
    seasonal_data = decomposition.seasonal.dropna()
    n_show = min(len(seasonal_data), config["period"] * 5)
    ax2.plot(seasonal_data.iloc[:n_show], color="#9C27B0", lw=0.8)
    ax2.set_title(f"Seasonal Component (period = {config['period']} steps, showing {n_show} steps)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(f"{config['unit']}")
    ax2.grid(True, alpha=0.3)

    # ─── Row 3: Residual ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.plot(decomposition.resid.dropna(), color="#607D8B", lw=0.5, alpha=0.8)
    ax3.set_title("Residual Component", fontsize=11, fontweight="bold")
    ax3.set_ylabel(f"{config['unit']}")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)

    # ─── Side panel: Statistics & Characteristics ────────────────────────────
    # Add a text box on the right side
    text_lines = [
        f"{'─'*30}",
        f"  Dataset Statistics",
        f"{'─'*30}",
        f"  Samples:      {stats['N']:,}",
        f"  Variables:    {stats['Variables']}",
        f"  Mean:         {stats['Mean']:.2f} {config['unit']}",
        f"  Std:          {stats['Std']:.2f} {config['unit']}",
        f"  Min:          {stats['Min']:.2f} {config['unit']}",
        f"  Max:          {stats['Max']:.2f} {config['unit']}",
        f"  Autocorr(1):  {stats['Autocorr(1)']:.4f}",
        f"",
        f"{'─'*30}",
        f"  Split Ratios",
        f"{'─'*30}",
        f"  Train: {stats['Train']} ({train_end:,})",
        f"  Val:   {stats['Val']} ({val_end - train_end:,})",
        f"  Test:  {stats['Test']} ({n - val_end:,})",
        f"",
        f"{'─'*30}",
        f"  Characteristics",
        f"{'─'*30}",
    ]
    for c in config["characteristics"]:
        text_lines.append(f"  • {c}")

    info_text = "\n".join(text_lines)

    fig.text(
        0.78, 0.5, info_text,
        fontsize=8.5,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD", alpha=0.9),
    )

    # ─── Save ────────────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, f"dataset_presentation_{name.lower()}.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Saved: {output_path}")
    print(f"    Samples: {n:,} | Variables: {stats['Variables']} | "
          f"Mean: {stats['Mean']:.2f} | Std: {stats['Std']:.2f}")

    return True


def main():
    print("\n" + "=" * 70)
    print("  DATASET PRESENTATION PLOTS FOR PFE REPORT")
    print("  (Time Series + Seasonal Decomposition)")
    print("=" * 70)

    success_count = 0

    for name, config in DATASETS.items():
        if not os.path.exists(config["path"]):
            print(f"\n  ✗ {name}: Dataset not found at {config['path']}")
            continue

        try:
            generate_dataset_plot(name, config)
            success_count += 1
        except Exception as e:
            print(f"\n  ✗ {name}: Error → {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  Done! {success_count}/{len(DATASETS)} plots generated.")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
