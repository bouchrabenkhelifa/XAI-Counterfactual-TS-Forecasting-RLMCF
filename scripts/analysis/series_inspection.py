"""
Generate time series plots and STL decomposition (Trend, Seasonal, Residual)
for ETTh1, ETTh2, and Weather datasets.

Usage:
    python plot_datasets.py
    python plot_datasets.py --data_root /path/to/data
    python plot_datasets.py --n_points 2000 --target OT
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import STL

# ─────────────────────────── colour palette ──────────────────────────────────
PALETTE = {
    "ETTh1":   "#2196F3",
    "ETTh2":   "#4CAF50",
    "Weather": "#FF5722",
}

DECOMP_COLORS = {
    "original": "#455A64",
    "trend":    "#E91E63",
    "seasonal": "#FF9800",
    "residual": "#9C27B0",
}


# ─────────────────────────── helpers ─────────────────────────────────────────

def load_series(path: str, target_col: str) -> pd.Series:
    """Load a CSV and return the target column as a pd.Series with DatetimeIndex."""
    df = pd.read_csv(path)

    # Try to parse a date column
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "datetime", "timestamp", "time"):
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    if target_col not in df.columns:
        available = list(df.columns)
        raise ValueError(
            f"Column '{target_col}' not found. Available: {available}"
        )

    series = df[target_col].dropna()
    return series


def find_csv(data_root: str, dataset: str) -> str | None:
    """Try common file-naming patterns for each dataset."""
    candidates = {
        "ETTh1":   ["ETTh1.csv", "etth1.csv", "ETT-small/ETTh1.csv"],
        "ETTh2":   ["ETTh2.csv", "etth2.csv", "ETT-small/ETTh2.csv"],
        "Weather": ["weather.csv", "Weather.csv", "WTH.csv", "wth.csv"],
    }
    for name in candidates.get(dataset, []):
        full = os.path.join(data_root, name)
        if os.path.exists(full):
            return full
    return None


def infer_period(series: pd.Series) -> int:
    """Guess a reasonable seasonal period from index frequency."""
    if hasattr(series.index, "freq") and series.index.freq is not None:
        freq = series.index.freq.name
        if "H" in freq:
            return 24        # hourly → daily
        if "10T" in freq or "10min" in freq:
            return 6 * 24   # 10-min → daily
        if "T" in freq or "min" in freq:
            return 60 * 24
    return 24   # default


def stl_decompose(series: pd.Series, period: int):
    """Run STL and return (trend, seasonal, residual) as arrays."""
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid


# ─────────────────────────── plot: raw series ────────────────────────────────

def plot_raw_series(datasets: dict, n_points: int, output_dir: str):
    """
    One figure with three panels (one per dataset) showing the raw time series.
    """
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(16, 4 * n_ds), sharex=False)
    if n_ds == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, datasets.items()):
        s = series.iloc[:n_points]
        x = np.arange(len(s))

        color = PALETTE.get(name, "#607D8B")
        ax.plot(x, s.values, color=color, lw=0.9, alpha=0.85)
        ax.fill_between(x, s.values, alpha=0.12, color=color)

        ax.set_title(f"{name}  —  {s.name}  ({len(s):,} points)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time step", fontsize=10)
        ax.set_ylabel(s.name, fontsize=10)
        ax.grid(True, alpha=0.25)

        # annotate basic stats
        stats = f"mean={s.mean():.2f}  std={s.std():.2f}  min={s.min():.2f}  max={s.max():.2f}"
        ax.annotate(stats, xy=(0.01, 0.96), xycoords="axes fraction",
                    fontsize=8, va="top", color="dimgray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6))

    fig.suptitle("Time Series Overview", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = os.path.join(output_dir, "01_raw_series.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Raw series    → {out}")


# ─────────────────────────── plot: decomposition ─────────────────────────────

def plot_decomposition(name: str, series: pd.Series, n_points: int,
                       period: int, output_dir: str):
    """
    4-panel STL decomposition plot for a single dataset.
    """
    s = series.iloc[:n_points]
    trend, seasonal, residual = stl_decompose(s, period)

    components = [
        ("Original",  s.values,   DECOMP_COLORS["original"]),
        ("Trend",     trend,       DECOMP_COLORS["trend"]),
        ("Seasonal",  seasonal,    DECOMP_COLORS["seasonal"]),
        ("Residual",  residual,    DECOMP_COLORS["residual"]),
    ]

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)
    x   = np.arange(len(s))

    for idx, (label, values, color) in enumerate(components):
        ax = fig.add_subplot(gs[idx])
        ax.plot(x, values, color=color, lw=0.85, alpha=0.9)

        if label in ("Original", "Trend"):
            ax.fill_between(x, values, alpha=0.10, color=color)
        else:
            ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
            ax.fill_between(x, values, 0, alpha=0.15, color=color)

        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.22)

        # Variance annotation for decomposed components
        if label != "Original":
            pct = np.var(values) / np.var(s.values) * 100
            ax.annotate(f"var share: {pct:.1f}%",
                        xy=(0.99, 0.94), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8, color="dimgray",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.6))

    axes_list = fig.get_axes()
    axes_list[-1].set_xlabel("Time step", fontsize=10)

    color_ds = PALETTE.get(name, "#607D8B")
    fig.suptitle(
        f"{name}  —  STL Decomposition  ({s.name},  period={period})",
        fontsize=13, fontweight="bold", color=color_ds
    )

    out = os.path.join(output_dir, f"02_decomposition_{name.lower()}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Decomposition  → {out}  (period={period})")


# ─────────────────────────── plot: side-by-side decomp ───────────────────────

def plot_decomposition_comparison(datasets: dict, n_points: int,
                                  periods: dict, output_dir: str):
    """
    Grid: rows = components (Original / Trend / Seasonal / Residual),
          cols = datasets.
    """
    ds_names = list(datasets.keys())
    n_cols    = len(ds_names)
    labels    = ["Original", "Trend", "Seasonal", "Residual"]

    fig, axes = plt.subplots(4, n_cols, figsize=(7 * n_cols, 12), sharex=False)

    for col, name in enumerate(ds_names):
        s      = datasets[name].iloc[:n_points]
        period = periods.get(name, 24)
        trend, seasonal, residual = stl_decompose(s, period)

        values_map = {
            "Original": s.values,
            "Trend":    trend,
            "Seasonal": seasonal,
            "Residual": residual,
        }

        color_ds = PALETTE.get(name, "#607D8B")

        for row, label in enumerate(labels):
            ax     = axes[row, col] if n_cols > 1 else axes[row]
            vals   = values_map[label]
            color  = DECOMP_COLORS[label.lower()]
            x      = np.arange(len(vals))

            ax.plot(x, vals, color=color, lw=0.8, alpha=0.88)

            if label in ("Original", "Trend"):
                ax.fill_between(x, vals, alpha=0.10, color=color)
            else:
                ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.4)
                ax.fill_between(x, vals, 0, alpha=0.14, color=color)

            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(name, fontsize=12, fontweight="bold", color=color_ds, pad=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Time step", fontsize=9)

    fig.suptitle(
        "STL Decomposition Comparison  —  ETTh1 · ETTh2 · Weather",
        fontsize=14, fontweight="bold", y=1.005
    )
    fig.tight_layout()

    out = os.path.join(output_dir, "03_decomposition_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Comparison     → {out}")


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ETTh1/ETTh2/Weather time series & decomposition")
    parser.add_argument("--data_root", default="dataset",
                        help="Root folder containing the CSV files (default: dataset/)")
    parser.add_argument("--target",    default="OT",
                        help="Target column name (default: OT)")
    parser.add_argument("--n_points",  type=int, default=2000,
                        help="Number of time steps to plot (default: 2000)")
    parser.add_argument("--output_dir", default="plots/dataset_overview",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Locate CSV files ──────────────────────────────────────────────────────
    dataset_names = ["ETTh1", "ETTh2", "Weather"]
    datasets      = {}
    periods       = {}

    print("\n" + "="*65)
    print("  Loading datasets …")
    print("="*65)

    for name in dataset_names:
        path = find_csv(args.data_root, name)

        if path is None:
            print(f"  ⚠  {name}: CSV not found under '{args.data_root}' — skipping.")
            continue

        try:
            series = load_series(path, args.target)
        except ValueError as e:
            print(f"  ⚠  {name}: {e}")
            continue

        datasets[name] = series
        periods[name]  = infer_period(series)
        print(f"  ✓  {name:<10}  {path}  |  {len(series):,} rows  |  col='{args.target}'  |  period≈{periods[name]}")

    if not datasets:
        print("\n  ✗ No datasets loaded. Check --data_root and --target.\n")
        return

    print("\n" + "="*65)
    print("  Generating figures …")
    print("="*65)

    # 1 — Raw series overview
    plot_raw_series(datasets, args.n_points, args.output_dir)

    # 2 — Individual decompositions
    for name, series in datasets.items():
        plot_decomposition(name, series, args.n_points, periods[name], args.output_dir)

    # 3 — Side-by-side comparison (only if ≥ 2 datasets)
    if len(datasets) >= 2:
        plot_decomposition_comparison(datasets, args.n_points, periods, args.output_dir)

    print("\n" + "="*65)
    print(f"  All figures saved in:  {args.output_dir}/")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()