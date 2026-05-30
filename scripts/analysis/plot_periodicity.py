"""
Plot periodicity analysis for the 3 datasets (ETTh1, ETTh2, Weather).
Shows FFT periodogram with dominant periods clearly marked.
Justifies the choice of seq_len=96 and pred_len=48.

Usage:
    python scripts/analysis/plot_periodicity.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

FIGURES_DIR = "assets/figures/data_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load datasets
# ─────────────────────────────────────────────────────────────────────────────

datasets = {
    "ETTh1": {
        "path": "assets/datasets/ETTh1.csv",
        "target": "OT",
        "freq_label": "hours",
        "freq_minutes": 60,
    },
    "ETTh2": {
        "path": "assets/datasets/ETTh2.csv",
        "target": "OT",
        "freq_label": "hours",
        "freq_minutes": 60,
    },
    "Weather": {
        "path": "assets/datasets/weather.csv",
        "target": "T (degC)",
        "freq_label": "steps (10 min)",
        "freq_minutes": 10,
    },
}


def compute_fft_periods(series, top_k=5):
    """Compute FFT and return top-k dominant periods (after detrending)."""
    from scipy.signal import detrend
    n = len(series)
    # Detrend to remove long-term trend that dominates FFT
    series = detrend(series)
    # FFT
    fft_vals = np.fft.rfft(series)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n)

    # Ignore DC component and very low frequencies (period > n/2)
    power[0] = 0
    # Also ignore periods > 500 steps (not useful for window selection)
    min_freq = 1.0 / 500
    power[freqs < min_freq] = 0

    # Get top-k peaks
    top_indices = np.argsort(power)[::-1][:top_k]
    periods = []
    for idx in top_indices:
        if freqs[idx] > 0:
            periods.append(1.0 / freqs[idx])

    return freqs, power, periods


def compute_autocorrelation(series, max_lag=200):
    """Compute autocorrelation up to max_lag."""
    n = len(series)
    series = series - series.mean()
    var = np.var(series)
    if var < 1e-10:
        return np.zeros(max_lag)

    acf = np.correlate(series, series, mode='full')
    acf = acf[n-1:]  # Keep positive lags only
    acf = acf[:max_lag] / (var * n)
    return acf


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: FFT Periodogram for all 3 datasets
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, (name, info) in enumerate(datasets.items()):
    ax = axes[i]
    df = pd.read_csv(info["path"])
    series = df[info["target"]].values.astype(np.float64)

    # Use first 70% (train set)
    n_train = int(0.7 * len(series))
    series = series[:n_train]

    freqs, power, top_periods = compute_fft_periods(series, top_k=5)

    # Convert frequency to period in timesteps
    periods_axis = np.zeros_like(freqs)
    periods_axis[1:] = 1.0 / freqs[1:]

    # Plot power vs period (only show periods < 500 for clarity)
    mask = (periods_axis > 1) & (periods_axis < 500)
    ax.plot(periods_axis[mask], power[mask] / power[mask].max(),
            lw=1.2, color="tab:blue")

    # Mark dominant periods
    for j, p in enumerate(top_periods[:3]):
        if p < 500:
            # Convert to hours
            hours = p * info["freq_minutes"] / 60
            ax.axvline(p, color="tab:red", ls="--", lw=1.5, alpha=0.7)
            ax.annotate(f"T={p:.0f} steps\n({hours:.0f}h)",
                        xy=(p, 0.85 - j*0.2), fontsize=9, color="tab:red",
                        ha="left")

    # Mark seq_len=96
    ax.axvline(96, color="tab:green", ls="-", lw=2, alpha=0.8)
    ax.annotate("seq_len=96", xy=(96, 0.95), fontsize=10, color="tab:green",
                fontweight="bold", ha="left")

    ax.set_xlabel(f"Period ({info['freq_label']})")
    ax.set_ylabel("Normalized Power")
    ax.set_title(f"{name} — FFT Periodogram (target: {info['target']})")
    ax.set_xlim(0, 400)
    ax.grid(alpha=0.3)

plt.tight_layout()
path1 = os.path.join(FIGURES_DIR, "periodicity_fft.png")
plt.savefig(path1, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Plot] → {path1}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Autocorrelation for all 3 datasets
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i, (name, info) in enumerate(datasets.items()):
    ax = axes[i]
    df = pd.read_csv(info["path"])
    series = df[info["target"]].values.astype(np.float64)

    n_train = int(0.7 * len(series))
    series = series[:n_train]

    max_lag = 300
    acf = compute_autocorrelation(series, max_lag=max_lag)
    lags = np.arange(max_lag)

    ax.plot(lags, acf, lw=1.2, color="tab:blue")
    ax.axhline(0, color="gray", lw=0.5)

    # Find first peak after lag 10 (dominant period)
    # Look for local maxima
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(acf[10:], height=0.1, distance=10)
    peaks = peaks + 10  # offset

    if len(peaks) > 0:
        dominant_period = peaks[0]
        hours = dominant_period * info["freq_minutes"] / 60
        ax.axvline(dominant_period, color="tab:red", ls="--", lw=1.5)
        ax.annotate(f"Period={dominant_period} steps ({hours:.0f}h)",
                    xy=(dominant_period, acf[dominant_period]),
                    xytext=(dominant_period + 10, acf[dominant_period] + 0.1),
                    fontsize=9, color="tab:red",
                    arrowprops=dict(arrowstyle="->", color="tab:red"))

    # Mark seq_len and pred_len
    ax.axvline(96, color="tab:green", ls="-", lw=2, alpha=0.8)
    ax.annotate("seq_len=96", xy=(96, 0.9), fontsize=10, color="tab:green",
                fontweight="bold")
    ax.axvline(48, color="tab:orange", ls="-", lw=2, alpha=0.8)
    ax.annotate("pred_len=48", xy=(48, 0.8), fontsize=10, color="tab:orange",
                fontweight="bold")

    ax.set_xlabel(f"Lag ({info['freq_label']})")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{name} — Autocorrelation Function (target: {info['target']})")
    ax.set_xlim(0, max_lag)
    ax.grid(alpha=0.3)

plt.tight_layout()
path2 = os.path.join(FIGURES_DIR, "periodicity_acf.png")
plt.savefig(path2, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Plot] → {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Periodicity Summary ──────────────────────────────────────")
print(f"{'Dataset':<10} {'Freq':<12} {'Dominant Period':<20} {'In Hours':<12} {'seq_len covers'}")
print("-" * 75)

for name, info in datasets.items():
    df = pd.read_csv(info["path"])
    series = df[info["target"]].values[:int(0.7*len(df))]
    _, _, top_periods = compute_fft_periods(series, top_k=3)

    p = top_periods[0]
    hours = p * info["freq_minutes"] / 60
    cycles = 96 / p
    print(f"{name:<10} {info['freq_minutes']} min      {p:.0f} steps             {hours:.0f}h           {cycles:.1f} cycles")

print(f"\n→ seq_len=96 captures {96/24:.0f} daily cycles for ETTh1/ETTh2")
print(f"→ pred_len=48 = 2 daily cycles (ETTh) = meaningful forecast horizon")


if __name__ == "__main__":
    pass
