"""
Evaluator aligned exactly with ForecastCF metrics.

Reference:
  "Counterfactual Explanations for Time Series Forecasting"
  Wang et al., ICDM 2023
  https://github.com/zhendong3wang/counterfactual-explanations-for-forecasting

All metric implementations are faithful replicas of their _helper.py:
  - validity_ratio      : fraction of forecast timesteps inside [lower, upper]
  - stepwise_auc        : AUC of cumulative valid-steps curve
  - proximity_l2        : mean L2 distance between x and x_cf
  - compactness         : fraction of unchanged timesteps (atol=0.01, as in their code)

NOTE: compactness uses atol=0.01 (their value), NOT 1e-3 (our unified_evaluator).
      Use this file for any comparison table against ForecastCF results.
"""

import numpy as np
import pandas as pd
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics  (exact replicas of ForecastCF _helper.py)
# ─────────────────────────────────────────────────────────────────────────────

def validity_ratio(y_cf: np.ndarray,
                   upper: np.ndarray,
                   lower: np.ndarray) -> float:
    """
    Fraction of forecast timesteps inside [lower, upper], averaged over samples.

    Shapes: y_cf / upper / lower — (N, H, 1)  or  (N, H)
    """
    valid = np.logical_and(y_cf <= upper, y_cf >= lower)
    return float(valid.mean(axis=1).mean())


def stepwise_auc(y_cf: np.ndarray,
                 upper: np.ndarray,
                 lower: np.ndarray) -> float:
    """
    AUC of the cumulative valid-steps curve.
    Replica of ForecastCF cumulative_valid_steps() → trapz AUC.

    Shapes: y_cf / upper / lower — (N, H, 1)
    """
    n_samples, n_steps, _ = y_cf.shape
    in_band = np.logical_and(y_cf <= upper, y_cf >= lower)[:, :, 0]  # (N, H)

    until_steps = np.zeros(n_samples)
    for i in range(n_samples):
        count = 0
        for s in range(n_steps):
            if in_band[i, s]:
                count += 1
                until_steps[i] = count
            else:
                until_steps[i] = count
                break

    valid_steps, counts = np.unique(until_steps, return_counts=True)
    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))

    # fillna — exact replica of fillna_cumsum_counts()
    df = pd.DataFrame(
        [{k: v for k, v in zip(valid_steps, cumsum_counts)}],
        columns=list(range(0, n_steps + 1)),
    )
    df = df.sort_index(ascending=True, axis=1)
    df = df.fillna(method="backfill", axis=1)
    df = df.fillna(value=0)
    vs = df.columns.to_numpy()
    cc = df.values[0]

    auc = float(np.trapz(cc[1:] / n_samples, vs[1:] / n_steps))
    return auc


def proximity_l2(x_orig: np.ndarray,
                 x_cf: np.ndarray) -> float:
    """
    Mean L2 distance per sample.
    Replica of ForecastCF euclidean_distance().

    Shapes: x_orig / x_cf — (N, BH, 1)  or  (N, BH)
    """
    # flatten to (N, BH) if needed
    if x_orig.ndim == 3:
        x_orig = x_orig[:, :, 0]
        x_cf   = x_cf[:, :, 0]
    return float(np.linalg.norm(x_orig - x_cf, axis=1).mean())


def compactness(x_orig: np.ndarray,
                x_cf: np.ndarray,
                atol: float = 0.01) -> float:
    """
    Fraction of unchanged timesteps.
    Replica of ForecastCF compactness_score(atol=0.01).

    Shapes: x_orig / x_cf — (N, BH, 1)  or  (N, BH)
    """
    c = np.isclose(x_orig, x_cf, atol=atol)
    return float(np.mean(c, axis=1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Bounds generation  (exact replica of ForecastCF generate_bounds)
# ─────────────────────────────────────────────────────────────────────────────

def polynomial_values(shift: float, change_percent: float,
                      poly_order: int, horizon: int) -> np.ndarray:
    """Replica of ForecastCF polynomial_values()."""
    if horizon == 1:
        return np.asarray([shift + change_percent])
    p_orders = [shift] + [0] * poly_order
    p_orders[-1] = change_percent / ((horizon - 1) ** poly_order)
    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    return np.asarray([np.polyval(p_coefs, i) for i in range(horizon)])


def generate_bounds(input_series: np.ndarray,
                    horizon: int,
                    shift: float = 0.0,
                    change_percent: float = -0.1,
                    poly_order: int = 1,
                    fraction_std: float = 1.0,
                    center: str = "median"):
    """
    Exact replica of ForecastCF generate_bounds().

    input_series : (BH, 1)  numpy array
    Returns upper, lower : (horizon, 1) numpy arrays
    """
    if center == "median":
        sv = np.median(input_series)
    elif center == "mean":
        sv = np.mean(input_series)
    elif center == "last":
        sv = float(input_series[-1, 0])
    else:
        sv = np.median(input_series)

    std  = np.std(input_series)
    poly = polynomial_values(shift, change_percent, poly_order, horizon)

    upper = sv * (1 + poly + fraction_std * std)
    lower = sv * (1 + poly - fraction_std * std)

    # swap if sv < 0  (their implicit behaviour via multiplication)
    if sv < 0:
        upper, lower = lower, upper

    return upper[:, np.newaxis], lower[:, np.newaxis]   # (H, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Unified evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_forecastcf_metrics(x_orig: np.ndarray,
                                 x_cf: np.ndarray,
                                 y_cf: np.ndarray,
                                 uppers: np.ndarray,
                                 lowers: np.ndarray,
                                 runtime_seconds: float = None,
                                 method_name: str = "Method",
                                 print_results: bool = True) -> dict:
    """
    Compute all ForecastCF metrics and return as dict.

    Parameters
    ----------
    x_orig  : (N, BH, 1)
    x_cf    : (N, BH, 1)
    y_cf    : (N, H,  1)
    uppers  : (N, H,  1)
    lowers  : (N, H,  1)
    """
    vr   = validity_ratio(y_cf, uppers, lowers)
    sauc = stepwise_auc(y_cf, uppers, lowers)
    prox = proximity_l2(x_orig, x_cf)
    comp = compactness(x_orig, x_cf)

    results = {
        "validity_ratio": vr,
        "stepwise_auc":   sauc,
        "proximity_l2":   prox,
        "compactness":    comp,
    }
    if runtime_seconds is not None:
        results["runtime_seconds"] = runtime_seconds

    if print_results:
        W = 60
        print(f"\n{'=' * W}")
        print(f"  {method_name}")
        print(f"{'=' * W}")
        print(f"  -- ForecastCF Paper Metrics (atol=0.01 for compactness)")
        print(f"  {'Validity Ratio  ↑':<36}: {vr:.4f}")
        print(f"  {'Step AUC        ↑':<36}: {sauc:.4f}")
        print(f"  {'Proximity L2    ↓':<36}: {prox:.4f}")
        print(f"  {'Compactness     ↑':<36}: {comp:.4f}")
        if runtime_seconds is not None:
            print(f"  {'Runtime (s)':<36}: {runtime_seconds:.1f}")
        print(f"{'=' * W}\n")

    return results


def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: round(float(v), 6) if isinstance(v, float) else v
                   for k, v in results.items()}, f, indent=4)
    print(f"[Saved] {path}")


def print_comparison_table(results_dict: dict, width: int = 72) -> None:
    """
    Print a side-by-side comparison table.

    results_dict : { "MethodName": {metric: value, ...}, ... }
    """
    methods   = list(results_dict.keys())
    col_w     = 14
    metrics   = [
        ("validity_ratio", "Validity Ratio  ↑"),
        ("stepwise_auc",   "Step AUC        ↑"),
        ("proximity_l2",   "Proximity L2    ↓"),
        ("compactness",    "Compactness     ↑"),
        ("runtime_seconds","Runtime (s)      "),
    ]

    header = f"  {'Metric':<36}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(f"\n{'=' * width}")
    print(f"  COMPARISON TABLE  (ForecastCF metrics, atol=0.01)")
    print(f"{'=' * width}")
    print(header)
    print("-" * width)

    for key, display in metrics:
        row = f"  {display:<36}"
        for m in methods:
            v = results_dict[m].get(key, float("nan"))
            if isinstance(v, float):
                row += f"{v:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)

    print(f"{'=' * width}\n")
