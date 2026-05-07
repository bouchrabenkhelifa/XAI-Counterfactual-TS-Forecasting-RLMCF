"""
baselines/ForecastCF/src/forecastcf_evaluator.py

Calcule les 4 métriques du papier ForecastCF (Wang et al., ICDM 2023) :
    1. validity_ratio   — proportion de timesteps dans [alpha, beta]
    2. stepwise_auc     — AUC de la courbe de validité cumulée
    3. proximity_l2     — distance L2 moyenne entre x_orig et x_cf
    4. compactness      — proportion de timesteps quasi-inchangés (atol=1e-2)

Ces métriques sont calculées en utilisant les bornes RL-MCF (alphas/betas)
pour rester cohérent avec CounterfactualEvaluator.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to2d(arr):
    """(B, T, 1) -> (B, T)  |  (B, T) -> (B, T)"""
    return arr[:, :, 0] if arr.ndim == 3 else arr


def _validity_ratio(y_cf, uppers, lowers):
    """Proportion de timesteps dans [lowers, uppers]."""
    yc = _to2d(y_cf)
    lo = _to2d(lowers)
    hi = _to2d(uppers)
    return float(((yc >= lo) & (yc <= hi)).mean())


def _stepwise_auc(y_cf, uppers, lowers):
    """
    AUC de la courbe phi(t) = P(nb_valid_steps >= t).
    Identique à stepwise_validity_auc(mode='proportion') de unified_evaluator.
    """
    yc = _to2d(y_cf)
    lo = _to2d(lowers)
    hi = _to2d(uppers)
    K, T = yc.shape

    valid  = (yc >= lo) & (yc <= hi)
    counts = valid.sum(axis=1)  # [K]

    phi    = np.array([np.mean(counts >= t) for t in range(T + 1)])
    t_norm = np.arange(T + 1) / T

    try:
        return float(np.trapezoid(phi, t_norm))
    except AttributeError:
        return float(np.trapz(phi, t_norm))


def _proximity_l2(x_orig, x_cf):
    """Distance L2 moyenne entre x_orig et x_cf."""
    xo = _to2d(x_orig)
    xc = _to2d(x_cf)
    return float(np.linalg.norm(xo - xc, axis=1).mean())


def _compactness(x_orig, x_cf, atol=1e-2):
    """Proportion de timesteps quasi-inchangés (|x_orig - x_cf| <= atol)."""
    xo = _to2d(x_orig)
    xc = _to2d(x_cf)
    return float((np.abs(xo - xc) <= atol).mean())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_forecastcf_metrics(
    x_orig,
    x_cf,
    y_cf,
    uppers,
    lowers,
    method_name="method",
    print_results=True,
):
    """
    Calcule les 4 métriques du papier ForecastCF.

    Parameters
    ----------
    x_orig      : np.ndarray [N, BH, 1] ou [N, BH]  — série originale
    x_cf        : np.ndarray [N, BH, 1] ou [N, BH]  — série contrefactuelle
    y_cf        : np.ndarray [N, H,  1] ou [N, H]   — forecast du CF
    uppers      : np.ndarray [N, H,  1] ou [N, H]   — borne haute (betas)
    lowers      : np.ndarray [N, H,  1] ou [N, H]   — borne basse (alphas)
    method_name : str  — nom de la méthode (pour affichage)
    print_results : bool — afficher les résultats dans la console

    Returns
    -------
    dict : {
        "validity_ratio": float,
        "stepwise_auc":   float,
        "proximity_l2":   float,
        "compactness":    float,
    }
    """
    results = {
        "validity_ratio": _validity_ratio(y_cf, uppers, lowers),
        "stepwise_auc":   _stepwise_auc(y_cf, uppers, lowers),
        "proximity_l2":   _proximity_l2(x_orig, x_cf),
        "compactness":    _compactness(x_orig, x_cf),
    }

    if print_results:
        print(f"\n-- ForecastCF Metrics [{method_name}] --")
        print(f"  Validity Ratio  : {results['validity_ratio']:.4f}")
        print(f"  Stepwise AUC    : {results['stepwise_auc']:.4f}")
        print(f"  Proximity L2    : {results['proximity_l2']:.4f}")
        print(f"  Compactness     : {results['compactness']:.4f}")

    return results
