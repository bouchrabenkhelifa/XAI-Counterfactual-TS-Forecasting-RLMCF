from __future__ import annotations

import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Optional


def _to2d(arr: np.ndarray) -> np.ndarray:
    """(B, T, 1) -> (B, T)  |  (B, T) -> (B, T)"""
    return arr[:, :, 0] if arr.ndim == 3 else arr


# =============================================================================
#  (a) Validity Ratio
# =============================================================================

def validity_ratio(
    preds_cf: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
) -> float:
    pc = _to2d(preds_cf)
    a  = _to2d(alphas)
    b  = _to2d(betas)
    return float(((pc >= a) & (pc <= b)).mean())


# =============================================================================
#  (b) Stepwise Validity AUC
# =============================================================================

def stepwise_validity_auc(
    preds_cf: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    mode: str = "proportion",
) -> float:
    pc = _to2d(preds_cf)
    a  = _to2d(alphas)
    b  = _to2d(betas)
    K, T = pc.shape

    if mode == "prefix":
        counts = np.zeros(K, dtype=int)
        for k in range(K):
            for t in range(T):
                if a[k, t] <= pc[k, t] <= b[k, t]:
                    counts[k] += 1
                else:
                    break

    elif mode == "best_run":
        counts = np.zeros(K, dtype=int)
        for k in range(K):
            best = cur = 0
            for t in range(T):
                if a[k, t] <= pc[k, t] <= b[k, t]:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            counts[k] = best

    else:  # "proportion"
        valid  = (pc >= a) & (pc <= b)
        counts = valid.sum(axis=1)

    phi    = np.array([np.mean(counts >= t) for t in range(T + 1)])
    t_norm = np.arange(T + 1) / T

    try:
        return float(np.trapezoid(phi, t_norm))
    except AttributeError:
        try:
            return float(np.trapz(phi, t_norm))
        except AttributeError:
            from scipy.integrate import trapezoid
            return float(trapezoid(phi, t_norm))


# =============================================================================
#  (c) Proximity
# =============================================================================

def proximity(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    norm: str = "l2",
) -> float:
    xo   = _to2d(X_orig)
    xc   = _to2d(X_cf)
    diff = np.abs(xo - xc)

    if norm == "l1":
        return float(diff.mean())
    elif norm == "linf":
        return float(diff.max(axis=1).mean())
    else:  # l2
        return float(np.linalg.norm(xo - xc, axis=1).mean())


# =============================================================================
#  (d) Compactness
# =============================================================================

def compactness(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    tol: float = 1e-3,
) -> float:
    xo = _to2d(X_orig)
    xc = _to2d(X_cf)
    return float((np.abs(xo - xc) <= tol).mean())


# =============================================================================
#  (e) Roughness
# =============================================================================

def roughness(arr: np.ndarray) -> tuple[float, float]:
    a = _to2d(arr)
    r = np.abs(np.diff(a, axis=1)).mean(axis=1)
    return float(r.mean()), float(r.std())


# =============================================================================
#  (f) Temporal Consistency
# =============================================================================

def temporal_consistency(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
) -> tuple[float, float]:
    xo  = _to2d(X_orig)
    xc  = _to2d(X_cf)
    out = []
    for i in range(len(xo)):
        dx, dc = np.diff(xo[i]), np.diff(xc[i])
        if dx.std() < 1e-8 or dc.std() < 1e-8:
            out.append(1.0)
        else:
            r, _ = pearsonr(dx, dc)
            out.append(float(np.clip((r + 1) / 2, 0, 1)))
    arr = np.array(out)
    return float(arr.mean()), float(arr.std())


# =============================================================================
#  (g) Relative Reduction
# =============================================================================

def relative_reduction(
    Y_hat: np.ndarray,
    Y_cf: np.ndarray,
) -> tuple[float, float]:
    yh  = _to2d(Y_hat)
    yc  = _to2d(Y_cf)
    arr = (yh.mean(axis=1) - yc.mean(axis=1)) / (np.abs(yh.mean(axis=1)) + 1e-8)
    return float(arr.mean()), float(arr.std())


# =============================================================================
#  (h) Plausibility
# =============================================================================

class PlausibilityEvaluator:

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.random_state  = random_state
        self._fitted       = False

    def fit(self, X_train: np.ndarray) -> "PlausibilityEvaluator":
        flat = _to2d(X_train).reshape(len(X_train), -1)
        self.sc_ = StandardScaler().fit(flat)
        xs = self.sc_.transform(flat)
        print("[Plausibility] Fitting IF, LOF, OC-SVM ...")
        self.IF_    = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(xs)
        self.LOF_   = LocalOutlierFactor(
            novelty=True,
            contamination=self.contamination,
        ).fit(xs)
        self.OCSVM_ = OneClassSVM(
            nu=self.contamination, kernel="rbf"
        ).fit(xs)
        self._fitted = True
        print("[Plausibility] Detectors ready OK")
        return self

    def score(self, X_cf: np.ndarray) -> dict[str, np.ndarray]:
        assert self._fitted, "Call fit() first."
        flat = _to2d(X_cf).reshape(len(X_cf), -1)
        xs   = self.sc_.transform(flat)

        def _norm(s: np.ndarray) -> np.ndarray:
            mn, mx = s.min(), s.max()
            if mx - mn < 1e-8:
                return np.zeros_like(s)
            return 1.0 - (s - mn) / (mx - mn)

        sc_if    = _norm(self.IF_.decision_function(xs))
        sc_lof   = _norm(self.LOF_.decision_function(xs))
        sc_ocsvm = _norm(self.OCSVM_.decision_function(xs))
        return {
            "if":       sc_if,
            "lof":      sc_lof,
            "ocsvm":    sc_ocsvm,
            "ensemble": (sc_if + sc_lof + sc_ocsvm) / 3,
        }


# =============================================================================
#  Evaluateur principal
# =============================================================================

class CounterfactualEvaluator:

    def __init__(
        self,
        x_train: Optional[np.ndarray] = None,
        tol_compact: float = 1e-3,
        auc_mode: str = "proportion",
        contamination: float = 0.1,
        fit_plausibility: bool = True,
        plausibility_checkpoint: Optional[str] = None,
    ):
        self.tol_compact = tol_compact
        self.auc_mode    = auc_mode
        self._plaus: Optional[PlausibilityEvaluator] = None
        self._ensemble_plaus = None  # For pre-trained EnsemblePlausibility

        # Charger détecteur pré-entraîné si fourni
        if plausibility_checkpoint is not None and os.path.exists(plausibility_checkpoint):
            import pickle
            print(f"[Plausibility] Loading pre-trained detector from {plausibility_checkpoint}")
            with open(plausibility_checkpoint, "rb") as f:
                ckpt = pickle.load(f)
            # The checkpoint contains an EnsemblePlausibility object
            self._ensemble_plaus = ckpt["ensemble"]
            print(f"[Plausibility] Pre-trained detector loaded OK")
        elif fit_plausibility and x_train is not None:
            self._plaus = PlausibilityEvaluator(contamination=contamination)
            self._plaus.fit(x_train)

    # -------------------------------------------------------------------------
    def evaluate(
        self,
        X_orig:  np.ndarray,   # [K, seq_len]    ou [K, seq_len, 1]
        X_cf:    np.ndarray,   # [K, seq_len]    ou [K, seq_len, 1]
        Y_hat:   np.ndarray,   # [K, pred_len]   ou [K, pred_len, 1]
        Y_cf:    np.ndarray,   # [K, pred_len]   ou [K, pred_len, 1]
        alphas:  np.ndarray,   # [K, pred_len]   — bornes basses
        betas:   np.ndarray,   # [K, pred_len]   — bornes hautes
    ) -> dict:
        """
        Calcule toutes les metriques.
        Chaque entree retournee : {"mean": float, "std": float}
        """
        R: dict = {}

        # (a) Validity Ratio
        vr = validity_ratio(Y_cf, alphas, betas)
        R["validity_ratio"] = {"mean": vr, "std": 0.0}

        # (b) Step-AUC
        sauc = stepwise_validity_auc(Y_cf, alphas, betas, mode=self.auc_mode)
        R["stepwise_auc"] = {"mean": sauc, "std": 0.0}

        # (c) Proximity
        R["proximity_l2"]   = {"mean": proximity(X_orig, X_cf, "l2"),   "std": 0.0}
        R["proximity_l1"]   = {"mean": proximity(X_orig, X_cf, "l1"),   "std": 0.0}
        R["proximity_linf"] = {"mean": proximity(X_orig, X_cf, "linf"), "std": 0.0}

        # (d) Compactness
        comp = compactness(X_orig, X_cf, tol=self.tol_compact)
        R["compactness"] = {"mean": comp, "std": 0.0}

        # (e) Roughness
        r_cf_m, r_cf_s = roughness(X_cf)
        r_x_m,  r_x_s  = roughness(X_orig)
        R["roughness_cf"]    = {"mean": r_cf_m, "std": r_cf_s}
        R["roughness_x"]     = {"mean": r_x_m,  "std": r_x_s}
        R["roughness_ratio"] = {"mean": r_cf_m / (r_x_m + 1e-8), "std": 0.0}

        # (f) Temporal Consistency
        tc_m, tc_s = temporal_consistency(X_orig, X_cf)
        R["temporal_consistency"] = {"mean": tc_m, "std": tc_s}

        # (g) Relative Reduction
        rr_m, rr_s = relative_reduction(Y_hat, Y_cf)
        R["relative_reduction"] = {"mean": rr_m, "std": rr_s}

        # (h) Plausibility
        if self._ensemble_plaus is not None:
            # Use pre-trained EnsemblePlausibility (torch-based)
            import torch
            X_cf_torch = torch.from_numpy(X_cf.astype(np.float32))
            sc_dict = self._ensemble_plaus.score_all(X_cf_torch)
            for k, v in sc_dict.items():
                v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                R[f"plausibility_{k}"] = {
                    "mean": float(v_np.mean()),
                    "std":  float(v_np.std()),
                }
        elif self._plaus is not None:
            # Use sklearn-based PlausibilityEvaluator
            sc = self._plaus.score(X_cf)
            for k, v in sc.items():
                R[f"plausibility_{k}"] = {
                    "mean": float(v.mean()),
                    "std":  float(v.std()),
                }

        return R

    # -------------------------------------------------------------------------
    @staticmethod
    def print_table(
        results: dict[str, dict],
        label: str = "Method",
        width: int = 60,
    ) -> None:
        SEP = "=" * width
        sections = [
            (
                "ForecastCF Paper Metrics",
                [
                    ("validity_ratio",   "Validity Ratio  UP", True),
                    ("stepwise_auc",     "Step AUC        UP", True),
                    ("proximity_l2",     "Proximity L2    DOWN", False),
                    ("compactness",      "Compactness     UP", True),
                ],
            ),
            (
                "Temporal Quality",
                [
                    ("roughness_ratio",       "Roughness ratio      DOWN", False),
                    ("temporal_consistency",  "Temporal Consistency UP", True),
                    ("relative_reduction",    "Relative Reduction   UP", True),
                ],
            ),
            (
                "Plausibility  (DOWN = more realistic)",
                [
                    ("plausibility_ensemble", "Ensemble DOWN", False),
                ],
            ),
        ]

        print(f"\n{'=' * width}")
        print(f"  {label}")
        print(f"{'=' * width}")

        for title, keys in sections:
            if not any(k in results for k, *_ in keys):
                continue
            print(f"\n  -- {title}")
            for key, display, higher_better in keys:
                if key not in results:
                    continue
                m    = results[key]["mean"]
                s    = results[key]["std"]
                flag = ""
                if higher_better is True  and m > 0.75:
                    flag = "  OK"
                elif higher_better is False and m < 0.25:
                    flag = "  OK"
                print(f"  {display:36s}: {m:.4f}  +/- {s:.4f}{flag}")

        print(f"\n{'=' * width}\n")

    # -------------------------------------------------------------------------
    @staticmethod
    def compare(
        results_dict: dict[str, dict],
        width: int = 80,
    ) -> None:
        methods   = list(results_dict.keys())
        col_w     = 12
        SEP       = "-" * width
        key_display = [
            ("validity_ratio",        "Validity Ratio  UP"),
            ("stepwise_auc",          "Step AUC        UP"),
            ("proximity_l2",          "Proximity L2    DOWN"),
            ("compactness",           "Compactness     UP"),
            ("roughness_ratio",       "Roughness ratio DOWN"),
            ("temporal_consistency",  "Temp. Consist.  UP"),
            ("relative_reduction",    "Rel. Reduction  UP"),
            ("plausibility_ensemble", "Plausibility DOWN  "),
        ]

        header = f"  {'Metric':<36}" + "".join(f"{m:>{col_w}}" for m in methods)
        print(f"\n{'=' * width}")
        print(f"  COMPARISON TABLE")
        print(f"{'=' * width}")
        print(header)
        print(SEP)

        for key, display in key_display:
            row = f"  {display:<36}"
            for m in methods:
                v = results_dict[m].get(key, {}).get("mean", float("nan"))
                row += f"{v:>{col_w}.4f}"
            print(row)

        print(f"{'=' * width}\n")