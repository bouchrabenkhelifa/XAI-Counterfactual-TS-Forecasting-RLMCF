import numpy as np
import torch

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM




def extract_features(W: np.ndarray) -> np.ndarray:
  
    x = W[:, :, 0].copy()                     # (N, L)

    # z-normalize each window individually
    mu  = x.mean(axis=1, keepdims=True)
    sig = x.std(axis=1,  keepdims=True) + 1e-8
    xn  = (x - mu) / sig                       # shape only, no level

    dx  = xn[:, 1:]  - xn[:, :-1]             # first differences
    dx2 = dx[:, 1:]  - dx[:, :-1]             # second differences

    # linear trend on normalized window
    L     = xn.shape[1]
    t     = np.linspace(-1, 1, L)
    trend = (xn * t).mean(axis=1)

    # autocorrelation lag-1 (smoothness)
    ac1   = (xn[:, :-1] * xn[:, 1:]).mean(axis=1)

    feat = np.stack([
        trend,                                 # trend direction
        ac1,                                   # smoothness
        xn.max(axis=1) - xn.min(axis=1),      # normalized range
        dx.std(axis=1)       + 1e-12,          # volatility
        np.abs(dx).mean(axis=1),               # avg abs change
        np.abs(dx).max(axis=1),                # max jump
        (np.abs(dx) > 1.0).mean(axis=1),      # fraction large jumps
        dx2.std(axis=1)      + 1e-12,          # smoothness of changes
        np.abs(dx2).mean(axis=1),              # avg acceleration
        (xn > 0).mean(axis=1) - 0.5,          # shape asymmetry
    ], axis=1).astype(np.float32)

    return feat


def series_to_windows(series: np.ndarray, window: int) -> np.ndarray:
    
    T   = len(series)
    idx = np.arange(0, T - window + 1, 1)
    return np.stack([series[i:i + window] for i in idx], axis=0)


def fit_calibration(raw: np.ndarray,
                    low_pct: float = 1.0,
                    high_pct: float = 99.0):
   
    p_low  = float(np.percentile(raw, low_pct))
    p_high = float(np.percentile(raw, high_pct))
    return p_low, p_high


def apply_calibration(raw: np.ndarray,
                      p_low: float,
                      p_high: float,
                      invert: bool,
                      device: torch.device) -> torch.Tensor:
   
    raw = raw.astype(np.float64)

    if abs(p_high - p_low) < 1e-12:
        return torch.zeros(len(raw), dtype=torch.float32).to(device)

    if invert:
        # IForest / LOF : low raw = normal → LOW plausibility score (more realistic)
        # Normalize to [0, 1] where 0 = most realistic, 1 = most anomalous
        p = (raw - p_low) / (p_high - p_low)
    else:
        # OC-SVM : high raw = normal (positive) → LOW plausibility score (more realistic)
        # Invert so that high raw → low score
        p = 1.0 - (raw - p_low) / (p_high - p_low)

    p = np.clip(p, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(p).to(device)



class IForestPlausibility:
   

    def __init__(self, window: int = 96, n_estimators: int = 500,
                 random_state: int = 0, n_jobs: int = -1):
        self.window  = window
        self.model   = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.p_low_  = None
        self.p_high_ = None
        self.fitted_ = False

    def _raw(self, feat: np.ndarray) -> np.ndarray:
        # invert : higher = more anomalous
        return (-self.model.score_samples(feat)).astype(np.float32)

    def fit(self, train_series_scaled: np.ndarray,
            val_series_scaled: np.ndarray = None) -> "IForestPlausibility":
        W    = series_to_windows(train_series_scaled, self.window)
        feat = extract_features(W)
        self.model.fit(feat)

        # calibrate on train + val for robustness
        raw = self._raw(feat)
        if val_series_scaled is not None:
            W_val    = series_to_windows(val_series_scaled, self.window)
            feat_val = extract_features(W_val)
            raw_val  = self._raw(feat_val)
            raw      = np.concatenate([raw, raw_val])

        self.p_low_, self.p_high_ = fit_calibration(raw)
        self.fitted_ = True
        print(f"[IForest]  fitted on {len(W)} windows | "
              f"p5={self.p_low_:.4f}  p95={self.p_high_:.4f}")
        return self

    def score(self, x_cf: torch.Tensor) -> torch.Tensor:
        assert self.fitted_, "Call fit() first"
        feat = extract_features(x_cf.detach().cpu().numpy().astype(np.float32))
        raw  = self._raw(feat)
        return apply_calibration(raw, self.p_low_, self.p_high_,
                                 invert=True, device=x_cf.device)

    def sanity_check(self, x_real: torch.Tensor,
                     label: str = "real") -> float:
        p      = self.score(x_real)
        m      = float(p.mean().item())
        status = "✓" if m > 0.6 else "✗ needs tuning"
        print(f"[IForest]  sanity({label}) mean={m:.4f}  {status}")
        return m




class LOFPlausibility:
  

    def __init__(self, window: int = 96, n_neighbors: int = 20,
                 n_jobs: int = -1):
        self.window  = window
        self.model   = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination="auto",
            n_jobs=n_jobs,
        )
        self.p_low_  = None
        self.p_high_ = None
        self.fitted_ = False

    def _raw(self, feat: np.ndarray) -> np.ndarray:
        return (-self.model.score_samples(feat)).astype(np.float32)

    def fit(self, train_series_scaled: np.ndarray,
            val_series_scaled: np.ndarray = None) -> "LOFPlausibility":
        W    = series_to_windows(train_series_scaled, self.window)
        feat = extract_features(W)
        self.model.fit(feat)

        raw = self._raw(feat)
        if val_series_scaled is not None:
            W_val    = series_to_windows(val_series_scaled, self.window)
            feat_val = extract_features(W_val)
            raw_val  = self._raw(feat_val)
            raw      = np.concatenate([raw, raw_val])

        self.p_low_, self.p_high_ = fit_calibration(raw)
        self.fitted_ = True
        print(f"[LOF]      fitted on {len(W)} windows | "
              f"p5={self.p_low_:.4f}  p95={self.p_high_:.4f}")
        return self

    def score(self, x_cf: torch.Tensor) -> torch.Tensor:
        assert self.fitted_, "Call fit() first"
        feat = extract_features(x_cf.detach().cpu().numpy().astype(np.float32))
        raw  = self._raw(feat)
        return apply_calibration(raw, self.p_low_, self.p_high_,
                                 invert=True, device=x_cf.device)

    def sanity_check(self, x_real: torch.Tensor,
                     label: str = "real") -> float:
        p      = self.score(x_real)
        m      = float(p.mean().item())
        status = "✓" if m > 0.6 else "✗ needs tuning"
        print(f"[LOF]      sanity({label}) mean={m:.4f}  {status}")
        return m




class OCSVMPlausibility:
  
    def __init__(self, window: int = 96, kernel: str = "rbf",
                 nu: float = 0.1, gamma: str = "scale",
                 max_fit_samples: int = 5000):
        self.window          = window
        self.max_fit_samples = max_fit_samples
        self.model           = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.p_low_          = None
        self.p_high_         = None
        self.fitted_         = False

    def _raw(self, feat: np.ndarray) -> np.ndarray:
    
        return self.model.score_samples(feat).astype(np.float32)

    def fit(self, train_series_scaled: np.ndarray,
            val_series_scaled: np.ndarray = None) -> "OCSVMPlausibility":
        W    = series_to_windows(train_series_scaled, self.window)
        feat = extract_features(W)

        if len(feat) > self.max_fit_samples:
            idx      = np.random.choice(len(feat), self.max_fit_samples,
                                        replace=False)
            feat_fit = feat[idx]
        else:
            feat_fit = feat

        self.model.fit(feat_fit)

        raw = self._raw(feat_fit)
        if val_series_scaled is not None:
            W_val    = series_to_windows(val_series_scaled, self.window)
            feat_val = extract_features(W_val)
            raw_val  = self._raw(feat_val)
            raw      = np.concatenate([raw, raw_val])

        self.p_low_, self.p_high_ = fit_calibration(raw)
        self.fitted_ = True
        print(f"[OC-SVM]   fitted on {len(feat_fit)} windows | "
              f"p5={self.p_low_:.4f}  p95={self.p_high_:.4f}")
        return self

    def score(self, x_cf: torch.Tensor) -> torch.Tensor:
        assert self.fitted_, "Call fit() first"
        feat = extract_features(x_cf.detach().cpu().numpy().astype(np.float32))
        raw  = self._raw(feat)
        return apply_calibration(raw, self.p_low_, self.p_high_,
                                 invert=False, device=x_cf.device)

    def sanity_check(self, x_real: torch.Tensor,
                     label: str = "real") -> float:
        p      = self.score(x_real)
        m      = float(p.mean().item())
        status = "✓" if m > 0.6 else "✗ needs tuning"
        print(f"[OC-SVM]   sanity({label}) mean={m:.4f}  {status}")
        return m



class EnsemblePlausibility:


    def __init__(self, window: int = 96):
        self.window  = window
        self.iforest = IForestPlausibility(window=window)
        self.lof     = LOFPlausibility(window=window)
        self.ocsvm   = OCSVMPlausibility(window=window)
        self.fitted_ = False

    def fit(self, train_series_scaled: np.ndarray,
            val_series_scaled: np.ndarray = None) -> "EnsemblePlausibility":
       
        print("=" * 50)
        print("[Ensemble] Fitting all plausibility detectors...")
        self.iforest.fit(train_series_scaled, val_series_scaled)
        self.lof.fit(train_series_scaled, val_series_scaled)
        self.ocsvm.fit(train_series_scaled, val_series_scaled)
        self.fitted_ = True
        print("[Ensemble] All detectors fitted ✓")
        print("=" * 50)
        return self

    def score(self, x_cf: torch.Tensor) -> torch.Tensor:
        assert self.fitted_, "Call fit() first"
        p_if  = self.iforest.score(x_cf)
        p_lof = self.lof.score(x_cf)
        p_svm = self.ocsvm.score(x_cf)
        return (p_if + p_lof + p_svm) / 3.0

    def score_all(self, x_cf: torch.Tensor) -> dict:
        assert self.fitted_, "Call fit() first"
        p_if  = self.iforest.score(x_cf)
        p_lof = self.lof.score(x_cf)
        p_svm = self.ocsvm.score(x_cf)
        return {
            "if":       p_if,
            "lof":      p_lof,
            "ocsvm":    p_svm,
            "ensemble": (p_if + p_lof + p_svm) / 3.0,
        }

    def sanity_check(self, x_real: torch.Tensor,
                     label: str = "real") -> dict:
        print(f"\n[Ensemble] Sanity check on {label} data :")
        m_if  = self.iforest.sanity_check(x_real, label)
        m_lof = self.lof.sanity_check(x_real, label)
        m_svm = self.ocsvm.sanity_check(x_real, label)
        m_ens = (m_if + m_lof + m_svm) / 3.0
        status = "✓" if m_ens > 0.6 else "✗ needs tuning"
        print(f"[Ensemble] mean={m_ens:.4f}  {status}\n")
        return {
            "if": m_if, "lof": m_lof,
            "ocsvm": m_svm, "ensemble": m_ens
        }