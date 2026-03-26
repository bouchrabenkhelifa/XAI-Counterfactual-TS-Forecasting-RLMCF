import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def first_diff(x):
    x = np.asarray(x)
    return x[:, 1:, ...] - x[:, :-1, ...]


def second_diff(x):
    dx = first_diff(x)
    return dx[:, 1:, ...] - dx[:, :-1, ...]


def roughness(x):
    x = np.asarray(x)
    dx = first_diff(x)
    axes = tuple(range(1, dx.ndim))
    return np.mean(np.abs(dx), axis=axes)


def roughness_ratio(x, x_cf, eps=1e-6):
    return roughness(x_cf) / (roughness(x) + eps)


def derivative_distance(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    dx = first_diff(x)
    dx_cf = first_diff(x_cf)
    axes = tuple(range(1, dx.ndim))
    return np.mean(np.abs(dx_cf - dx), axis=axes)


def second_derivative_distance(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    ddx = second_diff(x)
    ddx_cf = second_diff(x_cf)
    axes = tuple(range(1, ddx.ndim))
    return np.mean(np.abs(ddx_cf - ddx), axis=axes)


def temporal_consistency(x_cf, smoothness_threshold=1.0):
    x_cf = np.asarray(x_cf)
    dx = first_diff(x_cf)

    if dx.ndim == 3:
        mags = np.linalg.norm(dx, axis=2)
    else:
        mags = np.abs(dx)

    violations = np.sum(mags > smoothness_threshold, axis=1)
    consistency = 1.0 - (violations / np.maximum(mags.shape[1], 1))
    return np.clip(consistency, 0.0, 1.0)


def autocorrelation_preservation(x, x_cf, max_lag=10):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    B = x.shape[0]

    def _acf_1d(ts, max_lag_):
        vals = []
        for lag in range(1, max_lag_ + 1):
            if len(ts) <= lag:
                vals.append(0.0)
                continue
            corr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
            vals.append(0.0 if np.isnan(corr) else corr)
        return np.asarray(vals)

    sims = []
    for i in range(B):
        xi = x[i].reshape(-1)
        xci = x_cf[i].reshape(-1)
        acf_x = _acf_1d(xi, max_lag)
        acf_cf = _acf_1d(xci, max_lag)

        if np.std(acf_x) == 0 or np.std(acf_cf) == 0:
            sim = 1.0 if np.allclose(acf_x, acf_cf) else 0.0
        else:
            sim = np.corrcoef(acf_x, acf_cf)[0, 1]
            sim = 0.0 if np.isnan(sim) else max(0.0, sim)
        sims.append(sim)

    return np.asarray(sims, dtype=np.float32)


def spectral_similarity(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    B = x.shape[0]
    sims = []

    for i in range(B):
        xi = x[i].reshape(-1)
        xci = x_cf[i].reshape(-1)

        fx = np.abs(np.fft.fft(xi))
        fcf = np.abs(np.fft.fft(xci))

        fx = fx / (np.sum(fx) + 1e-12)
        fcf = fcf / (np.sum(fcf) + 1e-12)

        sim = np.corrcoef(fx, fcf)[0, 1]
        sim = 0.0 if np.isnan(sim) else max(0.0, sim)
        sims.append(sim)

    return np.asarray(sims, dtype=np.float32)


def statistical_similarity(reference_data, x_cf, method="ks_test"):
    ref = np.asarray(reference_data).reshape(-1)
    cf = np.asarray(x_cf).reshape(-1)

    if method == "ks_test":
        _, p_value = stats.ks_2samp(ref, cf)
        return float(p_value)

    if method == "wasserstein":
        return float(wasserstein_distance(ref, cf))

    if method == "kl_divergence":
        bins = np.linspace(min(ref.min(), cf.min()), max(ref.max(), cf.max()), 50)
        hist_ref, _ = np.histogram(ref, bins=bins, density=True)
        hist_cf, _ = np.histogram(cf, bins=bins, density=True)

        eps = 1e-10
        hist_ref = hist_ref + eps
        hist_cf = hist_cf + eps
        hist_ref = hist_ref / np.sum(hist_ref)
        hist_cf = hist_cf / np.sum(hist_cf)

        kl = np.sum(hist_cf * np.log(hist_cf / hist_ref))
        return float(kl)

    raise ValueError(f"Unknown method: {method}")


__all__ = [
    "first_diff",
    "second_diff",
    "roughness",
    "roughness_ratio",
    "derivative_distance",
    "second_derivative_distance",
    "temporal_consistency",
    "autocorrelation_preservation",
    "spectral_similarity",
    "statistical_similarity",
]