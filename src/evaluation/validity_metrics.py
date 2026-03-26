import numpy as np


def _to_1d_forecast(y):
    y = np.asarray(y)
    if y.ndim == 3:
        return y[..., 0]
    if y.ndim == 2:
        return y
    raise ValueError(f"Expected forecast shape [B,H] or [B,H,1], got {y.shape}")


def delta_mean(y_hat, y_cf):
    y_hat = _to_1d_forecast(y_hat)
    y_cf = _to_1d_forecast(y_cf)
    return y_hat.mean(axis=1) - y_cf.mean(axis=1)


def relative_reduction(y_hat, y_cf, eps=1e-6):
    y_hat = _to_1d_forecast(y_hat)
    d = delta_mean(y_hat, y_cf)
    base = np.abs(y_hat.mean(axis=1)) + eps
    return d / base


def target_gap(y_hat, y_cf, rho=0.10, eps=1e-6):
    rr = relative_reduction(y_hat, y_cf, eps=eps)
    return rr - rho


def success_indicator(y_hat, y_cf, rho=0.10, eps=1e-6):
    return (relative_reduction(y_hat, y_cf, eps=eps) >= rho).astype(np.float32)


def success_rate(y_hat, y_cf, rho=0.10, eps=1e-6):
    return float(success_indicator(y_hat, y_cf, rho=rho, eps=eps).mean())


__all__ = [
    "delta_mean",
    "relative_reduction",
    "target_gap",
    "success_indicator",
    "success_rate",
]