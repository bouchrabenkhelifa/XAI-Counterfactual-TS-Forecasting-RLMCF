import numpy as np

try:
    from dtaidistance import dtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False


def _flatten_per_sample(x):
    x = np.asarray(x)
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Expected shape [B,T] or [B,T,C], got {x.shape}")


def l1_distance(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    return np.mean(np.abs(x_cf - x), axis=tuple(range(1, x.ndim)))


def l2_distance(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    return np.sqrt(np.mean((x_cf - x) ** 2, axis=tuple(range(1, x.ndim))) + 1e-12)


def manhattan_distance(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    return np.sum(np.abs(x_cf - x), axis=tuple(range(1, x.ndim)))


def euclidean_distance(x, x_cf):
    xf = _flatten_per_sample(x)
    xcf = _flatten_per_sample(x_cf)
    return np.linalg.norm(xcf - xf, axis=1)


def dtw_distance(x, x_cf):
    if not DTW_AVAILABLE:
        raise ImportError(
            "dtaidistance is required for dtw_distance. Install with: pip install dtaidistance"
        )

    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    B = x.shape[0]
    out = []

    for i in range(B):
        xi = x[i]
        xci = x_cf[i]

        if xi.ndim > 1:
            xi = xi.reshape(-1)
            xci = xci.reshape(-1)

        out.append(float(dtw.distance(xi, xci)))

    return np.asarray(out, dtype=np.float32)


def normalized_l2_distance(x, x_cf, eps=1e-6):
    x = np.asarray(x)
    raw = l2_distance(x, x_cf)

    flat = _flatten_per_sample(x)
    ranges = flat.max(axis=1) - flat.min(axis=1)
    return raw / (ranges + eps)


__all__ = [
    "l1_distance",
    "l2_distance",
    "manhattan_distance",
    "euclidean_distance",
    "dtw_distance",
    "normalized_l2_distance",
]