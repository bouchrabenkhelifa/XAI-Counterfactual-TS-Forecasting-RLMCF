import numpy as np


def sparsity_ratio(x, x_cf, threshold=1e-3):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    diff = np.abs(x_cf - x)
    changed = (diff > threshold).astype(np.float32)
    axes = tuple(range(1, changed.ndim))
    return np.mean(changed, axis=axes)


def mean_change_magnitude(x, x_cf):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    axes = tuple(range(1, x.ndim))
    return np.mean(np.abs(x_cf - x), axis=axes)


def segment_sparsity(x, x_cf, threshold=1e-3):
    x = np.asarray(x)
    x_cf = np.asarray(x_cf)
    diff = np.abs(x_cf - x)

    if diff.ndim == 3:
        changed_t = (np.max(diff, axis=2) > threshold).astype(np.float32)
    elif diff.ndim == 2:
        changed_t = (diff > threshold).astype(np.float32)
    else:
        raise ValueError(f"Expected shape [B,T] or [B,T,C], got {diff.shape}")

    return np.mean(changed_t, axis=1)


__all__ = [
    "sparsity_ratio",
    "mean_change_magnitude",
    "segment_sparsity",
]