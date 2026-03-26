import pickle
import numpy as np
import torch


def load_plausibility_model(pkl_path):
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    if isinstance(model, dict):
        model = (
            model.get("ensemble")
            or model.get("model")
            or model.get("plausibility")
            or model
        )

    return model


def _to_torch_tensor(x):
    """
    Convert numpy array to torch tensor if needed.
    Keep torch tensor unchanged.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    return x


def _to_numpy(x):
    """
    Convert torch tensor to numpy if needed.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plausibility_score(plausibility_model, x_cf):
    """
    Compatible with both torch.Tensor and numpy.ndarray inputs.
    """
    x_cf = _to_torch_tensor(x_cf)
    scores = plausibility_model.score(x_cf)
    return _to_numpy(scores).astype(np.float32)


def plausibility_score_all(plausibility_model, x_cf):
    """
    If available, return all detector scores (IF, LOF, OCSVM, ensemble).
    Otherwise return only ensemble.
    """
    x_cf = _to_torch_tensor(x_cf)

    if hasattr(plausibility_model, "score_all"):
        out = plausibility_model.score_all(x_cf)
        clean = {}
        for k, v in out.items():
            clean[k] = _to_numpy(v).astype(np.float32)
        return clean

    return {"ensemble": plausibility_score(plausibility_model, x_cf)}


__all__ = [
    "load_plausibility_model",
    "plausibility_score",
    "plausibility_score_all",
]