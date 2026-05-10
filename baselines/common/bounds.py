"""
Module de calcul des bornes RL-MCF.
Réplique exacte de RLMaskTrainer._compute_bounds_np() de trainer_last.py.
"""

import json
import numpy as np
from types import SimpleNamespace


def compute_bounds_np(x_ot, y_hat, rho, fr, direction, global_sigma=None):
    """
    Calcule les bornes [alpha, beta] selon la formule RL-MCF.
    
    Réplique exacte de RLMaskTrainer._compute_bounds_np().
    
    Formule :
        sigma = global_sigma  (std fixe du train set, ou std(x_ot) si non disponible)
        gap   = rho * sigma   (séparation garantie entre y_hat et la bande)
        width = fr  * sigma   (largeur de la bande)
        
        direction < 0 (baisse) :  beta  = y_hat - gap
                                   alpha = beta  - width
        direction > 0 (hausse) :  alpha = y_hat + gap
                                   beta  = alpha + width
    
    Parameters
    ----------
    x_ot : np.ndarray
        Input observé, shape [N, BH, 1] ou [N, BH]
    y_hat : np.ndarray
        Forecast, shape [N, H, 1] ou [N, H]
    rho : float
        Paramètre gap (séparation entre y_hat et la bande)
    fr : float
        Paramètre width (largeur de la bande)
    direction : float
        Direction de la bande (-1.0 pour baisse, +1.0 pour hausse)
    global_sigma : float, optional
        Std fixe du train set. Si None, calcule std(x_ot) par sample.
    
    Returns
    -------
    alphas : np.ndarray
        Borne inférieure, shape [N, H]
    betas : np.ndarray
        Borne supérieure, shape [N, H]
    """
    # Flatten to 2D
    x2d = x_ot[:, :, 0] if x_ot.ndim == 3 else x_ot   # [N, BH]
    y2d = y_hat[:, :, 0] if y_hat.ndim == 3 else y_hat  # [N, H]
    
    # Compute sigma
    if global_sigma is not None:
        sigma = np.full(x2d.shape[0], global_sigma)
    else:
        sigma = x2d.std(axis=1, ddof=1).clip(min=1e-4)  # [N]
    
    # Compute gap and width
    gap   = (rho * sigma)[:, np.newaxis]   # [N, 1]
    width = (fr  * sigma)[:, np.newaxis]   # [N, 1]
    
    # Compute bounds based on direction
    if direction < 0:
        betas  = y2d - gap
        alphas = betas - width
    else:
        alphas = y2d + gap
        betas  = alphas + width
    
    return alphas, betas  # [N, H]


def load_bounds_params(rl_config_path):
    """
    Charge les paramètres de bornes (rho, fr, direction, global_sigma) depuis une config RL JSON.
    
    Parameters
    ----------
    rl_config_path : str
        Chemin vers le fichier de config RL JSON
    
    Returns
    -------
    dict
        Dictionnaire avec clés: rho, fr, direction, global_sigma
    """
    with open(rl_config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    
    cfg = SimpleNamespace(**cfg_dict)
    
    return {
        "rho":          getattr(cfg, "rho", 0.1),
        "fr":           getattr(cfg, "fr", 0.6),
        "direction":    getattr(cfg, "direction", -1.0),
        "global_sigma": getattr(cfg, "global_sigma", None),
    }


def get_rl_config_path(dataset, model):
    """
    Retourne le chemin de la config RL pour un (dataset, model) donné.
    
    Mapping :
        etth1   -> assets/configs/etth1_dataset/RL_ablations/config_{model}.json
        etth2   -> assets/configs/etth2_dataset/RL/config_{model}.json
        weather -> assets/configs/weather_dataset/RL/config_{model}.json
    
    Parameters
    ----------
    dataset : str
        Nom du dataset (etth1, etth2, weather)
    model : str
        Nom du modèle (itransformer, patchtst, dlinear, gru, timesnet)
    
    Returns
    -------
    str
        Chemin vers le fichier de config RL
    """
    dataset = dataset.lower()
    model = model.lower()
    
    if dataset == "etth1":
        # Special case for itransformer
        if model == "itransformer":
            return f"assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json"
        return f"assets/configs/etth1_dataset/RL_ablations/config_{model}.json"
    elif dataset == "etth2":
        return f"assets/configs/etth2_dataset/RL/config_{model}.json"
    elif dataset == "weather":
        return f"assets/configs/weather_dataset/RL/config_{model}.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
