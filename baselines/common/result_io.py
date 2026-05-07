"""
Module de gestion des résultats JSON.
Sauvegarde, chargement, cache MD5, agrégation inter-seeds.
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime
import sys
import torch


def compute_md5(file_path):
    """
    Calcule le hash MD5 d'un fichier.
    
    Parameters
    ----------
    file_path : str
        Chemin vers le fichier
    
    Returns
    -------
    str
        Hash MD5 en hexadécimal
    """
    if not os.path.exists(file_path):
        return None
    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def check_cache(output_path, checkpoint_path):
    """
    Vérifie si les résultats existent déjà avec le même checkpoint MD5.
    
    Parameters
    ----------
    output_path : str
        Chemin vers le fichier de résultats JSON
    checkpoint_path : str
        Chemin vers le checkpoint du forecaster
    
    Returns
    -------
    bool
        True si cache valide (résultats existent et MD5 correspond)
    """
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, "r") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False
    
    saved_md5 = saved.get("metadata", {}).get("checkpoint_md5")
    if saved_md5 is None:
        return False
    
    current_md5 = compute_md5(checkpoint_path)
    return saved_md5 == current_md5


def save_results(results, output_path):
    """
    Sauvegarde les résultats dans un fichier JSON.
    
    Parameters
    ----------
    results : dict
        Dictionnaire de résultats
    output_path : str
        Chemin vers le fichier de sortie
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved -> {output_path}")


def load_results(input_path):
    """
    Charge les résultats depuis un fichier JSON.
    
    Parameters
    ----------
    input_path : str
        Chemin vers le fichier de résultats
    
    Returns
    -------
    dict
        Dictionnaire de résultats
    """
    with open(input_path, "r") as f:
        return json.load(f)


def aggregate_seeds(seed_results):
    """
    Agrège les résultats de plusieurs seeds (calcule mean ± std).
    
    Parameters
    ----------
    seed_results : list of dict
        Liste de dictionnaires de résultats, un par seed.
        Chaque dict contient les métriques pour un seed donné.
    
    Returns
    -------
    dict
        Dictionnaire avec mean et std pour chaque métrique
    """
    if not seed_results:
        return {}
    
    # Collecter toutes les clés de métriques
    all_keys = set()
    for res in seed_results:
        all_keys.update(res.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [res[key] for res in seed_results if key in res]
        if not values:
            continue
        
        # Convertir en numpy array
        values_np = np.array(values, dtype=np.float32)
        
        aggregated[key] = {
            "mean": float(values_np.mean()),
            "std": float(values_np.std(ddof=1)) if len(values_np) > 1 else 0.0,
        }
    
    return aggregated


def create_metadata(checkpoint_path):
    """
    Crée le dictionnaire de metadata pour la reproductibilité.
    
    Parameters
    ----------
    checkpoint_path : str
        Chemin vers le checkpoint du forecaster
    
    Returns
    -------
    dict
        Metadata avec versions, checkpoint MD5, date
    """
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "checkpoint_name": os.path.basename(checkpoint_path),
        "checkpoint_md5": compute_md5(checkpoint_path),
        "run_date": datetime.now().isoformat(),
    }


def build_result_dict(method, dataset, model, n_samples, runtime_seconds,
                      avg_metrics, forecastcf_metrics, seeds, bounds_params,
                      checkpoint_path):
    """
    Construit le dictionnaire de résultats complet selon la structure définie.
    
    Parameters
    ----------
    method : str
        Nom de la méthode (ForecastCF_PyTorch, BaseNN, BaseShift, BaseGrad)
    dataset : str
        Nom du dataset
    model : str
        Nom du modèle
    n_samples : int
        Nombre de samples évalués
    runtime_seconds : float
        Temps d'exécution total en secondes
    avg_metrics : dict
        Métriques agrégées (mean ± std)
    forecastcf_metrics : dict or None
        Métriques ForecastCF (4 métriques) ou None
    seeds : list
        Liste des seeds utilisées
    bounds_params : dict
        Paramètres de bornes (rho, fr, direction, global_sigma)
    checkpoint_path : str
        Chemin vers le checkpoint du forecaster
    
    Returns
    -------
    dict
        Dictionnaire de résultats complet
    """
    result = {
        "method": method,
        "dataset": dataset,
        "model": model,
        "n_samples": n_samples,
        "runtime_seconds": runtime_seconds,
        "seeds": seeds,
        "bounds_params": bounds_params,
        "avg_metrics": avg_metrics,
        "metadata": create_metadata(checkpoint_path),
    }
    
    if forecastcf_metrics is not None:
        result["forecastcf_metrics"] = forecastcf_metrics
    
    return result
