"""
Module commun pour toutes les baselines.
Contient les fonctions partagées : compute_bounds_np, evaluator_wrapper, result_io.
"""

from .bounds import compute_bounds_np, load_bounds_params
from .evaluator_wrapper import run_evaluation
from .result_io import save_results, load_results, check_cache, aggregate_seeds

__all__ = [
    "compute_bounds_np",
    "load_bounds_params",
    "run_evaluation",
    "save_results",
    "load_results",
    "check_cache",
    "aggregate_seeds",
]
