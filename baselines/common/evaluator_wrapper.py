"""
Wrapper unifié pour l'évaluation des counterfactuals.
Appelle CounterfactualEvaluator (8 métriques) + evaluate_forecastcf_metrics (4 métriques).
"""

import numpy as np
from src.evaluation.unified_evaluator import CounterfactualEvaluator
from baselines.ForecastCF.src.forecastcf_evaluator import evaluate_forecastcf_metrics


def run_evaluation(x_orig, x_cf, y_hat, y_cf, alphas, betas,
                   x_train, method_name, seed):
    """
    Lance l'évaluation complète des counterfactuals.
    
    Calcule :
        - 8 métriques étendues via CounterfactualEvaluator
        - 4 métriques ForecastCF via evaluate_forecastcf_metrics
    
    Parameters
    ----------
    x_orig : np.ndarray
        Séries originales, shape [N, BH, 1]
    x_cf : np.ndarray
        Séries contrefactuelles, shape [N, BH, 1]
    y_hat : np.ndarray
        Forecasts originaux, shape [N, H, 1]
    y_cf : np.ndarray
        Forecasts contrefactuels, shape [N, H, 1]
    alphas : np.ndarray
        Bornes inférieures, shape [N, H]
    betas : np.ndarray
        Bornes supérieures, shape [N, H]
    x_train : np.ndarray or None
        Train set pour plausibilité, shape [M, BH, 1]
    method_name : str
        Nom de la méthode (pour affichage)
    seed : int
        Seed utilisée (pour affichage)
    
    Returns
    -------
    dict
        Dictionnaire avec clés "extended_metrics" et "forecastcf_metrics"
    """
    # Évaluation étendue (8 métriques)
    fit_plausibility = x_train is not None
    evaluator = CounterfactualEvaluator(
        x_train=x_train,
        fit_plausibility=fit_plausibility
    )
    
    extended_metrics = evaluator.evaluate(
        X_orig=x_orig,
        X_cf=x_cf,
        Y_hat=y_hat,
        Y_cf=y_cf,
        alphas=alphas,
        betas=betas
    )
    
    # Évaluation ForecastCF (4 métriques)
    # Convertir alphas/betas [N, H] -> [N, H, 1] pour evaluate_forecastcf_metrics
    uppers = betas[:, :, np.newaxis]   # [N, H, 1]
    lowers = alphas[:, :, np.newaxis]  # [N, H, 1]
    
    forecastcf_metrics = evaluate_forecastcf_metrics(
        x_orig=x_orig,
        x_cf=x_cf,
        y_cf=y_cf,
        uppers=uppers,
        lowers=lowers,
        method_name=f"{method_name} (seed={seed})",
        print_results=False
    )
    
    return {
        "extended_metrics": extended_metrics,
        "forecastcf_metrics": forecastcf_metrics
    }
