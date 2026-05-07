"""
Module de visualisation commune pour tous les baselines.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cf_example(x_orig, x_cf, y_orig, y_cf, alpha, beta, 
                    method_name, dataset, model, output_dir, sample_idx=0):
    """
    Génère une figure de visualisation pour un exemple de counterfactual.
    
    Parameters
    ----------
    x_orig : np.ndarray
        Série originale (OT channel), shape [BH]
    x_cf : np.ndarray
        Série contrefactuelle (OT channel), shape [BH]
    y_orig : np.ndarray
        Forecast original, shape [H]
    y_cf : np.ndarray
        Forecast contrefactuel, shape [H]
    alpha : np.ndarray
        Borne inférieure, shape [H]
    beta : np.ndarray
        Borne supérieure, shape [H]
    method_name : str
        Nom de la méthode (BaseNN, BaseGrad, ForecastCF-PyTorch, etc.)
    dataset : str
        Nom du dataset
    model : str
        Nom du modèle
    output_dir : str
        Répertoire de sortie
    sample_idx : int
        Index du sample
    """
    seq_len = len(x_orig)
    pred_len = len(y_orig)
    
    # Timesteps
    t_input = np.arange(seq_len)
    t_forecast = np.arange(seq_len, seq_len + pred_len)
    
    # Calculer validity
    in_band = ((y_cf >= alpha) & (y_cf <= beta)).sum()
    validity = in_band / len(y_cf)
    
    # Visualiser
    plt.figure(figsize=(12, 5))
    
    # Input sequence (x_orig et x_cf) - ORANGE pour original, BLEU pour CF
    plt.plot(t_input, x_orig, label='original x (input)', color='#FF8C00', linewidth=2, alpha=0.9)
    plt.plot(t_input, x_cf, label='CF x (input_cf)', color='#1E90FF', linewidth=2, linestyle='--', alpha=0.9)
    
    # Forecast region (bande verte)
    plt.fill_between(t_forecast, alpha, beta, color='green', alpha=0.2, label='valid bounds')
    
    # Forecasts - ORANGE pour original, BLEU pour CF (même couleurs que input)
    plt.plot(t_forecast, y_orig, label='original y (forecast)', color='#FF8C00', linewidth=2, alpha=0.7)
    plt.plot(t_forecast, y_cf, label='CF y (forecast_cf)', color='#1E90FF', linewidth=2, linestyle='--', alpha=0.7)
    
    # Ligne verticale séparant input et forecast
    plt.axvline(x=seq_len, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Labels et titre
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Value (OT channel)', fontsize=12)
    plt.title(f'CF example — {method_name} — {dataset}/{model}\nSample {sample_idx} — validity={validity:.2f}', fontsize=13)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.text(seq_len/2, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
             'Input Sequence', ha='center', fontsize=10, color='gray')
    plt.text(seq_len + pred_len/2, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
             'Forecast Horizon', ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cf_example_{method_name.lower().replace('-', '_')}_{dataset}_{model}_sample{sample_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path
