#!/usr/bin/env python
# coding: utf-8
"""
Analyse complète des résultats ForecastCF avec toutes les métriques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Ajouter le chemin src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.metrics import (
    compute_plausibility,
    compute_roughness,
    compute_temporal_consistency
)


def load_counterfactuals(result_csv, data_path, model_path, config_path, model_type):
    """
    Charge les résultats et régénère les counterfactuals pour analyse.
    """
    import torch
    import json
    
    # Charger les résultats
    df = pd.read_csv(result_csv)
    print(f"Loaded {len(df)} results from {result_csv}")
    
    # Charger le modèle
    if model_type.lower() == "itransformer":
        from src.models.Forecaster.iTransformer import Model
    elif model_type.lower() == "timesnet":
        from src.models.Forecaster.TimesNet import Model
    elif model_type.lower() == "dlinear":
        from src.models.Forecaster.DLinear import Model
    elif model_type.lower() == "gru":
        from src.models.Forecaster.GRU import Model
    elif model_type.lower() == "patchtst":
        from src.models.Forecaster.PatchTST import Model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Charger config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    config = Config(**config_dict)
    
    # Charger modèle
    model = Model(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Charger données
    data_df = pd.read_csv(data_path)
    target_col = config.target if hasattr(config, 'target') else data_df.columns[-1]
    data = data_df[[target_col]].values
    
    # Test split
    test_start = int(len(data) * 0.8)
    test_data = data[test_start:]
    
    # Normaliser
    mean = test_data.mean()
    std = test_data.std()
    test_data = (test_data - mean) / std
    
    return df, model, test_data, mean, std


def compute_all_metrics(cf_samples, original_samples, predictions, ae_model=None):
    """
    Calcule toutes les métriques: plausibility, roughness, temporal consistency.
    """
    metrics = {}
    
    # Plausibility (si AE disponible)
    if ae_model is not None:
        plaus_scores = []
        for cf in cf_samples:
            plaus = compute_plausibility(cf, ae_model)
            plaus_scores.append(plaus)
        metrics['plausibility'] = np.mean(plaus_scores)
        metrics['plausibility_std'] = np.std(plaus_scores)
    else:
        metrics['plausibility'] = None
        metrics['plausibility_std'] = None
    
    # Roughness
    rough_scores = []
    for cf in cf_samples:
        rough = compute_roughness(cf)
        rough_scores.append(rough)
    metrics['roughness'] = np.mean(rough_scores)
    metrics['roughness_std'] = np.std(rough_scores)
    
    # Temporal Consistency
    temp_scores = []
    for cf in cf_samples:
        temp = compute_temporal_consistency(cf)
        temp_scores.append(temp)
    metrics['temporal_consistency'] = np.mean(temp_scores)
    metrics['temporal_consistency_std'] = np.std(temp_scores)
    
    return metrics


def plot_results(df, output_dir='baselines/ForecastCF/figures'):
    """
    Génère des visualisations des résultats.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # Métriques à visualiser
    metrics = ['validity_ratio', 'proximity', 'compactness', 'step_validity_auc']
    
    # 1. Barplot des métriques
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[idx]
            
            # Calculer moyenne et std
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            
            # Barplot
            ax.bar(['ForecastCF'], [mean_val], yerr=[std_val], 
                   capsize=10, color='#3498db', alpha=0.7)
            
            # Titre et labels
            metric_names = {
                'validity_ratio': 'Validity Ratio (↑)',
                'proximity': 'Proximity L2 (↓)',
                'compactness': 'Compactness (↑)',
                'step_validity_auc': 'Step Validity AUC (↑)'
            }
            ax.set_title(metric_names.get(metric, metric), fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_ylim(0, 1)
            
            # Ajouter la valeur
            ax.text(0, mean_val + std_val + 0.05, f'{mean_val:.3f}±{std_val:.3f}',
                   ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecastcf_metrics.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/forecastcf_metrics.png")
    plt.close()
    
    # 2. Distribution des métriques (si plusieurs seeds)
    if len(df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[idx]
                
                # Boxplot
                ax.boxplot([df[metric]], labels=['ForecastCF'])
                
                metric_names = {
                    'validity_ratio': 'Validity Ratio (↑)',
                    'proximity': 'Proximity L2 (↓)',
                    'compactness': 'Compactness (↑)',
                    'step_validity_auc': 'Step Validity AUC (↑)'
                }
                ax.set_title(metric_names.get(metric, metric), fontsize=12, fontweight='bold')
                ax.set_ylabel('Value', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/forecastcf_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/forecastcf_distribution.png")
        plt.close()


def print_summary(df, extra_metrics=None):
    """
    Affiche un résumé des résultats.
    """
    print("\n" + "="*80)
    print("RÉSULTATS FORECASTCF")
    print("="*80)
    
    print(f"\nNombre de runs: {len(df)}")
    print(f"Seeds: {df['random_seed'].tolist() if 'random_seed' in df.columns else 'N/A'}")
    
    print("\n" + "-"*80)
    print(f"{'Métrique':<30} {'Moyenne':<15} {'Écart-type':<15}")
    print("-"*80)
    
    # Métriques principales
    metrics = {
        'validity_ratio': 'Validity Ratio (↑)',
        'step_validity_auc': 'Step Validity AUC (↑)',
        'proximity': 'Proximity L2 (↓)',
        'compactness': 'Compactness (↑)',
    }
    
    for col, name in metrics.items():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            print(f"{name:<30} {mean:>14.4f} {std:>14.4f}")
    
    # Métriques supplémentaires
    if extra_metrics:
        print("\n" + "-"*80)
        print("Métriques supplémentaires:")
        print("-"*80)
        
        if extra_metrics.get('plausibility') is not None:
            print(f"{'Plausibility (↑)':<30} {extra_metrics['plausibility']:>14.4f} "
                  f"{extra_metrics.get('plausibility_std', 0):>14.4f}")
        
        if extra_metrics.get('roughness') is not None:
            print(f"{'Roughness (↓)':<30} {extra_metrics['roughness']:>14.4f} "
                  f"{extra_metrics.get('roughness_std', 0):>14.4f}")
        
        if extra_metrics.get('temporal_consistency') is not None:
            print(f"{'Temporal Consistency (↑)':<30} {extra_metrics['temporal_consistency']:>14.4f} "
                  f"{extra_metrics.get('temporal_consistency_std', 0):>14.4f}")
    
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ForecastCF results")
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="Path to ForecastCF results CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/ForecastCF/figures",
        help="Directory to save figures"
    )
    parser.add_argument(
        "--compute-extra-metrics",
        action="store_true",
        help="Compute plausibility, roughness, temporal consistency (requires regenerating CFs)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint (required if --compute-extra-metrics)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to model config (required if --compute-extra-metrics)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to dataset (required if --compute-extra-metrics)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Model type (required if --compute-extra-metrics)"
    )
    
    args = parser.parse_args()
    
    # Charger les résultats
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} results from {args.results_csv}")
    
    # Afficher le résumé
    print_summary(df)
    
    # Générer les figures
    plot_results(df, args.output_dir)
    
    print(f"\n✓ Analyse terminée!")
    print(f"✓ Figures sauvegardées dans: {args.output_dir}")


if __name__ == "__main__":
    main()
