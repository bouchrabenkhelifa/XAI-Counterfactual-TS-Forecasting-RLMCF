#!/usr/bin/env python
# coding: utf-8
"""
Script pour comparer les résultats entre votre méthode RL et ForecastCF baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(rl_results_path, forecastcf_results_path):
    """Charge les résultats des deux méthodes."""
    
    rl_df = pd.read_csv(rl_results_path)
    fcf_df = pd.read_csv(forecastcf_results_path)
    
    return rl_df, fcf_df


def compute_statistics(df, method_name):
    """Calcule les statistiques (moyenne ± std) pour chaque métrique."""
    
    metrics = ['validity_ratio', 'proximity', 'compactness', 'step_validity_auc']
    
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            stats[metric] = {'mean': mean, 'std': std}
    
    return stats


def compare_methods(rl_df, fcf_df):
    """Compare les deux méthodes et affiche les résultats."""
    
    print("=" * 80)
    print("COMPARAISON: Votre méthode RL vs ForecastCF Baseline")
    print("=" * 80)
    print()
    
    # Statistiques pour chaque méthode
    rl_stats = compute_statistics(rl_df, "RL")
    fcf_stats = compute_statistics(fcf_df, "ForecastCF")
    
    # Affichage formaté
    metrics_info = {
        'validity_ratio': ('Validity Ratio', '↑', 'higher is better'),
        'step_validity_auc': ('Step Validity AUC', '↑', 'higher is better'),
        'proximity': ('Proximity (L2)', '↓', 'lower is better'),
        'compactness': ('Compactness', '↑', 'higher is better'),
    }
    
    print(f"{'Metric':<25} {'Your RL':<20} {'ForecastCF':<20} {'Winner':<15}")
    print("-" * 80)
    
    for metric, (name, direction, desc) in metrics_info.items():
        if metric in rl_stats and metric in fcf_stats:
            rl_mean = rl_stats[metric]['mean']
            rl_std = rl_stats[metric]['std']
            fcf_mean = fcf_stats[metric]['mean']
            fcf_std = fcf_stats[metric]['std']
            
            rl_str = f"{rl_mean:.4f} ± {rl_std:.4f}"
            fcf_str = f"{fcf_mean:.4f} ± {fcf_std:.4f}"
            
            # Déterminer le gagnant
            if direction == '↑':
                winner = "Your RL ✓" if rl_mean > fcf_mean else "ForecastCF"
                improvement = ((rl_mean - fcf_mean) / fcf_mean) * 100
            else:  # ↓
                winner = "Your RL ✓" if rl_mean < fcf_mean else "ForecastCF"
                improvement = ((fcf_mean - rl_mean) / fcf_mean) * 100
            
            print(f"{name:<25} {rl_str:<20} {fcf_str:<20} {winner:<15}")
            print(f"  {direction} {desc} | Improvement: {improvement:+.2f}%")
            print()
    
    print("=" * 80)


def plot_comparison(rl_df, fcf_df, output_dir='baselines/comparison_plots'):
    """Génère des graphiques de comparaison."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = ['validity_ratio', 'proximity', 'compactness', 'step_validity_auc']
    
    # Préparer les données
    rl_df['Method'] = 'Your RL'
    fcf_df['Method'] = 'ForecastCF'
    combined_df = pd.concat([rl_df, fcf_df], ignore_index=True)
    
    # Style
    sns.set_style("whitegrid")
    colors = {'Your RL': '#2ecc71', 'ForecastCF': '#e74c3c'}
    
    # 1. Barplot avec erreurs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in combined_df.columns:
            ax = axes[idx]
            sns.barplot(
                data=combined_df,
                x='Method',
                y=metric,
                palette=colors,
                ax=ax,
                capsize=0.1,
                errwidth=2
            )
            
            # Titre et labels
            metric_names = {
                'validity_ratio': 'Validity Ratio (↑)',
                'proximity': 'Proximity L2 (↓)',
                'compactness': 'Compactness (↑)',
                'step_validity_auc': 'Step Validity AUC (↑)'
            }
            ax.set_title(metric_names.get(metric, metric), fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_barplot.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparison_barplot.png")
    plt.close()
    
    # 2. Boxplot pour voir la distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in combined_df.columns:
            ax = axes[idx]
            sns.boxplot(
                data=combined_df,
                x='Method',
                y=metric,
                palette=colors,
                ax=ax
            )
            
            metric_names = {
                'validity_ratio': 'Validity Ratio (↑)',
                'proximity': 'Proximity L2 (↓)',
                'compactness': 'Compactness (↑)',
                'step_validity_auc': 'Step Validity AUC (↑)'
            }
            ax.set_title(metric_names.get(metric, metric), fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_boxplot.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparison_boxplot.png")
    plt.close()
    
    # 3. Radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Normaliser les métriques pour le radar chart
    metrics_for_radar = ['validity_ratio', 'step_validity_auc', 'compactness']
    
    rl_means = [rl_df[m].mean() for m in metrics_for_radar if m in rl_df.columns]
    fcf_means = [fcf_df[m].mean() for m in metrics_for_radar if m in fcf_df.columns]
    
    # Ajouter proximity inversée (1 - normalized proximity)
    if 'proximity' in rl_df.columns and 'proximity' in fcf_df.columns:
        max_prox = max(rl_df['proximity'].max(), fcf_df['proximity'].max())
        rl_means.append(1 - rl_df['proximity'].mean() / max_prox)
        fcf_means.append(1 - fcf_df['proximity'].mean() / max_prox)
        metrics_for_radar.append('proximity (inv)')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    rl_means += rl_means[:1]
    fcf_means += fcf_means[:1]
    angles += angles[:1]
    
    ax.plot(angles, rl_means, 'o-', linewidth=2, label='Your RL', color=colors['Your RL'])
    ax.fill(angles, rl_means, alpha=0.25, color=colors['Your RL'])
    ax.plot(angles, fcf_means, 'o-', linewidth=2, label='ForecastCF', color=colors['ForecastCF'])
    ax.fill(angles, fcf_means, alpha=0.25, color=colors['ForecastCF'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_radar.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparison_radar.png")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RL vs ForecastCF results")
    parser.add_argument(
        "--rl-results",
        type=str,
        required=True,
        help="Path to your RL results CSV"
    )
    parser.add_argument(
        "--fcf-results",
        type=str,
        required=True,
        help="Path to ForecastCF results CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/comparison_plots",
        help="Directory to save comparison plots"
    )
    
    args = parser.parse_args()
    
    # Charger les résultats
    print(f"Loading RL results from: {args.rl_results}")
    print(f"Loading ForecastCF results from: {args.fcf_results}")
    
    rl_df, fcf_df = load_results(args.rl_results, args.fcf_results)
    
    print(f"\nRL results: {len(rl_df)} runs")
    print(f"ForecastCF results: {len(fcf_df)} runs")
    print()
    
    # Comparer
    compare_methods(rl_df, fcf_df)
    
    # Générer les graphiques
    print("\nGenerating comparison plots...")
    plot_comparison(rl_df, fcf_df, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
