#!/usr/bin/env python
# coding: utf-8
"""
Génère un tableau LaTeX avec les métriques d'évaluation pour tous les modèles.
"""

import json
import os
from pathlib import Path


def load_evaluation(model_name, results_dir):
    """Charge les résultats d'évaluation pour un modèle."""
    eval_file = results_dir / f"rl_cf_{model_name}_etth1_evaluation.json"
    
    if not eval_file.exists():
        print(f"⚠ Fichier non trouvé: {eval_file}")
        return None
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    return data


def extract_metrics(data):
    """Extrait les métriques principales."""
    if data is None:
        return None
    
    metrics = {
        'validity': data.get('validity_ratio', {}).get('mean', 0.0),
        'validity_std': data.get('validity_ratio', {}).get('std', 0.0),
        'step_auc': data.get('stepwise_auc', {}).get('mean', 0.0),
        'step_auc_std': data.get('stepwise_auc', {}).get('std', 0.0),
        'proximity': data.get('proximity_l2', {}).get('mean', 0.0),
        'proximity_std': data.get('proximity_l2', {}).get('std', 0.0),
        'compactness': data.get('compactness', {}).get('mean', 0.0),
        'compactness_std': data.get('compactness', {}).get('std', 0.0),
        'temporal_consistency': data.get('temporal_consistency', {}).get('mean', 0.0),
        'temporal_consistency_std': data.get('temporal_consistency', {}).get('std', 0.0),
        'plausibility': data.get('plausibility_ensemble', {}).get('mean', 0.0),
        'plausibility_std': data.get('plausibility_ensemble', {}).get('std', 0.0),
    }
    
    return metrics


def format_metric(value, std, bold_best=False, lower_better=False):
    """Formate une métrique avec son écart-type."""
    if value == 0.0 and std == 0.0:
        return "---"
    
    formatted = f"{value:.4f}"
    if std > 0:
        formatted += f" $\\pm$ {std:.4f}"
    
    if bold_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted


def find_best_values(all_metrics, metric_name, lower_better=False):
    """Trouve la meilleure valeur pour une métrique."""
    values = [m[metric_name] for m in all_metrics.values() if m is not None]
    if not values:
        return None
    
    if lower_better:
        return min(values)
    else:
        return max(values)


def generate_latex_table(models, results_base_dir, output_file):
    """Génère le tableau LaTeX."""
    
    # Charger les données
    all_data = {}
    all_metrics = {}
    
    for model_name, model_display in models.items():
        data = load_evaluation(model_name, results_base_dir / f"etth1_{model_name}")
        all_data[model_name] = data
        all_metrics[model_name] = extract_metrics(data)
    
    # Trouver les meilleures valeurs
    best_values = {
        'validity': find_best_values(all_metrics, 'validity', lower_better=False),
        'step_auc': find_best_values(all_metrics, 'step_auc', lower_better=False),
        'proximity': find_best_values(all_metrics, 'proximity', lower_better=True),
        'compactness': find_best_values(all_metrics, 'compactness', lower_better=False),
        'temporal_consistency': find_best_values(all_metrics, 'temporal_consistency', lower_better=False),
        'plausibility': find_best_values(all_metrics, 'plausibility', lower_better=True),
    }
    
    # Générer le tableau LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Evaluation metrics for counterfactual generation on ETTh1 dataset}")
    latex.append("\\label{tab:cf_evaluation}")
    latex.append("\\begin{tabular}{l|cccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{Validity} $\\uparrow$ & \\textbf{Step AUC} $\\uparrow$ & \\textbf{Proximity} $\\downarrow$ & \\textbf{Compactness} $\\uparrow$ & \\textbf{Temp. Cons.} $\\uparrow$ & \\textbf{Plausibility} $\\downarrow$ \\\\")
    latex.append("\\midrule")
    
    # Ajouter les lignes pour chaque modèle
    for model_name, model_display in models.items():
        metrics = all_metrics[model_name]
        
        if metrics is None:
            latex.append(f"{model_display} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        
        # Formater chaque métrique
        validity = format_metric(
            metrics['validity'], metrics['validity_std'],
            bold_best=(metrics['validity'] == best_values['validity'])
        )
        step_auc = format_metric(
            metrics['step_auc'], metrics['step_auc_std'],
            bold_best=(metrics['step_auc'] == best_values['step_auc'])
        )
        proximity = format_metric(
            metrics['proximity'], metrics['proximity_std'],
            bold_best=(metrics['proximity'] == best_values['proximity']),
            lower_better=True
        )
        compactness = format_metric(
            metrics['compactness'], metrics['compactness_std'],
            bold_best=(metrics['compactness'] == best_values['compactness'])
        )
        temporal = format_metric(
            metrics['temporal_consistency'], metrics['temporal_consistency_std'],
            bold_best=(metrics['temporal_consistency'] == best_values['temporal_consistency'])
        )
        plausibility = format_metric(
            metrics['plausibility'], metrics['plausibility_std'],
            bold_best=(metrics['plausibility'] == best_values['plausibility']),
            lower_better=True
        )
        
        latex.append(f"{model_display} & {validity} & {step_auc} & {proximity} & {compactness} & {temporal} & {plausibility} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    # Sauvegarder
    latex_content = "\n".join(latex)
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Tableau LaTeX sauvegardé: {output_file}")
    print()
    print("=" * 80)
    print("TABLEAU LATEX")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)
    
    return latex_content


def main():
    # Définir les modèles
    models = {
        'v2': 'iTransformer',  # Utiliser v2 au lieu de final
        'dlinear': 'DLinear',
        'gru': 'GRU',
        'patchtst': 'PatchTST',
        'timesnet': 'TimesNet',
    }
    
    # Chemins
    results_base_dir = Path("assets/results")
    output_file = Path("assets/results/comparison/latex_table.tex")
    
    # Créer le répertoire de sortie
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Générer le tableau
    print("=" * 80)
    print("Génération du tableau LaTeX")
    print("=" * 80)
    print()
    
    generate_latex_table(models, results_base_dir, output_file)
    
    print()
    print("Pour utiliser dans votre document LaTeX:")
    print("  1. Ajoutez dans le préambule: \\usepackage{booktabs}")
    print("  2. Incluez le fichier: \\input{assets/results/comparison/latex_table.tex}")
    print()


if __name__ == "__main__":
    main()
