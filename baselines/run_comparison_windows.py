#!/usr/bin/env python
# coding: utf-8
"""
Version Windows de la comparaison RL vs ForecastCF.
N'utilise pas bash, fonctionne directement avec Python.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """Affiche un en-tête formaté."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def print_section(title):
    """Affiche une section."""
    print("\n" + "-"*80)
    print(title)
    print("-"*80)


def run_forecastcf_baseline():
    """Étape 1: Lancer ForecastCF baseline."""
    print_header("Étape 1/3: Exécution de ForecastCF baseline")
    
    # Paramètres
    model_path = "assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_S.pth"
    config_path = "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
    data_path = "assets/datasets/ETTh1.csv"
    output_file = "baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv"
    
    # Vérifier que les fichiers existent
    if not Path(model_path).exists():
        print(f"✗ Checkpoint non trouvé: {model_path}")
        print("\nVeuillez d'abord entraîner le modèle:")
        print("  python -m src.experiments.forecasting.run --config", config_path)
        return False
    
    if not Path(data_path).exists():
        print(f"✗ Dataset non trouvé: {data_path}")
        return False
    
    # Créer le répertoire de sortie
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Seeds à utiliser
    seeds = [1, 9, 30, 33, 39]
    
    print(f"\nLancement de ForecastCF pour {len(seeds)} seeds...")
    print(f"Modèle: {model_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_file}\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] Seed {seed}...")
        
        cmd = [
            "python", "baselines/ForecastCF/src/cf_search_pytorch.py",
            "--model-path", model_path,
            "--model-type", "itransformer",
            "--config-path", config_path,
            "--dataset", "etth1",
            "--data-path", data_path,
            "--horizon", "48",
            "--back-horizon", "96",
            "--center", "median",
            "--desired-shift", "0",
            "--desired-change", "-0.1",
            "--poly-order", "1",
            "--fraction-std", "1.0",
            "--random-seed", str(seed),
            "--output", output_file,
            "--device", "cpu",  # Utiliser CPU au lieu de cuda
            "--test-samples", "1000"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ Seed {seed} terminé")
            
            # Afficher les dernières lignes de sortie
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"    {line}")
        
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Erreur pour seed {seed}")
            if e.stderr:
                print(f"    Erreur: {e.stderr[:200]}")
            return False
    
    print(f"\n✓ ForecastCF terminé pour tous les seeds")
    print(f"  Résultats: {output_file}")
    return True


def check_rl_results(rl_results_path=None):
    """Étape 2: Vérifier que les résultats RL existent."""
    print_header("Étape 2/3: Vérification des résultats RL")
    
    if rl_results_path is None:
        # Chercher dans plusieurs emplacements possibles
        possible_paths = [
            "assets/results/rl_cf_etth1_itransformer.csv",
            "assets/results/rl/rl_cf_etth1_itransformer.csv",
            "results/rl_cf_etth1_itransformer.csv",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                rl_results_path = path
                break
    
    if rl_results_path is None or not Path(rl_results_path).exists():
        print("\n⚠ Fichier de résultats RL non trouvé")
        print("\nEmplacements vérifiés:")
        for path in possible_paths:
            print(f"  - {path}")
        
        print("\nOptions:")
        print("  1. Spécifiez le chemin avec --rl-results")
        print("  2. Ou exécutez d'abord votre méthode RL")
        print("\nExemple:")
        print("  python baselines/run_comparison_windows.py --rl-results path/to/results.csv")
        return False, None
    
    print(f"\n✓ Résultats RL trouvés: {rl_results_path}")
    
    # Vérifier le format du fichier
    try:
        import pandas as pd
        df = pd.read_csv(rl_results_path)
        print(f"  Nombre de lignes: {len(df)}")
        print(f"  Colonnes: {', '.join(df.columns[:5])}...")
    except Exception as e:
        print(f"  ⚠ Impossible de lire le fichier: {e}")
    
    return True, rl_results_path


def compare_results(rl_results, fcf_results=None, output_dir=None):
    """Étape 3: Comparer les résultats."""
    print_header("Étape 3/3: Comparaison des résultats")
    
    if fcf_results is None:
        fcf_results = "baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv"
    
    if output_dir is None:
        output_dir = "baselines/comparison_plots"
    
    fcf_results = Path(fcf_results)
    
    if not fcf_results.exists():
        print(f"\n✗ Résultats ForecastCF non trouvés: {fcf_results}")
        print("Veuillez d'abord exécuter l'étape 1.")
        return False
    
    print(f"\nComparaison:")
    print(f"  RL:          {rl_results}")
    print(f"  ForecastCF:  {fcf_results}")
    print(f"  Output:      {output_dir}\n")
    
    cmd = [
        "python", "baselines/compare_results.py",
        "--rl-results", str(rl_results),
        "--fcf-results", str(fcf_results),
        "--output-dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✓ Comparaison terminée avec succès")
        print(f"\nGraphiques sauvegardés dans: {output_dir}/")
        print("  - comparison_barplot.png")
        print("  - comparison_boxplot.png")
        print("  - comparison_radar.png")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erreur lors de la comparaison")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comparaison RL vs ForecastCF (version Windows)"
    )
    parser.add_argument(
        "--rl-results",
        type=str,
        help="Chemin vers les résultats RL (CSV)"
    )
    parser.add_argument(
        "--fcf-results",
        type=str,
        help="Chemin vers les résultats ForecastCF (CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baselines/comparison_plots",
        help="Répertoire de sortie pour les graphiques"
    )
    parser.add_argument(
        "--skip-fcf",
        action="store_true",
        help="Sauter l'exécution de ForecastCF (si déjà fait)"
    )
    parser.add_argument(
        "--only-compare",
        action="store_true",
        help="Seulement comparer (ForecastCF et RL déjà exécutés)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Comparaison RL vs ForecastCF")
    print("Modèle: iTransformer | Dataset: ETTh1")
    print("Version: Windows (Python pur)")
    print("="*80)
    
    # Si seulement comparaison
    if args.only_compare:
        if not args.rl_results:
            print("\n✗ --rl-results requis avec --only-compare")
            sys.exit(1)
        
        success = compare_results(args.rl_results, args.fcf_results, args.output_dir)
        sys.exit(0 if success else 1)
    
    # Étape 1: ForecastCF
    if not args.skip_fcf:
        if not run_forecastcf_baseline():
            print("\n✗ Échec de l'exécution de ForecastCF")
            sys.exit(1)
    else:
        print_section("Étape 1: Sautée (--skip-fcf)")
    
    # Étape 2: Vérification RL
    success, rl_results = check_rl_results(args.rl_results)
    if not success:
        print("\n✗ Résultats RL non trouvés")
        sys.exit(1)
    
    # Étape 3: Comparaison
    if not compare_results(rl_results, args.fcf_results, args.output_dir):
        print("\n✗ Échec de la comparaison")
        sys.exit(1)
    
    # Succès
    print("\n" + "="*80)
    print("✓ COMPARAISON TERMINÉE AVEC SUCCÈS")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
