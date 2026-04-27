#!/usr/bin/env python
# coding: utf-8
"""
Exécution étape par étape de la comparaison RL vs ForecastCF.
Permet de contrôler chaque étape individuellement.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def step1_run_forecastcf():
    """Étape 1: Lancer ForecastCF baseline."""
    print("\n" + "="*80)
    print("ÉTAPE 1: Exécution de ForecastCF baseline")
    print("="*80)
    
    fcf_dir = Path("baselines/ForecastCF")
    script = fcf_dir / "run_etth1_itransformer.sh"
    
    if not script.exists():
        print(f"✗ Script non trouvé: {script}")
        return False
    
    print(f"\nLancement de: {script}")
    print("Cela peut prendre 10-30 minutes selon votre GPU...\n")
    
    try:
        subprocess.run(
            ["bash", script.name],
            cwd=fcf_dir,
            check=True
        )
        print("\n✓ ForecastCF terminé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erreur lors de l'exécution: {e}")
        return False


def step2_check_rl_results(rl_results_path=None):
    """Étape 2: Vérifier que les résultats RL existent."""
    print("\n" + "="*80)
    print("ÉTAPE 2: Vérification des résultats RL")
    print("="*80)
    
    if rl_results_path is None:
        rl_results_path = "assets/results/rl_cf_etth1_itransformer.csv"
    
    rl_results = Path(rl_results_path)
    
    if not rl_results.exists():
        print(f"\n⚠ Fichier non trouvé: {rl_results}")
        print("\nVeuillez spécifier le chemin avec --rl-results")
        print("ou exécuter d'abord votre méthode RL.")
        return False, None
    
    print(f"\n✓ Résultats RL trouvés: {rl_results}")
    return True, rl_results


def step3_compare_results(rl_results, fcf_results=None, output_dir=None):
    """Étape 3: Comparer les résultats."""
    print("\n" + "="*80)
    print("ÉTAPE 3: Comparaison des résultats")
    print("="*80)
    
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
    
    try:
        subprocess.run([
            "python", "baselines/compare_results.py",
            "--rl-results", str(rl_results),
            "--fcf-results", str(fcf_results),
            "--output-dir", str(output_dir)
        ], check=True)
        
        print("\n✓ Comparaison terminée avec succès")
        print(f"\nGraphiques sauvegardés dans: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erreur lors de la comparaison: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Exécution étape par étape de la comparaison RL vs ForecastCF"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="Exécuter une étape spécifique (1=ForecastCF, 2=Vérif RL, 3=Comparaison)"
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
    
    args = parser.parse_args()
    
    print("="*80)
    print("Comparaison RL vs ForecastCF - Exécution étape par étape")
    print("="*80)
    
    # Si une étape spécifique est demandée
    if args.step:
        if args.step == 1:
            success = step1_run_forecastcf()
            sys.exit(0 if success else 1)
        
        elif args.step == 2:
            success, rl_results = step2_check_rl_results(args.rl_results)
            sys.exit(0 if success else 1)
        
        elif args.step == 3:
            if not args.rl_results:
                print("\n✗ --rl-results requis pour l'étape 3")
                sys.exit(1)
            success = step3_compare_results(
                args.rl_results,
                args.fcf_results,
                args.output_dir
            )
            sys.exit(0 if success else 1)
    
    # Sinon, exécuter toutes les étapes
    print("\nExécution de toutes les étapes...\n")
    
    # Étape 1: ForecastCF
    if not args.skip_fcf:
        if not step1_run_forecastcf():
            print("\n✗ Échec à l'étape 1")
            sys.exit(1)
    else:
        print("\n⊳ Étape 1 sautée (--skip-fcf)")
    
    # Étape 2: Vérification RL
    success, rl_results = step2_check_rl_results(args.rl_results)
    if not success:
        print("\n✗ Échec à l'étape 2")
        sys.exit(1)
    
    # Étape 3: Comparaison
    if not step3_compare_results(rl_results, args.fcf_results, args.output_dir):
        print("\n✗ Échec à l'étape 3")
        sys.exit(1)
    
    # Succès
    print("\n" + "="*80)
    print("✓ TOUTES LES ÉTAPES TERMINÉES AVEC SUCCÈS")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
