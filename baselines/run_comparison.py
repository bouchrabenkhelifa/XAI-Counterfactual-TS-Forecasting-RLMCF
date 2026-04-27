#!/usr/bin/env python
# coding: utf-8
"""
Version Python du script de comparaison.
Lance ForecastCF puis compare avec les résultats RL.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, cwd=None, description=""):
    """Exécute une commande shell et affiche la sortie."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Commande: {cmd}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} terminé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de {description}")
        print(f"Code de sortie: {e.returncode}")
        return False


def main():
    print("="*80)
    print("Comparaison RL vs ForecastCF")
    print("Modèle: iTransformer | Dataset: ETTh1")
    print("="*80)
    
    # Chemins
    root_dir = Path(__file__).parent.parent
    fcf_dir = root_dir / "baselines" / "ForecastCF"
    
    rl_results = root_dir / "assets" / "results" / "rl_cf_etth1_itransformer.csv"
    fcf_results = fcf_dir / "results" / "forecastcf_etth1_itransformer.csv"
    comparison_dir = root_dir / "baselines" / "comparison_plots"
    
    # Étape 1: Lancer ForecastCF baseline
    print("\nÉtape 1/3: Exécution de ForecastCF baseline...")
    print("-"*80)
    
    fcf_script = fcf_dir / "run_etth1_itransformer.sh"
    
    if not fcf_script.exists():
        print(f"✗ Script non trouvé: {fcf_script}")
        sys.exit(1)
    
    success = run_command(
        f"bash {fcf_script.name}",
        cwd=fcf_dir,
        description="ForecastCF baseline"
    )
    
    if not success:
        print("\n✗ Échec de l'exécution de ForecastCF")
        sys.exit(1)
    
    # Étape 2: Vérifier que les résultats RL existent
    print("\nÉtape 2/3: Vérification des résultats RL...")
    print("-"*80)
    
    if not rl_results.exists():
        print(f"⚠ Fichier de résultats RL non trouvé: {rl_results}")
        print("\nVeuillez d'abord exécuter votre méthode RL et sauvegarder les résultats.")
        print("\nExemple de commande pour lancer votre RL:")
        print("  python -m src.experiments.rl_cf.run --config <votre_config>")
        sys.exit(1)
    
    print(f"✓ Résultats RL trouvés: {rl_results}")
    
    # Étape 3: Comparer les résultats
    print("\nÉtape 3/3: Comparaison des résultats...")
    print("-"*80)
    
    compare_script = root_dir / "baselines" / "compare_results.py"
    
    cmd = (
        f"python {compare_script} "
        f"--rl-results {rl_results} "
        f"--fcf-results {fcf_results} "
        f"--output-dir {comparison_dir}"
    )
    
    success = run_command(
        cmd,
        cwd=root_dir,
        description="Comparaison des résultats"
    )
    
    if not success:
        print("\n✗ Échec de la comparaison")
        sys.exit(1)
    
    # Résumé final
    print("\n" + "="*80)
    print("✓ Comparaison terminée!")
    print("="*80)
    print()
    print(f"Résultats sauvegardés dans: {comparison_dir}")
    print("  - comparison_barplot.png")
    print("  - comparison_boxplot.png")
    print("  - comparison_radar.png")
    print()


if __name__ == "__main__":
    main()
