#!/usr/bin/env python
# coding: utf-8
"""
Script de vérification pour s'assurer que tout est prêt pour la comparaison.
"""

import os
import sys
from pathlib import Path


def check_file(path, description):
    """Vérifie qu'un fichier existe."""
    if os.path.exists(path):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Fichier manquant: {path}")
        return False


def check_directory(path, description):
    """Vérifie qu'un répertoire existe."""
    if os.path.isdir(path):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Répertoire manquant: {path}")
        return False


def check_python_package(package_name):
    """Vérifie qu'un package Python est installé."""
    try:
        __import__(package_name)
        print(f"✅ Package Python: {package_name}")
        return True
    except ImportError:
        print(f"❌ Package Python: {package_name}")
        print(f"   Installez avec: pip install {package_name}")
        return False


def main():
    print("=" * 80)
    print("VÉRIFICATION DE LA CONFIGURATION")
    print("Comparaison RL vs ForecastCF sur iTransformer + ETTh1")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # 1. Vérifier les fichiers de code
    print("1. Fichiers de code")
    print("-" * 80)
    all_ok &= check_file(
        "baselines/ForecastCF/src/pytorch_adapter.py",
        "Adapter PyTorch"
    )
    all_ok &= check_file(
        "baselines/ForecastCF/src/cf_search_pytorch.py",
        "Script CF search PyTorch"
    )
    all_ok &= check_file(
        "baselines/ForecastCF/src/forecastcf.py",
        "Code ForecastCF original"
    )
    all_ok &= check_file(
        "baselines/ForecastCF/src/_helper.py",
        "Helpers ForecastCF"
    )
    all_ok &= check_file(
        "baselines/ForecastCF/src/_utils.py",
        "Utils ForecastCF"
    )
    all_ok &= check_file(
        "baselines/compare_results.py",
        "Script de comparaison"
    )
    print()
    
    # 2. Vérifier les scripts
    print("2. Scripts d'exécution")
    print("-" * 80)
    all_ok &= check_file(
        "baselines/ForecastCF/run_etth1_itransformer.sh",
        "Script bash iTransformer"
    )
    all_ok &= check_file(
        "baselines/run_comparison.sh",
        "Script bash comparaison complète"
    )
    print()
    
    # 3. Vérifier les modèles et configs
    print("3. Modèles et configurations")
    print("-" * 80)
    all_ok &= check_file(
        "assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_S.pth",
        "Checkpoint iTransformer ETTh1"
    )
    all_ok &= check_file(
        "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "Config iTransformer ETTh1"
    )
    print()
    
    # 4. Vérifier les données
    print("4. Données")
    print("-" * 80)
    all_ok &= check_file(
        "assets/datasets/ETTh1.csv",
        "Dataset ETTh1"
    )
    print()
    
    # 5. Vérifier les répertoires
    print("5. Répertoires")
    print("-" * 80)
    all_ok &= check_directory(
        "baselines/ForecastCF/results",
        "Répertoire résultats ForecastCF"
    )
    if not os.path.isdir("baselines/ForecastCF/results"):
        print("   Création du répertoire...")
        os.makedirs("baselines/ForecastCF/results", exist_ok=True)
        all_ok = True
    
    all_ok &= check_directory(
        "src/models/Forecaster",
        "Répertoire modèles forecasting"
    )
    print()
    
    # 6. Vérifier les packages Python
    print("6. Packages Python")
    print("-" * 80)
    all_ok &= check_python_package("torch")
    all_ok &= check_python_package("numpy")
    all_ok &= check_python_package("pandas")
    all_ok &= check_python_package("matplotlib")
    all_ok &= check_python_package("seaborn")
    all_ok &= check_python_package("tensorflow")
    print()
    
    # 7. Vérifier CUDA (optionnel)
    print("7. GPU (optionnel)")
    print("-" * 80)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"   Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA non disponible (CPU sera utilisé)")
            print("   Les calculs seront plus lents mais fonctionneront")
    except:
        print("⚠️  Impossible de vérifier CUDA")
    print()
    
    # Résumé
    print("=" * 80)
    if all_ok:
        print("✅ TOUT EST PRÊT!")
        print()
        print("Vous pouvez maintenant lancer la comparaison:")
        print()
        print("  bash baselines/run_comparison.sh")
        print()
        print("Ou étape par étape:")
        print()
        print("  1. cd baselines/ForecastCF")
        print("  2. bash run_etth1_itransformer.sh")
        print("  3. cd ../..")
        print("  4. python baselines/compare_results.py \\")
        print("       --rl-results <vos_resultats_rl.csv> \\")
        print("       --fcf-results baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv")
        print()
    else:
        print("❌ CONFIGURATION INCOMPLÈTE")
        print()
        print("Veuillez corriger les problèmes ci-dessus avant de continuer.")
        print()
        print("Problèmes courants:")
        print("  - Modèle non entraîné: bash scripts/etth1_runs/run_itransformer.sh")
        print("  - Packages manquants: pip install torch numpy pandas matplotlib seaborn tensorflow")
        print()
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
