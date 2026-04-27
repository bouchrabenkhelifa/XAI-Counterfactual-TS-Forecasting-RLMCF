#!/usr/bin/env python
# coding: utf-8
"""
RL Counterfactual Generation for PatchTST on ETTh1
Version Python pour Windows
"""

import subprocess
import sys
from pathlib import Path


def main():
    # Configuration
    config_path = "assets/configs/models/etth1_dataset/RL_ablations/config_patchtst.json"
    
    # Vérifier que la config existe
    if not Path(config_path).exists():
        print(f"❌ Configuration non trouvée: {config_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("RL CF Training - PatchTST on ETTh1")
    print("=" * 80)
    print(f"Config: {config_path}")
    print()
    
    # Lancer l'entraînement
    try:
        subprocess.run([
            sys.executable, "-m", "src.experiments.rl_cf.run_last_v2",
            "--config", config_path
        ], check=True)
        
        print()
        print("=" * 80)
        print("✅ Training completed!")
        print("=" * 80)
        print("Results saved in:")
        print("  - Checkpoints: assets/checkpoints/etth1_chpts/RL_patchtst/")
        print("  - Figures: assets/figures/etth1_patchtst/")
        print("  - Results: assets/results/etth1_patchtst/")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("❌ Training failed!")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
