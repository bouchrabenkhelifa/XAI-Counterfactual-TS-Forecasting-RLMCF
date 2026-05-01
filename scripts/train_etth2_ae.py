"""
Étape 2a — Entraîne l'AE sur ETTh2.
À lancer APRÈS train_etth2_forecasters.py.

Usage:
    python scripts/train_etth2_ae.py
"""

import subprocess
import sys

AE_CONFIG = "assets/configs/models/etth2_dataset/ae/tcn_ae.json"

print(f"\n{'='*60}")
print("Training AE on ETTh2")
print(f"{'='*60}")

result = subprocess.run([
    sys.executable,
    "src/experiments/autoencoder/run.py",
    "--config", AE_CONFIG,
])

if result.returncode != 0:
    print("[ERROR] AE training failed.")
    sys.exit(result.returncode)

print("\nAE trained on ETTh2 ✓")
