"""
Entraîne les 5 forecasters sur ETTh2 séquentiellement.

Usage:
    python scripts/train_etth2_forecasters.py
"""

import subprocess
import sys

CONFIGS = [
    "assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
]

for cfg in CONFIGS:
    model = cfg.split("/")[-2]
    print(f"\n{'='*60}")
    print(f"Training: {model} on ETTh2")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "src/experiments/forecasting/run.py", "--config", cfg]
    )
    if result.returncode != 0:
        print(f"[ERROR] {model} failed — stopping.")
        sys.exit(result.returncode)

print("\nAll 5 forecasters trained on ETTh2 ✓")
