"""
Étape 2b — Entraîne les 5 agents RL sur ETTh2.
À lancer APRÈS train_etth2_ae.py.

Usage:
    python scripts/train_etth2_rl.py
"""

import subprocess
import sys

AE_CONFIG = "assets/configs/models/etth2_dataset/ae/tcn_ae.json"

RL_MODELS = [
    (
        "iTransformer",
        "assets/configs/models/etth2_dataset/RL/config_itransformer.json",
        "assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    ),
    (
        "PatchTST",
        "assets/configs/models/etth2_dataset/RL/config_patchtst.json",
        "assets/configs/models/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
    ),
    (
        "TimesNet",
        "assets/configs/models/etth2_dataset/RL/config_timesnet.json",
        "assets/configs/models/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
    ),
    (
        "GRU",
        "assets/configs/models/etth2_dataset/RL/config_gru.json",
        "assets/configs/models/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
    ),
    (
        "DLinear",
        "assets/configs/models/etth2_dataset/RL/config_dlinear.json",
        "assets/configs/models/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
    ),
]

for model, rl_cfg, f_cfg in RL_MODELS:
    print(f"\n{'='*60}")
    print(f"Training RL agent — {model} on ETTh2")
    print(f"{'='*60}")

    result = subprocess.run([
        sys.executable,
        "src/experiments/rl_cf/run_last.py",
        "--config",          rl_cfg,
        "--forecast_config", f_cfg,
        "--ae_config",       AE_CONFIG,
    ])

    if result.returncode != 0:
        print(f"[ERROR] {model} RL training failed — stopping.")
        sys.exit(result.returncode)

print("\nAll 5 RL agents trained on ETTh2 ✓")
