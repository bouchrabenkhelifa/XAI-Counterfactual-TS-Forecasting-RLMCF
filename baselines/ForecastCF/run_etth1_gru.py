"""
ForecastCF — ETTh1 avec modèle GRU
Lance les 5 seeds séquentiellement, équivalent du .sh sur Windows.

Usage (depuis la racine du projet) :
    python baselines/ForecastCF/run_etth1_gru.py
"""

import subprocess
import sys

MODEL_PATH  = "assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_gru_S.pth"
CONFIG_PATH = "assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json"
DATA_PATH   = "assets/datasets/ETTh1.csv"
OUTPUT      = "baselines/ForecastCF/results/forecastcf_etth1_gru.csv"
SEEDS       = [1, 9, 30]

for seed in SEEDS:
    print("=" * 50)
    print(f"ForecastCF | ETTh1 | GRU | seed={seed}")
    print("=" * 50)

    cmd = [
        sys.executable,
        "baselines/ForecastCF/src/cf_search_pytorch.py",
        "--model-path",     MODEL_PATH,
        "--model-type",     "gru",
        "--config-path",    CONFIG_PATH,
        "--dataset",        "etth1",
        "--data-path",      DATA_PATH,
        "--horizon",        "48",
        "--back-horizon",   "96",
        "--center",         "median",
        "--desired-shift",  "0",
        "--desired-change", "-0.1",
        "--poly-order",     "1",
        "--fraction-std",   "1.0",
        "--random-seed",    str(seed),
        "--output",         OUTPUT,
        "--device",         "cpu",
        "--test-samples",   "30",
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] seed={seed} failed with code {result.returncode}")
        sys.exit(result.returncode)

print(f"\nDone. Results → {OUTPUT}")
