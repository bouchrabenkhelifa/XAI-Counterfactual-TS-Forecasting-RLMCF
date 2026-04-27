#!/bin/bash

# ==========================================
# Available forecasting configs (ETTh1)
# ==========================================
# itransformer_96_48_S.json  → univariate, short horizon
# itransformer_96_96_S.json  → univariate, longer horizon

# itransformer_96_96.json    → multivariate, standard setup
# itransformer_192_96.json   → multivariate,longer input window
# ==========================================

CONFIG=${1:-assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json}

echo "Running config: $CONFIG"

python -m src.experiments.forecasting.run --config "$CONFIG"