#!/bin/bash

# ==========================================
# RL Counterfactual Generation for PatchTST
# Dataset: ETTh1
# Configuration: 96 -> 48, univariate
# ==========================================

CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/config_patchtst.json}

echo "=========================================="
echo "RL CF Training - PatchTST on ETTh1"
echo "=========================================="
echo "Config: $CONFIG"
echo

python -m src.experiments.rl_cf.run --config "$CONFIG"

echo
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Results saved in:"
echo "  - Checkpoints: assets/checkpoints/etth1_chpts/RL_patchtst/"
echo "  - Figures: assets/figures/etth1_patchtst/"
echo "  - Results: assets/results/etth1_patchtst/"
