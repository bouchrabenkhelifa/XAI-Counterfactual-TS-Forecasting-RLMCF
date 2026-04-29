#!/bin/bash
# ForecastCF — ETTh1 avec modèle GRU
# Paramètres alignés avec notre setup RL :
#   back_horizon=96, horizon=48, desired_change=-0.1, fraction_std=1.0
#
# Usage (depuis la racine du projet) :
#   bash baselines/ForecastCF/run_etth1_gru.sh

MODEL_PATH="assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_gru_S.pth"
CONFIG_PATH="assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json"
DATA_PATH="assets/datasets/ETTh1.csv"
OUTPUT="baselines/ForecastCF/results/forecastcf_etth1_gru.csv"

for seed in 1 9 30 33 39
do
    echo "========================================"
    echo "ForecastCF | ETTh1 | GRU | seed=$seed"
    echo "========================================"
    python baselines/ForecastCF/src/cf_search_pytorch.py \
        --model-path     "$MODEL_PATH" \
        --model-type     gru \
        --config-path    "$CONFIG_PATH" \
        --dataset        etth1 \
        --data-path      "$DATA_PATH" \
        --horizon        48 \
        --back-horizon   96 \
        --center         median \
        --desired-shift  0 \
        --desired-change -0.1 \
        --poly-order     1 \
        --fraction-std   1.0 \
        --random-seed    $seed \
        --output         "$OUTPUT" \
        --device         cpu \
        --test-samples   100
done

echo "Done. Results → $OUTPUT"
