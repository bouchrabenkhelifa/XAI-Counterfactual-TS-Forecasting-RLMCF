#!/bin/bash
# Run ForecastCF on ETTh1 with iTransformer (PyTorch)
# Paramètres alignés avec votre configuration RL:
#   back_horizon=96, horizon=48, desired_change=-0.1, fraction_std=1.0

MODEL_PATH="../../../assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_S.pth"
CONFIG_PATH="../../../assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
DATA_PATH="../../../assets/datasets/ETTh1.csv"
OUTPUT="results/forecastcf_etth1_itransformer.csv"

echo "========================================"
echo "ForecastCF + iTransformer on ETTh1"
echo "========================================"

for seed in 1 9 30 33 39
do
    echo "Running seed=$seed"
    python src/cf_search_pytorch.py \
        --model-path "$MODEL_PATH" \
        --model-type itransformer \
        --config-path "$CONFIG_PATH" \
        --dataset etth1 \
        --data-path "$DATA_PATH" \
        --horizon 48 \
        --back-horizon 96 \
        --center median \
        --desired-shift 0 \
        --desired-change -0.1 \
        --poly-order 1 \
        --fraction-std 1.0 \
        --random-seed $seed \
        --output "$OUTPUT" \
        --device cuda \
        --test-samples 1000
done

echo "Done. Results in $OUTPUT"
