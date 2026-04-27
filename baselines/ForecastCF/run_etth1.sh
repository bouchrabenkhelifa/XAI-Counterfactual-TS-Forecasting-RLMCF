#!/bin/bash
# Run ForecastCF on ETTh1 — parameters aligned with our RL setup:
#   back_horizon=96, horizon=48, desired_change=-0.1, fraction_std=1.0
#
# Usage (from baselines/ForecastCF/):
#   bash run_etth1.sh

for seed in 1 9 30 33 39
do
    echo "========================================"
    echo "ETTh1 | seed=$seed"
    echo "========================================"
    python src/cf_search.py \
        --dataset       etth1 \
        --horizon       48 \
        --back-horizon  96 \
        --split-size    0.7 0.15 0.15 \
        --stride-size   1 \
        --center        median \
        --desired-shift 0 \
        --desired-change -0.1 \
        --poly-order    1 \
        --fraction-std  1.0 \
        --random-seed   $seed \
        --output        results/forecastcf_etth1.csv
done

echo "Done. Results in results/forecastcf_etth1.csv"
