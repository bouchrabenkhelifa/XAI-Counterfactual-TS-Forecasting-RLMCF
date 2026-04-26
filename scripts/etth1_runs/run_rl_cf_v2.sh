#!/bin/bash

# GRU
python -m src.experiments.rl_cf.run_last_v2 \
    --config          assets/configs/models/etth1_dataset/RL_ablations/config_gru.json \
    --forecast_config assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json \
    --ae_config       assets/configs/models/etth1_dataset/ae/tcn_ae.json

# DLinear
python -m src.experiments.rl_cf.run_last_v2 \
    --config          assets/configs/models/etth1_dataset/RL_ablations/config_dlinear.json \
    --forecast_config assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json \
    --ae_config       assets/configs/models/etth1_dataset/ae/tcn_ae.json

# TimesNet
python -m src.experiments.rl_cf.run_last_v2 \
    --config          assets/configs/models/etth1_dataset/RL_ablations/config_timesnet.json \
    --forecast_config assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json \
    --ae_config       assets/configs/models/etth1_dataset/ae/tcn_ae.json
