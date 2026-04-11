#!/bin/bash

CONFIG=${1:-assets/configs/models/etth1_dataset/ae/tcn_ae.json}

echo "======================================="
echo "Running TCN AutoEncoder"
echo "Config: $CONFIG"
echo "======================================="

python -m src.experiments.autoencoder.run --config "$CONFIG"