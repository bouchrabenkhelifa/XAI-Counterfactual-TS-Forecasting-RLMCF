#!/bin/bash

CONFIG=${1:-assets/configs/models/etth1_dataset/ae/tcn_ae.json}

echo "Running TCN AutoEncoder Config: $CONFIG"

python -m src.experiments.autoencoder.run --config "$CONFIG"