#!/bin/bash

CONFIG=${1:-assets/configs/models/etth1_dataset/anomaly_detector/plausibility.json}

echo "Running plausibility / anomaly detector Config: $CONFIG"

python -m src.experiments.anomaly_detection.run --config "$CONFIG"
