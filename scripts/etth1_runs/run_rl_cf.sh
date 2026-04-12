# Available configs:
# full            -> full model           : CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/full.json}

# wo_proximity    -> without proximity    : CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/wo_proximity.json}

# wo_plausibility -> without plausibility : CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/wo_plausibility.json}

# wo_ae           -> without autoencoder  : CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/wo_ae.json}*

# wo_RL_ablations -> without RL update    : CONFIG=${1:-assets/configs/models/etth1_dataset/RL/wo_rl.json}

# ============================================================


CONFIG=${1:-assets/configs/models/etth1_dataset/RL_ablations/full.json}

echo "Running config: $CONFIG"

python -m src.experiments.rl_cf.run --config "$CONFIG"