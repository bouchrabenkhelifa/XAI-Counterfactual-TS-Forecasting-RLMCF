"""
Calculer global_sigma (std du train set) pour un dataset.

Usage:
    python scripts/compute_global_sigma.py --dataset etth1 --model gru
"""

import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.RL_trainers.trainer_last import prepare_rl_data
from src.utils.config import load_config


def compute_global_sigma(dataset, model):
    """
    Calcule le std global du train set (OT channel).
    """
    device = torch.device("cpu")
    
    # Charger configs
    cfg_f_path = f"assets/configs/models/{dataset}_dataset/forecasters/{model}/{dataset}_96_48_S.json"
    cfg_ae_path = f"assets/configs/models/{dataset}_dataset/ae/tcn_ae.json"
    
    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    
    print(f"[Compute global_sigma] Dataset={dataset}, Model={model}")
    
    # Charger données
    train_loader, _, _ = prepare_rl_data(cfg_f, cfg_ae, device)
    
    # Collecter toutes les séries du train set
    print("  Collecting train set...")
    all_x = []
    for batch in train_loader:
        batch_x, _, _, _ = batch
        batch_x = batch_x.float().to(device)
        x_ot = batch_x[:, :, -1].cpu().numpy()  # [B, BH]
        all_x.append(x_ot)
    
    all_x = np.concatenate(all_x, axis=0)  # [N, BH]
    print(f"  Train set: {len(all_x)} samples")
    
    # Calculer global sigma (std de toutes les séries)
    global_sigma = float(all_x.std())
    
    print(f"\n[Result]")
    print(f"  global_sigma = {global_sigma:.6f}")
    print(f"\nAdd this to your RL config JSON:")
    print(f'  "global_sigma": {global_sigma:.6f}')
    
    return global_sigma


def main():
    parser = argparse.ArgumentParser(description="Compute global_sigma for a dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, required=True, choices=["gru", "itransformer", "patchtst", "dlinear", "timesnet"])
    
    args = parser.parse_args()
    
    compute_global_sigma(args.dataset, args.model)


if __name__ == "__main__":
    main()
