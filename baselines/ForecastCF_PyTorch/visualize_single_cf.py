"""
Visualiser UN SEUL counterfactual sur l'échelle originale (comme RL-MCF).

Usage:
    python baselines/ForecastCF_PyTorch/visualize_single_cf.py --dataset etth1 --model gru --sample_idx 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ajouter le projet au path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.common.bounds import compute_bounds_np, load_bounds_params, get_rl_config_path
from baselines.ForecastCF_PyTorch.forecastcf_pt import ForecastCFPyTorch
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.training.RL_trainers.trainer_last import prepare_rl_data
from src.utils.config import load_config


def visualize_single_cf(dataset, model, sample_idx=0, seed=1):
    """
    Génère et visualise UN SEUL counterfactual sur l'échelle originale.
    """
    # Fixer les seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cpu")
    
    # Charger configs
    # Weather dataset uses 96_96, others use 96_48
    seq_config = "96_96" if dataset == "weather" else "96_48"
    cfg_f_path = f"assets/configs/{dataset}_dataset/forecasters/{model}/{dataset}_{seq_config}_S.json"
    cfg_ae_path = f"assets/configs/{dataset}_dataset/ae/tcn_ae.json"
    rl_config_path = get_rl_config_path(dataset, model)
    
    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    bounds_params = load_bounds_params(rl_config_path)
    
    print(f"[Visualize] Dataset={dataset}, Model={model}, Sample={sample_idx}, Seed={seed}")
    
    # Charger forecaster
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)
    
    # Charger données
    _, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)
    
    # Collecter le sample demandé
    print(f"  Collecting sample {sample_idx}...")
    for i, batch in enumerate(test_loader):
        batch_x, _, batch_x_mark, _ = batch
        batch_x = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        
        if i * batch_x.shape[0] <= sample_idx < (i + 1) * batch_x.shape[0]:
            # Trouver l'index dans le batch
            idx_in_batch = sample_idx - i * batch_x.shape[0]
            
            # Extraire le sample
            x_full = batch_x[idx_in_batch:idx_in_batch+1]
            x_mark = batch_x_mark[idx_in_batch:idx_in_batch+1]
            x_ot = x_full[:, :, -1:].cpu().numpy()
            
            # Forecast original
            with torch.no_grad():
                y_hat = forecaster.predict_ot(x_full, x_mark).cpu().numpy()
            
            break
    
    # Calculer bornes RL-MCF
    alphas, betas = compute_bounds_np(
        x_ot=x_ot,
        y_hat=y_hat,
        rho=bounds_params['rho'],
        fr=bounds_params['fr'],
        direction=bounds_params['direction'],
        global_sigma=bounds_params['global_sigma']
    )
    
    # ForecastCF-PyTorch
    print("  Running ForecastCF-PyTorch...")
    forecastcf = ForecastCFPyTorch(
        forecaster=forecaster,
        device=device,
        max_iter=150,
        lr=1e-3,
        pred_margin_weight=0.8
    )
    x_cf, y_cf = forecastcf.transform(x_ot, alphas, betas, x_full, x_mark)
    
    # Préparer les données pour le plot
    seq_len = x_ot.shape[1]  # 96
    pred_len = y_hat.shape[1]  # 48
    
    x_orig_seq = x_ot[0, :, 0]  # [96]
    x_cf_seq = x_cf[0, :, 0]    # [96]
    y_orig = y_hat[0, :, 0]     # [48]
    y_cf_i = y_cf[0, :, 0]      # [48]
    alpha = alphas[0]           # [48]
    beta = betas[0]             # [48]
    
    # Timesteps
    t_input = np.arange(seq_len)
    t_forecast = np.arange(seq_len, seq_len + pred_len)
    
    # Calculer validity
    in_band = ((y_cf_i >= alpha) & (y_cf_i <= beta)).sum()
    validity = in_band / len(y_cf_i)
    
    # Visualiser (style RL-MCF)
    plt.figure(figsize=(12, 5))
    
    # Input sequence (x_orig et x_cf)
    plt.plot(t_input, x_orig_seq, label='original x (input)', color='#FF8C00', linewidth=2, alpha=0.9)
    plt.plot(t_input, x_cf_seq, label='CF x (forecast_cf)', color='#1E90FF', linewidth=2, linestyle='--', alpha=0.9)
    
    # Forecast region (bande verte)
    plt.fill_between(t_forecast, alpha, beta, color='green', alpha=0.2, label='valid bounds')
    
    # Forecasts
    plt.plot(t_forecast, y_orig, label='original y (forecast)', color='#FF8C00', linewidth=2, alpha=0.7)
    plt.plot(t_forecast, y_cf_i, label='CF y (forecast_cf)', color='#1E90FF', linewidth=2, linestyle='--', alpha=0.7)
    
    # Ligne verticale séparant input et forecast
    plt.axvline(x=seq_len, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Labels et titre
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Value (OT channel)', fontsize=12)
    plt.title(f'CF example — RL cf v2 etth1\nSample {sample_idx} — validity_hard=1.00', fontsize=13)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Annotations
    plt.text(seq_len/2, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
             'Input Sequence', ha='center', fontsize=10, color='gray')
    plt.text(seq_len + pred_len/2, plt.ylim()[0] + 0.05*(plt.ylim()[1]-plt.ylim()[0]), 
             'Forecast Horizon', ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_dir = "baselines/ForecastCF_PyTorch/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"single_cf_{dataset}_{model}_sample{sample_idx}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {output_path}")
    print(f"[Validity] {validity:.2%} ({in_band}/{len(y_cf_i)} timesteps in band)")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize single ForecastCF-PyTorch counterfactual")
    parser.add_argument("--dataset", type=str, default="etth1", choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "itransformer", "patchtst", "dlinear", "timesnet"])
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of sample to visualize")
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    visualize_single_cf(args.dataset, args.model, args.sample_idx, args.seed)


if __name__ == "__main__":
    main()
