"""
Visualiser les counterfactuals générés par ForecastCF-PyTorch.

Usage:
    python baselines/ForecastCF_PyTorch/visualize_cf.py --dataset etth1 --model gru --n_samples 5
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


def visualize_counterfactuals(dataset, model, n_samples=5, seed=1):
    """
    Génère et visualise des counterfactuals.
    """
    # Fixer les seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cpu")
    
    # Charger configs
    cfg_f_path = f"assets/configs/models/{dataset}_dataset/forecasters/{model}/{dataset}_96_48_S.json"
    cfg_ae_path = f"assets/configs/models/{dataset}_dataset/ae/tcn_ae.json"
    rl_config_path = get_rl_config_path(dataset, model)
    
    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    bounds_params = load_bounds_params(rl_config_path)
    
    print(f"[Visualize] Dataset={dataset}, Model={model}, Seed={seed}")
    print(f"  Bounds: rho={bounds_params['rho']}, fr={bounds_params['fr']}, direction={bounds_params['direction']}")
    
    # Charger forecaster
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)
    
    # Charger données
    _, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)
    
    # Collecter test set
    print(f"  Collecting {n_samples} test samples...")
    X_test_list, Y_hat_test_list, X_full_list, X_mark_list = [], [], [], []
    for i, batch in enumerate(test_loader):
        if len(X_test_list) >= n_samples:
            break
        batch_x, _, batch_x_mark, _ = batch
        batch_x = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        
        with torch.no_grad():
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)
        
        # Prendre seulement n_samples
        remaining = n_samples - len(X_test_list)
        take = min(remaining, batch_x.shape[0])
        
        X_test_list.append(batch_x[:take, :, -1:].cpu().numpy())
        Y_hat_test_list.append(y_hat[:take].cpu().numpy())
        X_full_list.append(batch_x[:take])
        X_mark_list.append(batch_x_mark[:take])
    
    X_test = np.concatenate(X_test_list, axis=0)
    Y_hat_test = np.concatenate(Y_hat_test_list, axis=0)
    X_full = torch.cat(X_full_list, dim=0)
    X_mark = torch.cat(X_mark_list, dim=0)
    
    # Calculer bornes RL-MCF
    alphas, betas = compute_bounds_np(
        x_ot=X_test,
        y_hat=Y_hat_test,
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
    X_cf, Y_cf = forecastcf.transform(X_test, alphas, betas, X_full, X_mark)
    
    # Visualiser
    print("  Creating visualization...")
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        x_orig = X_test[i, :, 0]
        x_cf = X_cf[i, :, 0]
        y_orig = Y_hat_test[i, :, 0]
        y_cf_i = Y_cf[i, :, 0]
        alpha = alphas[i]
        beta = betas[i]
        
        # Plot 1: Input series (x_orig vs x_cf)
        ax1 = axes[i, 0]
        ax1.plot(x_orig, label='x_orig', color='blue', linewidth=1.5)
        ax1.plot(x_cf, label='x_cf', color='red', linewidth=1.5, linestyle='--')
        ax1.set_title(f'Sample {i+1}: Input Series (OT channel)', fontsize=10)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculer les changements
        diff = np.abs(x_cf - x_orig)
        changed = diff > 0.01
        n_changed = changed.sum()
        proximity = np.linalg.norm(x_cf - x_orig)
        
        ax1.text(0.02, 0.98, f'Changed: {n_changed}/{len(x_orig)} timesteps\nProximity L2: {proximity:.4f}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)
        
        # Plot 2: Forecast (y_orig vs y_cf vs bounds)
        ax2 = axes[i, 1]
        timesteps_forecast = np.arange(len(y_orig))
        
        # Bande cible
        ax2.fill_between(timesteps_forecast, alpha, beta, color='green', alpha=0.2, label='Target band [α, β]')
        
        # Forecasts
        ax2.plot(y_orig, label='y_orig (forecast original)', color='blue', linewidth=1.5, marker='o', markersize=3)
        ax2.plot(y_cf_i, label='y_cf (forecast CF)', color='red', linewidth=1.5, marker='s', markersize=3, linestyle='--')
        
        # Bornes
        ax2.plot(alpha, color='green', linewidth=1, linestyle=':', label='α (lower bound)')
        ax2.plot(beta, color='green', linewidth=1, linestyle=':', label='β (upper bound)')
        
        ax2.set_title(f'Sample {i+1}: Forecast', fontsize=10)
        ax2.set_xlabel('Forecast Horizon')
        ax2.set_ylabel('Value')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # Calculer validity
        in_band = ((y_cf_i >= alpha) & (y_cf_i <= beta)).sum()
        validity = in_band / len(y_cf_i)
        
        ax2.text(0.02, 0.98, f'Validity: {validity:.2%} ({in_band}/{len(y_cf_i)} timesteps)',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_dir = "baselines/ForecastCF_PyTorch/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cf_visualization_{dataset}_{model}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize ForecastCF-PyTorch counterfactuals")
    parser.add_argument("--dataset", type=str, default="etth1", choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "itransformer", "patchtst", "dlinear", "timesnet"])
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    visualize_counterfactuals(args.dataset, args.model, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
