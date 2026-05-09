"""
Debug: Afficher les valeurs des bornes pour comprendre le gap.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from baselines.common.bounds import compute_bounds_np, load_bounds_params, get_rl_config_path
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.training.RL_trainers.trainer_last import prepare_rl_data
from src.utils.config import load_config

# Config
dataset = "etth1"
model = "gru"
device = torch.device("cpu")
seed = 1

np.random.seed(seed)
torch.manual_seed(seed)

# Charger configs
# Weather dataset uses 96_96, others use 96_48
seq_config = "96_96" if dataset == "weather" else "96_48"
cfg_f_path = f"assets/configs/models/{dataset}_dataset/forecasters/{model}/{dataset}_{seq_config}_S.json"
cfg_ae_path = f"assets/configs/models/{dataset}_dataset/ae/tcn_ae.json"
rl_config_path = get_rl_config_path(dataset, model)

cfg_f = load_config(cfg_f_path)
cfg_ae = load_config(cfg_ae_path)
bounds_params = load_bounds_params(rl_config_path)

print(f"Bounds params: rho={bounds_params['rho']}, fr={bounds_params['fr']}, direction={bounds_params['direction']}")

# Charger forecaster
forecaster = ForecasterWrapperV2(cfg_f, device)
forecaster.model.eval()

# Charger données
_, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)

# Prendre 1 sample
for batch in test_loader:
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    
    x_ot = batch_x[0:1, :, -1:].cpu().numpy()
    
    with torch.no_grad():
        y_hat = forecaster.predict_ot(batch_x[0:1], batch_x_mark[0:1]).cpu().numpy()
    
    break

# Calculer bornes
alphas, betas = compute_bounds_np(
    x_ot=x_ot,
    y_hat=y_hat,
    rho=bounds_params['rho'],
    fr=bounds_params['fr'],
    direction=bounds_params['direction'],
    global_sigma=bounds_params['global_sigma']
)

alpha = alphas[0]
beta = betas[0]
y_orig = y_hat[0, :, 0]
x_orig = x_ot[0, :, 0]

# Calculer sigma
sigma_per_sample = x_orig.std()
sigma_used = bounds_params['global_sigma'] if bounds_params['global_sigma'] is not None else sigma_per_sample
gap = bounds_params['rho'] * sigma_used
width = bounds_params['fr'] * sigma_used

print(f"\n=== Sample Analysis ===")
print(f"x_ot: mean={x_orig.mean():.4f}, std={sigma_per_sample:.4f}, min={x_orig.min():.4f}, max={x_orig.max():.4f}")
print(f"y_hat: mean={y_orig.mean():.4f}, std={y_orig.std():.4f}, min={y_orig.min():.4f}, max={y_orig.max():.4f}")

print(f"\n=== Bounds Calculation ===")
if bounds_params['global_sigma'] is not None:
    print(f"Using GLOBAL sigma = {sigma_used:.4f} (from config)")
    print(f"  (per-sample sigma would be {sigma_per_sample:.4f})")
else:
    print(f"Using PER-SAMPLE sigma = {sigma_used:.4f}")
print(f"gap = rho * sigma = {bounds_params['rho']} * {sigma_used:.4f} = {gap:.4f}")
print(f"width = fr * sigma = {bounds_params['fr']} * {sigma_used:.4f} = {width:.4f}")

print(f"\ndirection = {bounds_params['direction']} (< 0 => réduction)")
print(f"beta = y_hat - gap = y_hat - {gap:.4f}")
print(f"alpha = beta - width = y_hat - {gap:.4f} - {width:.4f} = y_hat - {gap+width:.4f}")

print(f"\n=== Bounds Values ===")
print(f"alpha: mean={alpha.mean():.4f}, min={alpha.min():.4f}, max={alpha.max():.4f}")
print(f"beta:  mean={beta.mean():.4f}, min={beta.min():.4f}, max={beta.max():.4f}")
print(f"target (alpha+beta)/2: mean={(alpha+beta).mean()/2:.4f}")

print(f"\n=== Gap Analysis ===")
print(f"y_hat - beta (gap): mean={(y_orig - beta).mean():.4f}, min={(y_orig - beta).min():.4f}, max={(y_orig - beta).max():.4f}")
print(f"Expected gap: {gap:.4f}")

print(f"\n=== Reduction Required ===")
target = (alpha + beta) / 2.0
reduction_abs = y_orig - target
reduction_pct = (reduction_abs / np.abs(y_orig)) * 100
print(f"Absolute reduction needed: mean={reduction_abs.mean():.4f}, min={reduction_abs.min():.4f}, max={reduction_abs.max():.4f}")
print(f"Percentage reduction: mean={reduction_pct.mean():.2f}%, min={reduction_pct.min():.2f}%, max={reduction_pct.max():.2f}%")

print(f"\n=== Validity Check (original forecast) ===")
in_band_orig = ((y_orig >= alpha) & (y_orig <= beta)).sum()
validity_orig = in_band_orig / len(y_orig)
print(f"Original forecast validity: {validity_orig:.2%} ({in_band_orig}/{len(y_orig)} timesteps)")

if validity_orig > 0:
    print(f"\n⚠️ WARNING: Original forecast is already {validity_orig:.2%} valid!")
    print(f"   This means the gap ({gap:.4f}) is too small relative to forecast variability.")
    print(f"   The CF task is too easy - no significant reduction needed.")
