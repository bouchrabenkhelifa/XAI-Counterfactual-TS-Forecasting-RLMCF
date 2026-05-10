#!/usr/bin/env python
"""
Plot single RL sample without training.
Uses existing checkpoints to generate one counterfactual example.
Denormalizes to original scale for proper visualization.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.training.RL_trainers.trainer_last import RLMaskTrainer
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.data_provider.data_factory import data_provider

# Patch trainer
import src.training.RL_trainers.trainer_last as _trainer_mod
_trainer_mod.ForecasterWrapper = ForecasterWrapperV2

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*70)
print("Plotting RL iTransformer Sample (No Training)")
print("="*70)

dataset = 'etth1'
device = 'cpu'

# ============================================================================
# Load RL trainer with existing checkpoints
# ============================================================================
print("\nLoading RL trainer...")

ae_config_path = f'assets/configs/{dataset}_dataset/ae/tcn_ae.json'
rl_config_path = f'assets/configs/{dataset}_dataset/RL_ablations/config_v2.json'
f_config_path = f'assets/configs/{dataset}_dataset/forecasters/itransformer/{dataset}_96_48_S.json'

cfg_rl = load_config(rl_config_path)
cfg_f = load_config(f_config_path)
cfg_ae = load_config(ae_config_path)

trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

# Load RL checkpoint
ckpt_path = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
    print(f"  [OK] Loaded RL checkpoint: {ckpt_path}")
else:
    print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
    sys.exit(1)

# ============================================================================
# Get test set for denormalization
# ============================================================================
print("\nLoading test set...")

test_set, test_loader = data_provider(cfg_f, 'test')
scaler = test_set.scaler

# ============================================================================
# Evaluate and get counterfactuals
# ============================================================================
print("\nEvaluating RL agent...")

trainer.agent.eval()
summary, examples = trainer.evaluate(n_batches=1)

print(f"  [OK] Evaluation complete")

# Extract first example (keys: x_ot, x_cf, y_hat, y_cf)
if examples and len(examples) > 0:
    ex = examples[0]
    # Keys are: x_ot (original past), x_cf (counterfactual), y_hat (original forecast), y_cf (CF forecast)
    past_norm = ex['x_ot'][:, 0]  # (seq_len,)
    counterfactual_norm = ex['x_cf'][:, 0]  # (seq_len,)
    future_real_norm = ex['y_hat'][:, 0]  # (pred_len,)
    future_forecast_cf_norm = ex['y_cf'][:, 0]  # (pred_len,)
else:
    print("  [ERROR] No examples generated")
    sys.exit(1)

# ============================================================================
# Denormalize to original scale
# ============================================================================
print("\nDenormalizing to original scale...")

past_orig = scaler.inverse_transform(past_norm.reshape(-1, 1)).flatten()
counterfactual_orig = scaler.inverse_transform(counterfactual_norm.reshape(-1, 1)).flatten()
future_real_orig = scaler.inverse_transform(future_real_norm.reshape(-1, 1)).flatten()
future_forecast_cf_orig = scaler.inverse_transform(future_forecast_cf_norm.reshape(-1, 1)).flatten()

# ============================================================================
# Plot RL counterfactual
# ============================================================================
print("\nGenerating RL figure...")

fig, ax = plt.subplots(figsize=(14, 5))

seq_len = len(past_orig)
pred_len = len(future_real_orig)
total_len = seq_len + pred_len
t = np.arange(total_len)

# Plot past (original)
ax.plot(t[:seq_len], past_orig, color='#1f77b4', linewidth=2, label='Past Real')

# Plot counterfactual with shading
ax.plot(t[:seq_len], counterfactual_orig, color='#ff7f0e', linewidth=2, linestyle='--', label='Past CF')
ax.fill_between(t[:seq_len], past_orig, counterfactual_orig, alpha=0.3, color='#ff7f0e')

# Plot future real
ax.plot(t[seq_len:], future_real_orig, color='#1f77b4', linewidth=2, label='Future Real')

# Plot future forecast CF (objective)
ax.plot(t[seq_len:], future_forecast_cf_orig, color='#2ca02c', linewidth=2, linestyle='--', label='RL Objective')
ax.fill_between(t[seq_len:], future_real_orig, future_forecast_cf_orig, alpha=0.3, color='#2ca02c')

# Add vertical line at boundary
ax.axvline(x=seq_len - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Extract validity from summary
validity_rate = summary.get('validity_ratio', {}).get('mean', 0.0)
if isinstance(validity_rate, dict):
    validity_rate = validity_rate.get('mean', 0.0)

# Calculate sparsity (fraction of modified timesteps)
sparsity = np.sum(np.abs(counterfactual_orig - past_orig) > 0.01) / len(past_orig)

# Add title with metrics
ax.set_title(f'Sample 2 - Validity Hard = {validity_rate:.2f}', fontsize=12, fontweight='bold')
ax.set_xlabel('Timestep', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/rl_itransformer_etth1_sample.png', dpi=300, bbox_inches='tight')
plt.close()

# Save metrics
df = pd.DataFrame({
    'Model': ['RL-iTransformer'],
    'Dataset': ['ETTh1'],
    'Validity_Rate': [validity_rate],
    'Sparsity': [sparsity],
})
df.to_csv('results/metrics_rl_itransformer_etth1_sample.csv', index=False)

print(f"  [OK] RL sample figure saved: plots/rl_itransformer_etth1_sample.png")
print(f"       Validity: {validity_rate:.2f}")
print(f"       Sparsity: {sparsity:.2%}")

print("\n" + "="*70)
print("[OK] Done!")
print("="*70)
