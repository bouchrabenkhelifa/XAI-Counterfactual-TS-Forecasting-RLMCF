#!/usr/bin/env python
"""
Plot iTransformer Weather predictions on test sample in original scale (°C).
Uses the latest trained checkpoint and denormalizes to original scale.
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
from src.data_provider.data_factory import data_provider
from src.models.Forecaster.iTransformer import Model as iTransformerModel

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*70)
print("Plotting iTransformer Weather - Test Sample (Original Scale)")
print("="*70)

dataset = 'weather'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# Load config and model
# ============================================================================
print("\nLoading configuration and model...")

config_path = f'assets/configs/{dataset}_dataset/forecasters/itransformer/{dataset}_96_48_S.json'
config = load_config(config_path)

# Load test data
test_set, test_loader = data_provider(config, 'test')
scaler = test_set.scaler

# Load model
model_path = f'assets/checkpoints/{dataset}_chpts/forecaster/chpt_{dataset}_96_48_itransformer_S.pth'
model = iTransformerModel(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"  [OK] Config loaded: {config_path}")
print(f"  [OK] Model loaded: {model_path}")
print(f"  [OK] Device: {device}")

# ============================================================================
# Get first test batch
# ============================================================================
print("\nExtracting test sample...")

batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
batch_x = batch_x.float().to(device)
batch_y = batch_y.float().to(device)
batch_x_mark = batch_x_mark.float().to(device)
batch_y_mark = batch_y_mark.float().to(device)

# Get predictions
with torch.no_grad():
    dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1)
    output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

# Extract first sample (index 0)
sample_idx = 0
past_norm = batch_x[sample_idx, :, 0].cpu().numpy()  # (seq_len,)
future_real_norm = batch_y[sample_idx, -config.pred_len:, 0].cpu().numpy()  # (pred_len,)
future_pred_norm = output[sample_idx, :, 0].cpu().detach().numpy()  # (pred_len,)

print(f"  [OK] Sample extracted")
print(f"       Past length: {len(past_norm)}")
print(f"       Future length: {len(future_real_norm)}")

# ============================================================================
# Denormalize to original scale
# ============================================================================
print("\nDenormalizing to original scale...")

past_orig = scaler.inverse_transform(past_norm.reshape(-1, 1)).flatten()
future_real_orig = scaler.inverse_transform(future_real_norm.reshape(-1, 1)).flatten()
future_pred_orig = scaler.inverse_transform(future_pred_norm.reshape(-1, 1)).flatten()

print(f"  [OK] Denormalization complete")
print(f"       Past range: [{past_orig.min():.2f}, {past_orig.max():.2f}]°C")
print(f"       Future real range: [{future_real_orig.min():.2f}, {future_real_orig.max():.2f}]°C")
print(f"       Future pred range: [{future_pred_orig.min():.2f}, {future_pred_orig.max():.2f}]°C")

# ============================================================================
# Calculate metrics
# ============================================================================
mae = np.mean(np.abs(future_pred_orig - future_real_orig))
rmse = np.sqrt(np.mean((future_pred_orig - future_real_orig) ** 2))
mape = np.mean(np.abs((future_real_orig - future_pred_orig) / (np.abs(future_real_orig) + 1e-8))) * 100

print(f"\nMetrics on test sample:")
print(f"  MAE:  {mae:.4f}°C")
print(f"  RMSE: {rmse:.4f}°C")
print(f"  MAPE: {mape:.2f}%")

# ============================================================================
# Plot
# ============================================================================
print("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(14, 5))

seq_len = len(past_orig)
pred_len = len(future_real_orig)
total_len = seq_len + pred_len
t = np.arange(total_len)

# Plot past
ax.plot(t[:seq_len], past_orig, color='#1f77b4', linewidth=2, label='Past Real', zorder=2)

# Plot future real
ax.plot(t[seq_len:], future_real_orig, color='#1f77b4', linewidth=2, label='Future Real', zorder=2)

# Plot future predicted with shading
ax.plot(t[seq_len:], future_pred_orig, color='#ff7f0e', linewidth=2, linestyle='--', label='Prediction', zorder=2)
ax.fill_between(t[seq_len:], future_real_orig, future_pred_orig, alpha=0.3, color='#ff7f0e', zorder=1)

# Add vertical line at boundary
ax.axvline(x=seq_len - 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

# Labels and title
ax.set_title(f'iTransformer Weather - Test Sample | RMSE = {rmse:.4f}°C | MAE = {mae:.4f}°C', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Timestep', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'plots/itransformer_weather_test_sample.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"  [OK] Plot saved: {output_path}")

# ============================================================================
# Save metrics to CSV
# ============================================================================
df_metrics = pd.DataFrame({
    'Model': ['iTransformer'],
    'Dataset': ['Weather'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'Seq_Len': [config.seq_len],
    'Pred_Len': [config.pred_len],
})
df_metrics.to_csv('results/itransformer_weather_test_metrics.csv', index=False)
print(f"  [OK] Metrics saved: results/itransformer_weather_test_metrics.csv")

print("\n" + "="*70)
print("[OK] Done!")
print("="*70)
print(f"\nGenerated files:")
print(f"  - {output_path}")
print(f"  - results/itransformer_weather_test_metrics.csv")
