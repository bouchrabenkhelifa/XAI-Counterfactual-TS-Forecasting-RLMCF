"""
Generate forecast plots for ALL Weather forecasters in original scale (°C)
Uses trained checkpoints to generate predictions without retraining.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data_provider.data_factory import data_provider
from src.utils.train_tools import get_device


def load_config(config_path):
    """Load config from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def build_model(configs, device):
    """Build model based on model_type."""
    mt = getattr(configs, "model_type", "iTransformer").lower()
    
    if mt == "itransformer":
        from src.models.Forecaster.iTransformer import Model
    elif mt == "gru":
        from src.models.Forecaster.GRU import Model
    elif mt == "dlinear":
        from src.models.Forecaster.DLinear import Model
    elif mt == "patchtst":
        from src.models.Forecaster.PatchTST import Model
    elif mt == "timesnet":
        from src.models.Forecaster.TimesNet import Model
    else:
        raise ValueError(f"Unknown model_type: '{mt}'")
    
    return Model(configs).float().to(device)


# Model configurations (matching existing checkpoints)
MODELS = {
    "iTransformer": {
        "config": "assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_48_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_96_itransformer_S.pth",
    },
    "GRU": {
        "config": "assets/configs/models/weather_dataset/forecasters/gru/weather_96_48_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_48_gru_S.pth",
    },
    "PatchTST": {
        "config": "assets/configs/models/weather_dataset/forecasters/patchtst/weather_96_48_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_48_patchtst_S.pth",
    },
    "TimesNet": {
        "config": "assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_48_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_48_timesnet_S.pth",
    },
    "DLinear": {
        "config": "assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_96_dlinear_S.pth",
    },
}


def generate_plot(model_name, config_dict, checkpoint_path, output_dir, n_samples=4):
    """
    Generate forecast plot in original scale (°C).
    """
    
    # Convert dict to object-like structure
    class Config:
        def __init__(self, d):
            self.__dict__.update(d)
    
    configs = Config(config_dict)
    
    # Setup device
    device = get_device(configs)
    
    # Build model
    model = build_model(configs, device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠ {model_name}: Checkpoint not found → {checkpoint_path}")
        return False
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()
    
    # Load test data
    test_data, test_loader = data_provider(configs, "test")
    
    # Get first batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y = batch_y.float().to(device)
    
    # Forward pass
    with torch.no_grad():
        if model_name == "iTransformer":
            # iTransformer needs x_dec and x_mark_dec
            batch_y_dummy = torch.zeros_like(batch_y)
            outputs = model(batch_x, batch_x_mark, batch_y_dummy, batch_y_mark)
        else:
            outputs = model(batch_x, batch_x_mark)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
    
    # Extract predictions and targets
    f_dim = -1 if configs.features == "MS" else 0
    preds = outputs[:, -configs.pred_len:, f_dim:].cpu().numpy()
    targets = batch_y[:, -configs.pred_len:, f_dim:].cpu().numpy()
    inputs = batch_x[:, :, f_dim:].cpu().numpy()
    
    # Denormalize to original scale (°C)
    scaler = test_data.scaler
    preds_denorm = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    targets_denorm = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    inputs_denorm = scaler.inverse_transform(inputs.reshape(-1, 1)).reshape(inputs.shape)
    
    # Create plots
    os.makedirs(output_dir, exist_ok=True)
    
    n_plot = min(n_samples, preds_denorm.shape[0])
    seq_len = inputs_denorm.shape[1]
    pred_len = preds_denorm.shape[1]
    t_past = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3.5 * n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot past (input)
        ax.plot(t_past, inputs_denorm[i, :, 0], 
                color="grey", lw=1.2, label="Past (input)", alpha=0.8)
        
        # Plot real future
        ax.plot(t_future, targets_denorm[i, :, 0], 
                color="#2196F3", lw=2, label="Real future", marker="o", markersize=3)
        
        # Plot predicted future
        ax.plot(t_future, preds_denorm[i, :, 0], 
                color="#F44336", lw=2, linestyle="--", label="Predicted", marker="s", markersize=3)
        
        # Separator line
        ax.axvline(x=seq_len, color="black", linestyle=":", lw=1.5, alpha=0.5)
        
        # Calculate RMSE for this sample
        rmse = np.sqrt(np.mean((preds_denorm[i, :, 0] - targets_denorm[i, :, 0]) ** 2))
        mae = np.mean(np.abs(preds_denorm[i, :, 0] - targets_denorm[i, :, 0]))
        
        ax.set_title(f"Sample {i+1} | RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"{model_name} — Weather Test Forecast (Original Scale °C)", 
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, f"{model_name.lower()}_forecast_examples_original_scale.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Calculate overall metrics
    overall_rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
    overall_mae = np.mean(np.abs(preds_denorm - targets_denorm))
    
    print(f"  ✓ {model_name:15} → RMSE: {overall_rmse:.6f}°C | MAE: {overall_mae:.6f}°C")
    
    return True


def main():
    print("\n" + "="*80)
    print("  GENERATING WEATHER FORECASTER PLOTS (Original Scale °C)")
    print("="*80 + "\n")
    
    for model_name, model_info in MODELS.items():
        config_path = model_info["config"]
        checkpoint_path = model_info["checkpoint"]
        
        # Determine output directory
        output_dir = os.path.join("assets/figures/weather/forecaster", model_name.lower())
        
        # Load config
        if not os.path.exists(config_path):
            print(f"  ⚠ {model_name}: Config not found → {config_path}")
            continue
        
        config_dict = load_config(config_path)
        
        # Generate plot
        try:
            generate_plot(model_name, config_dict, checkpoint_path, output_dir, n_samples=4)
        except Exception as e:
            print(f"  ✗ {model_name}: Error → {str(e)}")
    
    print("\n" + "="*80)
    print("  All plots generated in original scale (°C)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
