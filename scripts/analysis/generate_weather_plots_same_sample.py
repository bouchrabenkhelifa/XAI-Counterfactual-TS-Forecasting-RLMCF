"""
Generate forecast plots for ALL Weather forecasters using the SAME test sample
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


# Model configurations (only models with seq_len=96)
MODELS = {
    "iTransformer": {
        "config": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_48_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_96_itransformer_S.pth",
    },
    "DLinear": {
        "config": "assets/configs/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
        "checkpoint": "assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_96_dlinear_S.pth",
    },
}


def generate_comparison_plot(models_data, output_dir, n_samples=4):
    """
    Generate comparison plots for all models using the same test data.
    
    models_data: dict with model_name -> {preds, targets, inputs, scaler}
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first model's data to determine dimensions
    first_model = list(models_data.keys())[0]
    first_data = models_data[first_model]
    
    n_plot = min(n_samples, first_data["preds"].shape[0])
    
    # Create figure with subplots for each model
    n_models = len(models_data)
    fig, axes = plt.subplots(n_models, n_plot, figsize=(16, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    elif n_plot == 1:
        axes = axes.reshape(-1, 1)
    
    for model_idx, (model_name, data) in enumerate(models_data.items()):
        preds_denorm = data["preds"]
        targets_denorm = data["targets"]
        inputs_denorm = data["inputs"]
        
        seq_len = inputs_denorm.shape[1]
        pred_len = preds_denorm.shape[1]
        t_past = np.arange(seq_len)
        t_future = np.arange(seq_len, seq_len + pred_len)
        
        for sample_idx in range(n_plot):
            ax = axes[model_idx, sample_idx]
            
            # Plot past (input)
            ax.plot(t_past, inputs_denorm[sample_idx, :, 0], 
                    color="grey", lw=1.2, label="Past (input)", alpha=0.8)
            
            # Plot real future
            ax.plot(t_future, targets_denorm[sample_idx, :, 0], 
                    color="#2196F3", lw=2, label="Real future", marker="o", markersize=3)
            
            # Plot predicted future
            ax.plot(t_future, preds_denorm[sample_idx, :, 0], 
                    color="#F44336", lw=2, linestyle="--", label="Predicted", marker="s", markersize=3)
            
            # Separator line
            ax.axvline(x=seq_len, color="black", linestyle=":", lw=1.5, alpha=0.5)
            
            # Calculate RMSE for this sample
            rmse = np.sqrt(np.mean((preds_denorm[sample_idx, :, 0] - targets_denorm[sample_idx, :, 0]) ** 2))
            mae = np.mean(np.abs(preds_denorm[sample_idx, :, 0] - targets_denorm[sample_idx, :, 0]))
            
            if sample_idx == 0:
                ax.set_ylabel(f"{model_name}\nTemp (°C)", fontsize=10, fontweight="bold")
            
            ax.set_title(f"RMSE: {rmse:.4f}°C | MAE: {mae:.4f}°C", fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if model_idx == 0 and sample_idx == 0:
                ax.legend(fontsize=8, loc="best")
    
    fig.suptitle("Weather Forecasters — Test Forecast Comparison (Original Scale °C)", 
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, "weather_forecasters_comparison_original_scale.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n✓ Comparison plot saved → {output_path}")


def main():
    print("\n" + "="*80)
    print("  GENERATING WEATHER FORECASTER COMPARISON (Same Test Sample)")
    print("="*80 + "\n")
    
    # Use iTransformer's config to load test data (seq_len=96)
    base_config_path = MODELS["iTransformer"]["config"]
    base_config_dict = load_config(base_config_path)
    
    class Config:
        def __init__(self, d):
            self.__dict__.update(d)
    
    base_configs = Config(base_config_dict)
    device = get_device(base_configs)
    
    # Load test data once (using iTransformer's seq_len=96)
    print(f"Loading test data with seq_len={base_configs.seq_len}...")
    test_data, test_loader = data_provider(base_configs, "test")
    
    # Get first batch
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    
    # Denormalize inputs and targets once
    scaler = test_data.scaler
    inputs_denorm = scaler.inverse_transform(batch_x[:, :, 0].cpu().numpy().reshape(-1, 1)).reshape(batch_x.shape[0], -1, 1)
    targets_denorm = scaler.inverse_transform(batch_y[:, -base_configs.pred_len:, 0].cpu().numpy().reshape(-1, 1)).reshape(batch_y.shape[0], base_configs.pred_len, 1)
    
    models_data = {}
    
    # Process each model
    for model_name, model_info in MODELS.items():
        config_path = model_info["config"]
        checkpoint_path = model_info["checkpoint"]
        
        print(f"\n  Processing {model_name}...")
        
        # Load config
        if not os.path.exists(config_path):
            print(f"    ⚠ Config not found → {config_path}")
            continue
        
        config_dict = load_config(config_path)
        configs = Config(config_dict)
        
        # Build model
        model = build_model(configs, device)
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"    ⚠ Checkpoint not found → {checkpoint_path}")
            continue
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        model.eval()
        
        # Forward pass
        try:
            with torch.no_grad():
                if model_name == "iTransformer":
                    batch_y_dummy = torch.zeros_like(batch_y)
                    outputs = model(batch_x, batch_x_mark, batch_y_dummy, batch_y_mark)
                else:
                    outputs = model(batch_x, batch_x_mark)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Extract predictions
            f_dim = -1 if configs.features == "MS" else 0
            preds = outputs[:, -configs.pred_len:, f_dim:].cpu().numpy()
            
            # Denormalize predictions
            preds_denorm = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
            
            # Store data
            models_data[model_name] = {
                "preds": preds_denorm,
                "targets": targets_denorm,
                "inputs": inputs_denorm,
                "scaler": scaler,
            }
            
            # Calculate overall metrics
            overall_rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
            overall_mae = np.mean(np.abs(preds_denorm - targets_denorm))
            
            print(f"    ✓ {model_name:15} → RMSE: {overall_rmse:.6f}°C | MAE: {overall_mae:.6f}°C")
        
        except Exception as e:
            print(f"    ✗ {model_name}: Error → {str(e)}")
    
    # Generate comparison plot
    if models_data:
        output_dir = "assets/figures/weather/forecaster"
        generate_comparison_plot(models_data, output_dir, n_samples=4)
    
    print("\n" + "="*80)
    print("  Comparison plot generated ✓")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
