"""
Summary of Weather Forecasters Performance
Extract RMSE/MSE from training history files
"""

import json
import os
import numpy as np

# Model configurations
MODELS = {
    "iTransformer": "assets/results/weather/forecaster/history_weather_96_96_itransformer_S.json",
    "GRU": "assets/results/weather/forecaster/history_weather_96_96_gru_S.json",
    "PatchTST": "assets/results/weather/forecaster/history_weather_96_96_patchtst_S.json",
    "TimesNet": "assets/results/weather/forecaster/history_weather_96_96_timesnet_S.json",
    "DLinear": "assets/results/weather/forecaster/history_weather_96_96_dlinear_S.json",
}

def load_history(path):
    """Load training history from JSON."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def mse_to_rmse(mse):
    """Convert MSE to RMSE."""
    return np.sqrt(mse)

def main():
    print("\n" + "="*80)
    print("  WEATHER FORECASTERS — PERFORMANCE SUMMARY")
    print("="*80)
    
    results = []
    
    for model_name, history_path in MODELS.items():
        history = load_history(history_path)
        
        if history is None:
            print(f"\n  ⚠ {model_name:15} — History file not found")
            continue
        
        # Get final losses (best epoch)
        train_loss = history["train_loss"][-1]
        val_loss = history["val_loss"][-1]
        test_loss = history["test_loss"][-1]
        
        # Convert MSE to RMSE
        train_rmse = mse_to_rmse(train_loss)
        val_rmse = mse_to_rmse(val_loss)
        test_rmse = mse_to_rmse(test_loss)
        
        results.append({
            "model": model_name,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "test_mse": test_loss,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
        })
        
        print(f"\n  {model_name}")
        print(f"  {'-'*76}")
        print(f"    Train Loss (MSE): {train_loss:.6f}  |  RMSE: {train_rmse:.6f}")
        print(f"    Val Loss (MSE):   {val_loss:.6f}  |  RMSE: {val_rmse:.6f}")
        print(f"    Test Loss (MSE):  {test_loss:.6f}  |  RMSE: {test_rmse:.6f}")
    
    # Ranking by test RMSE
    print("\n" + "="*80)
    print("  RANKING BY TEST RMSE (Lower is Better)")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda x: x["test_rmse"])
    
    for rank, result in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  {rank}."
        print(f"\n  {medal} {result['model']:15} — Test RMSE: {result['test_rmse']:.6f}")
        print(f"     Test MSE: {result['test_mse']:.6f}")
    
    # Summary table
    print("\n" + "="*80)
    print("  SUMMARY TABLE")
    print("="*80)
    print(f"\n  {'Model':<15} {'Train RMSE':<15} {'Val RMSE':<15} {'Test RMSE':<15}")
    print(f"  {'-'*60}")
    
    for result in sorted_results:
        print(f"  {result['model']:<15} {result['train_rmse']:<15.6f} {result['val_rmse']:<15.6f} {result['test_rmse']:<15.6f}")
    
    # Best and worst
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    print("\n" + "="*80)
    print(f"  Best Model:  {best['model']} (Test RMSE: {best['test_rmse']:.6f})")
    print(f"  Worst Model: {worst['model']} (Test RMSE: {worst['test_rmse']:.6f})")
    print(f"  Difference:  {worst['test_rmse'] - best['test_rmse']:.6f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
