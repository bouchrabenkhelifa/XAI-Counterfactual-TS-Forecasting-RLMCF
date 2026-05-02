#!/usr/bin/env python
"""
Grid search to find optimal configuration parameters for each dataset.

Usage:
    python scripts/analysis/grid_search_configs.py --dataset weather --model itransformer
    python scripts/analysis/grid_search_configs.py --dataset etth1 --model all
    
Output:
    CSV file with grid search results for LaTeX integration
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.data_provider.data_factory import data_provider
from src.models.Forecaster.DLinear import Model as DLinearModel
from src.models.Forecaster.GRU import Model as GRUModel
from src.models.Forecaster.PatchTST import Model as PatchTSTModel
from src.models.Forecaster.TimesNet import Model as TimesNetModel
from src.models.Forecaster.iTransformer import Model as iTransformerModel


# Grid search parameter ranges
GRID_SEARCH_PARAMS = {
    'itransformer': {
        'd_model': [64, 128, 256, 512],
        'n_heads': [2, 4, 8],
        'e_layers': [1, 2, 3],
        'd_ff': [128, 256, 512],
        'dropout': [0.05, 0.1, 0.2],
    },
    'patchtst': {
        'd_model': [32, 64, 128, 256],
        'n_heads': [2, 4, 8],
        'e_layers': [1, 2, 3],
        'd_ff': [64, 128, 256],
        'patch_len': [8, 16, 24, 32],
        'dropout': [0.05, 0.1, 0.2],
    },
    'gru': {
        'd_model': [64, 128, 256, 512],
        'e_layers': [1, 2, 3],
        'dropout': [0.05, 0.1, 0.2],
    },
    'timesnet': {
        'd_model': [8, 16, 32, 64],
        'd_ff': [16, 32, 64],
        'e_layers': [1, 2],
        'dropout': [0.05, 0.1],
    },
    'dlinear': {
        'moving_avg': [6, 12, 25],
        'learning_rate': [0.001, 0.005, 0.01],
    },
}


def create_config_variant(base_config, params_dict):
    """Create a config variant with modified parameters."""
    config = base_config.copy()
    for key, value in params_dict.items():
        config[key] = value
    return config


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_config(config_dict, model_type, dataset_name):
    """Evaluate a configuration."""
    
    try:
        # Create config object
        from types import SimpleNamespace
        config = SimpleNamespace(**config_dict)
        
        # Load data
        train_set, train_loader = data_provider(config, 'train')
        
        # Create model
        if model_type == 'itransformer':
            model = iTransformerModel(config)
        elif model_type == 'patchtst':
            model = PatchTSTModel(config)
        elif model_type == 'gru':
            model = GRUModel(config)
        elif model_type == 'timesnet':
            model = TimesNetModel(config)
        elif model_type == 'dlinear':
            model = DLinearModel(config)
        else:
            return None
        
        model.eval()
        
        # Count parameters
        n_params = count_parameters(model)
        
        # Test forward pass
        import torch
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        
        with torch.no_grad():
            if model_type == 'itransformer':
                # iTransformer needs decoder input
                dec_inp = torch.zeros_like(batch_y[:, -config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :config.label_len, :], dec_inp], dim=1)
                output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                output = model(batch_x, batch_x_mark)
        
        output_shape = output.shape if not isinstance(output, tuple) else output[0].shape
        
        return {
            'status': 'success',
            'n_params': n_params,
            'output_shape': str(output_shape),
            'memory_efficient': n_params < 1_000_000,
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)[:100],
        }


def grid_search(dataset_name, model_type, base_config_path, max_configs=50):
    """Perform grid search for a dataset and model."""
    
    print(f"\n{'='*80}")
    print(f"Grid Search: {dataset_name.upper()} - {model_type.upper()}")
    print(f"{'='*80}\n")
    
    # Load base config
    if not os.path.exists(base_config_path):
        print(f"Error: Config not found at {base_config_path}")
        return None
    
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Get parameter grid
    if model_type not in GRID_SEARCH_PARAMS:
        print(f"Error: Model type {model_type} not in grid search params")
        return None
    
    param_grid = GRID_SEARCH_PARAMS[model_type]
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    all_combinations = list(product(*param_values))
    print(f"Total combinations: {len(all_combinations)}")
    print(f"Sampling: {min(max_configs, len(all_combinations))} configs\n")
    
    # Sample combinations
    if len(all_combinations) > max_configs:
        indices = np.random.choice(len(all_combinations), max_configs, replace=False)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations
    
    results = []
    
    for i, combo in enumerate(sampled_combinations):
        params_dict = dict(zip(param_names, combo))
        config_variant = create_config_variant(base_config, params_dict)
        
        print(f"[{i+1}/{len(sampled_combinations)}] Testing: {params_dict}")
        
        eval_result = evaluate_config(config_variant, model_type, dataset_name)
        
        if eval_result:
            result = {
                'dataset': dataset_name,
                'model': model_type,
                **params_dict,
                **eval_result,
            }
            results.append(result)
            
            if eval_result['status'] == 'success':
                print(f"  ✓ Success - Params: {eval_result['n_params']:,}")
            else:
                print(f"  ✗ Error: {eval_result.get('error', 'Unknown')}")
        
        print()
    
    return pd.DataFrame(results)


def print_results_summary(df_results):
    """Print summary of grid search results."""
    
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80 + "\n")
    
    # Filter successful configs
    df_success = df_results[df_results['status'] == 'success']
    
    if len(df_success) == 0:
        print("No successful configurations found!")
        return
    
    print(f"Successful configs: {len(df_success)}/{len(df_results)}\n")
    
    # Group by model
    for model in df_success['model'].unique():
        df_model = df_success[df_success['model'] == model]
        print(f"\n{model.upper()}:")
        print(f"  Configs tested: {len(df_model)}")
        print(f"  Param range (n_params):")
        print(f"    Min: {df_model['n_params'].min():,}")
        print(f"    Max: {df_model['n_params'].max():,}")
        print(f"    Mean: {df_model['n_params'].mean():,.0f}")
        
        # Show best (smallest) configs
        df_small = df_model.nsmallest(3, 'n_params')
        print(f"\n  Top 3 smallest configs:")
        for idx, row in df_small.iterrows():
            print(f"    - {row['n_params']:,} params")


def save_results(df_results, output_path):
    """Save results to CSV."""
    
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for optimal configs")
    parser.add_argument('--dataset', type=str, default='weather', 
                       choices=['etth1', 'etth2', 'weather'],
                       help='Dataset to search')
    parser.add_argument('--model', type=str, default='itransformer',
                       choices=['itransformer', 'patchtst', 'gru', 'timesnet', 'dlinear', 'all'],
                       help='Model to search')
    parser.add_argument('--max-configs', type=int, default=50,
                       help='Maximum configs to test')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Map dataset to config path
    config_paths = {
        'etth1': 'assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json',
        'etth2': 'assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json',
        'weather': 'assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_48_S.json',
    }
    
    base_config_path = config_paths[args.dataset]
    
    # Determine models to search
    models = [args.model] if args.model != 'all' else ['itransformer', 'patchtst', 'gru', 'timesnet', 'dlinear']
    
    all_results = []
    
    for model in models:
        df_results = grid_search(args.dataset, model, base_config_path, args.max_configs)
        if df_results is not None:
            all_results.append(df_results)
    
    if all_results:
        df_combined = pd.concat(all_results, ignore_index=True)
        print_results_summary(df_combined)
        
        # Save results
        output_file = args.output or f'grid_search_{args.dataset}_{args.model}.csv'
        save_results(df_combined, output_file)
    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
