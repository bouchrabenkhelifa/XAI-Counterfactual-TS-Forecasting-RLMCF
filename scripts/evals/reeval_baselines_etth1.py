"""
Re-evaluate baselines (BaseNN, BaseGrad, ForecastCF) with corrected plausibility detector
=========================================================================================

This script re-evaluates existing baseline results using the corrected plausibility detector.
It loads saved counterfactuals and re-computes all metrics with the fixed detector.

Usage:
    python scripts/evals/reeval_baselines_etth1.py
"""

import os
import sys
import json
import numpy as np
import pickle
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.unified_evaluator import CounterfactualEvaluator
from src.data_provider.data_factory import data_provider
from src.utils.config import load_config

# Dataset config for loading train data
DATASET_CONFIG = "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"

# Baseline result directories
BASELINES = {
    "BaseNN": {
        "iTransformer": "baselines/BaseNN/results/basenn_itransformer_etth1_results.pkl",
        "DLinear": "baselines/BaseNN/results/basenn_dlinear_etth1_results.pkl",
    },
    "BaseGrad": {
        "iTransformer": "baselines/BaseGrad/results/basegrad_itransformer_etth1_results.pkl",
        "DLinear": "baselines/BaseGrad/results/basegrad_dlinear_etth1_results.pkl",
    },
    "ForecastCF": {
        "iTransformer": "baselines/ForecastCF_PyTorch/results/forecastcf_itransformer_etth1_results.pkl",
        "DLinear": "baselines/ForecastCF_PyTorch/results/forecastcf_dlinear_etth1_results.pkl",
    },
}

OUTPUT_DIR = "assets/results/etth1/baselines_reeval"


def load_baseline_results(result_path):
    """Load saved baseline results."""
    if not os.path.exists(result_path):
        print(f"  ⚠ Results not found: {result_path}")
        return None
    
    with open(result_path, "rb") as f:
        results = pickle.load(f)
    
    return results


def recompute_metrics(x_orig, x_cf, y_hat, y_cf, alphas, betas, x_train):
    """Recompute metrics with corrected plausibility detector."""
    
    # Create evaluator with corrected plausibility detector
    evaluator = CounterfactualEvaluator(
        x_train=x_train,
        fit_plausibility=True  # Will load pre-trained corrected detector
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        X_orig=x_orig,
        X_cf=x_cf,
        Y_hat=y_hat,
        Y_cf=y_cf,
        alphas=alphas,
        betas=betas
    )
    
    return metrics


def main():
    print("\n" + "="*80)
    print("  RE-EVALUATING BASELINES WITH CORRECTED PLAUSIBILITY DETECTOR")
    print("="*80 + "\n")
    
    # Load train data for plausibility detector
    print("Loading ETTh1 train data...")
    cfg = load_config(DATASET_CONFIG)
    train_data, train_loader = data_provider(cfg, "train")
    
    # Get all train data
    x_train_list = []
    for batch_x, _, _, _ in train_loader:
        x_train_list.append(batch_x.numpy())
    x_train = np.concatenate(x_train_list, axis=0)  # [N, seq_len, features]
    
    # Extract first feature only (OT)
    x_train = x_train[:, :, 0:1]  # [N, seq_len, 1]
    
    print(f"  Train data shape: {x_train.shape}")
    print(f"  ✓ Train data loaded\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = {}
    
    # Re-evaluate each baseline
    for method_name, models in BASELINES.items():
        print(f"\n{'='*80}")
        print(f"  {method_name}")
        print(f"{'='*80}\n")
        
        all_results[method_name] = {}
        
        for model_name, result_path in models.items():
            print(f"  Processing {model_name}...")
            
            # Load saved results
            saved_results = load_baseline_results(result_path)
            if saved_results is None:
                continue
            
            # Extract counterfactuals and forecasts
            x_orig = saved_results["x_orig"]
            x_cf = saved_results["x_cf"]
            y_hat = saved_results["y_hat"]
            y_cf = saved_results["y_cf"]
            alphas = saved_results["alphas"]
            betas = saved_results["betas"]
            
            print(f"    Loaded {len(x_orig)} samples")
            print(f"    x_orig shape: {x_orig.shape}")
            print(f"    x_cf shape: {x_cf.shape}")
            
            # Recompute metrics with corrected plausibility
            print(f"    Recomputing metrics with corrected plausibility...")
            new_metrics = recompute_metrics(
                x_orig, x_cf, y_hat, y_cf, alphas, betas, x_train
            )
            
            # Extract key metrics
            validity = new_metrics["validity_ratio"]["mean"]
            auc = new_metrics["stepwise_auc"]["mean"]
            proximity = new_metrics["proximity_l2"]["mean"]
            compactness = new_metrics["compactness"]["mean"]
            t_cons = new_metrics["temporal_consistency"]["mean"]
            plausibility = new_metrics["plausibility_ensemble"]["mean"]
            
            print(f"    ✓ Metrics recomputed:")
            print(f"      Validity:      {validity:.4f}")
            print(f"      AUC:           {auc:.4f}")
            print(f"      Proximity:     {proximity:.4f}")
            print(f"      Compactness:   {compactness:.4f}")
            print(f"      T-Cons:        {t_cons:.4f}")
            print(f"      Plausibility:  {plausibility:.4f}")
            
            # Store results
            all_results[method_name][model_name] = new_metrics
    
    # Save all results
    output_json = os.path.join(OUTPUT_DIR, "baselines_reeval_etth1.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"  ✓ All baselines re-evaluated")
    print(f"  Results saved → {output_json}")
    print(f"{'='*80}\n")
    
    # Print comparison table
    print("\n" + "="*80)
    print("  COMPARISON TABLE (with corrected plausibility)")
    print("="*80 + "\n")
    
    print(f"  {'Method':<15} {'Model':<15} {'Valid.':<10} {'AUC':<10} {'Prox.':<10} {'Plaus.':<10}")
    print(f"  {'-'*75}")
    
    for method_name in ["BaseNN", "BaseGrad", "ForecastCF"]:
        if method_name not in all_results:
            continue
        for model_name in ["iTransformer", "DLinear"]:
            if model_name not in all_results[method_name]:
                continue
            
            metrics = all_results[method_name][model_name]
            validity = metrics["validity_ratio"]["mean"]
            auc = metrics["stepwise_auc"]["mean"]
            proximity = metrics["proximity_l2"]["mean"]
            plausibility = metrics["plausibility_ensemble"]["mean"]
            
            print(f"  {method_name:<15} {model_name:<15} {validity:<10.4f} {auc:<10.4f} {proximity:<10.4f} {plausibility:<10.4f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
