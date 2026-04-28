#!/usr/bin/env python
# coding: utf-8
"""
Adaptation de ForecastCF pour les modèles PyTorch (iTransformer, etc.)
Version séparée qui n'écrase pas le code TensorFlow original.
"""

import logging
import os
import sys
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ajouter le chemin src du projet principal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Imports depuis le même répertoire
from baselines.ForecastCF.src.pytorch_adapter import PyTorchModelWrapper
from baselines.ForecastCF.src._helper import (
    add_extra_dim,
    cf_metrics,
)
from baselines.ForecastCF.src._utils import ResultWriter

warnings.filterwarnings(action="ignore")


def main():
    parser = ArgumentParser(
        description="Run ForecastCF with PyTorch models (iTransformer, etc.)"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to trained PyTorch model checkpoint (.pth)"
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        help="Model type: itransformer, timesnet, dlinear, gru, patchtst"
    )
    parser.add_argument(
        "--config-path", type=str, required=True,
        help="Path to model config JSON file"
    )
    parser.add_argument(
        "--dataset", type=str, default="etth1",
        help="Dataset name"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--horizon", type=int, required=True,
        help="Forecasting horizon"
    )
    parser.add_argument(
        "--back-horizon", type=int, required=True,
        help="Look-back window size"
    )
    parser.add_argument(
        "--center", type=str, default="median",
        help="Center parameter for bounds: median, mean, last, min, max"
    )
    parser.add_argument(
        "--desired-shift", type=float, default=0,
        help="Desired shift from center (e.g., 0.2 for 120% of center)"
    )
    parser.add_argument(
        "--desired-change", type=float, required=True,
        help="Desired trend change (e.g., -0.1 for 10% decrease)"
    )
    parser.add_argument(
        "--poly-order", type=int, default=1,
        help="Polynomial order for trend"
    )
    parser.add_argument(
        "--fraction-std", type=float, default=1.0,
        help="Fraction of std for bound width"
    )
    parser.add_argument(
        "--random-seed", type=int, default=39,
        help="Random seed"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu"
    )
    parser.add_argument(
        "--test-samples", type=int, default=None,
        help="Number of test samples to use (None = all)"
    )
    
    A = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting ForecastCF with PyTorch model: {A.model_type}")
    logger.info(f"Model path: {A.model_path}")
    logger.info(f"Config path: {A.config_path}")
    
    # Set random seeds
    np.random.seed(A.random_seed)
    torch.manual_seed(A.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(A.random_seed)
    
    # Load model config
    import json
    with open(A.config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    config = Config(**config_dict)
    
    # Load PyTorch model
    logger.info("Loading PyTorch model...")
    model = load_model(A.model_type, A.model_path, config, A.device)
    logger.info("Model loaded successfully")
    
    # Load and prepare data
    logger.info(f"Loading dataset from {A.data_path}")
    X_test, Y_test = load_test_data(
        A.data_path,
        A.back_horizon,
        A.horizon,
        config
    )
    logger.info(f"Test data shape: X={X_test.shape}, Y={Y_test.shape}")
    
    # Subsample if requested
    if A.test_samples is not None and A.test_samples < len(X_test):
        indices = np.random.choice(len(X_test), A.test_samples, replace=False)
        X_test = X_test[indices]
        Y_test = Y_test[indices]
        logger.info(f"Using {A.test_samples} test samples")
    
    # Generate bounds for each test sample
    logger.info("Generating target bounds...")
    desired_max_lst, desired_min_lst = [], []
    
    for i in range(len(X_test)):
        max_bound, min_bound = generate_bounds(
            center=A.center,
            shift=A.desired_shift,
            change_percent=A.desired_change,
            poly_order=A.poly_order,
            horizon=A.horizon,
            fraction_std=A.fraction_std,
            input_series=X_test[i]
        )
        desired_max_lst.append(max_bound)
        desired_min_lst.append(min_bound)
    
    # Run ForecastCF
    logger.info("Running ForecastCF optimization...")
    from baselines.ForecastCF.src.forecastcf import ForecastCF
    import tensorflow as tf
    
    cf_model = ForecastCF(
        max_iter=1000,  # Augmenté de 100 à 1000
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.01),  # Augmenté de 0.001 à 0.01
        pred_margin_weight=0.5,  # Augmenté de 0.25 à 0.5 pour forcer plus de changement
        step_weights=np.ones((1, A.back_horizon, 1)),
        random_state=A.random_seed,
    )
    
    # Fit with PyTorch model wrapper
    cf_model.fit(model)
    
    # Generate counterfactuals
    start_time = time.time()
    cf_samples, losses, _ = cf_model.transform(
        X_test, desired_max_lst, desired_min_lst
    )
    elapsed_time = time.time() - start_time
    logger.info(f"CF generation time: {elapsed_time:.4f}s")
    
    # Evaluate counterfactuals
    logger.info("Evaluating counterfactuals...")
    Y_cf_pred = model.predict(cf_samples)
    
    input_indices = range(0, A.back_horizon)
    label_indices = range(A.back_horizon, A.back_horizon + A.horizon)
    
    (
        validity,
        proximity,
        compactness,
        cumsum_valid_steps,
        cumsum_counts,
        cumsum_auc,
        slope_diff,
        slope_diff_preds,
    ) = cf_metrics(
        desired_max_lst,
        desired_min_lst,
        X_test,
        cf_samples,
        Y_cf_pred,
        input_indices,
        label_indices,
    )
    
    logger.info(f"Results:")
    logger.info(f"  Validity: {validity:.4f}")
    logger.info(f"  Step Validity AUC: {cumsum_auc:.4f}")
    logger.info(f"  Proximity: {proximity:.4f}")
    logger.info(f"  Compactness: {compactness:.4f}")
    
    # Save results
    result_writer = ResultWriter(file_name=A.output, dataset_name=A.dataset)
    if not os.path.isfile(A.output):
        result_writer.write_head()
    
    result_writer.write_result(
        random_seed=A.random_seed,
        method_name=A.model_type,
        cf_method_name="ForecastCF",
        horizon=A.horizon,
        desired_change=A.desired_change,
        fraction_std=A.fraction_std,
        forecast_smape=0.0,  # Not computed here
        forecast_mase=0.0,   # Not computed here
        validity_ratio=validity,
        proximity=proximity,
        compactness=compactness,
        step_validity_auc=cumsum_auc,
    )
    
    logger.info(f"Results saved to {A.output}")


def load_model(model_type, model_path, config, device):
    """Load PyTorch model based on type."""
    
    if model_type.lower() == "itransformer":
        from src.models.Forecaster.iTransformer import Model
        model_class = Model
    elif model_type.lower() == "timesnet":
        from src.models.Forecaster.TimesNet import Model
        model_class = Model
    elif model_type.lower() == "dlinear":
        from src.models.Forecaster.DLinear import Model
        model_class = Model
    elif model_type.lower() == "gru":
        from src.models.Forecaster.GRU import Model
        model_class = Model
    elif model_type.lower() == "patchtst":
        from src.models.Forecaster.PatchTST import Model
        model_class = Model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model
    model = model_class(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Wrap for TensorFlow compatibility
    return PyTorchModelWrapper(model, device=device)


def load_test_data(data_path, seq_len, pred_len, config):
    """Load and prepare test data."""
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Extract target column
    if hasattr(config, 'target'):
        target_col = config.target
    else:
        target_col = df.columns[-1]  # Last column by default
    
    data = df[[target_col]].values
    
    # Use last 20% as test set (simple split)
    test_start = int(len(data) * 0.8)
    test_data = data[test_start:]
    
    # Normalize
    mean = test_data.mean()
    std = test_data.std()
    test_data = (test_data - mean) / std
    
    # Create sequences
    X_test, Y_test = [], []
    for i in range(len(test_data) - seq_len - pred_len + 1):
        X_test.append(test_data[i:i+seq_len])
        Y_test.append(test_data[i+seq_len:i+seq_len+pred_len])
    
    return np.array(X_test), np.array(Y_test)


def polynomial_values(shift, change_percent, poly_order, horizon):
    """Generate polynomial trend values."""
    if horizon == 1:
        return np.asarray([shift + change_percent])
    
    p_orders = [shift]
    p_orders.extend([0 for _ in range(poly_order)])
    p_orders[-1] = change_percent / ((horizon - 1) ** poly_order)
    
    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    value_lst = np.asarray([np.polyval(p_coefs, i) for i in range(horizon)])
    
    return value_lst


def generate_bounds(center, shift, change_percent, poly_order, horizon, 
                   fraction_std, input_series):
    """Generate upper and lower bounds for counterfactual targets."""
    
    if center == "last":
        start_value = input_series[-1, 0]
    elif center == "median":
        start_value = np.median(input_series)
    elif center == "mean":
        start_value = np.mean(input_series)
    elif center == "min":
        start_value = np.min(input_series)
    elif center == "max":
        start_value = np.max(input_series)
    else:
        raise ValueError(f"Unknown center: {center}")
    
    std = np.std(input_series)
    
    trend = polynomial_values(shift, change_percent, poly_order, horizon)
    
    upper = add_extra_dim(
        start_value * (1 + trend + fraction_std * std)
    )
    lower = add_extra_dim(
        start_value * (1 + trend - fraction_std * std)
    )
    
    return upper, lower


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
