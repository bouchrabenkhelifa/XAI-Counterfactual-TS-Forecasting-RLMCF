#!/usr/bin/env python
"""
Extract dataset characteristics that determine configuration choices.

Usage:
    python scripts/analysis/extract_dataset_features.py
    
Output:
    Prints dataset statistics in table format for LaTeX integration
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)


def extract_features(csv_path, target_col, dataset_name):
    """Extract key features from a dataset."""
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get target series
    target = df[target_col].values
    
    # Calculate features
    features = {
        'dataset': dataset_name,
        'n_samples': len(df),
        'n_variables': len(df.columns) - 1,  # exclude date
        'target_col': target_col,
        'date_start': df['date'].min(),
        'date_end': df['date'].max(),
        'time_span_days': (df['date'].max() - df['date'].min()).days,
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'range': target.max() - target.min(),
        'autocorr_lag1': pd.Series(target).autocorr(lag=1),
        'autocorr_lag24': pd.Series(target).autocorr(lag=24) if len(target) > 24 else np.nan,
        'autocorr_lag96': pd.Series(target).autocorr(lag=96) if len(target) > 96 else np.nan,
    }
    
    # Calculate temporal resolution
    time_diffs = df['date'].diff().dropna()
    most_common_diff = time_diffs.value_counts().index[0]
    features['temporal_resolution'] = str(most_common_diff)
    
    # Calculate samples per day
    diff_str = str(most_common_diff)
    if 'days' in diff_str:
        # Extract hours and minutes from timedelta string
        parts = diff_str.split()
        if len(parts) >= 3:  # "0 days HH:MM:SS"
            time_part = parts[2]
            h, m, s = map(int, time_part.split(':'))
            total_minutes = h * 60 + m
            features['samples_per_day'] = 24 * 60 // total_minutes if total_minutes > 0 else 1
        else:
            features['samples_per_day'] = 1
    else:
        features['samples_per_day'] = 1
    
    return features


def print_features_table(features_list):
    """Print features in LaTeX table format."""
    
    print("\n" + "="*100)
    print("DATASET CHARACTERISTICS - LaTeX Table Format")
    print("="*100 + "\n")
    
    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    
    # Print as LaTeX table
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|}")
    print("\\hline")
    print("\\textbf{Characteristic} & \\textbf{ETTh1} & \\textbf{ETTh2} & \\textbf{Weather} \\\\")
    print("\\hline")
    
    # Key metrics to display
    metrics = [
        ('dataset', 'Dataset'),
        ('temporal_resolution', 'Temporal Resolution'),
        ('samples_per_day', 'Samples per Day'),
        ('n_samples', 'Total Samples'),
        ('n_variables', 'Number of Variables'),
        ('time_span_days', 'Time Span (days)'),
        ('mean', 'Mean Value'),
        ('std', 'Std Dev (Volatility)'),
        ('min', 'Min Value'),
        ('max', 'Max Value'),
        ('range', 'Range'),
        ('autocorr_lag1', 'Autocorr (lag=1)'),
        ('autocorr_lag24', 'Autocorr (lag=24)'),
        ('autocorr_lag96', 'Autocorr (lag=96)'),
    ]
    
    for key, label in metrics:
        values = []
        for feat in features_list:
            val = feat.get(key, 'N/A')
            if isinstance(val, float):
                if key.startswith('autocorr'):
                    values.append(f"{val:.4f}")
                else:
                    values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        
        print(f"{label} & {values[0]} & {values[1]} & {values[2]} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Dataset Characteristics Comparison}")
    print("\\label{tab:dataset_characteristics}")
    print("\\end{table}\n")


def print_features_csv(features_list):
    """Print features in CSV format."""
    
    print("\n" + "="*100)
    print("DATASET CHARACTERISTICS - CSV Format")
    print("="*100 + "\n")
    
    df_features = pd.DataFrame(features_list)
    print(df_features.to_csv(index=False))


def print_analysis_summary(features_list):
    """Print analysis summary with key insights."""
    
    print("\n" + "="*100)
    print("KEY INSIGHTS FOR CONFIGURATION")
    print("="*100 + "\n")
    
    etth1 = features_list[0]
    etth2 = features_list[1]
    weather = features_list[2]
    
    print("1. TEMPORAL RESOLUTION IMPACT:")
    print(f"   - ETTh1/ETTh2: {etth1['temporal_resolution']} (hourly)")
    print(f"   - Weather: {weather['temporal_resolution']} (10-minute)")
    print(f"   - Ratio: {weather['samples_per_day'] / etth1['samples_per_day']:.1f}x finer resolution\n")
    
    print("2. VOLATILITY ANALYSIS:")
    print(f"   - ETTh1 Std: {etth1['std']:.2f}°C")
    print(f"   - ETTh2 Std: {etth2['std']:.2f}°C")
    print(f"   - Weather Std: {weather['std']:.2f}°C")
    print(f"   - Weather/ETTh1 ratio: {weather['std']/etth1['std']:.2f}x (much more stable)\n")
    
    print("3. AUTOCORRELATION PATTERNS:")
    print(f"   - ETTh1 lag=1: {etth1['autocorr_lag1']:.4f}")
    print(f"   - Weather lag=1: {weather['autocorr_lag1']:.4f}")
    print(f"   - Weather has stronger autocorrelation (more predictable)\n")
    
    print("4. CONFIGURATION IMPLICATIONS:")
    seq_len_scale = weather['samples_per_day'] / etth1['samples_per_day']
    print(f"   - seq_len scaling: 96 × {seq_len_scale:.0f} = {int(96 * seq_len_scale)}")
    print(f"   - Model capacity: Reduce by ~{(1 - weather['std']/etth1['std'])*100:.0f}% (lower volatility)")
    print(f"   - Dropout: Can be lower (more stable data)")
    print(f"   - moving_avg: Scale by {seq_len_scale:.0f}x\n")


def main():
    """Main function."""
    
    datasets = [
        ('assets/datasets/ETTh1.csv', 'OT', 'ETTh1'),
        ('assets/datasets/ETTh2.csv', 'OT', 'ETTh2'),
        ('assets/datasets/weather.csv', 'T (degC)', 'Weather'),
    ]
    
    features_list = []
    
    print("\nExtracting dataset features...\n")
    
    for csv_path, target_col, name in datasets:
        if os.path.exists(csv_path):
            print(f"Processing {name}...")
            features = extract_features(csv_path, target_col, name)
            features_list.append(features)
            print(f"  [OK] {name} extracted\n")
        else:
            print(f"  ✗ {name} not found at {csv_path}\n")
    
    if len(features_list) == 3:
        # Print outputs
        print_features_table(features_list)
        print_features_csv(features_list)
        print_analysis_summary(features_list)
        
        print("\n" + "="*100)
        print("✓ Feature extraction complete!")
        print("="*100 + "\n")
    else:
        print("Error: Not all datasets found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
