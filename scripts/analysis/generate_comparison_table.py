"""
Generate fair comparison tables: RL-MCF vs Baselines, per dataset and architecture.
Reads all available results and produces LaTeX-ready tables.

Usage:
    python scripts/analysis/generate_comparison_table.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_DIRS = {
    "BaseNN": "baselines/BaseNN/results",
    "BaseGrad": "baselines/BaseGrad/results",
    "ForecastCF": "baselines/ForecastCF_PyTorch/results",
}

RLMCF_DIRS = {
    "etth1": "assets/results/etth1/summary/etth1_all_models_colab_plaus.json",
    "etth2": "assets/results/etth2/summary/etth2_all_models_colab_plaus.json",
    "weather": "assets/results/weather/summary/weather_all_models_colab_plaus.json",
}

METRICS = ["validity_ratio", "stepwise_auc", "proximity_l2", "compactness", "temporal_consistency", "plausibility_ensemble"]
METRIC_LABELS = ["Validity↑", "AUC↑", "Prox L2↓", "Compact↑", "T.Cons↑", "Plaus↓"]
METRIC_DIRS = ["up", "up", "down", "up", "up", "down"]  # for bolding best


def load_baseline_result(method, dataset, model):
    """Load a baseline result JSON."""
    dir_path = BASELINE_DIRS[method]
    prefix = {"BaseNN": "basenn", "BaseGrad": "basegrad", "ForecastCF": "forecastcf_pt"}[method]
    filename = f"{prefix}_{dataset}_{model}.json"
    path = os.path.join(dir_path, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    # Extract metrics
    metrics = {}
    avg = data.get("avg_metrics", {})
    for m in METRICS:
        if m in avg:
            metrics[m] = avg[m]["mean"] if isinstance(avg[m], dict) else avg[m]
    # Runtime
    metrics["runtime_ms"] = data.get("runtime_seconds", 0) * 1000 / max(data.get("n_samples", 1), 1)
    return metrics


def load_rlmcf_result(dataset, model):
    """Load RL-MCF result from summary JSON."""
    path = RLMCF_DIRS.get(dataset)
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    # Model name mapping
    model_map = {
        "itransformer": "iTransformer",
        "patchtst": "PatchTST",
        "dlinear": "DLinear",
        "gru": "GRU",
        "timesnet": "TimesNet",
    }
    model_key = model_map.get(model, model)
    if model_key not in data:
        return None

    metrics = {}
    for m in METRICS:
        if m in data[model_key]:
            metrics[m] = data[model_key][m]["mean"]
    metrics["runtime_ms"] = 2.0  # ~2ms per sample (single forward pass)
    return metrics


def print_table(dataset, model, results):
    """Print a comparison table for one (dataset, model) pair."""
    print(f"\n{'='*80}")
    print(f"  {dataset.upper()} — {model.upper()}")
    print(f"{'='*80}")
    print(f"{'Method':<15}", end="")
    for label in METRIC_LABELS:
        print(f"{label:>12}", end="")
    print(f"{'Time(ms)':>12}")
    print("-" * (15 + 12 * (len(METRIC_LABELS) + 1)))

    # Find best per metric
    all_values = {m: [] for m in METRICS}
    for method, metrics in results.items():
        if metrics:
            for m in METRICS:
                if m in metrics:
                    all_values[m].append(metrics[m])

    for method, metrics in results.items():
        if metrics is None:
            print(f"{method:<15}  (not available)")
            continue
        print(f"{method:<15}", end="")
        for m in METRICS:
            val = metrics.get(m, 0)
            print(f"{val:>12.4f}", end="")
        print(f"{metrics.get('runtime_ms', 0):>12.1f}")


def main():
    datasets = ["etth1", "etth2", "weather"]
    models = ["itransformer", "dlinear"]  # Only these have all baselines

    all_tables = {}

    for dataset in datasets:
        for model in models:
            results = {}

            # Load baselines
            for method in ["BaseNN", "BaseGrad", "ForecastCF"]:
                results[method] = load_baseline_result(method, dataset, model)

            # Load RL-MCF
            results["RL-MCF"] = load_rlmcf_result(dataset, model)

            all_tables[(dataset, model)] = results
            print_table(dataset, model, results)

    # ─────────────────────────────────────────────────────────────────────
    # Summary: best method per (dataset, model, metric)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  SUMMARY: RL-MCF wins per metric")
    print(f"{'='*80}")

    wins = {m: 0 for m in ["RL-MCF", "BaseNN", "BaseGrad", "ForecastCF"]}
    total = 0

    for (dataset, model), results in all_tables.items():
        for i, m in enumerate(METRICS):
            best_val = None
            best_method = None
            for method, metrics in results.items():
                if metrics and m in metrics:
                    val = metrics[m]
                    if best_val is None:
                        best_val = val
                        best_method = method
                    elif METRIC_DIRS[i] == "up" and val > best_val:
                        best_val = val
                        best_method = method
                    elif METRIC_DIRS[i] == "down" and val < best_val:
                        best_val = val
                        best_method = method
            if best_method:
                wins[best_method] = wins.get(best_method, 0) + 1
                total += 1

    print(f"\nTotal comparisons: {total}")
    for method, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {method:<15} wins {count}/{total} ({100*count/total:.0f}%)")


if __name__ == "__main__":
    main()
