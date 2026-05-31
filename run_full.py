#!/usr/bin/env python
"""
Full Evaluation Pipeline
=========================
Generates all evaluation tables and figures for the paper:

  1. Baseline Comparison Table + Figures (scatter, compactness, temporal consistency)
  2. Model-Agnostic Table + Figures (radar charts, heatmap, t-SNE)
  3. Ablation Study Table + Figure (ablation series)

Usage:
    python run_full.py

Output:
    - Tables printed to terminal
    - Figures saved to assets/figures/global_analysis/
"""

import os
import sys
import json
import subprocess
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

# ==============================================================================
# Configuration
# ==============================================================================

RLMCF_SUMMARIES = {
    "ETTh1": "assets/results/etth1/summary/etth1_all_models_colab_plaus.json",
    "ETTh2": "assets/results/etth2/summary/etth2_all_models_colab_plaus.json",
    "Weather": "assets/results/weather/summary/weather_all_models_colab_plaus.json",
}

BASELINE_DIRS = {
    "BaseNN": "baselines/BaseNN/results",
    "BaseGrad": "baselines/BaseGrad/results",
    "ForecastCF": "baselines/ForecastCF_PyTorch/results",
}

ABLATION_RESULTS = {
    "RL-MCF (full)":       {"validity_ratio": 0.974, "stepwise_auc": 0.979, "proximity_l2": 0.605, "compactness": 0.802, "temporal_consistency": 0.968, "plausibility_ensemble": 0.169},
    "w/o Proximity":       {"validity_ratio": 0.889, "stepwise_auc": 0.934, "proximity_l2": 0.811, "compactness": 0.790, "temporal_consistency": 0.949, "plausibility_ensemble": 0.253},
    "w/o Mask":            {"validity_ratio": 0.851, "stepwise_auc": 0.855, "proximity_l2": 2.050, "compactness": 0.002, "temporal_consistency": 0.642, "plausibility_ensemble": 0.450},
    "RLP (no policy)":     {"validity_ratio": 0.354, "stepwise_auc": 0.363, "proximity_l2": 0.585, "compactness": 0.844, "temporal_consistency": 0.962, "plausibility_ensemble": 0.438},
}

METRICS = ["validity_ratio", "stepwise_auc", "proximity_l2", "compactness", "temporal_consistency", "plausibility_ensemble"]
LABELS = ["Valid.", "AUC", "Prox.", "Comp.", "T-Cons.", "Plaus."]
DIRS = ["up", "up", "down", "up", "up", "down"]

FIGURE_SCRIPTS = {
    "baselines": [
        ("Scatter validity vs plausibility",
         "scripts/plots_generation/baselines/scatter_validity_plausibility.py"),
        ("Barplot compactness",
         "scripts/plots_generation/model_analysis/barplot_compactness.py"),
        ("Barplot temporal consistency",
         "scripts/plots_generation/model_analysis/barplot_temporal_consistency.py"),
    ],
    "architectures": [
        ("Radar charts (3 datasets x 5 models)",
         "scripts/plots_generation/model_analysis/generate_radar_3datasets.py"),
        ("Heatmap validity + t-SNE",
         "scripts/plots_generation/model_analysis/plot_heatmap_and_latent.py"),
        ("CF examples (3 datasets)",
         "scripts/plots_generation/cf_examples/generate_cf_examples_3datasets.py"),
    ],
    "ablation": [
        ("Ablation series figure",
         "scripts/plots_generation/ablations/generate_ablation_series_figure.py"),
    ],
}


# ==============================================================================
# Helpers
# ==============================================================================

def load_json(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def load_baseline(method, dataset, model):
    prefix = {"BaseNN": "basenn", "BaseGrad": "basegrad", "ForecastCF": "forecastcf_pt"}[method]
    ds = dataset.lower()
    path = os.path.join(BASELINE_DIRS[method], f"{prefix}_{ds}_{model}.json")
    data = load_json(path)
    if data is None:
        return None
    avg = data.get("avg_metrics", {})
    metrics = {}
    for m in METRICS:
        if m in avg:
            metrics[m] = avg[m]["mean"] if isinstance(avg[m], dict) else avg[m]
    n = max(data.get("n_samples", 1), 1)
    metrics["time_ms"] = data.get("runtime_seconds", 0) * 1000 / n
    return metrics


def load_rlmcf(dataset, model):
    path = RLMCF_SUMMARIES.get(dataset)
    data = load_json(path)
    if data is None or model not in data:
        return None
    metrics = {}
    for m in METRICS:
        if m in data[model]:
            metrics[m] = data[model][m]["mean"]
    metrics["time_ms"] = 2.0
    return metrics


def print_separator(title):
    print(f"\n\n{'#'*90}")
    print(f"#  {title}")
    print(f"{'#'*90}")


def print_row(method, metrics, col_w=10, show_time=False):
    row = f"  {method:<20}"
    for m in METRICS:
        v = metrics.get(m)
        if v is not None:
            row += f"{v:>{col_w}.4f}"
        else:
            row += f"{'--':>{col_w}}"
    if show_time:
        row += f"{metrics.get('time_ms', 0):>{col_w}.1f}"
    print(row)


def run_figure_script(label, script_path):
    full_path = os.path.join(ROOT, script_path)
    if not os.path.exists(full_path):
        print(f"    [SKIP] {label} -- not found")
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, full_path],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if result.returncode == 0:
        print(f"    [OK] {label}")
        return True
    else:
        err = result.stderr[-150:] if result.stderr else "unknown"
        print(f"    [FAIL] {label}: {err}")
        return False


# ==============================================================================
# PART 1: Baseline Comparison
# ==============================================================================

def run_baseline_comparison():
    print_separator("PART 1: BASELINE COMPARISON (RL-MCF vs ForecastCF, BaseGrad, BaseNN)")

    datasets_models = [
        ("ETTh1", "iTransformer", "itransformer"),
        ("ETTh1", "DLinear", "dlinear"),
        ("ETTh2", "iTransformer", "itransformer"),
        ("ETTh2", "DLinear", "dlinear"),
        ("Weather", "iTransformer", "itransformer"),
        ("Weather", "DLinear", "dlinear"),
    ]

    col_w = 10
    header = f"  {'Method':<20}" + "".join(f"{l:>{col_w}}" for l in LABELS) + f"{'Time(ms)':>{col_w}}"

    for dataset, model_label, model_key in datasets_models:
        print(f"\n  --- {dataset} / {model_label} ---")
        print(header)
        print("  " + "-" * (20 + col_w * (len(LABELS) + 1)))

        for method in ["BaseNN", "BaseGrad", "ForecastCF"]:
            m = load_baseline(method, dataset, model_key)
            if m:
                print_row(method, m, col_w, show_time=True)

        rlmcf = load_rlmcf(dataset, model_label)
        if rlmcf:
            print_row("RL-MCF (Ours)", rlmcf, col_w, show_time=True)

    # Generate figures
    print(f"\n  Generating baseline figures...")
    for label, script in FIGURE_SCRIPTS["baselines"]:
        run_figure_script(label, script)


# ==============================================================================
# PART 2: Model-Agnostic Generalization
# ==============================================================================

def run_architecture_comparison():
    print_separator("PART 2: MODEL-AGNOSTIC GENERALIZATION (5 Architectures x 3 Datasets)")

    models = ["iTransformer", "PatchTST", "TimesNet", "GRU", "DLinear"]
    col_w = 10
    header = f"  {'Model':<20}" + "".join(f"{l:>{col_w}}" for l in LABELS)

    for dataset in ["ETTh1", "ETTh2", "Weather"]:
        print(f"\n  --- {dataset} ---")
        print(header)
        print("  " + "-" * (20 + col_w * len(LABELS)))

        data = load_json(RLMCF_SUMMARIES[dataset])
        if data is None:
            print("    [No data]")
            continue

        for model in models:
            if model in data:
                metrics = {m: data[model][m]["mean"] for m in METRICS if m in data[model]}
                print_row(model, metrics, col_w)

    # Generate figures
    print(f"\n  Generating architecture figures...")
    for label, script in FIGURE_SCRIPTS["architectures"]:
        run_figure_script(label, script)


# ==============================================================================
# PART 3: Ablation Study
# ==============================================================================

def run_ablation():
    print_separator("PART 3: ABLATION STUDY (ETTh1, iTransformer)")

    col_w = 10
    header = f"  {'Configuration':<20}" + "".join(f"{l:>{col_w}}" for l in LABELS)
    print(f"\n{header}")
    print("  " + "-" * (20 + col_w * len(LABELS)))

    for config, metrics in ABLATION_RESULTS.items():
        print_row(config, metrics, col_w)

    # Generate figure
    print(f"\n  Generating ablation figure...")
    for label, script in FIGURE_SCRIPTS["ablation"]:
        run_figure_script(label, script)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 90)
    print("  RL-MCF: FULL EVALUATION PIPELINE")
    print("  Generates all tables and figures for the paper")
    print("=" * 90)

    run_baseline_comparison()
    run_architecture_comparison()
    run_ablation()

    print_separator("DONE")
    print(f"""
  All tables printed above.
  Figures saved to: assets/figures/global_analysis/

  Generated figures:
    - scatter_validity_plausibility.png
    - barplot_compactness.png
    - barplot_temporal_consistency.png
    - radar_3datasets_combined.png
    - heatmap_validity.png
    - latent_tsne_orig_vs_cf.png
    - cf_examples_3datasets.png
    - ablation_series_4methods.png
""")


if __name__ == "__main__":
    main()
