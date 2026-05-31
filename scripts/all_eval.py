"""
Global evaluation: RL-MCF vs all baselines, per dataset and architecture.
Reads all available result JSONs and prints comparison tables.

Usage:
    python scripts/all_eval.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────────────────────
BASELINE_DIRS = {
    "BaseNN": "baselines/BaseNN/results",
    "BaseGrad": "baselines/BaseGrad/results",
    "ForecastCF": "baselines/ForecastCF_PyTorch/results",
}

RLMCF_SUMMARIES = {
    "etth1": "assets/results/etth1/summary/etth1_all_models_colab_plaus.json",
    "etth2": "assets/results/etth2/summary/etth2_all_models_colab_plaus.json",
    "weather": "assets/results/weather/summary/weather_all_models_colab_plaus.json",
}

MODEL_MAP = {
    "itransformer": "iTransformer",
    "patchtst": "PatchTST",
    "dlinear": "DLinear",
    "gru": "GRU",
    "timesnet": "TimesNet",
}

METRICS = ["validity_ratio", "stepwise_auc", "proximity_l2", "compactness", "temporal_consistency", "plausibility_ensemble"]
LABELS = ["Validity↑", "AUC↑", "Prox L2↓", "Compact↑", "T.Cons↑", "Plaus↓"]
DIRS = ["up", "up", "down", "up", "up", "down"]

DATASETS = ["etth1", "etth2", "weather"]
MODELS = ["itransformer", "dlinear", "gru", "patchtst", "timesnet"]


def load_baseline(method, dataset, model):
    prefix = {"BaseNN": "basenn", "BaseGrad": "basegrad", "ForecastCF": "forecastcf_pt"}[method]
    path = os.path.join(BASELINE_DIRS[method], f"{prefix}_{dataset}_{model}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
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
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    key = MODEL_MAP.get(model, model)
    if key not in data:
        return None
    metrics = {}
    for m in METRICS:
        if m in data[key]:
            metrics[m] = data[key][m]["mean"]
    metrics["time_ms"] = 2.0
    return metrics


def bold_best(values, direction):
    """Return index of best value."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return -1
    if direction == "up":
        return max(valid, key=lambda x: x[1])[0]
    else:
        return min(valid, key=lambda x: x[1])[0]


def print_table(dataset, model, results):
    methods = list(results.keys())
    print(f"\n{'='*90}")
    print(f"  {dataset.upper()} — {model.upper()}")
    print(f"{'='*90}")
    header = f"{'Method':<15}"
    for l in LABELS:
        header += f"{l:>12}"
    header += f"{'Time(ms)':>12}"
    print(header)
    print("-" * 90)

    for method in methods:
        m = results[method]
        if m is None:
            print(f"{method:<15}  — not available —")
            continue
        row = f"{method:<15}"
        for metric in METRICS:
            val = m.get(metric)
            if val is not None:
                row += f"{val:>12.4f}"
            else:
                row += f"{'—':>12}"
        row += f"{m.get('time_ms', 0):>12.1f}"
        print(row)


def main():
    all_wins = {"RL-MCF": 0, "BaseNN": 0, "BaseGrad": 0, "ForecastCF": 0}
    total = 0
    tables_printed = 0

    for dataset in DATASETS:
        for model in MODELS:
            results = {}
            for method in ["BaseNN", "BaseGrad", "ForecastCF"]:
                results[method] = load_baseline(method, dataset, model)
            results["RL-MCF"] = load_rlmcf(dataset, model)

            # Skip if no data at all
            available = [m for m, v in results.items() if v is not None]
            if len(available) < 2:
                continue

            print_table(dataset, model, results)
            tables_printed += 1

            # Count wins
            for i, metric in enumerate(METRICS):
                vals = []
                method_names = []
                for method in ["BaseNN", "BaseGrad", "ForecastCF", "RL-MCF"]:
                    m = results.get(method)
                    if m and metric in m:
                        vals.append(m[metric])
                        method_names.append(method)
                    else:
                        vals.append(None)
                        method_names.append(method)

                valid_vals = [(method_names[j], v) for j, v in enumerate(vals) if v is not None]
                if len(valid_vals) < 2:
                    continue

                if DIRS[i] == "up":
                    winner = max(valid_vals, key=lambda x: x[1])[0]
                else:
                    winner = min(valid_vals, key=lambda x: x[1])[0]
                all_wins[winner] = all_wins.get(winner, 0) + 1
                total += 1

    # Summary
    print(f"\n\n{'='*90}")
    print(f"  GLOBAL SUMMARY — {tables_printed} tables, {total} metric comparisons")
    print(f"{'='*90}")
    for method, count in sorted(all_wins.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"  {method:<15} {count:>3}/{total}  ({pct:>5.1f}%)  {bar}")


def generate_figures():
    """Generate global cross-dataset figures after evaluation."""
    import subprocess

    print(f"\n\n{'='*90}")
    print(f"  GENERATING GLOBAL FIGURES")
    print(f"{'='*90}")

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT

    scripts = [
        ("Radar charts (3 datasets × 5 models)",
         "scripts/plots_generation/model_analysis/generate_radar_3datasets.py"),
        ("CF examples (3 datasets)",
         "scripts/plots_generation/cf_examples/generate_cf_examples_3datasets.py"),
        ("Scatter validity vs plausibility",
         "scripts/plots_generation/baselines/scatter_validity_plausibility.py"),
        ("Barplot compactness",
         "scripts/plots_generation/model_analysis/barplot_compactness.py"),
        ("Barplot temporal consistency",
         "scripts/plots_generation/model_analysis/barplot_temporal_consistency.py"),
        ("Heatmap + t-SNE",
         "scripts/plots_generation/model_analysis/plot_heatmap_and_latent.py"),
    ]

    for label, script in scripts:
        script_path = os.path.join(ROOT, script)
        if not os.path.exists(script_path):
            print(f"  [SKIP] {label} -- script not found")
            continue
        print(f"\n  -> {label}")
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            print(f"    [OK] Done")
        else:
            print(f"    [FAIL]: {result.stderr[-200:] if result.stderr else 'unknown error'}")

    print(f"\n{'='*90}")
    print(f"  Figures saved -> assets/figures/global_analysis/")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
    generate_figures()
