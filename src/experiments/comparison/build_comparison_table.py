"""
Build the final comparison table: ForecastCF vs Our RL method on ETTh1.

Usage:
    python -m src.experiments.comparison.build_comparison_table \
        --forecastcf_csv  baselines/ForecastCF/results/forecastcf_etth1.csv \
        --rl_results      assets/results/etth1_v2/rl_cf_v2_etth1_evaluation.json

Output:
    - Prints the comparison table to stdout
    - Saves assets/results/comparison/comparison_etth1.json
"""

import argparse
import json
import os
import pandas as pd
import numpy as np


def load_forecastcf_results(csv_path: str) -> dict:
    """
    Parse their CSV and return per-model metrics for ForecastCF method only
    (excludes BaseShift and BaseNN baselines).
    """
    df = pd.read_csv(csv_path)

    # Keep only ForecastCF rows (not BaseShift / BaseNN)
    df_fcf = df[df["cf_model"] == "ForecastCF"].copy()

    if df_fcf.empty:
        raise ValueError(f"No ForecastCF rows found in {csv_path}")

    results = {}
    for model in df_fcf["forecast_model"].unique():
        sub = df_fcf[df_fcf["forecast_model"] == model]
        results[f"ForecastCF+{model}"] = {
            "validity_ratio": round(sub["validity_ratio"].mean(), 4),
            "stepwise_auc":   round(sub["step_validity_auc"].mean(), 4),
            "proximity_l2":   round(sub["proximity"].mean(), 4),
            "compactness":    round(sub["compactness"].mean(), 4),
            "forecast_smape": round(sub["forecast_smape"].mean(), 4),
            "n_seeds":        len(sub),
        }

    return results


def load_rl_results(json_path: str, method_name: str = "Ours (RL+iTransformer)") -> dict:
    """Load our RL evaluation JSON."""
    with open(json_path) as f:
        data = json.load(f)

    return {
        method_name: {
            "validity_ratio": data.get("validity_ratio",  data.get("Validity Ratio  ↑", {}).get("mean", float("nan"))),
            "stepwise_auc":   data.get("stepwise_auc",    data.get("Step AUC        ↑", {}).get("mean", float("nan"))),
            "proximity_l2":   data.get("proximity_l2",    data.get("Proximity L2    ↓", {}).get("mean", float("nan"))),
            "compactness":    data.get("compactness",      data.get("Compactness     ↑", {}).get("mean", float("nan"))),
        }
    }


def print_table(all_results: dict, width: int = 80) -> None:
    methods = list(all_results.keys())
    col_w   = 16
    metrics = [
        ("validity_ratio", "Validity Ratio  ↑"),
        ("stepwise_auc",   "Step AUC        ↑"),
        ("proximity_l2",   "Proximity L2    ↓"),
        ("compactness",    "Compactness     ↑"),
    ]

    header = f"  {'Metric':<36}" + "".join(f"{m:>{col_w}}" for m in methods)
    sep    = "-" * max(width, len(header) + 4)

    print(f"\n{'=' * max(width, len(header) + 4)}")
    print(f"  COMPARISON TABLE — ETTh1  (ForecastCF metrics, atol=0.01)")
    print(f"{'=' * max(width, len(header) + 4)}")
    print(header)
    print(sep)

    for key, display in metrics:
        row = f"  {display:<36}"
        best_val = None
        higher_better = key in ("validity_ratio", "stepwise_auc", "compactness")

        vals = [float(all_results[m].get(key, float("nan"))) for m in methods]
        finite = [v for v in vals if not np.isnan(v)]
        if not finite:
            continue
        if higher_better:
            best_val = max(finite)
        else:
            best_val = min(finite)

        for m in methods:
            v = float(all_results[m].get(key, float("nan")))
            marker = " *" if (not np.isnan(v) and v == best_val) else "  "
            row += f"{v:>{col_w-2}.4f}{marker}"
        print(row)

    print(sep)
    print(f"  * = best value")
    print(f"{'=' * max(width, len(header) + 4)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecastcf_csv", default="baselines/ForecastCF/results/forecastcf_etth1.csv")
    parser.add_argument("--rl_results",     default="assets/results/etth1_v2/rl_cf_v2_etth1_evaluation.json")
    parser.add_argument("--rl_name",        default="Ours (RL+iTransformer)")
    parser.add_argument("--output_dir",     default="assets/results/comparison")
    A = parser.parse_args()

    print(f"Loading ForecastCF results from: {A.forecastcf_csv}")
    fcf_results = load_forecastcf_results(A.forecastcf_csv)

    print(f"Loading RL results from:         {A.rl_results}")
    rl_results  = load_rl_results(A.rl_results, A.rl_name)

    all_results = {**fcf_results, **rl_results}

    print_table(all_results)

    # Save
    os.makedirs(A.output_dir, exist_ok=True)
    out_path = os.path.join(A.output_dir, "comparison_etth1.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
