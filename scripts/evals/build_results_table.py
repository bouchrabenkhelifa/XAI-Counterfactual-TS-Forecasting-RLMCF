"""
build_results_table.py
-----------------------
Génère le tableau comparatif final pour le papier.

Colonnes : ForecastCF (N-BEATS) | Ours (iTransformer) | Ours (DLinear) | Ours (GRU) | Ours (TimesNet)
Métriques : Validity Ratio, Step AUC, Proximity L2, Compactness,
            Roughness Ratio, Temporal Consistency, Plausibility

Usage :
    python scripts/evals/build_results_table.py
    python scripts/evals/build_results_table.py --save_csv
    python scripts/evals/build_results_table.py --save_latex
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────

RL_RESULTS = {
    "Ours (iTransformer)": "assets/results/etth1_v2/rl_cf_v2_etth1_evaluation.json",
    "Ours (DLinear)":      "assets/results/etth1_dlinear/rl_cf_dlinear_etth1_evaluation.json",
    "Ours (GRU)":          "assets/results/etth1_gru/rl_cf_gru_etth1_evaluation.json",
    "Ours (TimesNet)":     "assets/results/etth1_timesnet/rl_cf_timesnet_etth1_evaluation.json",
}

FORECASTCF_CSV = "baselines/ForecastCF/results/forecastcf_etth1.csv"

OUTPUT_DIR = "assets/results/comparison"

# ─────────────────────────────────────────────────────────────────────────────
# Metrics definition
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    # (json_key,              display_name,              higher_is_better)
    ("validity_ratio",        "Validity Ratio ↑",        True),
    ("stepwise_auc",          "Step AUC ↑",              True),
    ("proximity_l2",          "Proximity L2 ↓",          False),
    ("compactness",           "Compactness ↑",           True),
    ("roughness_ratio",       "Roughness Ratio ↓",       False),
    ("temporal_consistency",  "Temporal Consistency ↑",  True),
    ("plausibility_ensemble", "Plausibility ↓",          False),
]

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_rl_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: float(v["mean"]) for k, v in data.items()}


def load_forecastcf_results(csv_path: str) -> dict:
    """Extract ForecastCF (not baselines) row from the CSV."""
    import csv
    results = {}
    if not os.path.exists(csv_path):
        print(f"[Warning] ForecastCF CSV not found: {csv_path}")
        return results

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("cf_model", "").strip() == "ForecastCF":
                results["validity_ratio"]       = float(row["validity_ratio"])
                results["stepwise_auc"]         = float(row["step_validity_auc"])
                results["proximity_l2"]         = float(row["proximity"])
                results["compactness"]          = float(row["compactness"])
                # ForecastCF does not report these — mark as N/A
                results["roughness_ratio"]      = float("nan")
                results["temporal_consistency"] = float("nan")
                results["plausibility_ensemble"]= float("nan")
                break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_table(all_results: dict) -> pd.DataFrame:
    methods = list(all_results.keys())
    rows = []
    for json_key, display_name, higher in METRICS:
        row = {"Metric": display_name}
        vals = []
        for m in methods:
            v = all_results[m].get(json_key, float("nan"))
            row[m] = v
            if not np.isnan(v):
                vals.append((v, m))
        # Find best
        if vals:
            best_val, best_m = max(vals, key=lambda x: x[0]) if higher \
                               else min(vals, key=lambda x: x[0])
            row["_best"] = best_m
        else:
            row["_best"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, all_results: dict):
    methods = [m for m in all_results.keys()]
    col_w   = 22

    header = f"  {'Metric':<30}" + "".join(f"{m:>{col_w}}" for m in methods)
    sep    = "=" * (30 + col_w * len(methods) + 2)

    print(f"\n{sep}")
    print(f"  COMPARISON TABLE — ETTh1  (RL-CF vs ForecastCF)")
    print(f"{sep}")
    print(header)
    print("-" * len(sep))

    sections = [
        ("ForecastCF Paper Metrics",
         ["Validity Ratio ↑", "Step AUC ↑", "Proximity L2 ↓", "Compactness ↑"]),
        ("Temporal Quality (Ours only)",
         ["Roughness Ratio ↓", "Temporal Consistency ↑"]),
        ("Plausibility (Ours only)",
         ["Plausibility ↓"]),
    ]

    for section_title, metric_names in sections:
        print(f"\n  -- {section_title}")
        for _, row in df[df["Metric"].isin(metric_names)].iterrows():
            line = f"  {row['Metric']:<30}"
            for m in methods:
                v = row[m]
                if np.isnan(v):
                    cell = "N/A"
                else:
                    cell = f"{v:.4f}"
                    if row["_best"] == m:
                        cell = f"*{cell}*"   # mark best
                line += f"{cell:>{col_w}}"
            print(line)

    print(f"\n{sep}")
    print("  * = best value for that metric")
    print(f"{sep}\n")


def save_csv(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_table_etth1.csv")
    df.drop(columns=["_best"]).to_csv(path, index=False)
    print(f"[Saved] CSV → {path}")


def save_latex(df: pd.DataFrame, all_results: dict, output_dir: str):
    """Generate a LaTeX table ready for the paper."""
    os.makedirs(output_dir, exist_ok=True)
    methods = list(all_results.keys())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Counterfactual explanation results on ETTh1.}")
    lines.append(r"\label{tab:results_etth1}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    col_spec = "l" + "c" * len(methods)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = "Metric & " + " & ".join(methods) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    section_map = {
        "Validity Ratio ↑":       "ForecastCF Metrics",
        "Step AUC ↑":             None,
        "Proximity L2 ↓":         None,
        "Compactness ↑":          None,
        "Roughness Ratio ↓":      r"\midrule Temporal Quality",
        "Temporal Consistency ↑": None,
        "Plausibility ↓":         r"\midrule Plausibility",
    }

    for _, row in df.iterrows():
        metric = row["Metric"]
        if section_map.get(metric):
            sec = section_map[metric]
            if sec.startswith(r"\midrule"):
                lines.append(r"\midrule")
        cells = []
        for m in methods:
            v = row[m]
            if np.isnan(v):
                cell = "--"
            else:
                cell = f"{v:.4f}"
                if row["_best"] == m:
                    cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{metric} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    path = os.path.join(output_dir, "comparison_table_etth1.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved] LaTeX → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_csv",   action="store_true")
    parser.add_argument("--save_latex", action="store_true")
    args = parser.parse_args()

    # Load all results
    all_results = {}

    # ForecastCF baseline first
    fcf = load_forecastcf_results(FORECASTCF_CSV)
    if fcf:
        all_results["ForecastCF (N-BEATS)"] = fcf

    # Our method
    for name, path in RL_RESULTS.items():
        if os.path.exists(path):
            all_results[name] = load_rl_results(path)
        else:
            print(f"[Warning] Not found: {path}")

    if not all_results:
        print("No results found.")
        return

    # Build and print table
    df = build_table(all_results)
    print_table(df, all_results)

    # Save
    if args.save_csv:
        save_csv(df, OUTPUT_DIR)
    if args.save_latex:
        save_latex(df, all_results, OUTPUT_DIR)

    # Always save both
    save_csv(df, OUTPUT_DIR)
    save_latex(df, all_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
