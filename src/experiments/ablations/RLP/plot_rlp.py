"""
Re-génère les figures RLP depuis un fichier JSON existant.
Note : la figure d'exemples CF (rlp_cf_examples.png) nécessite un re-run
       complet car les séries ne sont pas stockées dans le JSON.

Usage:
    python src/experiments/ablations/RLP/plot_rlp.py --input src/experiments/ablations/RLP/results/rlp_etth1.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiments.ablations.RLP.run_rlp import plot_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="src/experiments/ablations/RLP/results/rlp_etth1.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Reconstruire avg_summary depuis all_trials si avg_metrics est vide
    avg_summary = data.get("avg_metrics", {})
    all_summaries = data.get("all_trials", [])

    if not avg_summary and all_summaries:
        import numpy as np
        for k in all_summaries[0]:
            means = [s[k]["mean"] for s in all_summaries if isinstance(s.get(k, {}).get("mean"), (int, float))]
            if means:
                avg_summary[k] = {"mean": float(np.mean(means)), "std": float(np.std(means))}

    out_dir = os.path.dirname(args.input)
    plot_results(avg_summary, all_summaries, out_dir=out_dir)


if __name__ == "__main__":
    main()
