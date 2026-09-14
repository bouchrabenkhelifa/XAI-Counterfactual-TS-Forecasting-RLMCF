"""
Multi-seed robustness experiment — Table 5 (ICONIP revision)
=============================================================
Trains 5 RL agents per model with different training seeds,
evaluating each on the SAME fixed test set (deterministic by
construction in data_loader.py — no shuffling, fixed borders).

This isolates training stochasticity from test-segment variance,
as requested by Reviewers 1211 and 1213.

Usage:
    $env:PYTHONPATH = "."
    python scripts/evals/run_multiseed_table5.py --model itransformer
    python scripts/evals/run_multiseed_table5.py --model all
    python scripts/evals/run_multiseed_table5.py --model all --eval_only

Results saved to:
    assets/results/multiseed/<model>/seed_<s>_evaluation.json
    assets/results/multiseed/table5_summary.json
"""

import os
import sys
import json
import random
import argparse
import copy
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_main import RLMaskTrainer

# ─── Seeds ────────────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]

# ─── Model configs ────────────────────────────────────────────────────────────
MODELS = {
    "itransformer": {
        "rl_config":       "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "ae_config":       "assets/configs/etth1_dataset/ae/tcn_ae.json",
    },
    "patchtst": {
        "rl_config":       "assets/configs/etth1_dataset/RL_ablations/config_patchtst.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
        "ae_config":       "assets/configs/etth1_dataset/ae/tcn_ae.json",
    },
    "timesnet": {
        "rl_config":       "assets/configs/etth1_dataset/RL_ablations/config_timesnet.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
        "ae_config":       "assets/configs/etth1_dataset/ae/tcn_ae.json",
    },
    "gru": {
        "rl_config":       "assets/configs/etth1_dataset/RL_ablations/config_gru.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
        "ae_config":       "assets/configs/etth1_dataset/ae/tcn_ae.json",
    },
    "dlinear": {
        "rl_config":       "assets/configs/etth1_dataset/RL_ablations/config_dlinear.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
        "ae_config":       "assets/configs/etth1_dataset/ae/tcn_ae.json",
    },
}

# Metrics to collect for Table 5
TABLE_METRICS = [
    "validity_ratio",
    "stepwise_auc",
    "proximity_l2",
    "compactness",
    "temporal_consistency",
    "plausibility_ensemble",
]


def set_seed(seed: int):
    """Fix all sources of randomness for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may slow down GPU training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed_ckpt_dir(base_dir: str, seed: int) -> str:
    """Return a seed-specific subdirectory to avoid overwriting runs."""
    return os.path.join(base_dir, f"seed_{seed}")


def run_one_seed(model_name: str, seed: int, paths: dict,
                 eval_batches: int = 20, eval_only: bool = False):
    """Train (or load) one agent with a fixed seed, then evaluate."""

    print(f"\n{'='*65}")
    print(f"  Model={model_name.upper()}  |  Seed={seed}")
    print(f"{'='*65}")

    # ── Fix seed BEFORE any model/data initialisation ─────────────────────────
    set_seed(seed)

    # ── Load configs ──────────────────────────────────────────────────────────
    cfg_rl = load_config(paths["rl_config"])
    cfg_f  = load_config(paths["forecast_config"])
    cfg_ae = load_config(paths["ae_config"])

    # ── Redirect checkpoints / results / figures to seed-specific dirs ────────
    base_ckpt    = cfg_rl.checkpoint_dir_lp
    base_figures = cfg_rl.figures_dir_lp
    base_results = cfg_rl.results_dir_lp

    cfg_rl.checkpoint_dir_lp = get_seed_ckpt_dir(base_ckpt,    seed)
    cfg_rl.figures_dir_lp    = get_seed_ckpt_dir(base_figures,  seed)
    cfg_rl.results_dir_lp    = get_seed_ckpt_dir(base_results,  seed)

    # Also tag the experiment name so JSON files are seed-specific
    cfg_rl.name = f"{cfg_rl.name}_seed{seed}"

    device = get_device(cfg_f)

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    # ── Train or load ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(cfg_rl.checkpoint_dir_lp,
                             f"{cfg_rl.name}_agent_best.pt")

    if eval_only:
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            return None
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
        trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"  [Eval-only] Loaded checkpoint → {ckpt_path}")
    else:
        trainer.train()

    # ── Evaluate on fixed test set ────────────────────────────────────────────
    # The test set is deterministic: no shuffle, fixed border indices in
    # data_loader.py — so results are directly comparable across seeds.
    summary, _ = trainer.evaluate(n_batches=eval_batches)
    return summary


def aggregate_seeds(all_seed_results: list) -> dict:
    """
    Compute mean ± std across seeds for each metric.
    all_seed_results: list of dicts {metric: {mean: float, std: float, ...}}
    """
    aggregated = {}
    for metric in TABLE_METRICS:
        values = []
        for res in all_seed_results:
            if res is None:
                continue
            v = res.get(metric, {})
            # Use the per-run mean as the point estimate
            if isinstance(v, dict):
                values.append(v.get("mean", float("nan")))
            else:
                values.append(float(v))

        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[metric] = {
                "mean_across_seeds": float(np.mean(values)),
                "std_across_seeds":  float(np.std(values, ddof=1)),
                "per_seed_values":   values,
                "n_seeds":           len(values),
            }
        else:
            aggregated[metric] = {
                "mean_across_seeds": float("nan"),
                "std_across_seeds":  float("nan"),
                "per_seed_values":   [],
                "n_seeds":           0,
            }
    return aggregated


def print_table5(summary: dict):
    """Print Table 5 in terminal — mean ± std across seeds."""
    col_w = 18
    header = f"  {'Model':<16}" + "".join(f"{m:>{col_w}}" for m in TABLE_METRICS)
    print(f"\n{'='*90}")
    print("  TABLE 5 — RL-MCF robustness across 5 training seeds (fixed test set, ETTh1)")
    print(f"{'='*90}")
    print(header)
    print("-" * 90)

    for model_name, metrics in summary.items():
        row = f"  {model_name:<16}"
        for m in TABLE_METRICS:
            stats = metrics.get(m, {})
            mu  = stats.get("mean_across_seeds", float("nan"))
            std = stats.get("std_across_seeds",  float("nan"))
            cell = f"{mu:.3f}±{std:.3f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"{'='*90}")
    print("  Note: mean ± std over 5 training seeds; test segments are identical")
    print("        across all runs, isolating training stochasticity.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed Table 5 robustness experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Training seeds to use (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=20,
        help="Number of test batches for evaluation (default: 20)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load existing seed checkpoints and evaluate only",
    )
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    out_dir = "assets/results/multiseed"
    os.makedirs(out_dir, exist_ok=True)

    full_summary = {}

    for model_name in models_to_run:
        paths = MODELS[model_name]
        model_out_dir = os.path.join(out_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        seed_results = []

        for seed in args.seeds:
            result = run_one_seed(
                model_name=model_name,
                seed=seed,
                paths=paths,
                eval_batches=args.eval_batches,
                eval_only=args.eval_only,
            )

            # Save individual seed result
            if result is not None:
                seed_json = os.path.join(model_out_dir, f"seed_{seed}_evaluation.json")
                with open(seed_json, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved → {seed_json}")

            seed_results.append(result)

        # Aggregate across seeds
        agg = aggregate_seeds(seed_results)
        full_summary[model_name] = agg

        # Save per-model aggregated results
        model_json = os.path.join(model_out_dir, "aggregated.json")
        with open(model_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n  Aggregated ({model_name}) → {model_json}")

    # Save full Table 5 summary
    table5_path = os.path.join(out_dir, "table5_summary.json")
    with open(table5_path, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"\nFull Table 5 saved → {table5_path}")

    # Print final table
    print_table5(full_summary)


if __name__ == "__main__":
    main()
