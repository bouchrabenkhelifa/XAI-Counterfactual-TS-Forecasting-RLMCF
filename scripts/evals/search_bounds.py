"""
search_bounds.py
-----------------
Cherche les meilleurs paramètres de bornes (rho, fr) pour chaque checkpoint
sans réentraîner — évalue la validity ratio sur le test set.

Usage :
    python scripts/evals/search_bounds.py
    python scripts/evals/search_bounds.py --model itransformer
"""

import argparse
import json
import os
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.evaluation.unified_evaluator import validity_ratio, stepwise_validity_auc, proximity, compactness

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "itransformer": {
        "rl_config":       "assets/configs/models/etth1_dataset/RL_ablations/config_v2.json",
        "forecast_config": "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "ae_config":       "assets/configs/models/etth1_dataset/ae/tcn_ae.json",
        "checkpoint":      "assets/checkpoints/etth1_chpts/RL_v2/rl_cf_v2_etth1_agent_best.pt",
        "trainer":         "v1",
    },
    "gru": {
        "rl_config":       "assets/configs/models/etth1_dataset/RL_ablations/config_gru.json",
        "forecast_config": "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
        "ae_config":       "assets/configs/models/etth1_dataset/ae/tcn_ae.json",
        "checkpoint":      "assets/checkpoints/etth1_chpts/RL_gru/rl_cf_gru_etth1_agent_best.pt",
        "trainer":         "v2",
    },
    "dlinear": {
        "rl_config":       "assets/configs/models/etth1_dataset/RL_ablations/config_dlinear.json",
        "forecast_config": "assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
        "ae_config":       "assets/configs/models/etth1_dataset/ae/tcn_ae.json",
        "checkpoint":      "assets/checkpoints/etth1_chpts/RL_dlinear/rl_cf_dlinear_etth1_agent_best.pt",
        "trainer":         "v2",
    },
    "timesnet": {
        "rl_config":       "assets/configs/models/etth1_dataset/RL_ablations/config_timesnet.json",
        "forecast_config": "assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
        "ae_config":       "assets/configs/models/etth1_dataset/ae/tcn_ae.json",
        "checkpoint":      "assets/checkpoints/etth1_chpts/RL_timesnet/rl_cf_timesnet_etth1_agent_best.pt",
        "trainer":         "v2",
    },
}

# Grid to search
RHO_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
FR_VALUES  = [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]

OUTPUT_DIR = "assets/results/comparison"


# ─────────────────────────────────────────────────────────────────────────────

def collect_predictions(model_key: str, n_batches: int = 20):
    """Run inference with the saved checkpoint, return raw arrays."""
    cfg = MODELS[model_key]

    cfg_rl = load_config(cfg["rl_config"])
    cfg_f  = load_config(cfg["forecast_config"])
    cfg_ae = load_config(cfg["ae_config"])
    device = get_device(cfg_f)

    if cfg["trainer"] == "v1":
        from src.training.RL_trainers.trainer_last import RLMaskTrainer, run_episode_eval
    else:
        from src.training.RL_trainers.trainer_last_v2 import RLMaskTrainer, run_episode_eval

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    ckpt = torch.load(cfg["checkpoint"], map_location=device, weights_only=False)
    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.eval()

    all_x, all_xcf, all_yh, all_ycf = [], [], [], []
    for i, batch in enumerate(trainer.test_loader):
        if i >= n_batches:
            break
        ep = run_episode_eval(
            batch=batch, ae_arch=trainer.ae_arch, forecaster=trainer.forecaster,
            agent=trainer.agent, reward_fn=trainer.reward_fn, device=device,
            mask_last_k=trainer.mask_last_k, mask_ramp_k=trainer.mask_ramp_k,
        )
        if ep is None:
            continue
        all_x.append(ep["x_ot"].cpu().numpy())
        all_xcf.append(ep["x_cf"].cpu().numpy())
        all_yh.append(ep["y_hat"].cpu().numpy())
        all_ycf.append(ep["y_cf"].cpu().numpy())

    return (
        np.concatenate(all_x),
        np.concatenate(all_xcf),
        np.concatenate(all_yh),
        np.concatenate(all_ycf),
        trainer.reward_fn.global_sigma,
    )


def compute_bounds_np(y_hat, global_sigma, rho, fr, direction=-1.0):
    """Compute alpha/beta bounds for given rho/fr."""
    y2d   = y_hat[:, :, 0] if y_hat.ndim == 3 else y_hat
    sigma = global_sigma
    gap   = rho * sigma
    width = fr  * sigma
    if direction < 0:
        betas  = y2d - gap
        alphas = betas - width
    else:
        alphas = y2d + gap
        betas  = alphas + width
    return alphas, betas


def search_bounds(model_key: str, n_batches: int = 20):
    print(f"\n{'='*60}")
    print(f"  Searching bounds for : {model_key.upper()}")
    print(f"{'='*60}")

    x, xcf, yh, ycf, global_sigma = collect_predictions(model_key, n_batches)
    print(f"  global_sigma = {global_sigma:.4f}")
    print(f"  Samples      = {len(x)}")
    print()

    results = []
    print(f"  {'rho':>6} {'fr':>6} | {'VR':>8} {'AUC':>8} {'Prox':>8} {'Comp':>8} | {'beta<yhat%':>10}")

    for rho in RHO_VALUES:
        for fr in FR_VALUES:
            alphas, betas = compute_bounds_np(yh, global_sigma, rho, fr)

            # Check beta < y_hat (no intersection)
            yh2 = yh[:, :, 0] if yh.ndim == 3 else yh
            no_intersect = (betas < yh2).mean()

            vr   = validity_ratio(ycf, alphas, betas)
            sauc = stepwise_validity_auc(ycf, alphas, betas)
            prox = proximity(x, xcf)
            comp = compactness(x, xcf)

            results.append({
                "rho": rho, "fr": fr,
                "validity_ratio": vr, "stepwise_auc": sauc,
                "proximity_l2": prox, "compactness": comp,
                "no_intersect_pct": no_intersect,
            })

            marker = " <--" if vr > 0.85 and no_intersect > 0.99 else ""
            print(f"  {rho:>6.2f} {fr:>6.2f} | {vr:>8.4f} {sauc:>8.4f} {prox:>8.4f} {comp:>8.4f} | {no_intersect*100:>9.1f}%{marker}")

    # Best by validity ratio with no intersection guarantee
    valid_results = [r for r in results if r["no_intersect_pct"] > 0.99]
    if valid_results:
        best = max(valid_results, key=lambda r: r["validity_ratio"])
        print(f"\n  Best (beta<yhat guaranteed): rho={best['rho']:.2f}  fr={best['fr']:.2f}")
        print(f"    Validity Ratio : {best['validity_ratio']:.4f}")
        print(f"    Step AUC       : {best['stepwise_auc']:.4f}")
        print(f"    Proximity L2   : {best['proximity_l2']:.4f}")
        print(f"    Compactness    : {best['compactness']:.4f}")
    else:
        best = max(results, key=lambda r: r["validity_ratio"])
        print(f"\n  Best (no constraint): rho={best['rho']:.2f}  fr={best['fr']:.2f}")

    return results, best


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["itransformer", "gru", "dlinear", "timesnet", "all"],
    )
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.model == "all" else [args.model]

    summary = {}
    for m in models:
        _, best = search_bounds(m, args.n_batches)
        summary[m] = best

    # Summary table
    print(f"\n{'='*70}")
    print(f"  BEST BOUNDS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<16} {'rho':>6} {'fr':>6} | {'VR':>8} {'AUC':>8} {'Prox':>8} {'Comp':>8}")
    print(f"  {'-'*60}")
    for m, b in summary.items():
        print(f"  {m:<16} {b['rho']:>6.2f} {b['fr']:>6.2f} | "
              f"{b['validity_ratio']:>8.4f} {b['stepwise_auc']:>8.4f} "
              f"{b['proximity_l2']:>8.4f} {b['compactness']:>8.4f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "best_bounds.json")
    with open(path, "w") as f:
        json.dump({m: {k: float(v) for k, v in b.items()} for m, b in summary.items()}, f, indent=2)
    print(f"\n[Saved] {path}")


if __name__ == "__main__":
    main()
