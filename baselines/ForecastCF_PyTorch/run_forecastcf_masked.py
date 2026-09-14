"""
Runner for ForecastCF-Masked baseline (ICONIP revision).
=========================================================
Runs ForecastCF with the same temporal mask as RL-MCF, applied at
every optimisation step (not post-hoc).

Usage:
    $env:PYTHONPATH = "."
    python baselines/ForecastCF_PyTorch/run_forecastcf_masked.py \
        --dataset etth1 --model itransformer
    python baselines/ForecastCF_PyTorch/run_forecastcf_masked.py \
        --dataset etth1 --model all
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.common.bounds import compute_bounds_np, load_bounds_params, get_rl_config_path
from baselines.common.evaluator_wrapper import run_evaluation
from baselines.ForecastCF_PyTorch.forecastcf_masked import ForecastCFMasked
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.training.RL_trainers.trainer_last import prepare_rl_data
from src.utils.config import load_config

MODELS = ["itransformer", "patchtst", "timesnet", "gru", "dlinear"]
OUTPUT_DIR = "baselines/ForecastCF_PyTorch/results_masked"


def run_single(dataset, model, seeds, device, n_batches=20):
    seq_config  = "96_96" if dataset == "weather" else "96_48"
    cfg_f_path  = f"assets/configs/{dataset}_dataset/forecasters/{model}/{dataset}_{seq_config}_S.json"
    cfg_ae_path = f"assets/configs/{dataset}_dataset/ae/tcn_ae.json"
    rl_cfg_path = get_rl_config_path(dataset, model)

    cfg_f        = load_config(cfg_f_path)
    cfg_ae       = load_config(cfg_ae_path)
    bounds_params = load_bounds_params(rl_cfg_path)

    # Read mask params from RL config
    rl_cfg = load_config(rl_cfg_path)
    mask_last_k = getattr(rl_cfg, "mask_last_k", 12)
    mask_ramp_k = getattr(rl_cfg, "mask_ramp_k", 4)

    print(f"\n[ForecastCF-Masked] {dataset}/{model} "
          f"| mask_last_k={mask_last_k} mask_ramp_k={mask_ramp_k}")

    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    _, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)

    seed_metrics = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Collect test batches
        X_test, Y_hat, X_full, X_mark = [], [], [], []
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            bx, _, bxm, _ = batch
            bx  = bx.float().to(device)
            bxm = bxm.float().to(device)
            with torch.no_grad():
                yh = forecaster.predict_ot(bx, bxm)
            X_test.append(bx[:, :, -1:].cpu().numpy())
            Y_hat.append(yh.cpu().numpy())
            X_full.append(bx)
            X_mark.append(bxm)

        X_test = np.concatenate(X_test, axis=0)
        Y_hat  = np.concatenate(Y_hat,  axis=0)
        X_full = torch.cat(X_full, dim=0)
        X_mark = torch.cat(X_mark, dim=0)

        alphas, betas = compute_bounds_np(
            x_ot=X_test, y_hat=Y_hat,
            rho=bounds_params["rho"], fr=bounds_params["fr"],
            direction=bounds_params["direction"],
            global_sigma=bounds_params["global_sigma"],
        )

        t0 = time.time()
        cf = ForecastCFMasked(
            forecaster=forecaster, device=device,
            max_iter=150, lr=1e-3, pred_margin_weight=0.8,
            mask_last_k=mask_last_k, mask_ramp_k=mask_ramp_k,
        )
        X_cf, Y_cf = cf.transform(X_test, alphas, betas, X_full, X_mark)
        runtime = time.time() - t0
        print(f"  seed={seed} | {runtime:.1f}s "
              f"({runtime/len(X_test)*1000:.1f}ms/sample)")

        res = run_evaluation(
            x_orig=X_test, x_cf=X_cf, y_hat=Y_hat, y_cf=Y_cf,
            alphas=alphas, betas=betas,
            x_train=None, method_name="ForecastCF-Masked",
            seed=seed, dataset=dataset,
        )
        m = {k: v["mean"] for k, v in res["extended_metrics"].items()
             if isinstance(v, dict) and "mean" in v}
        seed_metrics.append(m)

    # Aggregate
    all_keys = seed_metrics[0].keys()
    agg = {k: {"mean": float(np.mean([s[k] for s in seed_metrics])),
               "std":  float(np.std( [s[k] for s in seed_metrics], ddof=1))}
           for k in all_keys}

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR,
                            f"forecastcf_masked_{dataset}_{model}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset, "model": model,
                   "mask_last_k": mask_last_k, "mask_ramp_k": mask_ramp_k,
                   "seeds": seeds, "metrics": agg}, f, indent=2)
    print(f"  Saved → {out_path}")

    # Print
    print(f"\n  {'Metric':<26} {'mean':>8} {'std':>8}")
    print("  " + "-"*44)
    for k, v in agg.items():
        print(f"  {k:<26} {v['mean']:>8.4f} {v['std']:>8.4f}")

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etth1",
                        choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", default="itransformer",
                        choices=MODELS + ["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 9, 30])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    models = MODELS if args.model == "all" else [args.model]

    for m in models:
        run_single(args.dataset, m, args.seeds, device, args.n_batches)


if __name__ == "__main__":
    main()
