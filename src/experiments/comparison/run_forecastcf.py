"""
Run ForecastCF baseline (PyTorch) on ETTh1.

Usage:
    python -m src.experiments.comparison.run_forecastcf \
        --rl_config   assets/configs/models/etth1_dataset/RL_ablations/config_v2.json \
        --fcf_config  assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json \
        --ae_config   assets/configs/models/etth1_dataset/ae/tcn_ae.json
"""

import argparse
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.experiments.comparison.forecastcf_pytorch import run_forecastcf_etth1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_config",  default="assets/configs/models/etth1_dataset/RL_ablations/config_v2.json")
    parser.add_argument("--fcf_config", default="assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json")
    parser.add_argument("--ae_config",  default="assets/configs/models/etth1_dataset/ae/tcn_ae.json")
    # ForecastCF hyperparameters (paper defaults)
    parser.add_argument("--desired_change",  type=float, default=-0.1)
    parser.add_argument("--fraction_std",    type=float, default=1.0)
    parser.add_argument("--poly_order",      type=int,   default=1)
    parser.add_argument("--shift",           type=float, default=0.0)
    parser.add_argument("--center",          type=str,   default="median")
    parser.add_argument("--max_iter",        type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--pred_margin_weight", type=float, default=0.25)
    parser.add_argument("--n_eval_batches",  type=int,   default=20)
    parser.add_argument("--results_dir",     type=str,   default="assets/results/forecastcf_baseline")
    parser.add_argument("--figures_dir",     type=str,   default="assets/figures/forecastcf_baseline")
    A = parser.parse_args()

    cfg_rl  = load_config(A.rl_config)
    cfg_f   = load_config(A.fcf_config)
    cfg_ae  = load_config(A.ae_config)
    device  = get_device(cfg_f)

    run_forecastcf_etth1(
        cfg_forecaster      = cfg_f,
        cfg_ae              = cfg_ae,
        cfg_rl              = cfg_rl,
        device              = device,
        desired_change      = A.desired_change,
        fraction_std        = A.fraction_std,
        poly_order          = A.poly_order,
        shift               = A.shift,
        center              = A.center,
        max_iter            = A.max_iter,
        lr                  = A.lr,
        pred_margin_weight  = A.pred_margin_weight,
        n_eval_batches      = A.n_eval_batches,
        results_dir         = A.results_dir,
        figures_dir         = A.figures_dir,
    )


if __name__ == "__main__":
    main()
