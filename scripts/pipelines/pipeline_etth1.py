"""
Pipeline complet ETTh1 — train + eval + plots
=============================================
Orchestre toutes les étapes dans l'ordre :
  1. Train forecasters (5 modèles)
  2. Train AE
  3. Train RL agents (5 modèles)
  4. Eval tous les modèles + tableau récapitulatif + plots

Usage:
    # Pipeline complet
    python scripts/run_etth1.py

    # Eval seulement (forecasters + AE + RL déjà entraînés)
    python scripts/run_etth1.py --eval_only

    # Sauter le train des forecasters (déjà entraînés)
    python scripts/run_etth1.py --skip_forecasters

    # Sauter le train de l'AE (déjà entraîné)
    python scripts/run_etth1.py --skip_ae
"""

import argparse
import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Configs ───────────────────────────────────────────────────────────────────

AE_CONFIG = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"

FORECASTER_CONFIGS = [
    "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    "assets/configs/models/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
    "assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
    "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
    "assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
]

RL_MODELS = [
    (
        "iTransformer",
        "assets/configs/models/etth1_dataset/RL_ablations/config_v2.json",
        "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    ),
    (
        "PatchTST",
        "assets/configs/models/etth1_dataset/RL_ablations/config_patchtst.json",
        "assets/configs/models/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
    ),
    (
        "TimesNet",
        "assets/configs/models/etth1_dataset/RL_ablations/config_timesnet.json",
        "assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
    ),
    (
        "GRU",
        "assets/configs/models/etth1_dataset/RL_ablations/config_gru.json",
        "assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
    ),
    (
        "DLinear",
        "assets/configs/models/etth1_dataset/RL_ablations/config_dlinear.json",
        "assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
    ),
]

EVAL_SCRIPT    = "scripts/evals/eval_etth1_all_models.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, label, stop_on_error=True):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"[ERROR] {label} failed (code {result.returncode})")
        if stop_on_error:
            sys.exit(result.returncode)
    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETTh1 full pipeline")
    parser.add_argument("--eval_only",         action="store_true", help="Skip all training")
    parser.add_argument("--skip_forecasters",  action="store_true", help="Skip forecaster training")
    parser.add_argument("--skip_ae",           action="store_true", help="Skip AE training")
    parser.add_argument("--skip_rl",           action="store_true", help="Skip RL training")
    parser.add_argument("--eval_batches",      type=int, default=20)
    args = parser.parse_args()

    # ── 1. Forecasters ────────────────────────────────────────────────────────
    if not args.eval_only and not args.skip_forecasters:
        for cfg in FORECASTER_CONFIGS:
            model = cfg.split("/")[-2]
            run(["src/experiments/forecasting/run.py", "--config", cfg],
                f"Train forecaster — {model} / ETTh1")
    else:
        print("\n[Skip] Forecaster training")

    # ── 2. AE ─────────────────────────────────────────────────────────────────
    if not args.eval_only and not args.skip_ae:
        run(["src/experiments/autoencoder/run.py", "--config", AE_CONFIG],
            "Train AE / ETTh1")
    else:
        print("\n[Skip] AE training")

    # ── 3. RL agents ──────────────────────────────────────────────────────────
    if not args.eval_only and not args.skip_rl:
        for model, rl_cfg, f_cfg in RL_MODELS:
            run([
                "src/experiments/rl_cf/run_last.py",
                "--config",          rl_cfg,
                "--forecast_config", f_cfg,
                "--ae_config",       AE_CONFIG,
            ], f"Train RL agent — {model} / ETTh1")
    else:
        print("\n[Skip] RL training")

    # ── 4. Eval + plots ───────────────────────────────────────────────────────
    run([EVAL_SCRIPT], "Eval all models + plots / ETTh1")

    print("\n" + "="*60)
    print("  ETTh1 pipeline complete ✓")
    print("  Results → assets/results/etth1_summary/")
    print("  Figures → assets/results/etth1_summary/")
    print("="*60)


if __name__ == "__main__":
    main()
