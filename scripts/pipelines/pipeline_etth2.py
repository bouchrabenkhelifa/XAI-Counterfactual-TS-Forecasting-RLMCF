"""
Pipeline complet ETTh2 — train + eval + plots
=============================================
Orchestre toutes les étapes dans l'ordre :
  1. Train forecasters (5 modèles)
  2. Train AE
  3. Train RL agents (5 modèles)
  4. Eval tous les modèles + tableau récapitulatif + plots

Usage:
    # Pipeline complet
    python scripts/pipelines/pipeline_etth2.py

    # Eval seulement (tout déjà entraîné)
    python scripts/pipelines/pipeline_etth2.py --eval_only

    # Sauter certaines étapes
    python scripts/pipelines/pipeline_etth2.py --skip_forecasters
    python scripts/pipelines/pipeline_etth2.py --skip_ae
    python scripts/pipelines/pipeline_etth2.py --skip_rl
"""

import argparse
import subprocess
import sys

# ── Configs ───────────────────────────────────────────────────────────────────

AE_CONFIG = "assets/configs/models/etth2_dataset/ae/tcn_ae.json"

FORECASTER_CONFIGS = [
    "assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
    "assets/configs/models/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
]

RL_MODELS = [
    (
        "iTransformer",
        "assets/configs/models/etth2_dataset/RL/config_itransformer.json",
        "assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
    ),
    (
        "PatchTST",
        "assets/configs/models/etth2_dataset/RL/config_patchtst.json",
        "assets/configs/models/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
    ),
    (
        "TimesNet",
        "assets/configs/models/etth2_dataset/RL/config_timesnet.json",
        "assets/configs/models/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
    ),
    (
        "GRU",
        "assets/configs/models/etth2_dataset/RL/config_gru.json",
        "assets/configs/models/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
    ),
    (
        "DLinear",
        "assets/configs/models/etth2_dataset/RL/config_dlinear.json",
        "assets/configs/models/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
    ),
]

EVAL_SCRIPT = "scripts/evals/eval_etth2_all_models.py"


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, label, stop_on_error=True):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"[ERROR] {label} failed (code {result.returncode})")
        if stop_on_error:
            sys.exit(result.returncode)
    return result.returncode


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETTh2 full pipeline")
    parser.add_argument("--eval_only",        action="store_true")
    parser.add_argument("--skip_forecasters", action="store_true")
    parser.add_argument("--skip_ae",          action="store_true")
    parser.add_argument("--skip_rl",          action="store_true")
    args = parser.parse_args()

    # ── 1. Forecasters ────────────────────────────────────────────────────────
    if not args.eval_only and not args.skip_forecasters:
        for cfg in FORECASTER_CONFIGS:
            model = cfg.split("/")[-2]
            run(["src/experiments/forecasting/run.py", "--config", cfg],
                f"Train forecaster — {model} / ETTh2")
    else:
        print("\n[Skip] Forecaster training")

    # ── 2. AE ─────────────────────────────────────────────────────────────────
    if not args.eval_only and not args.skip_ae:
        run(["src/experiments/autoencoder/run.py", "--config", AE_CONFIG],
            "Train AE / ETTh2")
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
            ], f"Train RL agent — {model} / ETTh2")
    else:
        print("\n[Skip] RL training")

    # ── 4. Eval + plots ───────────────────────────────────────────────────────
    run([EVAL_SCRIPT], "Eval all models + plots / ETTh2")

    print("\n" + "="*60)
    print("  ETTh2 pipeline complete ✓")
    print("  Results → assets/results/etth2_summary/")
    print("  Figures → assets/results/etth2_summary/")
    print("="*60)


if __name__ == "__main__":
    main()
