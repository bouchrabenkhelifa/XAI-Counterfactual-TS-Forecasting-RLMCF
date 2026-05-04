"""
Pipeline complet Weather — train forecasters + eval + plots
(AE and RL training skipped by default)

Usage:
    python scripts/pipelines/pipeline_weather.py                    # Train forecasters only
    python scripts/pipelines/pipeline_weather.py --eval_only        # Eval only
    python scripts/pipelines/pipeline_weather.py --skip_forecasters # Skip forecaster training
    python scripts/pipelines/pipeline_weather.py --train_ae         # Include AE training
    python scripts/pipelines/pipeline_weather.py --train_rl         # Include RL training
"""

import argparse, subprocess, sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

AE_CONFIG = "assets/configs/models/weather_dataset/ae/tcn_ae.json"

# Configs for retraining (all 96_96)
FORECASTER_CONFIGS = [
    "assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_48_S.json",
    "assets/configs/models/weather_dataset/forecasters/gru/weather_96_96_S.json",
    "assets/configs/models/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
    "assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_96_S.json",
    "assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
]

RL_MODELS = [
    ("iTransformer",
     "assets/configs/models/weather_dataset/RL/config_itransformer.json",
     "assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_48_S.json"),
    ("PatchTST",
     "assets/configs/models/weather_dataset/RL/config_patchtst.json",
     "assets/configs/models/weather_dataset/forecasters/patchtst/weather_96_96_S.json"),
    ("TimesNet",
     "assets/configs/models/weather_dataset/RL/config_timesnet.json",
     "assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_96_S.json"),
    ("GRU",
     "assets/configs/models/weather_dataset/RL/config_gru.json",
     "assets/configs/models/weather_dataset/forecasters/gru/weather_96_96_S.json"),
    ("DLinear",
     "assets/configs/models/weather_dataset/RL/config_dlinear.json",
     "assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json"),
]

EVAL_SCRIPT = "scripts/evals/eval_weather_all_models.py"


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


def main():
    parser = argparse.ArgumentParser(description="Weather full pipeline")
    parser.add_argument("--eval_only",        action="store_true")
    parser.add_argument("--skip_forecasters", action="store_true")
    parser.add_argument("--skip_ae",          action="store_true")
    parser.add_argument("--skip_rl",          action="store_true")
    args = parser.parse_args()

    if not args.eval_only and not args.skip_forecasters:
        for cfg in FORECASTER_CONFIGS:
            model = cfg.split("/")[-2]
            run(["src/experiments/forecasting/run.py", "--config", cfg],
                f"Train forecaster — {model} / Weather")
    else:
        print("\n[Skip] Forecaster training")

    if not args.eval_only and not args.skip_ae:
        run(["src/experiments/autoencoder/run.py", "--config", AE_CONFIG],
            "Train AE / Weather")
    else:
        print("\n[Skip] AE training")

    if not args.eval_only and not args.skip_rl:
        for model, rl_cfg, f_cfg in RL_MODELS:
            run(["src/experiments/rl_cf/run.py",
                 "--config", rl_cfg, "--forecast_config", f_cfg, "--ae_config", AE_CONFIG],
                f"Train RL agent — {model} / Weather")
    else:
        print("\n[Skip] RL training")

    run([EVAL_SCRIPT], "Eval all models + plots / Weather")

    print("\n" + "="*60)
    print("  Weather pipeline complete ✓")
    print("  Results → assets/results/weather_summary/")
    print("="*60)


if __name__ == "__main__":
    main()
