"""
Retrain 5 RL agents that have poor generalization.
Run this on Colab with GPU.

Targets:
  1. ETTh1/TimesNet  (Valid=0.549, full_test=0.758 — inversed, needs more epochs)
  2. ETTh1/GRU       (Valid=0.940, full_test=0.084 — catastrophic overfitting)
  3. ETTh2/GRU       (Valid=0.695, full_test=0.440 — poor generalization)
  4. Weather/PatchTST (Valid=0.821, full_test=0.545 — large gap)
  5. Weather/TimesNet (Valid=0.949, full_test=0.714 — under-trained, only 2 epochs)

Strategy:
  - Increase epochs to 50 (was 30 or 2)
  - Use eval_train_batches=100 (expose agent to more diverse training data)
  - Standardize mask to 24/8 for all (was 16/6 for GRU)

Usage (Colab):
    !pip install -r requirements.txt
    import os; os.environ['PYTHONPATH'] = '.'
    !python retrain_5agents.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_main import RLMaskTrainer


# ══════════════════════════════════════════════════════════════════════════════
# Configs to retrain — with improved hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

RETRAIN_CONFIGS = [
    {
        "name": "ETTh1/TimesNet",
        "rl_config": "assets/configs/etth1_dataset/RL_ablations/config_timesnet.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
        "ae_config": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        # Override: more epochs, larger eval exposure
        "overrides": {
            "epochs": 50,
            "eval_train_batches": 100,
            "mask_last_k": 24,
            "mask_ramp_k": 8,
        }
    },
    {
        "name": "ETTh1/GRU",
        "rl_config": "assets/configs/etth1_dataset/RL_ablations/config_gru.json",
        "forecast_config": "assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
        "ae_config": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        # Override: standardize mask, more epochs
        "overrides": {
            "epochs": 50,
            "eval_train_batches": 100,
            "mask_last_k": 24,
            "mask_ramp_k": 8,
        }
    },
    {
        "name": "ETTh2/GRU",
        "rl_config": "assets/configs/etth2_dataset/RL/config_gru.json",
        "forecast_config": "assets/configs/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
        "ae_config": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "overrides": {
            "epochs": 50,
            "eval_train_batches": 100,
        }
    },
    {
        "name": "Weather/PatchTST",
        "rl_config": "assets/configs/weather_dataset/RL/config_patchtst.json",
        "forecast_config": "assets/configs/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
        "ae_config": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "overrides": {
            "epochs": 50,
            "eval_train_batches": 100,
            "mask_last_k": 24,
            "mask_ramp_k": 8,
        }
    },
    {
        "name": "Weather/TimesNet",
        "rl_config": "assets/configs/weather_dataset/RL/config_timesnet.json",
        "forecast_config": "assets/configs/weather_dataset/forecasters/timesnet/weather_96_96_S.json",
        "ae_config": "assets/configs/weather_dataset/ae/tcn_ae.json",
        # Was only 2 epochs! Increase significantly
        "overrides": {
            "epochs": 50,
            "eval_train_batches": 100,
            "lr_actor": 3e-4,
            "lr_critic": 1e-4,
        }
    },
]


def retrain_one(config_entry):
    """Retrain a single RL agent with overrides."""
    name = config_entry["name"]
    print(f"\n\n{'#'*70}")
    print(f"#  RETRAINING: {name}")
    print(f"{'#'*70}")

    cfg_rl = load_config(config_entry["rl_config"])
    cfg_f = load_config(config_entry["forecast_config"])
    cfg_ae = load_config(config_entry["ae_config"])

    # Apply overrides
    for key, value in config_entry.get("overrides", {}).items():
        setattr(cfg_rl, key, value)
        print(f"  [Override] {key} = {value}")

    device = get_device(cfg_f)
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg_rl.epochs}")
    print(f"  mask_last_k: {cfg_rl.mask_last_k}, mask_ramp_k: {cfg_rl.mask_ramp_k}")
    print(f"  eval_train_batches: {cfg_rl.eval_train_batches}")

    # Train
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    trainer.train()

    # Evaluate on full test set (all batches)
    print(f"\n  Evaluating on FULL test set...")
    trainer.evaluate(n_batches=9999)

    print(f"\n  [DONE] {name}")
    print(f"  Checkpoint: {cfg_rl.checkpoint_dir_lp}/{cfg_rl.name}_agent_best.pt")


def main():
    print("=" * 70)
    print("  RETRAIN 5 RL AGENTS (improved generalization)")
    print("  Run on Colab with GPU for speed")
    print("=" * 70)
    print(f"\n  Agents to retrain:")
    for i, c in enumerate(RETRAIN_CONFIGS, 1):
        print(f"    {i}. {c['name']}")

    for config_entry in RETRAIN_CONFIGS:
        retrain_one(config_entry)

    print(f"\n\n{'='*70}")
    print("  ALL RETRAINING COMPLETE")
    print("  New checkpoints saved. Re-run evaluation to update tables.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
