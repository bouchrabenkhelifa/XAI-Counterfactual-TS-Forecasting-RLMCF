import os
import json
import argparse
from types import SimpleNamespace

from src.training.ad_trainers.ad_trainer import main as run_plausibility


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.figures_dir, exist_ok=True)

    print(f"[Run] Config loaded: {args.config}")

    run_plausibility(args.config)


if __name__ == "__main__":
    main()
