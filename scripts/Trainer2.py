import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.itransformer_trainer import ITransformerTrainer
from src.utils.config import load_config


if __name__ == "__main__":
    config_path = "configs/models/itransformer/etth1_192_96.json"

    configs = load_config(config_path)
    trainer = ITransformerTrainer(configs)
    model, history = trainer.train()

    print("Training finished.")
    print("Best checkpoint saved to:", os.path.join(configs.checkpoint_dir, configs.checkpoint_name))
    print("History saved to:", os.path.join(configs.results_dir, configs.history_name))