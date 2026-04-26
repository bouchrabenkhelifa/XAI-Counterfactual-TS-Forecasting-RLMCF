import os
import argparse

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to forecasting config file",
    )
    args = parser.parse_args()

    configs = load_config(args.config)

    os.makedirs(configs.checkpoint_dir, exist_ok=True)
    os.makedirs(configs.results_dir, exist_ok=True)
    if hasattr(configs, "figures_dir"):
        os.makedirs(configs.figures_dir, exist_ok=True)

    # Route to the right trainer based on model_type
    model_type = getattr(configs, "model_type", "iTransformer").lower()

    if model_type == "itransformer":
        from src.training.forecasting.itransformer_trainer import ITransformerTrainer
        trainer = ITransformerTrainer(configs)
    elif model_type in ("gru", "dlinear", "timesnet"):
        from src.training.forecaster_trainers.generic_forecaster_trainer import GenericForecasterTrainer
        trainer = GenericForecasterTrainer(configs)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         f"Choose from: iTransformer, GRU, DLinear, TimesNet")

    model, history = trainer.train()

    print("\nTraining finished.")
    print("Config    :", args.config)
    print("Checkpoint:", os.path.join(configs.checkpoint_dir, configs.checkpoint_name))
    print("History   :", os.path.join(configs.results_dir, configs.history_name))


if __name__ == "__main__":
    main()