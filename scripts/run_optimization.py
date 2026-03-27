from src.training.optim_trainer import OptimizationTrainer
from src.utils.config import load_config
from src.utils.train_tools import get_device


def main():
    cfg_forecaster = load_config("assets/configs/etth1_96_48_S.json", "forecaster")
    cfg_ae = load_config("assets/configs/latent01.json", "autoencoder")
    cfg_optim = load_config(
        "assets/configs/optimization_strategy/optim1.json",
        "optimization",
    )

    device = get_device()

    trainer = OptimizationTrainer(
        cfg_forecaster=cfg_forecaster,
        cfg_ae=cfg_ae,
        cfg_optim=cfg_optim,
        device=device,
    )
    trainer.evaluate(n_batches=20)


if __name__ == "__main__":
    main()