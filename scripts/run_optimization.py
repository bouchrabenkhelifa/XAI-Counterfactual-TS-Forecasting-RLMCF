from types import SimpleNamespace
import torch
from src.training.optim_trainer import OptimizationTrainer
from src.utils.config import load_config

CONFIG_FORECASTER = "assets/configs/models/itransformer/etth1_96_48_S.json"
CONFIG_AE = "assets/configs/models/ae/tcn_ae.json"
CONFIG_OPTIM = "assets/configs/models/optimization_strategy/optim1.json"


def main():
    cfg_forecaster = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_optim = SimpleNamespace(**load_config(CONFIG_OPTIM).optimization)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = OptimizationTrainer(
        cfg_forecaster=cfg_forecaster,
        cfg_ae=cfg_ae,
        cfg_optim=cfg_optim,
        device=device,
    )
    trainer.evaluate(n_batches=20)


if __name__ == "__main__":
    main()