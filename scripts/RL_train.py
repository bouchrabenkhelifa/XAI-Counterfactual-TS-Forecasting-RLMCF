import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))

from src.utils.config      import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainer import RLTrainer


CONFIG_FORECASTER = "configs/models/itransformer/etth1_96_96.json"
CONFIG_AE         = "configs/models/ae/tcn_ae.json"
CONFIG_RL         = "configs/models/RL/RL_etth1.json"


if __name__ == "__main__":

    cfg_f  = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)
    device = get_device(cfg_f)

    print(f"Device : {device}")

    trainer = RLTrainer(
        cfg_forecaster = cfg_f,
        cfg_ae         = cfg_ae,
        cfg_rl         = cfg_rl,
        device         = device,
    )

    history = trainer.train()

    examples = trainer.evaluate(n_batches=20)

    print("\n✅ Done !")
    print(f"  Checkpoint → assets/checkpoints/rl/rl_agent_best.pt")
    print(f"  Figures    → figures/rl/")
    print(f"  Results    → assets/results/rl/rl_history.json")