import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config      import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainer_without_ae import RLTrainerInputSpace

CONFIG_FORECASTER = "assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json"
CONFIG_AE         = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CONFIG_RL         = "assets/configs/models/etth1_dataset/RL/RL_without_ae.json"

if __name__ == "__main__":
    cfg_f  = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)
    device = get_device(cfg_f)

    print(f"Device    : {device}")
    print(f"Pipeline  : Input Space (sans AE)")
    print(f"Objectif  : réduire mean(forecast) de {cfg_rl.rho*100:.0f}%")
    print(f"eta_input : {cfg_rl.eta_input}")

    trainer  = RLTrainerInputSpace(cfg_f, cfg_ae, cfg_rl, device)
    history  = trainer.train()
    examples = trainer.evaluate(n_batches=20)

    print(f"\n✅ Done !")
    print(f"  Checkpoint → {cfg_rl.checkpoint_dir_input}/rl_input_agent_best.pt")
    print(f"  Figures    → {cfg_rl.figures_dir_input}/")
    print(f"  Results    → {cfg_rl.results_dir_input}/rl_input_history.json")