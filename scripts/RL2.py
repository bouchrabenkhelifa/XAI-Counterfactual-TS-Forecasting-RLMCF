import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config      import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainer_S import RLTrainer

CONFIG_FORECASTER = "assets/configs/models/itransformer/etth1_96_48_S.json"
CONFIG_AE         = "assets/configs/models/ae/tcn_ae.json"
CONFIG_RL         = "assets/configs/models/RL/RL2.json"

if __name__ == "__main__":
    cfg_f  = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)
    device = get_device(cfg_f)

    print(f"Device : {device}")

    trainer  = RLTrainer(cfg_f, cfg_ae, cfg_rl, device)
    history  = trainer.train()
    examples = trainer.evaluate(n_batches=20)

    print(f"\n✅ Done !")
    print(f"  Checkpoint → {cfg_rl.checkpoint_dir}/rl_agent_best.pt")
    print(f"  Figures    → {cfg_rl.figures_dir}/")
    print(f"  Results    → {cfg_rl.results_dir}/rl_history_S.json")