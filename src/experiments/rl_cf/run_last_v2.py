import argparse
import glob
import os

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_last_v2 import RLMaskTrainer


def run_one(
    rl_config_path,
    forecast_config_path,
    ae_config_path,
    eval_batches=20,
    eval_only=False,
):
    cfg_rl = load_config(rl_config_path)
    
    # Use forecast_config_path from RL config if available, otherwise use command-line arg
    if hasattr(cfg_rl, 'forecast_config_path') and cfg_rl.forecast_config_path:
        forecast_config_path = cfg_rl.forecast_config_path
    
    print("\n" + "=" * 70)
    print(f"RL config       : {rl_config_path}")
    print(f"Forecast config : {forecast_config_path}")
    print(f"AE config       : {ae_config_path}")
    print("=" * 70)

    cfg_f = load_config(forecast_config_path)
    cfg_ae = load_config(ae_config_path)
    device = get_device(cfg_f)

    print(f"Device          : {device}")
    print(f"Experiment      : {getattr(cfg_rl, 'name', 'unnamed')}")
    print("=" * 70)

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

    if not eval_only:
        trainer.train()
    else:
        # Charger le meilleur checkpoint
        import torch
        ckpt_path = os.path.join(
            cfg_rl.checkpoint_dir_lp,
            f"{getattr(cfg_rl, 'name', 'exp')}_agent_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
            trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
            print(f"[Eval] Loaded best checkpoint → {ckpt_path}")
        else:
            print(f"[Eval] WARNING: checkpoint not found at {ckpt_path}, using random weights")

    trainer.evaluate(n_batches=eval_batches)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL counterfactual training — α/β objective"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to a single RL config JSON")
    group.add_argument(
        "--config_dir", type=str, help="Directory of RL config JSON files"
    )

    parser.add_argument(
        "--forecast_config",
        type=str,
        default="assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    )
    parser.add_argument(
        "--ae_config",
        type=str,
        default="assets/configs/models/etth1_dataset/ae/tcn_ae.json",
    )
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    if args.config is not None:
        run_one(
            rl_config_path=args.config,
            forecast_config_path=args.forecast_config,
            ae_config_path=args.ae_config,
            eval_batches=args.eval_batches,
            eval_only=args.eval_only,
        )
        return

    config_paths = sorted(glob.glob(os.path.join(args.config_dir, "*.json")))
    if not config_paths:
        raise FileNotFoundError(f"No JSON configs found in: {args.config_dir}")

    print(f"\nFound {len(config_paths)} configs in {args.config_dir}")
    for cfg_path in config_paths:
        run_one(
            rl_config_path=cfg_path,
            forecast_config_path=args.forecast_config,
            ae_config_path=args.ae_config,
            eval_batches=args.eval_batches,
            eval_only=args.eval_only,
        )

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
