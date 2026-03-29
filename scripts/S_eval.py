import os
import json
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.train_tools import get_device

from src.training.RL_trainer_S import RLTrainer
from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


CONFIG_FORECASTER = "assets/configs/models/itransformer/etth1_96_48_S.json"
CONFIG_AE = "assets/configs/models/ae/tcn_ae.json"
CONFIG_RL = "assets/configs/models/RL/etth1_rl_S.json"

CHECKPOINT_PATH = "assets/checkpoints/rl_S/rl_agent_best.pt"
EXTERNAL_PLAUS_PATH = "assets/checkpoints/anomaly detector/plausibility_etth1.pkl"


# =========================
# LOAD CHECKPOINT
# =========================
def load_checkpoint_into_trainer(trainer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])

    print(f"[Eval] Loaded checkpoint: {ckpt_path}")


# =========================
# EPISODE
# =========================
@torch.no_grad()
def trainer_run_episode(trainer, batch, filter_quantile=0.75):

    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(trainer.device)
    batch_x_mark = batch_x_mark.float().to(trainer.device)

    x_ot = batch_x[:, :, -1:]

    z = trainer.ae.encode(x_ot)
    y_hat = trainer.forecaster.predict_ot(batch_x, batch_x_mark)

    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    mask = mean_yhat >= torch.quantile(mean_yhat, filter_quantile)

    if mask.sum() == 0:
        return None

    x_ot = x_ot[mask]
    batch_x = batch_x[mask]
    batch_x_mark = batch_x_mark[mask]
    y_hat = y_hat[mask]
    z = z[mask]

    s = trainer.agent.build_state(z, y_hat)

    # deterministic policy
    mu, _ = trainer.agent.actor(s)
    a = mu

    z_cf = torch.clamp(z + trainer.agent.eta * a, -1.0, 1.0)
    x_cf = trainer.ae.decode(z_cf)

    y_cf = trainer.forecaster.predict_from_ot(
        x_ot=x_cf,
        x_full=batch_x,
        x_mark=batch_x_mark
    )

    reward_dict = trainer.reward_fn(x_ot, x_cf, y_hat, y_cf)

    return {
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
    }


# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate_frozen_model(trainer, evaluator, n_batches=20):

    trainer.agent.eval()

    all_x, all_x_cf = [], []
    all_y_hat, all_y_cf = [], []
    examples = []

    for i, batch in enumerate(trainer.test_loader):

        if i >= n_batches:
            break

        ep = trainer_run_episode(trainer, batch)

        if ep is None:
            continue

        all_x.append(ep["x_ot"].cpu().numpy())
        all_x_cf.append(ep["x_cf"].cpu().numpy())
        all_y_hat.append(ep["y_hat"].cpu().numpy())
        all_y_cf.append(ep["y_cf"].cpu().numpy())

        if len(examples) < 4:
            examples.append({
                "x_ot": ep["x_ot"][0].cpu().numpy(),
                "x_cf": ep["x_cf"][0].cpu().numpy(),
                "y_hat": ep["y_hat"][0].cpu().numpy(),
                "y_cf": ep["y_cf"][0].cpu().numpy(),
            })

    all_x = np.concatenate(all_x)
    all_x_cf = np.concatenate(all_x_cf)
    all_y_hat = np.concatenate(all_y_hat)
    all_y_cf = np.concatenate(all_y_cf)

    metrics = evaluator.evaluate_batch(all_x, all_x_cf, all_y_hat, all_y_cf)
    summary = evaluator.summarize_with_std(metrics)

    return summary, examples


# =========================
# MAIN
# =========================
def main():

    cfg_f = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)

    device = get_device(cfg_f)
    print(f"Device: {device}")

    trainer = RLTrainer(cfg_f, cfg_ae, cfg_rl, device)
    load_checkpoint_into_trainer(trainer, CHECKPOINT_PATH, device)

    plaus_model = load_plausibility_model(EXTERNAL_PLAUS_PATH)

    evaluator = CounterfactualEvaluator(
        plausibility_model=plaus_model,
        rho=cfg_rl.rho,
    )

    summary, examples = evaluate_frozen_model(trainer, evaluator)

    # flatten for pretty print
    summary_flat = {k: v["mean"] for k, v in summary.items()}

    print("\n── Final Evaluation Metrics ─────────────────────────")
    evaluator.pretty_print_summary(summary_flat)

    print("\n── Mean ± Std ───────────────────────────────────────")

    for k, v in summary.items():
        print(f"{k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Save
    os.makedirs(cfg_rl.results_dir, exist_ok=True)

    with open(os.path.join(cfg_rl.results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Eval] Saved results ✔")


if __name__ == "__main__":
    main()