import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.train_tools import get_device

from src.training.RL_learned_mask_trainer import RLLearnedMaskTrainer
from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


CONFIG_FORECASTER = "assets/configs/models/itransformer/etth1_96_48_S.json"
CONFIG_AE = "assets/configs/models/ae/tcn_ae.json"
CONFIG_RL = "assets/configs/models/RL/rl_learned_mask.json"

CHECKPOINT_PATH = "assets/checkpoints/RL_mask_learned/rl_mask_learned_agent_best.pt"
EXTERNAL_PLAUS_PATH = "assets/checkpoints/anomaly detector/plausibility_etth1.pkl"


def load_checkpoint_into_trainer(trainer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "actor_state_dict" not in ckpt or "critic_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain actor_state_dict and critic_state_dict")

    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])

    print(f"[Eval] Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def _ensure_mask_shape(mask_t, x_ot, mask_min=0.05, mask_max=0.95):
    if mask_t.dim() == 2:
        mask_t = mask_t.unsqueeze(2)
    elif mask_t.dim() != 3:
        raise ValueError(f"mask_t must have shape [B,T] or [B,T,1], got {mask_t.shape}")

    if mask_t.shape[1] != x_ot.shape[1]:
        raise ValueError(
            f"Mask time dimension mismatch: mask_t.shape={mask_t.shape}, x_ot.shape={x_ot.shape}"
        )

    mask_t = torch.clamp(mask_t, min=mask_min, max=mask_max)
    return mask_t


@torch.no_grad()
def trainer_run_episode(trainer, batch, filter_quantile=0.75):
    """
    Reproduit exactement l'épisode du trainer learned-mask en mode déterministe.
    """
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(trainer.device)
    batch_x_mark = batch_x_mark.float().to(trainer.device)
    x_ot = batch_x[:, :, -1:]   # (B, seq_len, 1)

    z = trainer.ae.encode(x_ot)
    y_hat = trainer.forecaster.predict_ot(batch_x, batch_x_mark)

    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    keep = mean_yhat >= torch.quantile(mean_yhat, filter_quantile)
    if keep.sum() == 0:
        return None

    x_ot = x_ot[keep]
    batch_x = batch_x[keep]
    batch_x_mark = batch_x_mark[keep]
    y_hat = y_hat[keep]
    z = z[keep]

    s = trainer.agent.build_state(z, y_hat)

    # action déterministe
    mu, _log_std, mask_t = trainer.agent.actor(s)
    a = mu

    z_cf = torch.clamp(z + trainer.agent.eta * a, -1.0, 1.0)

    # proposition décodée
    x_prop = trainer.ae.decode(z_cf)
    delta = x_prop - x_ot

    # HF skip identique au training
    x_lf = trainer.ae.decode(trainer.ae.encode(x_ot))
    x_hf = x_ot - x_lf

    mask_t = _ensure_mask_shape(
        mask_t=mask_t,
        x_ot=x_ot,
        mask_min=trainer.mask_min,
        mask_max=trainer.mask_max,
    )

    masked_delta = delta + trainer.alpha_hf * x_hf
    x_cf = x_ot + mask_t * masked_delta

    y_cf = trainer.forecaster.predict_from_ot(
        x_ot=x_cf,
        x_full=batch_x,
        x_mark=batch_x_mark
    )

    reward_dict = trainer.reward_fn(
        x_ot, x_cf, y_hat, y_cf,
        mask_t=mask_t,
        z_cf=z_cf
    )

    return {
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "z": z.detach(),
        "z_cf": z_cf.detach(),
        "mask_t": mask_t.detach(),
        "reward_dict": reward_dict,
        "n_valid": int(keep.sum()),
    }


@torch.no_grad()
def evaluate_frozen_model(trainer, evaluator, n_batches=20, include_dtw=False):
    trainer.agent.eval()

    all_x = []
    all_x_cf = []
    all_y_hat = []
    all_y_cf = []

    cf_examples = []

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

        if len(cf_examples) < 4:
            cf_examples.append({
                "x_ot": ep["x_ot"][0].cpu().numpy(),
                "x_cf": ep["x_cf"][0].cpu().numpy(),
                "y_hat": ep["y_hat"][0].cpu().numpy(),
                "y_cf": ep["y_cf"][0].cpu().numpy(),
                "mask_t": ep["mask_t"][0].cpu().numpy(),
            })

    if len(all_x) == 0:
        raise RuntimeError("No valid CFs generated during evaluation.")

    all_x = np.concatenate(all_x, axis=0)
    all_x_cf = np.concatenate(all_x_cf, axis=0)
    all_y_hat = np.concatenate(all_y_hat, axis=0)
    all_y_cf = np.concatenate(all_y_cf, axis=0)

    metrics = evaluator.evaluate_batch(
        x=all_x,
        x_cf=all_x_cf,
        y_hat=all_y_hat,
        y_cf=all_y_cf,
        include_dtw=include_dtw,
    )

    summary = evaluator.summarize_with_std(metrics)
    return summary, cf_examples


def save_examples_plot(examples, out_path, rho=0.10):
    if not examples:
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, ex in enumerate(examples):
        x_ot = ex["x_ot"][:, 0]
        x_cf = ex["x_cf"][:, 0]
        y_hat = ex["y_hat"][:, 0]
        y_cf = ex["y_cf"][:, 0]

        mask_t = ex["mask_t"]
        if mask_t.ndim == 2:
            mask_t = mask_t[:, 0]

        full_orig = np.concatenate([x_ot, y_hat])
        full_cf = np.concatenate([x_cf, y_cf])
        t_all = np.arange(len(full_orig))
        reduction = (y_hat.mean() - y_cf.mean()) / (abs(y_hat.mean()) + 1e-8) * 100
        ok = "✓" if reduction >= rho * 100 else "✗"

        axes[i, 0].plot(t_all, full_orig, color="steelblue", lw=1.5, label="x + forecast(x)")
        axes[i, 0].plot(t_all, full_cf, color="coral", lw=1.5, ls="--", label="x_cf + forecast(x_cf)")
        axes[i, 0].axvline(len(x_ot), color="gray", ls="--", lw=1.2)
        axes[i, 0].fill_between(t_all, full_orig, full_cf, alpha=0.12, color="coral")
        axes[i, 0].set_title(f"Sample {i+1} — {reduction:+.1f}% {ok}", fontsize=11)
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(alpha=0.3)

        axes[i, 1].plot(mask_t, color="purple", lw=1.8)
        axes[i, 1].set_ylim(-0.05, 1.05)
        axes[i, 1].set_title("Learned temporal mask", fontsize=11)
        axes[i, 1].grid(alpha=0.3)

    plt.suptitle("Frozen checkpoint evaluation — RL Learned Mask", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Examples plot -> {out_path}")


def main():
    cfg_f = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)

    device = get_device(cfg_f)
    print(f"Device: {device}")

    trainer = RLLearnedMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    load_checkpoint_into_trainer(trainer, CHECKPOINT_PATH, device)

    plaus_model = load_plausibility_model(EXTERNAL_PLAUS_PATH)
    evaluator = CounterfactualEvaluator(
        plausibility_model=plaus_model,
        rho=cfg_rl.rho,
    )

    summary, examples = evaluate_frozen_model(
        trainer=trainer,
        evaluator=evaluator,
        n_batches=20,
        include_dtw=False,
    )

    summary_flat = {k: v["mean"] for k, v in summary.items()}

    print("\n── Final Evaluation Metrics ─────────────────────────")
    evaluator.pretty_print_summary(summary_flat)

    print("\n── Mean ± Std ───────────────────────────────────────")
    keys_to_show = [
        "success",
        "delta_mean",
        "relative_reduction",
        "target_gap",
        "l1",
        "l2",
        "euclidean",
        "manhattan",
        "plausibility_x",
        "plausibility_cf",
        "plausibility_x_ensemble",
        "plausibility_cf_ensemble",
        "plausibility_x_if",
        "plausibility_cf_if",
        "plausibility_x_lof",
        "plausibility_cf_lof",
        "plausibility_x_ocsvm",
        "plausibility_cf_ocsvm",
        "roughness_x",
        "roughness_cf",
        "roughness_ratio",
        "derivative_distance",
        "second_derivative_distance",
        "temporal_consistency",
        "autocorrelation_similarity",
        "spectral_similarity",
        "sparsity_ratio",
        "change_magnitude",
        "segment_sparsity",
    ]

    for k in keys_to_show:
        if k in summary:
            print(f"{k:28s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    os.makedirs(cfg_rl.results_dir_lp, exist_ok=True)
    os.makedirs(cfg_rl.figures_dir_lp, exist_ok=True)

    metrics_path = os.path.join(cfg_rl.results_dir_lp, "frozen_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Eval] Metrics saved -> {metrics_path}")

    fig_path = os.path.join(cfg_rl.figures_dir_lp, "frozen_eval_examples.png")
    save_examples_plot(examples, fig_path, rho=cfg_rl.rho)


if __name__ == "__main__":
    main()