import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.train_tools import get_device

from src.training.RL_trainers.RL_trainer import RLMaskTrainer
from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


CONFIG_FORECASTER = "assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json"
CONFIG_AE = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CONFIG_RL = "assets/configs/models/etth1_dataset/RL_ablations/wo_plausibility.json"

CHECKPOINT_PATH = "assets/checkpoints/etth1_chpts/RL_ablations/wo_plausibility/wo_plausibility_agent_best.pt"
EXTERNAL_PLAUS_PATH = "assets/checkpoints/etth1_chpts/anomaly detector/plausibility_etth1.pkl"


def load_checkpoint_into_trainer(trainer, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "actor_state_dict" not in ckpt or "critic_state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain actor_state_dict and critic_state_dict")

    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])

    print(f"[Eval] Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def build_temporal_mask(batch_size, seq_len, channels, last_k, ramp_k, device):
    m = torch.zeros((batch_size, seq_len, channels), device=device)

    start_full = max(0, seq_len - last_k)
    m[:, start_full:, :] = 1.0

    if ramp_k > 0:
        start_ramp = max(0, start_full - ramp_k)
        ramp_len = start_full - start_ramp
        if ramp_len > 0:
            ramp = torch.linspace(0.0, 1.0, ramp_len, device=device).view(1, ramp_len, 1)
            m[:, start_ramp:start_full, :] = ramp

    return m


def forecast_monotonicity_with_context(
    x_ot,
    x_cf,
    x_full,
    x_mark,
    forecaster,
    threshold=5e-2,
):
    device = next(forecaster.model.parameters()).device

    x_ot_np = x_ot.detach().cpu().numpy()
    x_cf_np = x_cf.detach().cpu().numpy()

    x_full = x_full.detach().to(device)
    x_mark = x_mark.detach().to(device)

    B = x_ot_np.shape[0]
    scores = []

    for i in range(B):
        xi = x_ot_np[i:i + 1]
        xi_cf = x_cf_np[i:i + 1]

        xfull_i = x_full[i:i + 1]
        xmark_i = x_mark[i:i + 1]

        diff = np.abs(xi_cf - xi)
        if diff.ndim == 3:
            mask = np.where(np.max(diff[0], axis=-1) > threshold)[0]
        else:
            mask = np.where(diff[0] > threshold)[0]

        if len(mask) == 0:
            scores.append(1.0)
            continue

        xi_t = torch.from_numpy(xi).float().to(device)
        xi_cf_t = torch.from_numpy(xi_cf).float().to(device)

        with torch.no_grad():
            baseline_forecast = forecaster.predict_from_ot(
                x_ot=xi_t,
                x_full=xfull_i,
                x_mark=xmark_i,
            )
        baseline_mean = baseline_forecast.mean().item()

        contributions = []
        for t in mask:
            x_ablated = xi_cf_t.clone()
            x_ablated[0, t, :] = xi_t[0, t, :]

            with torch.no_grad():
                ablated_forecast = forecaster.predict_from_ot(
                    x_ot=x_ablated,
                    x_full=xfull_i,
                    x_mark=xmark_i,
                )

            contribution = baseline_mean - ablated_forecast.mean().item()
            contributions.append(contribution)

        scores.append(float(np.mean(np.array(contributions) > 0)))

    return np.asarray(scores, dtype=np.float32)


@torch.no_grad()
def trainer_run_episode(trainer, batch, filter_quantile=None):
    """
    Reproduit exactement l'épisode du RL-Mask trainer en mode déterministe.
    """
    if filter_quantile is None:
        filter_quantile = getattr(trainer, "filter_quantile", 0.75)

    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(trainer.device)
    batch_x_mark = batch_x_mark.float().to(trainer.device)
    x_ot = batch_x[:, :, -1:]  # (B, seq_len, 1)

    # IMPORTANT: use ae_arch, not ae
    z = trainer.ae_arch.encode(x_ot)
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

    # deterministic action
    mu, _ = trainer.agent.actor(s)
    a = mu
    z_cf = torch.clamp(z + trainer.agent.eta * a, -1.0, 1.0)

    # decoded proposal
    x_prop = trainer.ae_arch.decode(z_cf)
    delta = x_prop - x_ot

    # same HF skip as training
    x_lf = trainer.ae_arch.decode(trainer.ae_arch.encode(x_ot))
    x_hf = x_ot - x_lf

    # same temporal mask as training
    temp_mask = build_temporal_mask(
        batch_size=x_ot.shape[0],
        seq_len=x_ot.shape[1],
        channels=x_ot.shape[2],
        last_k=trainer.mask_last_k,
        ramp_k=trainer.mask_ramp_k,
        device=trainer.device,
    )

    masked_delta = delta + trainer.alpha_hf * x_hf
    x_cf = x_ot + temp_mask * masked_delta

    y_cf = trainer.forecaster.predict_from_ot(
        x_ot=x_cf,
        x_full=batch_x,
        x_mark=batch_x_mark,
    )

    reward_dict = trainer.reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)

    return {
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "batch_x": batch_x.detach(),
        "batch_x_mark": batch_x_mark.detach(),
        "z": z.detach(),
        "z_cf": z_cf.detach(),
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
    all_monot = []

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

        mono = forecast_monotonicity_with_context(
            x_ot=ep["x_ot"],
            x_cf=ep["x_cf"],
            x_full=ep["batch_x"],
            x_mark=ep["batch_x_mark"],
            forecaster=trainer.forecaster,
        )
        all_monot.append(mono)

        if len(cf_examples) < 4:
            cf_examples.append(
                {
                    "x_ot": ep["x_ot"][0].cpu().numpy(),
                    "x_cf": ep["x_cf"][0].cpu().numpy(),
                    "y_hat": ep["y_hat"][0].cpu().numpy(),
                    "y_cf": ep["y_cf"][0].cpu().numpy(),
                }
            )

    if len(all_x) == 0:
        raise RuntimeError("No valid CFs generated during evaluation.")

    all_x = np.concatenate(all_x, axis=0)
    all_x_cf = np.concatenate(all_x_cf, axis=0)
    all_y_hat = np.concatenate(all_y_hat, axis=0)
    all_y_cf = np.concatenate(all_y_cf, axis=0)
    all_monot = np.concatenate(all_monot, axis=0)

    metrics = evaluator.evaluate_batch(
        x=all_x,
        x_cf=all_x_cf,
        y_hat=all_y_hat,
        y_cf=all_y_cf,
        include_dtw=include_dtw,
        include_reachability=True,
        forecaster=None,
    )
    metrics["forecast_monotonicity"] = all_monot

    summary = evaluator.summarize_with_std(metrics)
    return summary, cf_examples


def save_examples_plot(examples, out_path, rho=0.10):
    if not examples:
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i, ex in enumerate(examples):
        x_ot = ex["x_ot"][:, 0]
        x_cf = ex["x_cf"][:, 0]
        y_hat = ex["y_hat"][:, 0]
        y_cf = ex["y_cf"][:, 0]

        full_orig = np.concatenate([x_ot, y_hat])
        full_cf = np.concatenate([x_cf, y_cf])
        t_all = np.arange(len(full_orig))
        reduction = (y_hat.mean() - y_cf.mean()) / (abs(y_hat.mean()) + 1e-8) * 100
        ok = "✓" if reduction >= rho * 100 else "✗"

        axes[i].plot(t_all, full_orig, color="steelblue", lw=1.5, label="x + forecast(x)")
        axes[i].plot(t_all, full_cf, color="coral", lw=1.5, ls="--", label="x_cf + forecast(x_cf)")
        axes[i].axvline(len(x_ot), color="gray", ls="--", lw=1.2)
        axes[i].fill_between(t_all, full_orig, full_cf, alpha=0.12, color="coral")
        axes[i].set_title(f"Sample {i + 1} — {reduction:+.1f}% {ok}", fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    plt.suptitle("Frozen checkpoint evaluation — RL Mask", fontsize=13)
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

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    load_checkpoint_into_trainer(trainer, CHECKPOINT_PATH, device)

    plaus_model = None
    if os.path.exists(EXTERNAL_PLAUS_PATH):
        plaus_model = load_plausibility_model(EXTERNAL_PLAUS_PATH)
        print(f"[Eval] Plausibility model loaded: {EXTERNAL_PLAUS_PATH}")
    else:
        print("[Eval] No plausibility model found.")

    evaluator = CounterfactualEvaluator(
        plausibility_model=plaus_model,
        rho=cfg_rl.rho,
        x_train=trainer.x_train_eval,
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
        "rate_of_change_feasibility",
        "action_efficiency_ratio",
        "reachability_score",
        "causal_compactness",
        "forecast_monotonicity",
    ]

    for k in keys_to_show:
        if k in summary:
            print(f"{k:28s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    os.makedirs(cfg_rl.results_dir_lp, exist_ok=True)
    os.makedirs(cfg_rl.figures_dir_lp, exist_ok=True)

    metrics_path = os.path.join(cfg_rl.results_dir_lp, "frozen_eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Eval] Metrics saved -> {metrics_path}")

    fig_path = os.path.join(cfg_rl.figures_dir_lp, "frozen_eval_examples.png")
    save_examples_plot(examples, fig_path, rho=cfg_rl.rho)


if __name__ == "__main__":
    main()