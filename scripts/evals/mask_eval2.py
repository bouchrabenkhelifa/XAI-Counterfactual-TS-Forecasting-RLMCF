import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from src.utils.config import load_config
from src.utils.train_tools import get_device

from src.training.RL_trainers.RL_mask_trainer import RLMaskTrainer
from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


CONFIG_FORECASTER = "assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json"
CONFIG_AE = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CONFIG_RL = "assets/configs/models/etth1_dataset/RL/rl_mask.json"

CHECKPOINT_PATH = "assets/checkpoints/RL_mask/rl_mask_agent_best.pt"
EXTERNAL_PLAUS_PATH = "assets/checkpoints/anomaly detector/plausibility_etth1.pkl"


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


@torch.no_grad()
def trainer_run_episode(trainer, batch, filter_quantile=0.75, use_hf=True):
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(trainer.device)
    batch_x_mark = batch_x_mark.float().to(trainer.device)
    x_ot = batch_x[:, :, -1:]

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

    mu, _ = trainer.agent.actor(s)
    a = mu

    z_cf = torch.clamp(z + trainer.agent.eta * a, -1.0, 1.0)

    x_prop = trainer.ae.decode(z_cf)
    delta = x_prop - x_ot

    temp_mask = build_temporal_mask(
        batch_size=x_ot.shape[0],
        seq_len=x_ot.shape[1],
        channels=x_ot.shape[2],
        last_k=trainer.mask_last_k,
        ramp_k=trainer.mask_ramp_k,
        device=trainer.device,
    )

    if use_hf:
        x_lf = trainer.ae.decode(trainer.ae.encode(x_ot))
        x_hf = x_ot - x_lf
        masked_delta = delta + trainer.alpha_hf * x_hf
    else:
        masked_delta = delta

    x_cf = x_ot + temp_mask * masked_delta

    y_cf = trainer.forecaster.predict_from_ot(
        x_ot=x_cf,
        x_full=batch_x,
        x_mark=batch_x_mark
    )

    reward_dict = trainer.reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)

    return {
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "z_ot": z.detach(),
        "z_cf": z_cf.detach(),
        "reward_dict": reward_dict,
        "mask": temp_mask.detach(),
    }


@torch.no_grad()
def collect_train_reference(trainer, n_batches=50):
    xs = []
    zs = []

    for i, batch in enumerate(trainer.train_loader):
        if i >= n_batches:
            break

        batch_x, _, _, _ = batch
        batch_x = batch_x.float().to(trainer.device)
        x_ot = batch_x[:, :, -1:]
        z_ot = trainer.ae.encode(x_ot)

        xs.append(x_ot.cpu().numpy())
        zs.append(z_ot.cpu().numpy())

    if len(xs) == 0:
        raise RuntimeError("Could not collect x_train reference.")

    return np.concatenate(xs, axis=0), np.concatenate(zs, axis=0)


@torch.no_grad()
def forecasting_callable(trainer, x_np):
    x_t = torch.from_numpy(np.asarray(x_np)).float().to(trainer.device)

    bsz, seq_len, _ = x_t.shape
    x_full = torch.zeros((bsz, seq_len, trainer.cfg_f.enc_in), device=trainer.device)
    x_full[:, :, -1:] = x_t

    x_mark = torch.zeros((bsz, seq_len, 4), device=trainer.device)

    y = trainer.forecaster.predict_from_ot(
        x_ot=x_t,
        x_full=x_full,
        x_mark=x_mark
    )
    return y.detach().cpu().numpy()


def flatten_3d(x):
    return x.reshape(x.shape[0], -1)


def build_knn_index(ref, k=5):
    k = min(k, len(ref))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(flatten_3d(ref))
    return knn


def knn_mean_dist(knn, samples):
    dists, idx = knn.kneighbors(flatten_3d(samples))
    return dists.mean(axis=1), idx


def summarize_knn(orig, cf, ref, prefix="input", k=5):
    knn = build_knn_index(ref, k=k)

    d_orig, idx_orig = knn_mean_dist(knn, orig)
    d_cf, idx_cf = knn_mean_dist(knn, cf)

    summary = {
        f"{prefix}_orig_knn_mean": float(np.mean(d_orig)),
        f"{prefix}_orig_knn_std": float(np.std(d_orig)),
        f"{prefix}_cf_knn_mean": float(np.mean(d_cf)),
        f"{prefix}_cf_knn_std": float(np.std(d_cf)),
        f"{prefix}_knn_delta": float(np.mean(d_cf) - np.mean(d_orig)),
        f"{prefix}_knn_ratio": float(np.mean(d_cf) / (np.mean(d_orig) + 1e-8)),
    }
    return summary, idx_orig, idx_cf, knn


def save_examples_plot(examples, out_path, rho=0.10, title="Frozen checkpoint evaluation — RL Mask + Actionability"):
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
        axes[i].set_title(f"Sample {i+1} — {reduction:+.1f}% {ok}", fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Examples plot -> {out_path}")


def save_neighbor_plots(
    examples,
    ref_x,
    idx_cf,
    out_path,
    n_examples=4,
    title="CF vs nearest real neighbors"
):
    n = min(n_examples, len(examples))
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        x_ot = examples[i]["x_ot"][:, 0]
        x_cf = examples[i]["x_cf"][:, 0]

        ax.plot(x_ot, color="steelblue", lw=1.5, label="Original")
        ax.plot(x_cf, color="coral", lw=2.0, ls="--", label="CF")

        for j, nn_idx in enumerate(idx_cf[i][:3]):
            nn_series = ref_x[nn_idx][:, 0]
            ax.plot(nn_series, lw=1.0, alpha=0.65, label=f"NN{j+1}" if i == 0 else None)

        ax.set_title(f"Sample {i+1}")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Neighbor plot -> {out_path}")


@torch.no_grad()
def evaluate_frozen_model(
    trainer,
    evaluator,
    n_batches=20,
    include_dtw=False,
    include_reachability=True,
    include_forecast_monotonicity=True,
    use_hf=True,
):
    trainer.agent.eval()

    all_x = []
    all_x_cf = []
    all_y_hat = []
    all_y_cf = []
    all_z = []
    all_z_cf = []

    cf_examples = []

    for i, batch in enumerate(trainer.test_loader):
        if i >= n_batches:
            break

        ep = trainer_run_episode(trainer, batch, use_hf=use_hf)
        if ep is None:
            continue

        all_x.append(ep["x_ot"].cpu().numpy())
        all_x_cf.append(ep["x_cf"].cpu().numpy())
        all_y_hat.append(ep["y_hat"].cpu().numpy())
        all_y_cf.append(ep["y_cf"].cpu().numpy())
        all_z.append(ep["z_ot"].cpu().numpy())
        all_z_cf.append(ep["z_cf"].cpu().numpy())

        if len(cf_examples) < 4:
            cf_examples.append({
                "x_ot": ep["x_ot"][0].cpu().numpy(),
                "x_cf": ep["x_cf"][0].cpu().numpy(),
                "y_hat": ep["y_hat"][0].cpu().numpy(),
                "y_cf": ep["y_cf"][0].cpu().numpy(),
            })

    if len(all_x) == 0:
        raise RuntimeError("No valid CFs generated during evaluation.")

    all_x = np.concatenate(all_x, axis=0)
    all_x_cf = np.concatenate(all_x_cf, axis=0)
    all_y_hat = np.concatenate(all_y_hat, axis=0)
    all_y_cf = np.concatenate(all_y_cf, axis=0)
    all_z = np.concatenate(all_z, axis=0)
    all_z_cf = np.concatenate(all_z_cf, axis=0)

    forecaster_fn = None
    if include_forecast_monotonicity:
        forecaster_fn = lambda x: forecasting_callable(trainer, x)

    metrics = evaluator.evaluate_batch(
        x=all_x,
        x_cf=all_x_cf,
        y_hat=all_y_hat,
        y_cf=all_y_cf,
        include_dtw=include_dtw,
        forecaster=forecaster_fn,
        include_reachability=include_reachability,
    )

    summary = evaluator.summarize_with_std(metrics)

    outputs = {
        "summary": summary,
        "examples": cf_examples,
        "all_x": all_x,
        "all_x_cf": all_x_cf,
        "all_y_hat": all_y_hat,
        "all_y_cf": all_y_cf,
        "all_z": all_z,
        "all_z_cf": all_z_cf,
    }
    return outputs


def print_knn_block(title, stats):
    print(f"\n── {title} ─────────────────────────────────────")
    for k, v in stats.items():
        print(f"{k:28s}: {v:.4f}")


def main():
    cfg_f = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)

    device = get_device(cfg_f)
    print(f"Device: {device}")

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    load_checkpoint_into_trainer(trainer, CHECKPOINT_PATH, device)

    x_train_ref, z_train_ref = collect_train_reference(trainer, n_batches=50)

    plaus_model = load_plausibility_model(EXTERNAL_PLAUS_PATH)
    evaluator = CounterfactualEvaluator(
        plausibility_model=plaus_model,
        rho=cfg_rl.rho,
        x_train=x_train_ref,
    )

    # ===== Evaluation WITH HF =====
    out_hf = evaluate_frozen_model(
        trainer=trainer,
        evaluator=evaluator,
        n_batches=20,
        include_dtw=False,
        include_reachability=True,
        include_forecast_monotonicity=True,
        use_hf=True,
    )

    summary_hf = out_hf["summary"]
    examples_hf = out_hf["examples"]

    summary_flat_hf = {k: v["mean"] for k, v in summary_hf.items()}

    # ===== Evaluation NO HF =====
    out_nohf = evaluate_frozen_model(
        trainer=trainer,
        evaluator=evaluator,
        n_batches=20,
        include_dtw=False,
        include_reachability=True,
        include_forecast_monotonicity=True,
        use_hf=False,
    )

    # ===== KNN diagnostics =====
    knn_input_hf, _, idx_cf_hf, _ = summarize_knn(
        out_hf["all_x"], out_hf["all_x_cf"], x_train_ref, prefix="input", k=5
    )
    knn_latent_hf, _, _, _ = summarize_knn(
        out_hf["all_z"], out_hf["all_z_cf"], z_train_ref, prefix="latent", k=5
    )

    knn_input_nohf, _, idx_cf_nohf, _ = summarize_knn(
        out_nohf["all_x"], out_nohf["all_x_cf"], x_train_ref, prefix="input", k=5
    )
    knn_latent_nohf, _, _, _ = summarize_knn(
        out_nohf["all_z"], out_nohf["all_z_cf"], z_train_ref, prefix="latent", k=5
    )

    print("\n── Final Evaluation Metrics (WITH HF) ─────────────────────────")
    evaluator.pretty_print_summary(summary_flat_hf)

    print("\n── Mean ± Std (WITH HF) ───────────────────────────────────────")
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
        if k in summary_hf:
            print(f"{k:28s}: {summary_hf[k]['mean']:.4f} ± {summary_hf[k]['std']:.4f}")

    print_knn_block("Reachability Diagnostics (WITH HF) — input space", knn_input_hf)
    print_knn_block("Reachability Diagnostics (WITH HF) — latent space", knn_latent_hf)
    print_knn_block("Reachability Diagnostics (NO HF) — input space", knn_input_nohf)
    print_knn_block("Reachability Diagnostics (NO HF) — latent space", knn_latent_nohf)

    os.makedirs(cfg_rl.results_dir_lp, exist_ok=True)
    os.makedirs(cfg_rl.figures_dir_lp, exist_ok=True)

    metrics_path = os.path.join(cfg_rl.results_dir_lp, "frozen_eval_metrics_actionability.json")
    with open(metrics_path, "w") as f:
        json.dump(summary_hf, f, indent=2)
    print(f"\n[Eval] Metrics saved -> {metrics_path}")

    diag_path = os.path.join(cfg_rl.results_dir_lp, "reachability_knn_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(
            {
                "with_hf": {
                    "input": knn_input_hf,
                    "latent": knn_latent_hf,
                },
                "no_hf": {
                    "input": knn_input_nohf,
                    "latent": knn_latent_nohf,
                },
            },
            f,
            indent=2,
        )
    print(f"[Eval] KNN diagnostics saved -> {diag_path}")

    fig_path = os.path.join(cfg_rl.figures_dir_lp, "frozen_eval_examples_actionability.png")
    save_examples_plot(
        examples_hf,
        fig_path,
        rho=cfg_rl.rho,
        title="Frozen checkpoint evaluation — RL Mask + Actionability (WITH HF)"
    )

    neigh_hf_path = os.path.join(cfg_rl.figures_dir_lp, "reachability_neighbors_with_hf.png")
    save_neighbor_plots(
        examples_hf,
        x_train_ref,
        idx_cf_hf,
        neigh_hf_path,
        n_examples=4,
        title="CF vs nearest real neighbors (WITH HF)"
    )

    examples_nohf = out_nohf["examples"]
    fig_nohf_path = os.path.join(cfg_rl.figures_dir_lp, "frozen_eval_examples_no_hf.png")
    save_examples_plot(
        examples_nohf,
        fig_nohf_path,
        rho=cfg_rl.rho,
        title="Frozen checkpoint evaluation — RL Mask + Actionability (NO HF)"
    )

    neigh_nohf_path = os.path.join(cfg_rl.figures_dir_lp, "reachability_neighbors_no_hf.png")
    save_neighbor_plots(
        examples_nohf,
        x_train_ref,
        idx_cf_nohf,
        neigh_nohf_path,
        n_examples=4,
        title="CF vs nearest real neighbors (NO HF)"
    )


if __name__ == "__main__":
    main()