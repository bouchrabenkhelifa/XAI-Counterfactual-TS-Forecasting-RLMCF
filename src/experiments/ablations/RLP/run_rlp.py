"""
Random Latent Perturbation (RLP) Baseline
==========================================
Même pipeline que le framework RL :
  x → AE.encode() → z → z + eta*noise → AE.decode() → masque temporel → forecaster → y_cf

La seule différence : l'action est un bruit gaussien aléatoire N(0,1)
au lieu de la politique apprise par le RL.

Usage (depuis la racine du projet) :
    python baselines/RLP/run_rlp.py \
        --forecast_config assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json \
        --ae_config       assets/configs/etth1_dataset/ae/tcn_ae.json \
        --rl_config       assets/configs/etth1_dataset/RL_ablations/config_final.json \
        --eval_batches    20 \
        --n_trials        10 \
        --seed            42 \
        --output          baselines/RLP/results/rlp_etth1.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.RL.reward_last import CFReward
from src.models.Forecaster.forecaster_wrapper import ForecasterWrapper
from src.evaluation.unified_evaluator import CounterfactualEvaluator
from src.training.RL_trainers.trainer_last import (
    prepare_rl_data,
    filter_batch,
    build_temporal_mask,
)

# ── 6 métriques principales ───────────────────────────────────────────────────
MAIN_METRICS = [
    ("validity_ratio",        "Validity Ratio ↑",        True),
    ("stepwise_auc",          "Stepwise AUC ↑",          True),
    ("proximity_l2",          "Proximity L2 ↓",          False),
    ("compactness",           "Compactness ↑",           True),
    ("temporal_consistency",  "Temporal Consistency ↑",  True),
    ("plausibility_ensemble", "Plausibility Ensemble ↓", False),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_bounds_np(y_hat, x_ot, rho, fr, direction, global_sigma=None):
    """Reproduit exactement RLMaskTrainer._compute_bounds_np."""
    x2d = x_ot[:, :, 0]
    y2d = y_hat[:, :, 0]

    if global_sigma is not None:
        sigma = np.full(x2d.shape[0], global_sigma)
    else:
        sigma = x2d.std(axis=1, ddof=1).clip(min=1e-4)

    gap   = (rho * sigma)[:, np.newaxis]
    width = (fr  * sigma)[:, np.newaxis]

    if direction < 0:
        betas  = y2d - gap
        alphas = betas - width
    else:
        alphas = y2d + gap
        betas  = alphas + width

    return alphas, betas


def run_one_trial(
    ae_arch, forecaster, test_loader, device,
    eta, mask_last_k, mask_ramp_k, filter_quantile, n_batches,
    n_examples=4,
):
    """Un seul run de perturbation aléatoire.
    Retourne (all_x, all_x_cf, all_y_hat, all_y_cf, examples).
    """
    all_x, all_x_cf, all_y_hat, all_y_cf = [], [], [], []
    examples = []

    for i, batch in enumerate(test_loader):
        if i >= n_batches:
            break

        batch_x, _, batch_x_mark, _ = batch
        batch_x      = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        x_ot         = batch_x[:, :, -1:]

        with torch.no_grad():
            z     = ae_arch.encode(x_ot)
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

        mask_keep = filter_batch(y_hat, filter_quantile)
        if mask_keep.sum() == 0:
            continue

        x_ot         = x_ot[mask_keep]
        batch_x      = batch_x[mask_keep]
        batch_x_mark = batch_x_mark[mask_keep]
        y_hat        = y_hat[mask_keep]
        z            = z[mask_keep]

        with torch.no_grad():
            noise = torch.randn_like(z)
            z_cf  = torch.clamp(z + eta * noise, -1.0, 1.0)

            x_prop    = ae_arch.decode(z_cf)
            delta     = x_prop - x_ot
            temp_mask = build_temporal_mask(
                x_ot.shape[0], x_ot.shape[1], x_ot.shape[2],
                mask_last_k, mask_ramp_k, device,
            )
            x_cf = x_ot + temp_mask * delta
            y_cf = forecaster.predict_from_ot(
                x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark
            )

        all_x.append(x_ot.cpu().numpy())
        all_x_cf.append(x_cf.cpu().numpy())
        all_y_hat.append(y_hat.cpu().numpy())
        all_y_cf.append(y_cf.cpu().numpy())

        if len(examples) < n_examples:
            for b in range(min(n_examples - len(examples), x_ot.shape[0])):
                examples.append({
                    "x_ot":  x_ot[b].cpu().numpy(),
                    "x_cf":  x_cf[b].cpu().numpy(),
                    "y_hat": y_hat[b].cpu().numpy(),
                    "y_cf":  y_cf[b].cpu().numpy(),
                })

    if not all_x:
        raise RuntimeError("No valid samples found in test_loader.")

    return (
        np.concatenate(all_x,    axis=0),
        np.concatenate(all_x_cf, axis=0),
        np.concatenate(all_y_hat, axis=0),
        np.concatenate(all_y_cf, axis=0),
        examples,
    )


# ── figures ───────────────────────────────────────────────────────────────────

def plot_results(avg_summary: dict, all_summaries: list, out_dir: str):
    """
    Figure 1 : barplot des 6 métriques (avg ± std over trials).
    Figure 2 : variabilité par trial pour les 6 métriques.
    """
    os.makedirs(out_dir, exist_ok=True)

    keys   = [k for k, _, _ in MAIN_METRICS]
    labels = [l for _, l, _ in MAIN_METRICS]
    means  = [avg_summary[k]["mean"] if k in avg_summary else 0.0 for k in keys]
    stds   = [avg_summary[k]["std"]  if k in avg_summary else 0.0 for k in keys]
    colors = ["#4C9BE8" if higher else "#E8754C" for _, _, higher in MAIN_METRICS]

    # Figure 1 — barplot
    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(keys))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.55)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Random Latent Perturbation — 6 Main Metrics (avg ± std over trials)")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(handles=[
        Patch(facecolor="#4C9BE8", alpha=0.85, label="Higher is better ↑"),
        Patch(facecolor="#E8754C", alpha=0.85, label="Lower is better ↓"),
    ], fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "rlp_metrics_barplot.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved → {p1}")

    # Figure 2 — variabilité par trial
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for idx, (key, label, higher) in enumerate(MAIN_METRICS):
        ax = axes[idx]
        trial_means = [s[key]["mean"] for s in all_summaries if key in s]
        ax.plot(range(1, len(trial_means) + 1), trial_means,
                marker="o", linewidth=1.8,
                color="#4C9BE8" if higher else "#E8754C")
        avg = np.mean(trial_means)
        ax.axhline(avg, color="gray", linestyle="--", linewidth=1.0, label=f"avg={avg:.3f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Trial")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("RLP — Per-trial variability", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "rlp_metrics_trials.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved → {p2}")


def plot_cf_examples(examples: list, out_dir: str, rho: float, fr: float,
                     direction: float, global_sigma: float):
    """
    Figure qualitative : série originale vs CF + leurs forecasts + bande α/β.
    """
    n = len(examples)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for i, ex in enumerate(examples):
        x_ot  = ex["x_ot"][:, 0]
        x_cf  = ex["x_cf"][:, 0]
        y_hat = ex["y_hat"][:, 0]
        y_cf  = ex["y_cf"][:, 0]

        BH = len(x_ot)
        t_back = np.arange(BH)
        t_fore = np.arange(BH, BH + len(y_hat))

        sigma    = global_sigma if global_sigma is not None else float(x_ot.std())
        gap      = rho * sigma
        width    = fr  * sigma
        if direction < 0:
            beta_np  = y_hat - gap
            alpha_np = beta_np - width
        else:
            alpha_np = y_hat + gap
            beta_np  = alpha_np + width

        valid_ratio = float(((y_cf >= alpha_np) & (y_cf <= beta_np)).mean())

        ax = axes[i]
        ax.plot(t_back, x_ot,  color="#2196F3", lw=1.8, label="x original")
        ax.plot(t_back, x_cf,  color="#FF5722", lw=1.8, ls="--", label="x CF (random)")
        ax.plot(t_fore, y_hat, color="#2196F3", lw=1.5, ls="-.",  label="forecast original")
        ax.plot(t_fore, y_cf,  color="#FF5722", lw=1.5, ls=":",   label="forecast CF")
        ax.fill_between(t_fore, alpha_np, beta_np,
                        alpha=0.20, color="#4CAF50", label="α/β target band")
        ax.axvline(BH, color="gray", lw=1.2, ls="--", alpha=0.7)
        ax.set_title(f"Sample {i+1} — validity={valid_ratio:.2f}", fontsize=11)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    fig.suptitle("RLP — CF Examples (original vs random perturbation)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "rlp_cf_examples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random Latent Perturbation Baseline")
    parser.add_argument(
        "--forecast_config",
        default="assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
    )
    parser.add_argument(
        "--ae_config",
        default="assets/configs/etth1_dataset/ae/tcn_ae.json",
    )
    parser.add_argument(
        "--rl_config",
        default="assets/configs/etth1_dataset/RL_ablations/config_final.json",
    )
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--n_trials",     type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output", default="src/experiments/ablations/RLP/results/rlp_etth1.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_f  = load_config(args.forecast_config)
    cfg_ae = load_config(args.ae_config)
    cfg_rl = load_config(args.rl_config)
    device = get_device(cfg_f)

    eta             = cfg_rl.eta
    mask_last_k     = cfg_rl.mask_last_k
    mask_ramp_k     = cfg_rl.mask_ramp_k
    filter_quantile = getattr(cfg_rl, "filter_quantile", 0.0)
    direction       = getattr(cfg_rl, "direction", -1.0)
    rho             = getattr(cfg_rl, "rho",  0.20)
    fr              = getattr(cfg_rl, "fr",   0.60)

    print(f"Device      : {device}")
    print(f"eta         : {eta}  mask_last_k: {mask_last_k}  mask_ramp_k: {mask_ramp_k}")
    print(f"n_trials    : {args.n_trials}  eval_batches: {args.eval_batches}")

    # Modèles gelés
    print("\nLoading frozen models...")
    ae_arch = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae_arch.eval()
    for p in ae_arch.parameters():
        p.requires_grad_(False)

    forecaster = ForecasterWrapper(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    train_loader, test_loader, _ = prepare_rl_data(cfg_f, cfg_ae, device)

    # global_sigma
    _x_stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        bx, _, _, _ = batch
        _x_stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_x_stds).mean())
    print(f"global_sigma: {global_sigma:.4f}")

    # x_train pour plausibilité
    x_train_batches = []
    for i, batch in enumerate(train_loader):
        if i >= getattr(cfg_rl, "eval_train_batches", 20):
            break
        bx, _, _, _ = batch
        x_train_batches.append(bx[:, :, -1:].cpu().numpy())
    x_train_eval = np.concatenate(x_train_batches, axis=0) if x_train_batches else None

    evaluator = CounterfactualEvaluator(
        x_train=x_train_eval,
        fit_plausibility=(x_train_eval is not None),
    )

    # n_trials runs
    print(f"\nRunning {args.n_trials} random trials...")
    all_summaries = []
    cf_examples   = []

    for trial in range(args.n_trials):
        torch.manual_seed(args.seed + trial)
        np.random.seed(args.seed + trial)

        all_x, all_x_cf, all_y_hat, all_y_cf, examples = run_one_trial(
            ae_arch=ae_arch,
            forecaster=forecaster,
            test_loader=test_loader,
            device=device,
            eta=eta,
            mask_last_k=mask_last_k,
            mask_ramp_k=mask_ramp_k,
            filter_quantile=filter_quantile,
            n_batches=args.eval_batches,
        )

        if trial == 0:
            cf_examples = examples

        alphas, betas = compute_bounds_np(
            all_y_hat, all_x, rho, fr, direction, global_sigma
        )

        summary = evaluator.evaluate(
            X_orig=all_x, X_cf=all_x_cf,
            Y_hat=all_y_hat, Y_cf=all_y_cf,
            alphas=alphas, betas=betas,
        )
        all_summaries.append(summary)

        validity_key = next((k for k in summary if "valid" in k.lower()), None)
        v = f"{summary[validity_key]['mean']:.4f}" if validity_key else "N/A"
        print(f"  Trial {trial+1:2d}/{args.n_trials} — validity={v}")

    # Moyenner sur les trials
    avg_summary = {}
    std_summary = {}
    for k in all_summaries[0]:
        means = [s[k]["mean"] for s in all_summaries
                 if isinstance(s.get(k, {}).get("mean"), (int, float))]
        if means:
            avg_summary[k] = {"mean": float(np.mean(means)), "std": float(np.std(means))}
            std_summary[k] = float(np.std(means))

    print(f"\n── RLP Average over {args.n_trials} trials ──────────────────────")
    CounterfactualEvaluator.print_table(avg_summary, label="Random Latent Perturbation")

    # Sauvegarder JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "baseline":     "Random Latent Perturbation",
            "n_trials":     args.n_trials,
            "eval_batches": args.eval_batches,
            "eta":          eta,
            "mask_last_k":  mask_last_k,
            "mask_ramp_k":  mask_ramp_k,
            "avg_metrics":  avg_summary,
            "std_metrics":  std_summary,
            "all_trials":   all_summaries,
        }, f, indent=2)
    print(f"Saved → {args.output}")

    # Figures
    out_dir = os.path.dirname(args.output)
    plot_results(avg_summary, all_summaries, out_dir=out_dir)
    plot_cf_examples(cf_examples, out_dir=out_dir,
                     rho=rho, fr=fr, direction=direction, global_sigma=global_sigma)


if __name__ == "__main__":
    main()
