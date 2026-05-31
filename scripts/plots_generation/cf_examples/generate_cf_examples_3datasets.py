#!/usr/bin/env python
"""
Generate CF Example Plots for 3 Datasets
==========================================
Produces one counterfactual example per dataset (ETTh1, ETTh2, Weather)
using the iTransformer forecaster, displayed as a single combined figure.

Usage:
    python scripts/generate_cf_examples_3datasets.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_last import build_temporal_mask

# ── Dataset Configurations ────────────────────────────────────────────────────
DATASETS = {
    "ETTh1": {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json",
        "ckpt_rl": "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
        "ylabel": "OT (°C)",
        "batch_idx": 42,
    },
    "ETTh2": {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/etth2_dataset/RL/config_itransformer.json",
        "ckpt_rl": "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
        "ylabel": "OT (°C)",
        "batch_idx": 30,
    },
    "Weather": {
        "cfg_f": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "cfg_rl": "assets/configs/weather_dataset/RL/config_itransformer.json",
        "ckpt_rl": "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
        "ylabel": "T (°C)",
        "batch_idx": 25,
    },
}

OUT_PATH = "assets/figures/global_analysis/cf_examples_3datasets.png"


def generate_cf(x_ot, bx, bx_mark, ae, forecaster, agent, lk, rk, device):
    """Generate counterfactual using RL-MCF."""
    with torch.no_grad():
        z = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(bx, bx_mark)
        z_cf, _, _ = agent.act_deterministic(z, y_hat)
        x_prop = ae.decode(z_cf)
        mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
        x_cf = x_ot + mask * (x_prop - x_ot)
        y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
    return x_cf, y_cf, y_hat


def process_dataset(dataset_name, cfg_dict, device):
    """Load models and generate CF for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name}")
    print(f"{'='*60}")

    cfg_f = load_config(os.path.join(ROOT, cfg_dict["cfg_f"]))
    cfg_ae = load_config(os.path.join(ROOT, cfg_dict["cfg_ae"]))
    cfg_rl = load_config(os.path.join(ROOT, cfg_dict["cfg_rl"]))

    # Load AE
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # Load Forecaster
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    # Load RL Agent
    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim,
        pred_len=cfg_f.pred_len,
        eta=cfg_rl.eta,
        entropy_coef=cfg_rl.entropy_coef,
        direction=getattr(cfg_rl, "direction", -1.0),
    ).to(device)

    ckpt = torch.load(os.path.join(ROOT, cfg_dict["ckpt_rl"]), map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Compute global sigma
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        bx_b, _, _, _ = batch
        _stds.append(bx_b[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())
    print(f"  global_sigma = {global_sigma:.4f}")

    reward_fn = CFReward(
        fr=cfg_rl.fr,
        rho=cfg_rl.rho,
        direction=getattr(cfg_rl, "direction", -1.0),
        global_sigma=global_sigma,
    ).to(device)

    lk = cfg_rl.mask_last_k
    rk = cfg_rl.mask_ramp_k

    # Load test sample
    test_dataset, test_loader = data_provider(cfg_f, "test")
    scaler = test_dataset.scaler

    batch_idx = cfg_dict["batch_idx"]
    for i, batch in enumerate(test_loader):
        if i == batch_idx:
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]
            # Take first sample
            x_ot = x_ot[0:1]
            bx = bx[0:1]
            bx_mark = bx_mark[0:1]
            break

    # Compute bounds
    with torch.no_grad():
        y_hat_bounds = forecaster.predict_ot(bx, bx_mark)
        alpha, beta, _ = reward_fn.compute_bounds(y_hat_bounds, x_ot=x_ot)

    # Generate CF
    x_cf, y_cf, y_hat = generate_cf(x_ot, bx, bx_mark, ae, forecaster, agent, lk, rk, device)

    # Convert to numpy
    x_np = x_ot[0, :, 0].cpu().numpy()
    yh_np = y_hat[0, :, 0].cpu().numpy()
    xcf_np = x_cf[0, :, 0].cpu().numpy()
    ycf_np = y_cf[0, :, 0].cpu().numpy()
    alpha_np = alpha[0].cpu().numpy()
    beta_np = beta[0].cpu().numpy()

    # Denormalize
    def denorm(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    result = {
        "x": denorm(x_np),
        "y_hat": denorm(yh_np),
        "x_cf": denorm(xcf_np),
        "y_cf": denorm(ycf_np),
        "alpha": denorm(alpha_np),
        "beta": denorm(beta_np),
        "mask_start": len(x_np) - lk - rk,
        "seq_len": len(x_np),
        "pred_len": len(yh_np),
    }

    vr = float(((ycf_np >= alpha_np) & (ycf_np <= beta_np)).mean())
    print(f"  Validity Ratio: {vr:.3f}")
    result["vr"] = vr

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Process all datasets
    results = {}
    for name, cfg in DATASETS.items():
        results[name] = process_dataset(name, cfg, device)

    # ── Create combined figure ────────────────────────────────────────────────
    print("\n\n[*] Creating combined figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (dataset_name, r) in enumerate(results.items()):
        ax = axes[idx]

        BH = r["seq_len"]
        H = r["pred_len"]
        t_back = np.arange(BH)
        t_fore = np.arange(BH, BH + H)

        orig_full = np.concatenate([r["x"], r["y_hat"]])
        cf_full = np.concatenate([r["x_cf"], r["y_cf"]])

        # Shaded mask region
        mask_start = r["mask_start"]
        ax.axvspan(mask_start, BH, alpha=0.10, color="#FF6F00", zorder=0)

        # Original: blue
        ax.plot(t_back, orig_full[:BH], color="#1565C0", lw=2.2, ls="--")
        ax.plot(t_fore, orig_full[BH:], color="#1565C0", lw=2.2, ls="-",
                label=r"$x$ / $\hat{y}$ (original)")
        ax.plot([BH - 1, BH], orig_full[BH - 1:BH + 1], color="#1565C0", lw=2.2, ls="-")

        # CF: orange
        ax.plot(t_back, cf_full[:BH], color="#E65100", lw=1.8, ls="--")
        ax.plot(t_fore, cf_full[BH:], color="#E65100", lw=1.8, ls="-",
                label=f"$x_{{cf}}$ / $\\hat{{y}}_{{cf}}$ (VR={r['vr']:.2f})")
        ax.plot([BH - 1, BH], cf_full[BH - 1:BH + 1], color="#E65100", lw=1.8, ls="-")

        # Target band
        ax.fill_between(t_fore, r["alpha"], r["beta"], alpha=0.18, color="#757575",
                        label=r"Target band [$\alpha$, $\beta$]")

        # Vertical separator
        ax.axvline(BH, color="black", lw=1.0, ls="--", alpha=0.4)

        ax.set_xlabel("Time step", fontsize=11)
        ax.set_ylabel(DATASETS[dataset_name]["ylabel"], fontsize=11)
        ax.set_title(f"{dataset_name} (iTransformer)", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2, ls="--")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()

    out_full = os.path.join(ROOT, OUT_PATH)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    plt.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Combined figure saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
