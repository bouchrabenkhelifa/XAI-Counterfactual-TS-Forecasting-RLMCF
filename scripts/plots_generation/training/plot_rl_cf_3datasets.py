"""
Plot RL counterfactual examples for all 3 datasets (ETTh1, ETTh2, Weather).
Shows original series + CF series + target band [α, β] side by side.

Usage:
    python scripts/analysis/plot_rl_cf_3datasets.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_main import build_temporal_mask

FIGURES_DIR = "assets/figures/global_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Configs per dataset
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "ETTh1": {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
        "rho": 0.2, "fr": 0.5, "mask_last_k": 12, "mask_ramp_k": 4, "eta": 0.15,
        "ylabel": "OT (°C)",
    },
    "ETTh2": {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
        "rho": 0.2, "fr": 0.5, "mask_last_k": 12, "mask_ramp_k": 4, "eta": 0.15,
        "ylabel": "OT (°C)",
    },
    "Weather": {
        "cfg_f": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
        "rho": 0.2, "fr": 0.5, "mask_last_k": 12, "mask_ramp_k": 4, "eta": 0.15,
        "ylabel": "T (°C)",
    },
}


def generate_cf_example(info, device):
    """Generate one good CF example for a dataset."""
    cfg_f = load_config(info["cfg_f"])
    cfg_ae = load_config(info["cfg_ae"])
    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"

    # Load models
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    # Load RL agent
    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=info["eta"], entropy_coef=0.02, direction=-1.0,
    ).to(device)
    ckpt = torch.load(info["rl_ckpt"], map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Compute global sigma
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(
        fr=info["fr"], rho=info["rho"], direction=-1.0, global_sigma=global_sigma
    ).to(device)

    # Get scaler for denormalization
    test_data, test_loader = data_provider(cfg_f, "test")
    scaler = test_data.scaler

    lk = info["mask_last_k"]
    rk = info["mask_ramp_k"]

    # Find a good sample (high validity + interesting visually)
    best_score = -1
    best_result = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 300:
                break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]

            z = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf, _, _ = agent.act_deterministic(z, y_hat)
            x_prop = ae.decode(z_cf)
            mask = build_temporal_mask(1, x_ot.shape[1], 1, lk, rk, device)
            x_cf = x_ot + mask * (x_prop - x_ot)
            y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr = float(((y_cf[0, :, 0] >= alpha[0]) & (y_cf[0, :, 0] <= beta[0])).float().mean())
            variance = float(x_ot[0, :, 0].std())

            # Score: high validity AND high variance (interesting)
            score = vr * 0.7 + min(variance / 0.5, 1.0) * 0.3

            if score > best_score and vr >= 0.8:
                best_score = score
                # Denormalize
                x_np = scaler.inverse_transform(x_ot[0, :, 0].cpu().numpy().reshape(-1, 1)).flatten()
                xcf_np = scaler.inverse_transform(x_cf[0, :, 0].cpu().numpy().reshape(-1, 1)).flatten()
                yh_np = scaler.inverse_transform(y_hat[0, :, 0].cpu().numpy().reshape(-1, 1)).flatten()
                ycf_np = scaler.inverse_transform(y_cf[0, :, 0].cpu().numpy().reshape(-1, 1)).flatten()
                a_np = scaler.inverse_transform(alpha[0].cpu().numpy().reshape(-1, 1)).flatten()
                b_np = scaler.inverse_transform(beta[0].cpu().numpy().reshape(-1, 1)).flatten()

                best_result = {
                    "x": x_np, "x_cf": xcf_np,
                    "y_hat": yh_np, "y_cf": ycf_np,
                    "alpha": a_np, "beta": b_np,
                    "vr": vr,
                    "seq_len": cfg_f.seq_len, "pred_len": cfg_f.pred_len,
                }

    return best_result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for col, (name, info) in enumerate(DATASETS.items()):
        print(f"\n── {name} ──")

        # Check if RL checkpoint exists
        if not os.path.exists(info["rl_ckpt"]):
            print(f"  ⚠ Checkpoint not found: {info['rl_ckpt']}")
            axes[col].set_title(f"{name}\n(checkpoint not found)")
            continue

        result = generate_cf_example(info, device)
        if result is None:
            axes[col].set_title(f"{name}\n(no valid CF found)")
            continue

        ax = axes[col]
        seq_len = result["seq_len"]
        pred_len = result["pred_len"]

        # Build full series
        orig_full = np.concatenate([result["x"], result["y_hat"]])
        cf_full = np.concatenate([result["x_cf"], result["y_cf"]])
        t_all = np.arange(len(orig_full))
        t_back = np.arange(seq_len)
        t_fore = np.arange(seq_len, seq_len + pred_len)

        # Plot original (blue)
        ax.plot(t_back, orig_full[:seq_len], color="#1565C0", lw=2, ls="-")
        ax.plot(t_fore, orig_full[seq_len:], color="#1565C0", lw=2, ls="--",
                label="$x$ / $\hat{y}$ (original)")
        ax.plot([seq_len-1, seq_len], orig_full[seq_len-1:seq_len+1], color="#1565C0", lw=2)

        # Plot CF (orange/red)
        ax.plot(t_back, cf_full[:seq_len], color="#E65100", lw=2, ls="-")
        ax.plot(t_fore, cf_full[seq_len:], color="#E65100", lw=2, ls="--",
                label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ RL-MCF (VR={result['vr']:.2f})")
        ax.plot([seq_len-1, seq_len], cf_full[seq_len-1:seq_len+1], color="#E65100", lw=2)

        # Target band
        ax.fill_between(t_fore, result["alpha"], result["beta"],
                        alpha=0.2, color="gray", label=r"Target band [$\alpha$, $\beta$]")

        # Vertical line
        ax.axvline(seq_len, color="gray", ls=":", lw=1)

        ax.set_title(f"{name}", fontsize=12)
        ax.set_xlabel("Timestep")
        if col == 0:
            ax.set_ylabel(info["ylabel"])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25, ls="--")

    plt.suptitle("RL-MCF Counterfactual Examples — 3 Datasets (iTransformer)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rl_cf_3datasets.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] → {path}")


if __name__ == "__main__":
    main()
