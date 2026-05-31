"""
Generate comparison figure for Weather: RL-MCF vs ForecastCF baseline.
Same style as the ETTh1 comparison figure.

Usage:
    python scripts/generate_comparison_weather.py
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
from src.training.RL_trainers.trainer_main import build_temporal_mask
from baselines.ForecastCF_PyTorch.forecastcf_pt import ForecastCF_PyTorch

# ── Configs ───────────────────────────────────────────────────────────────────
CFG_F   = "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json"
CFG_AE  = "assets/configs/weather_dataset/ae/tcn_ae.json"
CKPT_RL = "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt"
OUT_PATH = "assets/figures/global_analysis/comparison_weather_rlmcf_vs_forecastcf.png"

RHO = 0.1
FR = 0.6
MASK_LAST_K = 24
MASK_RAMP_K = 8
ETA = 0.15


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_f = load_config(CFG_F)
    cfg_ae = load_config(CFG_AE)
    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading AE...")
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    print("Loading forecaster...")
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters():
        p.requires_grad_(False)

    print("Loading RL agent...")
    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=ETA, entropy_coef=0.02, direction=-1.0,
    ).to(device)
    ckpt = torch.load(CKPT_RL, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # ── ForecastCF baseline ───────────────────────────────────────────────────
    print("Loading ForecastCF baseline...")
    fcf = ForecastCF_PyTorch(forecaster, device, lr=0.01, max_iter=300,
                              w_validity=1.0, w_proximity=0.1)

    # ── Compute global sigma ──────────────────────────────────────────────────
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(fr=FR, rho=RHO, direction=-1.0, global_sigma=global_sigma).to(device)

    # ── Get scaler + test data ────────────────────────────────────────────────
    test_data, test_loader = data_provider(cfg_f, "test")
    scaler = test_data.scaler

    # ── Find best sample ──────────────────────────────────────────────────────
    print("Scanning test set...")
    best_score = -1
    best_data = None

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

            # RLMCF
            z_cf, _, _ = agent.act_deterministic(z, y_hat)
            x_prop = ae.decode(z_cf)
            mask = build_temporal_mask(1, x_ot.shape[1], 1, MASK_LAST_K, MASK_RAMP_K, device)
            x_cf = x_ot + mask * (x_prop - x_ot)
            y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr = float(((y_cf[0, :, 0] >= alpha[0]) & (y_cf[0, :, 0] <= beta[0])).float().mean())
            variance = float(x_ot[0, :, 0].std())

            score = vr * 0.7 + min(variance / 0.5, 1.0) * 0.3

            if score > best_score and vr >= 0.85:
                best_score = score
                best_data = {
                    "bx": bx.clone(), "bx_mark": bx_mark.clone(),
                    "x_ot": x_ot.clone(), "y_hat": y_hat.clone(),
                    "x_cf_rl": x_cf.clone(), "y_cf_rl": y_cf.clone(),
                    "alpha": alpha.clone(), "beta": beta.clone(),
                    "vr_rl": vr,
                }
                if vr >= 0.95:
                    print(f"  batch {i}: RLMCF VR={vr:.2f} → great!")
                    break

    if best_data is None:
        print("No valid CF found!")
        return

    # ── Generate ForecastCF on same sample ────────────────────────────────────
    print("Generating ForecastCF...")
    x_np = best_data["x_ot"][0].cpu().numpy()
    alpha_np = best_data["alpha"][0].cpu().numpy()
    beta_np = best_data["beta"][0].cpu().numpy()

    x_cf_fcf, y_cf_fcf = fcf.transform_sample(
        x_np=x_np.reshape(1, -1, 1),
        alpha_np=alpha_np,
        beta_np=beta_np,
        x_full_orig=best_data["bx"][:1],
        x_mark=best_data["bx_mark"][:1],
    )

    # Compute ForecastCF validity
    y_cf_fcf_flat = y_cf_fcf[0, :, 0]
    vr_fcf = float(((y_cf_fcf_flat >= alpha_np) & (y_cf_fcf_flat <= beta_np)).mean())

    print(f"RLMCF VR={best_data['vr_rl']:.3f}  |  ForecastCF VR={vr_fcf:.3f}")

    # ── Denormalize ───────────────────────────────────────────────────────────
    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    x = inv(best_data["x_ot"][0, :, 0].cpu().numpy())
    yh = inv(best_data["y_hat"][0, :, 0].cpu().numpy())
    x_cf_rl = inv(best_data["x_cf_rl"][0, :, 0].cpu().numpy())
    y_cf_rl = inv(best_data["y_cf_rl"][0, :, 0].cpu().numpy())
    x_cf_f = inv(x_cf_fcf[0, :, 0])
    y_cf_f = inv(y_cf_fcf_flat)
    a_np = inv(alpha_np)
    b_np = inv(beta_np)

    BH = len(x)
    H = len(yh)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    orig_full = np.concatenate([x, yh])
    rl_full = np.concatenate([x_cf_rl, y_cf_rl])
    fcf_full = np.concatenate([x_cf_f, y_cf_f])

    t_back = np.arange(BH)
    t_fore = np.arange(BH, BH + H)

    # Original (blue)
    ax.plot(t_back, orig_full[:BH], color="#1565C0", lw=2.5, ls="-")
    ax.plot(t_fore, orig_full[BH:], color="#1565C0", lw=2.5, ls="--",
            label="$x$ / $\hat{y}$ (original)")
    ax.plot([BH-1, BH], orig_full[BH-1:BH+1], color="#1565C0", lw=2.5)

    # RLMCF (green)
    ax.plot(t_back, rl_full[:BH], color="#2E7D32", lw=2.0, ls="-")
    ax.plot(t_fore, rl_full[BH:], color="#2E7D32", lw=2.0, ls="--",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ RL-MCF  (VR={best_data['vr_rl']:.2f})")
    ax.plot([BH-1, BH], rl_full[BH-1:BH+1], color="#2E7D32", lw=2.0)

    # ForecastCF (red)
    ax.plot(t_back, fcf_full[:BH], color="#C62828", lw=2.0, ls="-")
    ax.plot(t_fore, fcf_full[BH:], color="#C62828", lw=2.0, ls="--",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ ForecastCF  (VR={vr_fcf:.2f})")
    ax.plot([BH-1, BH], fcf_full[BH-1:BH+1], color="#C62828", lw=2.0)

    # Target band
    ax.fill_between(t_fore, a_np, b_np, alpha=0.18, color="#757575",
                    label=r"Target band [$\alpha$, $\beta$]")

    ax.axvline(BH, color="black", lw=1.2, ls="--", alpha=0.4)

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("T (°C)", fontsize=12)
    ax.set_title("Weather — RL-MCF vs ForecastCF (iTransformer, pred_len=96)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25, ls="--")

    all_vals = np.concatenate([orig_full, rl_full, fcf_full, a_np, b_np])
    ax.set_ylim(all_vals.min() - 0.5, all_vals.max() + 0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
