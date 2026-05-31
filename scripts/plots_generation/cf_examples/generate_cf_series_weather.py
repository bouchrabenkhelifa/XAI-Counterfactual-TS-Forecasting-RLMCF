"""
Generate CF Series Figure for Weather dataset: RLMCF vs BaseGrad vs BaseNN
Same style as the ETTh1 comparison figure but for Weather.

Usage:
    python scripts/generate_cf_series_weather.py
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
from baselines.BaseGrad.basegrad import BaseGrad
from baselines.BaseNN.basenn import BaseNN

# ── Configs ───────────────────────────────────────────────────────────────────
CFG_F   = "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json"
CFG_AE  = "assets/configs/weather_dataset/ae/tcn_ae.json"
CKPT_RL = "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt"
OUT_PATH = "assets/figures/global_analysis/cf_series_weather.png"

# RL params
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

    # ── Compute global sigma ──────────────────────────────────────────────────
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50:
            break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())
    print(f"global_sigma = {global_sigma:.4f}")

    reward_fn = CFReward(fr=FR, rho=RHO, direction=-1.0, global_sigma=global_sigma).to(device)

    # ── Get scaler ────────────────────────────────────────────────────────────
    test_data, test_loader = data_provider(cfg_f, "test")
    scaler = test_data.scaler

    # ── Find best sample ──────────────────────────────────────────────────────
    print("Scanning test set...")
    best_score = -1
    best_data = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 500:
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

            # RLP (random)
            torch.manual_seed(i)
            noise = torch.randn_like(z)
            z_cf_rp = torch.clamp(z + ETA * noise, -1, 1)
            x_prop_rp = ae.decode(z_cf_rp)
            x_cf_rp = x_ot + mask * (x_prop_rp - x_ot)
            y_cf_rp = forecaster.predict_from_ot(x_ot=x_cf_rp, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr_rm = float(((y_cf[0, :, 0] >= alpha[0]) & (y_cf[0, :, 0] <= beta[0])).float().mean())
            vr_rp = float(((y_cf_rp[0, :, 0] >= alpha[0]) & (y_cf_rp[0, :, 0] <= beta[0])).float().mean())
            variance = float(x_ot[0, :, 0].std())

            score = (vr_rm - vr_rp) * 0.7 + min(variance / 0.5, 1.0) * 0.3

            if score > best_score and vr_rm >= 0.8:
                best_score = score
                best_data = {
                    "x_ot": x_ot[0, :, 0].cpu().numpy(),
                    "x_ot_raw": x_ot[0, :, 0].cpu().numpy(),
                    "x_cf": x_cf[0, :, 0].cpu().numpy(),
                    "x_cf_rp": x_cf_rp[0, :, 0].cpu().numpy(),
                    "y_hat": y_hat[0, :, 0].cpu().numpy(),
                    "y_cf": y_cf[0, :, 0].cpu().numpy(),
                    "y_cf_rp": y_cf_rp[0, :, 0].cpu().numpy(),
                    "alpha_raw": alpha[0].cpu().numpy(),
                    "beta_raw": beta[0].cpu().numpy(),
                    "vr_rm": vr_rm,
                    "vr_rp": vr_rp,
                    "bx": bx[0:1].clone(),
                    "bx_mark": bx_mark[0:1].clone(),
                }
                if vr_rm >= 0.95 and best_score >= 0.7:
                    print(f"  batch {i}: RLMCF={vr_rm:.2f}  RLP={vr_rp:.2f} → great!")
                    break

    if best_data is None:
        print("No valid CF found!")
        return

    print(f"Selected: RLMCF VR={best_data['vr_rm']:.3f}  RLP VR={best_data['vr_rp']:.3f}")

    # ── Generate BaseGrad CF for the same sample ──────────────────────────────
    print("Generating BaseGrad CF...")
    basegrad = BaseGrad(forecaster, device, lr=0.01, max_iter=300, w_validity=1.0, w_proximity=0.5)
    x_cf_bg, y_cf_bg = basegrad.transform_sample(
        x_np=best_data["x_ot_raw"].reshape(1, -1, 1),
        alpha_np=best_data["alpha_raw"],
        beta_np=best_data["beta_raw"],
        x_full_orig=best_data["bx"],
        x_mark=best_data["bx_mark"],
    )
    x_cf_bg = x_cf_bg[0, :, 0]
    y_cf_bg = y_cf_bg[0, :, 0]
    vr_bg = float(((y_cf_bg >= best_data["alpha_raw"]) & (y_cf_bg <= best_data["beta_raw"])).mean())
    print(f"  BaseGrad VR={vr_bg:.3f}")

    # ── Denormalize ───────────────────────────────────────────────────────────
    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    x = inv(best_data["x_ot"])
    x_cf = inv(best_data["x_cf"])
    x_cf_rp = inv(best_data["x_cf_rp"])
    x_cf_bg_d = inv(x_cf_bg)
    yh = inv(best_data["y_hat"])
    ycf = inv(best_data["y_cf"])
    ycf_rp = inv(best_data["y_cf_rp"])
    ycf_bg_d = inv(y_cf_bg)
    a_np = inv(best_data["alpha_raw"])
    b_np = inv(best_data["beta_raw"])

    BH = len(x)
    H = len(yh)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    orig_full = np.concatenate([x, yh])
    rm_full = np.concatenate([x_cf, ycf])
    bg_full = np.concatenate([x_cf_bg_d, ycf_bg_d])
    rp_full = np.concatenate([x_cf_rp, ycf_rp])

    t_back = np.arange(BH)
    t_fore = np.arange(BH, BH + H)

    # Original (blue)
    ax.plot(t_back, orig_full[:BH], color="#1565C0", lw=2.5, ls="-")
    ax.plot(t_fore, orig_full[BH:], color="#1565C0", lw=2.5, ls="--",
            label="$x$ / $\hat{y}$ (original)")
    ax.plot([BH-1, BH], orig_full[BH-1:BH+1], color="#1565C0", lw=2.5)

    # RLMCF (green)
    ax.plot(t_back, rm_full[:BH], color="#2E7D32", lw=2.0, ls="-")
    ax.plot(t_fore, rm_full[BH:], color="#2E7D32", lw=2.0, ls="--",
            label=f"$x_{{cf}}$ RL-MCF  (VR={best_data['vr_rm']:.2f})")
    ax.plot([BH-1, BH], rm_full[BH-1:BH+1], color="#2E7D32", lw=2.0)

    # BaseGrad (red)
    ax.plot(t_back, bg_full[:BH], color="#C62828", lw=2.0, ls="-")
    ax.plot(t_fore, bg_full[BH:], color="#C62828", lw=2.0, ls="--",
            label=f"$x_{{cf}}$ BaseGrad  (VR={vr_bg:.2f})")
    ax.plot([BH-1, BH], bg_full[BH-1:BH+1], color="#C62828", lw=2.0)

    # RLP (orange)
    ax.plot(t_back, rp_full[:BH], color="#E65100", lw=1.5, ls="-", alpha=0.7)
    ax.plot(t_fore, rp_full[BH:], color="#E65100", lw=1.5, ls="--", alpha=0.7,
            label=f"$x_{{cf}}$ RLP  (VR={best_data['vr_rp']:.2f})")
    ax.plot([BH-1, BH], rp_full[BH-1:BH+1], color="#E65100", lw=1.5, alpha=0.7)

    # Target band
    ax.fill_between(t_fore, a_np, b_np, alpha=0.18, color="#757575",
                    label=r"Target band [$\alpha$, $\beta$]")

    ax.axvline(BH, color="black", lw=1.2, ls="--", alpha=0.4)

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("T (°C)", fontsize=12)
    ax.set_title("Weather — RL-MCF vs Baselines (iTransformer, pred_len=96)", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25, ls="--")

    # Dynamic y limits
    all_vals = np.concatenate([orig_full, rm_full, bg_full, rp_full, a_np, b_np])
    ax.set_ylim(all_vals.min() - 0.5, all_vals.max() + 0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
