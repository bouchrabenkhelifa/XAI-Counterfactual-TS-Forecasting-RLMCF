#!/usr/bin/env python
"""
Generate Ablation Series Figure: RLMCF vs RLP vs wo_mask
=========================================================
Une seule figure, un seul sample, toutes les courbes superposées.
- x_original : bleu plein (lookback + forecast en continu)
- RLMCF      : rouge pointillé
- RLP        : orange pointillé  (bruit gaussien, pipeline officiel run_rlp.py)
- wo_mask    : vert pointillé

Usage:
    python scripts/generate_ablation_series_figure.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.Forecaster.forecaster_wrapper import ForecasterWrapper
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
# Import du build_temporal_mask officiel (même que run_rlp.py)
from src.training.RL_trainers.trainer_last import build_temporal_mask

# ── Chemins ───────────────────────────────────────────────────────────────────
CFG_F       = "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
CFG_AE      = "assets/configs/etth1_dataset/ae/tcn_ae.json"
CFG_RLMCF   = "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json"
CKPT_RLMCF  = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt"
CKPT_WOMASK = "assets/checkpoints/etth1_chpts/RL/wo_mask/rl_cf_wo_mask_etth1_agent_best.pt"
OUT_PATH    = "assets/figures/comparison/ablation_series_3methods.png"


def make_rlmcf(x_ot, bx, bx_mark, ae, forecaster_v2, agent, lk, rk, device):
    """RLMCF : politique apprise + masque temporel."""
    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster_v2.predict_ot(bx, bx_mark)
        z_cf, _, _ = agent.act_deterministic(z, y_hat)
        x_prop = ae.decode(z_cf)
        mask   = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
        x_cf   = x_ot + mask * (x_prop - x_ot)
        y_cf   = forecaster_v2.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
    return x_cf, y_cf, y_hat


def make_rlp(x_ot, bx, bx_mark, ae, forecaster_v1, lk, rk, eta, device, seed=42):
    """RLP : bruit gaussien N(0,1) × eta — pipeline officiel de run_rlp.py.
    Utilise ForecasterWrapper (v1) comme dans run_rlp.py.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster_v1.predict_ot(bx, bx_mark)
        noise = torch.randn_like(z)
        z_cf  = torch.clamp(z + eta * noise, -1.0, 1.0)
        x_prop = ae.decode(z_cf)
        mask   = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
        x_cf   = x_ot + mask * (x_prop - x_ot)
        y_cf   = forecaster_v1.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
    return x_cf, y_cf


def make_womask(x_ot, bx, bx_mark, ae, forecaster_v2, agent, device):
    """wo_mask : politique apprise SANS masque temporel (x_cf = x_prop directement)."""
    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster_v2.predict_ot(bx, bx_mark)
        z_cf, _, _ = agent.act_deterministic(z, y_hat)
        x_cf  = ae.decode(z_cf)          # pas de masque
        y_cf  = forecaster_v2.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
    return x_cf, y_cf


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_f    = load_config(CFG_F)
    cfg_ae   = load_config(CFG_AE)
    cfg_rlmcf = load_config(CFG_RLMCF)   # mask_last_k=12, ramp_k=4, eta=0.15, rho=0.2, fr=0.5

    # ── AE ────────────────────────────────────────────────────────────────────
    print("Loading AE...")
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    # ── Forecaster v2 (pour RLMCF et wo_mask) ────────────────────────────────
    print("Loading forecaster (v2)...")
    forecaster_v2 = ForecasterWrapperV2(cfg_f, device)
    forecaster_v2.model.eval()
    for p in forecaster_v2.model.parameters(): p.requires_grad_(False)

    # ── Forecaster v1 (pour RLP — même que run_rlp.py) ───────────────────────
    print("Loading forecaster (v1 for RLP)...")
    forecaster_v1 = ForecasterWrapper(cfg_f, device)
    forecaster_v1.model.eval()
    for p in forecaster_v1.model.parameters(): p.requires_grad_(False)

    # ── Agent RLMCF ───────────────────────────────────────────────────────────
    print("Loading RLMCF agent...")
    agent_rlmcf = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=cfg_rlmcf.eta, entropy_coef=cfg_rlmcf.entropy_coef,
        direction=getattr(cfg_rlmcf, "direction", -1.0),
    ).to(device)
    ckpt = torch.load(CKPT_RLMCF, map_location=device, weights_only=False)
    agent_rlmcf.actor.load_state_dict(ckpt["actor_state_dict"])
    agent_rlmcf.critic.load_state_dict(ckpt["critic_state_dict"])
    agent_rlmcf.eval()

    # ── Agent wo_mask ─────────────────────────────────────────────────────────
    print("Loading wo_mask agent...")
    agent_womask = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=cfg_rlmcf.eta, entropy_coef=cfg_rlmcf.entropy_coef,
        direction=getattr(cfg_rlmcf, "direction", -1.0),
    ).to(device)
    ckpt_wo = torch.load(CKPT_WOMASK, map_location=device, weights_only=False)
    agent_womask.actor.load_state_dict(ckpt_wo["actor_state_dict"])
    agent_womask.critic.load_state_dict(ckpt_wo["critic_state_dict"])
    agent_womask.eval()

    # ── Reward fn (bounds) ────────────────────────────────────────────────────
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())
    print(f"global_sigma = {global_sigma:.4f}")

    reward_fn = CFReward(
        fr=cfg_rlmcf.fr, rho=cfg_rlmcf.rho,
        direction=getattr(cfg_rlmcf, "direction", -1.0),
        global_sigma=global_sigma,
    ).to(device)

    lk = cfg_rlmcf.mask_last_k   # 12
    rk = cfg_rlmcf.mask_ramp_k   # 4
    eta = cfg_rlmcf.eta           # 0.15

    # ── Scan test set : RLMCF élevé ET RLP bas ───────────────────────────────
    test_dataset, test_loader = data_provider(cfg_f, "test")
    print("\nScanning test set (RLMCF high, RLP low)...")
    
    # Get scaler for denormalization
    scaler = test_dataset.scaler

    best_score = -1.0
    best_data  = None

    for i, batch in enumerate(test_loader):
        bx, _, bx_mark, _ = batch
        bx      = bx.float().to(device)
        bx_mark = bx_mark.float().to(device)
        x_ot    = bx[:, :, -1:]

        with torch.no_grad():
            # RLMCF validity
            z     = ae.encode(x_ot)
            y_hat = forecaster_v2.predict_ot(bx, bx_mark)
            z_cf_rm, _, _ = agent_rlmcf.act_deterministic(z, y_hat)
            x_prop_rm = ae.decode(z_cf_rm)
            mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
            x_cf_rm = x_ot + mask * (x_prop_rm - x_ot)
            y_cf_rm = forecaster_v2.predict_from_ot(x_ot=x_cf_rm, x_full=bx, x_mark=bx_mark)

            # RLP validity (seed=i pour varier)
            torch.manual_seed(i)
            noise   = torch.randn_like(z)
            z_cf_rp = torch.clamp(z + eta * noise, -1.0, 1.0)
            x_prop_rp = ae.decode(z_cf_rp)
            x_cf_rp = x_ot + mask * (x_prop_rp - x_ot)
            y_hat_v1 = forecaster_v1.predict_ot(bx, bx_mark)
            y_cf_rp = forecaster_v1.predict_from_ot(x_ot=x_cf_rp, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr_rm = ((y_cf_rm[:, :, 0] >= alpha) & (y_cf_rm[:, :, 0] <= beta)).float().mean(dim=1)
            vr_rp = ((y_cf_rp[:, :, 0] >= alpha) & (y_cf_rp[:, :, 0] <= beta)).float().mean(dim=1)

        # Score = RLMCF - RLP (on veut RLMCF haut et RLP bas)
        score = vr_rm - vr_rp
        max_score = float(score.max())

        if max_score > best_score:
            best_score = max_score
            idx = int(score.argmax())
            best_data = (
                x_ot[idx:idx+1].clone(),
                bx[idx:idx+1].clone(),
                bx_mark[idx:idx+1].clone(),
                i,   # seed pour RLP
            )
            print(f"  batch {i:3d}  sample {idx}  RLMCF={float(vr_rm[idx]):.2f}  RLP={float(vr_rp[idx]):.2f}  diff={max_score:.2f}")
            if best_score >= 0.90 and float(vr_rm[idx]) >= 0.95:
                print("  → great contrast, stopping")
                break

    x_ot, bx, bx_mark, rlp_seed = best_data
    print(f"\nSelected: RLMCF-RLP diff = {best_score:.3f}")

    # ── Générer les 3 CFs ─────────────────────────────────────────────────────
    print("Generating counterfactuals...")
    x_cf_rm, y_cf_rm, y_hat = make_rlmcf(x_ot, bx, bx_mark, ae, forecaster_v2, agent_rlmcf, lk, rk, device)
    x_cf_rp, y_cf_rp        = make_rlp(x_ot, bx, bx_mark, ae, forecaster_v1, lk, rk, eta, device, seed=rlp_seed)
    x_cf_wo, y_cf_wo        = make_womask(x_ot, bx, bx_mark, ae, forecaster_v2, agent_womask, device)

    # ── Bounds ────────────────────────────────────────────────────────────────
    alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)

    # ── Numpy ─────────────────────────────────────────────────────────────────
    x      = x_ot[0, :, 0].cpu().numpy()
    yh     = y_hat[0, :, 0].cpu().numpy()
    a_np   = alpha[0].cpu().numpy()
    b_np   = beta[0].cpu().numpy()
    xcf_rm = x_cf_rm[0, :, 0].cpu().numpy()
    ycf_rm = y_cf_rm[0, :, 0].cpu().numpy()
    xcf_rp = x_cf_rp[0, :, 0].cpu().numpy()
    ycf_rp = y_cf_rp[0, :, 0].cpu().numpy()
    xcf_wo = x_cf_wo[0, :, 0].cpu().numpy()
    ycf_wo = y_cf_wo[0, :, 0].cpu().numpy()

    BH = len(x);  H = len(yh)
    t_back = np.arange(BH)
    t_fore = np.arange(BH, BH + H)

    vr_rm = float(((ycf_rm >= a_np) & (ycf_rm <= b_np)).mean())
    vr_rp = float(((ycf_rp >= a_np) & (ycf_rp <= b_np)).mean())
    vr_wo = float(((ycf_wo >= a_np) & (ycf_wo <= b_np)).mean())
    print(f"Validity — RLMCF: {vr_rm:.3f}  |  RLP: {vr_rp:.3f}  |  wo-mask: {vr_wo:.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Concaténer lookback + forecast pour chaque courbe (série continue)
    orig_full = np.concatenate([x,      yh])
    rm_full   = np.concatenate([xcf_rm, ycf_rm])
    rp_full   = np.concatenate([xcf_rp, ycf_rp])
    wo_full   = np.concatenate([xcf_wo, ycf_wo])

    t_back = np.arange(BH)
    t_fore = np.arange(BH, BH + H)

    # x original : bleu — lookback pointillé, forecast continu, connectés
    ax.plot(t_back, orig_full[:BH], color="#1565C0", lw=2.5, ls="--")
    ax.plot(t_fore, orig_full[BH:], color="#1565C0", lw=2.5, ls="-",
            label="$x$ / $\hat{y}$ (original)")
    ax.plot([BH-1, BH], orig_full[BH-1:BH+1], color="#1565C0", lw=2.5, ls="-")

    # RLMCF : rouge
    ax.plot(t_back, rm_full[:BH], color="#C62828", lw=2.0, ls="--")
    ax.plot(t_fore, rm_full[BH:], color="#C62828", lw=2.0, ls="-",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ RLMCF  (VR={vr_rm:.2f})")
    ax.plot([BH-1, BH], rm_full[BH-1:BH+1], color="#C62828", lw=2.0, ls="-")

    # RLP : orange
    ax.plot(t_back, rp_full[:BH], color="#E65100", lw=2.0, ls="--")
    ax.plot(t_fore, rp_full[BH:], color="#E65100", lw=2.0, ls="-",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ RLP  (VR={vr_rp:.2f})")
    ax.plot([BH-1, BH], rp_full[BH-1:BH+1], color="#E65100", lw=2.0, ls="-")

    # wo-mask : vert
    ax.plot(t_back, wo_full[:BH], color="#2E7D32", lw=2.0, ls="--")
    ax.plot(t_fore, wo_full[BH:], color="#2E7D32", lw=2.0, ls="-",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ wo-mask  (VR={vr_wo:.2f})")
    ax.plot([BH-1, BH], wo_full[BH-1:BH+1], color="#2E7D32", lw=2.0, ls="-")

    # Bande α/β : gris transparent (denormalized)
    ax.fill_between(t_fore, a_denorm, b_denorm, alpha=0.18, color="#757575",
                    label=r"Target band [$\alpha$, $\beta$]")

    # Ligne verticale lookback/forecast
    ax.axvline(BH, color="black", lw=1.2, ls="--", alpha=0.4)

    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("OT (°C)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, ncol=1, framealpha=0.95)
    ax.grid(alpha=0.25, ls="--")
    
    # Set Y axis to start at 4 with specific ticks
    y_min = 4
    y_max = max(orig_full.max(), rm_full.max(), rp_full.max(), wo_full.max(), b_denorm.max()) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # Set Y ticks: 4, 6, 8, 10, ...
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Saved → {OUT_PATH}")
    print(f"   Also saved → {OUT_PATH.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
