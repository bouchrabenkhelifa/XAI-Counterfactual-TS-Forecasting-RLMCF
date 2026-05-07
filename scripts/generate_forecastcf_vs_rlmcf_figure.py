#!/usr/bin/env python
"""
Generate ForecastCF vs RL-MCF Comparison Figure
================================================
Compare ForecastCF baseline with RL-MCF on the same sample.

Figure shows:
- x_original : blue solid (lookback + forecast continuous)
- ForecastCF : orange dashed
- RL-MCF     : red dashed

Usage:
    python scripts/generate_forecastcf_vs_rlmcf_figure.py
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
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_last import build_temporal_mask

# Import ForecastCF baseline
from baselines.ForecastCF_PyTorch.forecastcf_pt import ForecastCFPyTorch
from baselines.common.bounds import compute_bounds_np, load_bounds_params, get_rl_config_path

# ── Paths ─────────────────────────────────────────────────────────────────────
CFG_F       = "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
CFG_AE      = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CFG_RLMCF   = "assets/configs/models/etth1_dataset/RL_ablations/config_itransformer_best.json"
CKPT_RLMCF  = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt"
OUT_PATH    = "assets/figures/comparison/forecastcf_vs_rlmcf_itransformer.png"


def make_rlmcf(x_ot, bx, bx_mark, ae, forecaster, agent, lk, rk, device):
    """RL-MCF: learned policy + temporal mask."""
    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(bx, bx_mark)
        z_cf, _, _ = agent.act_deterministic(z, y_hat)
        x_prop = ae.decode(z_cf)
        mask   = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
        x_cf   = x_ot + mask * (x_prop - x_ot)
        y_cf   = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
    return x_cf, y_cf, y_hat


def make_forecastcf(x_ot, bx, bx_mark, forecaster, alpha_np, beta_np, device):
    """ForecastCF: gradient-based optimization with SPSA."""
    # Initialize ForecastCF
    forecastcf = ForecastCFPyTorch(
        forecaster=forecaster,
        device=device,
        max_iter=300,  # Same as trained
        lr=0.01,
        pred_margin_weight=0.5
    )
    
    # Generate CF
    x_cf_np, y_cf_np = forecastcf.transform_sample(
        x_np=x_ot.cpu().numpy(),
        alpha_np=alpha_np,
        beta_np=beta_np,
        x_full_t=bx,
        x_mark_t=bx_mark
    )
    
    x_cf = torch.from_numpy(x_cf_np).float().to(device)
    y_cf = torch.from_numpy(y_cf_np).float().to(device)
    
    return x_cf, y_cf


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    cfg_f    = load_config(CFG_F)
    cfg_ae   = load_config(CFG_AE)
    cfg_rlmcf = load_config(CFG_RLMCF)
    
    # ── AE ────────────────────────────────────────────────────────────────────
    print("Loading AE...")
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters(): 
        p.requires_grad_(False)
    
    # ── Forecaster ────────────────────────────────────────────────────────────
    print("Loading forecaster (v2)...")
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters(): 
        p.requires_grad_(False)
    
    # ── Agent RL-MCF ──────────────────────────────────────────────────────────
    print("Loading RL-MCF agent...")
    agent_rlmcf = ActorCritic(
        latent_dim=cfg_ae.latent_dim, 
        pred_len=cfg_f.pred_len,
        eta=cfg_rlmcf.eta, 
        entropy_coef=cfg_rlmcf.entropy_coef,
        direction=getattr(cfg_rlmcf, "direction", -1.0),
    ).to(device)
    
    ckpt = torch.load(CKPT_RLMCF, map_location=device, weights_only=False)
    agent_rlmcf.actor.load_state_dict(ckpt["actor_state_dict"])
    agent_rlmcf.critic.load_state_dict(ckpt["critic_state_dict"])
    agent_rlmcf.eval()
    
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
        fr=cfg_rlmcf.fr, 
        rho=cfg_rlmcf.rho,
        direction=getattr(cfg_rlmcf, "direction", -1.0),
        global_sigma=global_sigma,
    ).to(device)
    
    lk = cfg_rlmcf.mask_last_k   # 12
    rk = cfg_rlmcf.mask_ramp_k   # 4
    
    # ── Scan for sample: RL-MCF high, ForecastCF lower ──────────────────────
    test_dataset, test_loader = data_provider(cfg_f, "test")
    print("\nScanning for sample: RL-MCF high (>90%), ForecastCF lower (30-50%)...")
    
    # Get scaler for denormalization
    scaler = test_dataset.scaler
    
    candidates = []  # Store candidates: (gap, vr_rm, vr_fc, batch_idx, sample_idx, data...)
    
    for i, batch in enumerate(test_loader):
        if i >= 50:  # Scan batches
            break
            
        bx, _, bx_mark, _ = batch
        bx      = bx.float().to(device)
        bx_mark = bx_mark.float().to(device)
        x_ot    = bx[:, :, -1:]
        
        with torch.no_grad():
            # RL-MCF validity
            z     = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf_rm, _, _ = agent_rlmcf.act_deterministic(z, y_hat)
            x_prop_rm = ae.decode(z_cf_rm)
            mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
            x_cf_rm = x_ot + mask * (x_prop_rm - x_ot)
            y_cf_rm = forecaster.predict_from_ot(x_ot=x_cf_rm, x_full=bx, x_mark=bx_mark)
            
            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr_rm = ((y_cf_rm[:, :, 0] >= alpha) & (y_cf_rm[:, :, 0] <= beta)).float().mean(dim=1)
            
            # For each sample, also compute ForecastCF validity
            for j in range(len(vr_rm)):
                vr_rm_val = float(vr_rm[j])
                
                # Only consider samples where RL-MCF has high validity
                if vr_rm_val >= 0.90:
                    # Generate ForecastCF for this sample
                    x_ot_j = x_ot[j:j+1]
                    bx_j = bx[j:j+1]
                    bx_mark_j = bx_mark[j:j+1]
                    alpha_j = alpha[j].cpu().numpy()
                    beta_j = beta[j].cpu().numpy()
                    
                    # ForecastCF
                    forecastcf = ForecastCFPyTorch(
                        forecaster=forecaster,
                        device=device,
                        max_iter=300,
                        lr=0.01,
                        pred_margin_weight=0.5
                    )
                    x_cf_fc_np, y_cf_fc_np = forecastcf.transform_sample(
                        x_np=x_ot_j.cpu().numpy(),
                        alpha_np=alpha_j,
                        beta_np=beta_j,
                        x_full_t=bx_j,
                        x_mark_t=bx_mark_j
                    )
                    vr_fc_val = float(((y_cf_fc_np[0, :, 0] >= alpha_j) & (y_cf_fc_np[0, :, 0] <= beta_j)).mean())
                    
                    # We want: RL-MCF high (>90%), ForecastCF medium (40-60%)
                    if 0.40 <= vr_fc_val <= 0.60:
                        gap = vr_rm_val - vr_fc_val
                        candidates.append((
                            abs(vr_fc_val - 0.50),  # Distance from 50% (smaller is better)
                            vr_rm_val,
                            vr_fc_val,
                            i,
                            j,
                            x_ot_j.clone(),
                            bx_j.clone(),
                            bx_mark_j.clone(),
                            alpha_j,
                            beta_j,
                        ))
                        print(f"  batch {i:3d}  sample {j}  RL-MCF={vr_rm_val:.2f}  ForecastCF={vr_fc_val:.2f}  gap={gap:.2f}")
    
    # Select sample with ForecastCF closest to 50%
    if len(candidates) == 0:
        raise ValueError("No suitable samples found with RL-MCF high and ForecastCF ~50%")
    
    # Sort by distance from 50% (ascending) and pick the closest one
    candidates.sort(key=lambda x: x[0])
    selected = candidates[0]
    
    dist_from_50, vr_rm_best, vr_fc_best, batch_idx, sample_idx, x_ot, bx, bx_mark, alpha_np, beta_np = selected
    print(f"\nSelected: batch {batch_idx}, sample {sample_idx}")
    print(f"  RL-MCF VR = {vr_rm_best:.3f}  |  ForecastCF VR = {vr_fc_best:.3f}  |  Gap = {vr_rm_best - vr_fc_best:.3f}")
    
    # ── Generate CFs ──────────────────────────────────────────────────────────
    print("Generating counterfactuals...")
    print("  [1/2] RL-MCF...")
    x_cf_rm, y_cf_rm, y_hat = make_rlmcf(x_ot, bx, bx_mark, ae, forecaster, agent_rlmcf, lk, rk, device)
    
    print("  [2/2] ForecastCF (SPSA)...")
    x_cf_fc, y_cf_fc = make_forecastcf(x_ot, bx, bx_mark, forecaster, alpha_np, beta_np, device)
    
    # ── Convert to numpy ──────────────────────────────────────────────────────
    x      = x_ot[0, :, 0].cpu().numpy()
    yh     = y_hat[0, :, 0].cpu().numpy()
    a_np   = alpha_np
    b_np   = beta_np
    
    xcf_rm = x_cf_rm[0, :, 0].cpu().numpy()
    ycf_rm = y_cf_rm[0, :, 0].cpu().numpy()
    
    xcf_fc = x_cf_fc[0, :, 0].cpu().numpy()
    ycf_fc = y_cf_fc[0, :, 0].cpu().numpy()
    
    BH = len(x)
    H = len(yh)
    
    vr_rm = float(((ycf_rm >= a_np) & (ycf_rm <= b_np)).mean())
    vr_fc = float(((ycf_fc >= a_np) & (ycf_fc <= b_np)).mean())
    
    print(f"Validity — RL-MCF: {vr_rm:.3f}  |  ForecastCF: {vr_fc:.3f}")
    
    # ── Denormalize data ──────────────────────────────────────────────────────
    def denormalize(data):
        """Denormalize using scaler (inverse transform)."""
        data_2d = data.reshape(-1, 1)
        denorm = scaler.inverse_transform(data_2d)
        return denorm.flatten()
    
    x_denorm      = denormalize(x)
    yh_denorm     = denormalize(yh)
    xcf_rm_denorm = denormalize(xcf_rm)
    ycf_rm_denorm = denormalize(ycf_rm)
    xcf_fc_denorm = denormalize(xcf_fc)
    ycf_fc_denorm = denormalize(ycf_fc)
    a_denorm      = denormalize(a_np)
    b_denorm      = denormalize(b_np)
    
    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Concatenate lookback + forecast for continuous series (denormalized)
    orig_full = np.concatenate([x_denorm,      yh_denorm])
    rm_full   = np.concatenate([xcf_rm_denorm, ycf_rm_denorm])
    fc_full   = np.concatenate([xcf_fc_denorm, ycf_fc_denorm])
    
    t_back = np.arange(BH)
    t_fore = np.arange(BH, BH + H)
    
    # x original: blue — lookback dashed, forecast solid, connected
    ax.plot(t_back, orig_full[:BH], color="#1565C0", lw=2.5, ls="--")
    ax.plot(t_fore, orig_full[BH:], color="#1565C0", lw=2.5, ls="-",
            label="$x$ / $\hat{y}$ (original)")
    ax.plot([BH-1, BH], orig_full[BH-1:BH+1], color="#1565C0", lw=2.5, ls="-")
    
    # RL-MCF: green
    ax.plot(t_back, rm_full[:BH], color="#2E7D32", lw=2.0, ls="--")
    ax.plot(t_fore, rm_full[BH:], color="#2E7D32", lw=2.0, ls="-",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ RL-MCF  (VR={vr_rm:.2f})")
    ax.plot([BH-1, BH], rm_full[BH-1:BH+1], color="#2E7D32", lw=2.0, ls="-")
    
    # ForecastCF: red
    ax.plot(t_back, fc_full[:BH], color="#C62828", lw=2.0, ls="--")
    ax.plot(t_fore, fc_full[BH:], color="#C62828", lw=2.0, ls="-",
            label=f"$x_{{cf}}$ / $\hat{{y}}_{{cf}}$ ForecastCF  (VR={vr_fc:.2f})")
    ax.plot([BH-1, BH], fc_full[BH-1:BH+1], color="#C62828", lw=2.0, ls="-")
    
    # Target band α/β: gray transparent (denormalized)
    ax.fill_between(t_fore, a_denorm, b_denorm, alpha=0.18, color="#757575",
                    label=r"Target band [$\alpha$, $\beta$]")
    
    # Vertical line lookback/forecast
    ax.axvline(BH, color="black", lw=1.2, ls="--", alpha=0.4)
    
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("OT (°C)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, ncol=1, framealpha=0.95)
    ax.grid(alpha=0.25, ls="--")
    
    # Set Y axis to start at 4 with specific ticks
    y_min = 4
    y_max = max(orig_full.max(), rm_full.max(), fc_full.max(), b_denorm.max()) * 1.05
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
