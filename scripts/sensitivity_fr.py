#!/usr/bin/env python
"""
Sensitivity Analysis: fr vs Validity Ratio (iTransformer, ETTh1)
fr in [0.40, 0.42, ..., 0.60] — same checkpoint, only fr changes in reward fn.

Usage:
    python scripts/sensitivity_fr.py
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
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

CFG_F   = "assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
CFG_AE  = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CFG_RL  = "assets/configs/models/etth1_dataset/RL_ablations/config_v2.json"
CKPT    = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt"
OUT     = "assets/figures/comparison/sensitivity_fr_vr.png"
FR_LIST = [round(v, 2) for v in np.arange(0.40, 0.62, 0.02)]
N_BATCHES = 20

def main():
    device = torch.device("cpu")
    print(f"fr values: {FR_LIST}")

    cfg_f  = load_config(CFG_F)
    cfg_ae = load_config(CFG_AE)
    cfg_rl = load_config(CFG_RL)

    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()
    for p in forecaster.model.parameters(): p.requires_grad_(False)

    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=cfg_rl.eta, entropy_coef=cfg_rl.entropy_coef,
        direction=getattr(cfg_rl, "direction", -1.0),
    ).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    _, train_loader = data_provider(cfg_f, "train")
    _, test_loader  = data_provider(cfg_f, "test")

    # global_sigma (calculé une seule fois)
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())
    print(f"global_sigma = {global_sigma:.4f}\n")

    lk = cfg_rl.mask_last_k
    rk = cfg_rl.mask_ramp_k

    results = {}
    for fr in FR_LIST:
        reward_fn = CFReward(
            fr=fr, rho=cfg_rl.rho,
            direction=getattr(cfg_rl, "direction", -1.0),
            global_sigma=global_sigma,
        ).to(device)

        all_vr = []
        for i, batch in enumerate(test_loader):
            if i >= N_BATCHES: break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device); bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]
            with torch.no_grad():
                z     = ae.encode(x_ot)
                y_hat = forecaster.predict_ot(bx, bx_mark)
                z_cf, _, _ = agent.act_deterministic(z, y_hat)
                x_prop = ae.decode(z_cf)
                mask   = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], lk, rk, device)
                x_cf   = x_ot + mask * (x_prop - x_ot)
                y_cf   = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
                alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
                vr = ((y_cf[:, :, 0] >= alpha) & (y_cf[:, :, 0] <= beta)).float().mean(dim=1)
                all_vr.extend(vr.cpu().numpy().tolist())

        mean_vr = float(np.mean(all_vr))
        results[fr] = mean_vr
        print(f"  fr={fr:.2f}  VR={mean_vr:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fr_arr = list(results.keys())
    vr_arr = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fr_arr, vr_arr, color="#C62828", lw=2.5, marker="o",
            markersize=8, markerfacecolor="white", markeredgewidth=2.5)
    for fr, vr in zip(fr_arr, vr_arr):
        ax.annotate(f"{vr:.3f}", (fr, vr),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#C62828")
    ax.axvline(0.6, color="gray", lw=1.2, ls="--", alpha=0.6, label="$fr=0.6$ (paper)")
    ax.set_xlabel("$fr$ (band width parameter)", fontsize=12)
    ax.set_ylabel("Validity Ratio ↑", fontsize=12)
    ax.set_xlim(0.38, 0.62); ax.set_ylim(0, 1.05)
    ax.set_xticks(fr_arr)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, ls="--")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved → {OUT}")

if __name__ == "__main__":
    main()
