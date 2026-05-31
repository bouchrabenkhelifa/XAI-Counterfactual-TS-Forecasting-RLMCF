"""
Test cross-dataset transfer: use ETTh1 RL agent on ETTh2 test set.
Shows if the policy generalizes across datasets.

Usage:
    python scripts/analysis/test_transfer.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_main import build_temporal_mask


def evaluate_agent(agent, ae, forecaster, test_loader, reward_fn, mask_k, ramp_k, device, n_batches=20):
    """Evaluate an agent on a test set and return validity."""
    agent.eval()
    all_vr = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]

            z = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf, _, _ = agent.act_deterministic(z, y_hat)
            x_prop = ae.decode(z_cf)
            mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], 1, mask_k, ramp_k, device)
            x_cf = x_ot + mask * (x_prop - x_ot)
            y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)

            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            vr = ((y_cf[:, :, 0] >= alpha) & (y_cf[:, :, 0] <= beta)).float().mean(dim=1)
            all_vr.extend(vr.cpu().numpy().tolist())

    return float(np.mean(all_vr))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)
    print("CROSS-DATASET TRANSFER TEST")
    print("Agent trained on ETTh1 → evaluated on ETTh2 (iTransformer)")
    print("=" * 60)

    # ── Load ETTh2 components (target dataset) ────────────────────────────
    cfg_f_etth2 = load_config("assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json")
    cfg_ae_etth2 = load_config("assets/configs/etth2_dataset/ae/tcn_ae.json")
    if not hasattr(cfg_f_etth2, "model_type"):
        cfg_f_etth2.model_type = "iTransformer"

    ae_etth2 = TCNAutoEncoder.from_checkpoint(cfg_ae_etth2.checkpoint_path, device=device)
    ae_etth2.eval()

    forecaster_etth2 = ForecasterWrapperV2(cfg_f_etth2, device)
    forecaster_etth2.model.eval()

    _, test_loader_etth2 = data_provider(cfg_f_etth2, "test")

    # Global sigma for ETTh2
    _, train_loader_etth2 = data_provider(cfg_f_etth2, "train")
    _stds = []
    for i, batch in enumerate(train_loader_etth2):
        if i >= 50: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    sigma_etth2 = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(fr=0.5, rho=0.2, direction=-1.0, global_sigma=sigma_etth2).to(device)

    # ── Load ETTh1 agent (source) ────────────────────────────────────────
    print("\n── Loading ETTh1 agent ──")
    agent_etth1 = ActorCritic(latent_dim=64, pred_len=48, eta=0.15, entropy_coef=0.02, direction=-1.0).to(device)
    ckpt = torch.load("assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
                      map_location=device, weights_only=False)
    agent_etth1.actor.load_state_dict(ckpt["actor_state_dict"])
    agent_etth1.critic.load_state_dict(ckpt["critic_state_dict"])

    # ── Load ETTh2 agent (native) ────────────────────────────────────────
    print("── Loading ETTh2 agent (native) ──")
    agent_etth2 = ActorCritic(latent_dim=64, pred_len=48, eta=0.15, entropy_coef=0.02, direction=-1.0).to(device)
    ckpt2 = torch.load("assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
                       map_location=device, weights_only=False)
    agent_etth2.actor.load_state_dict(ckpt2["actor_state_dict"])
    agent_etth2.critic.load_state_dict(ckpt2["critic_state_dict"])

    # ── Evaluate both on ETTh2 test set ───────────────────────────────────
    print("\n── Evaluating on ETTh2 test set ──")

    vr_native = evaluate_agent(agent_etth2, ae_etth2, forecaster_etth2, test_loader_etth2,
                               reward_fn, mask_k=12, ramp_k=4, device=device)
    print(f"  ETTh2 agent (native):     Validity = {vr_native:.4f}")

    vr_transfer = evaluate_agent(agent_etth1, ae_etth2, forecaster_etth2, test_loader_etth2,
                                 reward_fn, mask_k=12, ramp_k=4, device=device)
    print(f"  ETTh1 agent (transfer):   Validity = {vr_transfer:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n── Transfer Result ──")
    print(f"  Native (ETTh2→ETTh2):   {vr_native:.4f}")
    print(f"  Transfer (ETTh1→ETTh2): {vr_transfer:.4f}")
    print(f"  Retention:              {vr_transfer/vr_native*100:.1f}%")

    if vr_transfer > 0.5:
        print("  ✓ Transfer works — policy partially generalizes across datasets")
    else:
        print("  ✗ Transfer fails — policy is dataset-specific")


if __name__ == "__main__":
    main()
