"""
Plot CF examples for all 5 architectures × 3 datasets.
Each dataset = one figure with 5 subplots (one per forecaster).
Shows: original + forecast (blue), CF + forecast_cf (orange), target band (gray).

Usage:
    python scripts/analysis/plot_cf_all_models_all_datasets.py
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

MODELS = ["itransformer", "patchtst", "dlinear", "gru", "timesnet"]
MODEL_LABELS = ["iTransformer", "PatchTST", "DLinear", "GRU", "TimesNet"]

DATASETS = {
    "etth1": {
        "ae_cfg": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "forecaster_cfgs": {
            "itransformer": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
            "patchtst": "assets/configs/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json",
            "dlinear": "assets/configs/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json",
            "gru": "assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json",
            "timesnet": "assets/configs/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json",
        },
        "rl_ckpts": {
            "itransformer": "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
            "patchtst": "assets/checkpoints/etth1_chpts/RL/patchtst/rl_cf_patchtst_etth1_agent_best.pt",
            "dlinear": "assets/checkpoints/etth1_chpts/RL/dlinear/rl_cf_dlinear_etth1_agent_best.pt",
            "gru": "assets/checkpoints/etth1_chpts/RL/gru/rl_cf_gru_etth1_agent_best.pt",
            "timesnet": "assets/checkpoints/etth1_chpts/RL/timesnet/rl_cf_timesnet_etth1_agent_best.pt",
        },
        "rl_cfgs": {
            "itransformer": "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json",
            "patchtst": "assets/configs/etth1_dataset/RL_ablations/config_patchtst.json",
            "dlinear": "assets/configs/etth1_dataset/RL_ablations/config_dlinear.json",
            "gru": "assets/configs/etth1_dataset/RL_ablations/config_gru.json",
            "timesnet": "assets/configs/etth1_dataset/RL_ablations/config_timesnet.json",
        },
        "ylabel": "OT (°C)",
        "title": "ETTh1",
    },
    "etth2": {
        "ae_cfg": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "forecaster_cfgs": {
            "itransformer": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
            "patchtst": "assets/configs/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json",
            "dlinear": "assets/configs/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json",
            "gru": "assets/configs/etth2_dataset/forecasters/gru/etth2_96_48_S.json",
            "timesnet": "assets/configs/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json",
        },
        "rl_ckpts": {
            "itransformer": "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
            "patchtst": "assets/checkpoints/etth2_chpts/RL/patchtst/rl_cf_patchtst_etth2_agent_best.pt",
            "dlinear": "assets/checkpoints/etth2_chpts/RL/dlinear/rl_cf_dlinear_etth2_agent_best.pt",
            "gru": "assets/checkpoints/etth2_chpts/RL/gru/rl_cf_gru_etth2_agent_best.pt",
            "timesnet": "assets/checkpoints/etth2_chpts/RL/timesnet/rl_cf_timesnet_etth2_agent_best.pt",
        },
        "rl_cfgs": {
            "itransformer": "assets/configs/etth2_dataset/RL/config_itransformer.json",
            "patchtst": "assets/configs/etth2_dataset/RL/config_patchtst.json",
            "dlinear": "assets/configs/etth2_dataset/RL/config_dlinear.json",
            "gru": "assets/configs/etth2_dataset/RL/config_gru.json",
            "timesnet": "assets/configs/etth2_dataset/RL/config_timesnet.json",
        },
        "ylabel": "OT (°C)",
        "title": "ETTh2",
    },
    "weather": {
        "ae_cfg": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "forecaster_cfgs": {
            "itransformer": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
            "patchtst": "assets/configs/weather_dataset/forecasters/patchtst/weather_96_96_S.json",
            "dlinear": "assets/configs/weather_dataset/forecasters/dlinear/weather_96_96_S.json",
            "gru": "assets/configs/weather_dataset/forecasters/gru/weather_96_96_S.json",
            "timesnet": "assets/configs/weather_dataset/forecasters/timesnet/weather_96_96_S.json",
        },
        "rl_ckpts": {
            "itransformer": "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
            "patchtst": "assets/checkpoints/weather_chpts/RL/RL_patchtst/rl_cf_patchtst_weather_agent_best.pt",
            "dlinear": "assets/checkpoints/weather_chpts/RL/RL_dlinear/rl_cf_dlinear_weather_agent_best.pt",
            "gru": "assets/checkpoints/weather_chpts/RL/RL_gru/rl_cf_gru_weather_agent_best.pt",
            "timesnet": "assets/checkpoints/weather_chpts/RL/RL_timesnet/rl_cf_timesnet_weather_agent_best.pt",
        },
        "rl_cfgs": {
            "itransformer": "assets/configs/weather_dataset/RL/config_itransformer.json",
            "patchtst": "assets/configs/weather_dataset/RL/config_patchtst.json",
            "dlinear": "assets/configs/weather_dataset/RL/config_dlinear.json",
            "gru": "assets/configs/weather_dataset/RL/config_gru.json",
            "timesnet": "assets/configs/weather_dataset/RL/config_timesnet.json",
        },
        "ylabel": "T (°C)",
        "title": "Weather",
    },
}


def generate_one_cf(ds_info, model_name, device):
    """Generate one CF example for a (dataset, model) pair."""
    cfg_f_path = ds_info["forecaster_cfgs"][model_name]
    cfg_ae_path = ds_info["ae_cfg"]
    rl_ckpt_path = ds_info["rl_ckpts"][model_name]
    rl_cfg_path = ds_info["rl_cfgs"][model_name]

    if not os.path.exists(rl_ckpt_path):
        print(f"    ⚠ Checkpoint not found: {rl_ckpt_path}")
        return None

    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    cfg_rl = load_config(rl_cfg_path)
    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"

    # Load AE
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()

    # Load forecaster
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()

    # Load agent
    agent = ActorCritic(
        latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
        eta=cfg_rl.eta, entropy_coef=0.02, direction=-1.0,
    ).to(device)
    ckpt = torch.load(rl_ckpt_path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Global sigma
    _, train_loader = data_provider(cfg_f, "train")
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 30: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(fr=cfg_rl.fr, rho=cfg_rl.rho, direction=-1.0, global_sigma=global_sigma).to(device)

    # Test data
    test_data, test_loader = data_provider(cfg_f, "test")
    scaler = test_data.scaler
    lk = cfg_rl.mask_last_k
    rk = cfg_rl.mask_ramp_k

    # Find good sample
    best_score = -1
    best_result = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 200: break
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
            var = float(x_ot[0, :, 0].std())
            score = vr * 0.7 + min(var / 0.4, 1.0) * 0.3

            if score > best_score and vr >= 0.7:
                best_score = score
                def inv(arr):
                    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()
                best_result = {
                    "x": inv(x_ot[0, :, 0].cpu().numpy()),
                    "x_cf": inv(x_cf[0, :, 0].cpu().numpy()),
                    "y_hat": inv(y_hat[0, :, 0].cpu().numpy()),
                    "y_cf": inv(y_cf[0, :, 0].cpu().numpy()),
                    "alpha": inv(alpha[0].cpu().numpy()),
                    "beta": inv(beta[0].cpu().numpy()),
                    "vr": vr,
                    "seq_len": cfg_f.seq_len,
                    "pred_len": cfg_f.pred_len,
                }

    return best_result


def plot_dataset(ds_name, ds_info, device):
    """Generate a 5-subplot figure for one dataset."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))

    for col, (model_name, model_label) in enumerate(zip(MODELS, MODEL_LABELS)):
        ax = axes[col]
        print(f"  {model_label}...")

        result = generate_one_cf(ds_info, model_name, device)

        if result is None:
            ax.set_title(f"{model_label}\n(no checkpoint)")
            ax.grid(alpha=0.2)
            continue

        seq_len = result["seq_len"]
        pred_len = result["pred_len"]
        t_back = np.arange(seq_len)
        t_fore = np.arange(seq_len, seq_len + pred_len)

        # Original
        ax.plot(t_back, result["x"], color="#1565C0", lw=1.5)
        ax.plot(t_fore, result["y_hat"], color="#1565C0", lw=1.5, ls="--")

        # CF
        ax.plot(t_back, result["x_cf"], color="#E65100", lw=1.5)
        ax.plot(t_fore, result["y_cf"], color="#E65100", lw=1.5, ls="--")

        # Band
        ax.fill_between(t_fore, result["alpha"], result["beta"], alpha=0.2, color="gray")

        ax.axvline(seq_len, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{model_label}\nVR={result['vr']:.2f}", fontsize=10)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel(ds_info["ylabel"])

    plt.suptitle(f"RL-MCF Counterfactual Examples — {ds_info['title']}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"cf_5models_{ds_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] → {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for ds_name, ds_info in DATASETS.items():
        print(f"{'='*50}")
        print(f"Dataset: {ds_info['title']}")
        print(f"{'='*50}")
        plot_dataset(ds_name, ds_info, device)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
