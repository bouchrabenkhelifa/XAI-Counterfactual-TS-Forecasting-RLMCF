"""
CF examples — iTransformer across ETTh1 / ETTh2 / Weather
==========================================================
Génère une figure avec 3 subplots côte à côte (un par dataset).
Chaque subplot montre un exemple de counterfactual :
  - original input (bleu)
  - counterfactual input (orange)
  - forecast original ŷ (bleu pointillé)
  - forecast CF  ŷ_cf (vert pointillé)
  - bande alpha/beta (zone verte transparente)

Usage:
    python scripts/analysis/plot_cf_itransformer_3datasets.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.training.RL_trainers.trainer_main import RLMaskTrainer, run_episode_eval
from src.data_provider.data_factory import data_provider

# ── Config par dataset ────────────────────────────────────────────────────────
DATASETS = [
    {
        "label":      "ETTh1",
        "ae_cfg":     "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "rl_cfg":     "assets/configs/etth1_dataset/RL_ablations/config_v2.json",
        "f_cfg":      "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "ckpt":       "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt",
        "ylabel":     "°C",
    },
    {
        "label":      "ETTh2",
        "ae_cfg":     "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "rl_cfg":     "assets/configs/etth2_dataset/RL/config_itransformer.json",
        "f_cfg":      "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
        "ckpt":       "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
        "ylabel":     "°C",
    },
    {
        "label":      "Weather",
        "ae_cfg":     "assets/configs/weather_dataset/ae/tcn_ae.json",
        "rl_cfg":     "assets/configs/weather_dataset/RL/config_itransformer.json",
        "f_cfg":      "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        "ckpt":       "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
        "ylabel":     "°C",
    },
]

OUTPUT_DIR  = "assets/figures/global_analysis"
N_BATCHES   = 5          # batches à parcourir pour trouver un bon exemple
SAMPLE_IDX  = 0          # index dans le batch retenu
DEVICE      = "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_trainer_and_ckpt(ds):
    cfg_rl  = load_config(ds["rl_cfg"])
    cfg_f   = load_config(ds["f_cfg"])
    cfg_ae  = load_config(ds["ae_cfg"])

    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, DEVICE)

    ckpt = torch.load(ds["ckpt"], map_location=DEVICE, weights_only=False)
    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.agent.eval()

    # Charger le scaler depuis le dataset (pas depuis l'AE checkpoint)
    test_dataset, _ = data_provider(cfg_f, "test")
    trainer._plot_scaler = test_dataset.scaler

    print(f"  [{ds['label']}] checkpoint loaded")
    return trainer


def get_cf_example(trainer, n_batches=N_BATCHES, sample_idx=SAMPLE_IDX):
    """
    Parcourt les batches du test_loader et retourne le premier épisode valide.
    Retourne un dict avec les arrays numpy dénormalisés (°C).
    """
    scaler = trainer._plot_scaler

    def denorm(arr):
        """arr shape (T,) → dénormalisé (T,)"""
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    for i, batch in enumerate(trainer.test_loader):
        if i >= n_batches:
            break
        ep = run_episode_eval(
            batch=batch,
            ae_arch=trainer.ae_arch,
            forecaster=trainer.forecaster,
            agent=trainer.agent,
            reward_fn=trainer.reward_fn,
            device=DEVICE,
            use_rl=True,
            mask_last_k=trainer.mask_last_k,
            mask_ramp_k=trainer.mask_ramp_k,
            filter_quantile=trainer.filter_quantile,
        )
        if ep is None:
            continue

        b = min(sample_idx, ep["x_ot"].shape[0] - 1)

        x_np = ep["x_ot"][b:b+1].cpu().numpy()
        y_np = ep["y_hat"][b:b+1].cpu().numpy()
        alphas, betas = trainer._compute_bounds_np(x_np, y_np)

        return {
            "x_orig": denorm(ep["x_ot"][b, :, 0].cpu().numpy()),
            "x_cf":   denorm(ep["x_cf"][b, :, 0].cpu().numpy()),
            "y_hat":  denorm(ep["y_hat"][b, :, 0].cpu().numpy()),
            "y_cf":   denorm(ep["y_cf"][b, :, 0].cpu().numpy()),
            "alpha":  denorm(alphas[0]),
            "beta":   denorm(betas[0]),
        }
    raise RuntimeError("No valid CF episode found in test batches.")


def plot_one(ax, ex, label, ylabel):
    seq_len  = len(ex["x_orig"])
    pred_len = len(ex["y_hat"])
    t_past   = np.arange(seq_len)
    t_fut    = np.arange(seq_len, seq_len + pred_len)

    # ── past ──────────────────────────────────────────────────────────────────
    ax.plot(t_past, ex["x_orig"], color="#2196F3", lw=1.8,
            label="original + forecast", zorder=3)
    ax.plot(t_past, ex["x_cf"],  color="#FF7043", lw=1.8, ls="--",
            label="CF + forecast_CF", zorder=3)
    ax.fill_between(t_past, ex["x_orig"], ex["x_cf"],
                    color="#FF7043", alpha=0.15, zorder=2)

    # ── future ────────────────────────────────────────────────────────────────
    ax.plot(t_fut, ex["y_hat"], color="#2196F3", lw=1.8, zorder=3)
    ax.plot(t_fut, ex["y_cf"],  color="#4CAF50", lw=1.8, ls="--",
            label="CF bound", zorder=3)

    # bande alpha/beta (zone cible)
    ax.fill_between(t_fut, ex["alpha"], ex["beta"],
                    color="#4CAF50", alpha=0.20, zorder=1, label="target band")

    # séparateur passé / futur
    ax.axvline(x=seq_len - 0.5, color="black", ls=":", lw=1.2, alpha=0.6)

    ax.set_title("")
    ax.set_xlabel("Time steps", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25, ls="--")
    ax.tick_params(labelsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ds in DATASETS:
        print(f"\n{'='*50}")
        print(f"  {ds['label']}")
        print(f"{'='*50}")
        trainer = load_trainer_and_ckpt(ds)
        ex      = get_cf_example(trainer)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        plot_one(ax, ex, ds["label"], ds["ylabel"])
        print(f"  validity check: y_cf in [alpha, beta] = "
              f"{np.mean((ex['y_cf'] >= ex['alpha']) & (ex['y_cf'] <= ex['beta'])):.2f}")

        name    = ds["label"].lower()
        out_png = os.path.join(OUTPUT_DIR, f"cf_itransformer_{name}.png")
        fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_pdf}")
        print(f"  Saved -> {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
