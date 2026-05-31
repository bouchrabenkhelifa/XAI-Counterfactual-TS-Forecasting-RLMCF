"""
t-SNE of Latent Space: Original z vs Counterfactual z_cf — 3 Datasets side by side.
Replicates the style of latent_tsne_orig_vs_cf.png but for ETTh1, ETTh2, Weather.

Usage:
    python scripts/analysis/plot_latent_tsne_3datasets.py

Output:
    assets/figures/global_analysis/latent_tsne_3datasets.png
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.data_provider.data_factory import data_provider

FIGURES_DIR = os.path.join(ROOT, "assets", "figures", "global_analysis")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Dataset configurations ──────────────────────────────────────────────────────
DATASETS = {
    "ETTh1": {
        "cfg_f": "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json",
        "cfg_ae": "assets/configs/etth1_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt",
        "pred_len": 48,
        "eta": 0.15,
        "direction": -1.0,
    },
    "ETTh2": {
        "cfg_f": "assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json",
        "cfg_ae": "assets/configs/etth2_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt",
        "pred_len": 48,
        "eta": 0.15,
        "direction": -1.0,
    },
    "Weather": {
        "cfg_f": "assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json",
        "cfg_ae": "assets/configs/weather_dataset/ae/tcn_ae.json",
        "rl_ckpt": "assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt",
        "pred_len": 96,
        "eta": 0.15,
        "direction": -1.0,
    },
}


def collect_latent_vectors(ds_config, device, max_batches=300, n_samples=500):
    """Collect original, CF, and random-noise latent vectors for one dataset."""

    cfg_f = load_config(ds_config["cfg_f"])
    cfg_ae = load_config(ds_config["cfg_ae"])

    # Load AE
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # Load forecaster
    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()

    # Load RL agent
    agent = ActorCritic(
        latent_dim=64,
        pred_len=ds_config["pred_len"],
        eta=ds_config["eta"],
        entropy_coef=0.02,
        direction=ds_config["direction"],
    ).to(device)

    ckpt = torch.load(ds_config["rl_ckpt"], map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Collect latent vectors from test set
    _, test_loader = data_provider(cfg_f, "test")

    z_orig_list, z_cf_list, z_noise_list = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]

            z = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf, _, _ = agent.act_deterministic(z, y_hat)

            # Random noise baseline
            z_noise = torch.clamp(z + 0.15 * torch.randn_like(z), -1, 1)

            z_orig_list.append(z.cpu().numpy())
            z_cf_list.append(z_cf.cpu().numpy())
            z_noise_list.append(z_noise.cpu().numpy())

    z_orig = np.concatenate(z_orig_list, axis=0)
    z_cf = np.concatenate(z_cf_list, axis=0)
    z_noise = np.concatenate(z_noise_list, axis=0)

    # Limit to n_samples for t-SNE speed
    n = min(n_samples, len(z_orig))
    return z_orig[:n], z_cf[:n], z_noise[:n]


def main():
    print("=" * 60)
    print("  Latent t-SNE: Original vs CF — 3 Datasets")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # ─── Collect data for all datasets ────────────────────────────────────────
    all_data = {}
    for ds_name, ds_config in DATASETS.items():
        print(f"  [{ds_name}] Collecting latent vectors...")
        try:
            z_orig, z_cf, z_noise = collect_latent_vectors(ds_config, device)
            all_data[ds_name] = (z_orig, z_cf, z_noise)
            print(f"  [{ds_name}] OK — {len(z_orig)} samples")
        except Exception as e:
            print(f"  [{ds_name}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_data:
        print("No data collected. Exiting.")
        return

    # ─── Create figure ────────────────────────────────────────────────────────
    n_datasets = len(all_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]

    fig.suptitle(
        "Latent Space Visualization (t-SNE) — Original vs Counterfactual",
        fontsize=14, fontweight="bold", y=0.98,
    )

    colors = {"Original": "#1976D2", "CF (RL-MCF)": "#E53935", "Random perturbation": "#9E9E9E"}
    alphas_map = {"Original": 0.5, "CF (RL-MCF)": 0.5, "Random perturbation": 0.25}
    sizes = {"Original": 20, "CF (RL-MCF)": 20, "Random perturbation": 12}

    for ax, (ds_name, (z_orig, z_cf, z_noise)) in zip(axes, all_data.items()):
        n = len(z_orig)
        z_all = np.vstack([z_orig, z_cf, z_noise])
        labels = np.array(["Original"] * n + ["CF (RL-MCF)"] * n + ["Random perturbation"] * n)

        print(f"  [{ds_name}] Running t-SNE ({3*n} points)...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        z_2d = tsne.fit_transform(z_all)

        # Plot in order: noise first (background), then original, then CF
        for label in ["Random perturbation", "Original", "CF (RL-MCF)"]:
            mask = labels == label
            ax.scatter(
                z_2d[mask, 0], z_2d[mask, 1],
                c=colors[label], alpha=alphas_map[label], s=sizes[label],
                label=label, edgecolors="none",
            )

        ax.set_xlabel("t-SNE dim 1", fontsize=10)
        ax.set_ylabel("t-SNE dim 2", fontsize=10)
        ax.set_title(f"{ds_name} — iTransformer, latent dim=64", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, markerscale=2, loc="best")
        ax.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = os.path.join(FIGURES_DIR, "latent_tsne_3datasets.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {output_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
