"""
Generate two figures for the PFE report:
  1. Heatmap: Datasets × Architectures colored by Validity Ratio
  2. t-SNE of latent space: original z vs counterfactual z_cf

Usage:
    python scripts/analysis/plot_heatmap_and_latent.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

FIGURES_DIR = "assets/figures/global_analysis"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Heatmap Datasets × Architectures (Validity Ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heatmap():
    """Load evaluation results and plot heatmap."""

    # Load results for each dataset
    results_paths = {
        "ETTh1": "assets/results/etth1/summary/etth1_all_models_colab_plaus.json",
        "ETTh2": "assets/results/etth2/summary/etth2_all_models_colab_plaus.json",
        "Weather": "assets/results/weather/summary/weather_all_models_colab_plaus.json",
    }

    models = ["iTransformer", "PatchTST", "DLinear", "TimesNet", "GRU"]
    datasets = list(results_paths.keys())

    # Build validity matrix
    validity_matrix = np.zeros((len(datasets), len(models)))

    for i, (ds, path) in enumerate(results_paths.items()):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for j, model in enumerate(models):
                if model in data:
                    validity_matrix[i, j] = data[model]["validity_ratio"]["mean"]

    # ── Plot heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(validity_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=11)

    # Annotate cells
    for i in range(len(datasets)):
        for j in range(len(models)):
            val = validity_matrix[i, j]
            color = "white" if val < 0.4 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_title("Validity Ratio — Datasets × Forecaster Architectures", fontsize=13)
    plt.colorbar(im, ax=ax, label="Validity Ratio ↑", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "heatmap_validity.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Heatmap -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: t-SNE of Latent Space (z_orig vs z_cf)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_latent_tsne():
    """Generate t-SNE visualization of original vs CF latent vectors."""
    from src.utils.config import load_config
    from src.utils.train_tools import get_device
    from src.models.autoencoder.tcn_ae import TCNAutoEncoder
    from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
    from src.models.RL.agent import ActorCritic
    from src.data_provider.data_factory import data_provider
    from src.training.RL_trainers.trainer_main import build_temporal_mask

    cfg_f = load_config("assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json")
    cfg_ae = load_config("assets/configs/etth1_dataset/ae/tcn_ae.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    if not hasattr(cfg_f, "model_type"):
        cfg_f.model_type = "iTransformer"
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()

    # Load RL agent
    rl_ckpt = "assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt"
    agent = ActorCritic(latent_dim=64, pred_len=48, eta=0.15, entropy_coef=0.02, direction=-1.0).to(device)
    ckpt = torch.load(rl_ckpt, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor_state_dict"])
    agent.critic.load_state_dict(ckpt["critic_state_dict"])
    agent.eval()

    # Collect latent vectors
    _, test_loader = data_provider(cfg_f, "test")

    z_orig_list = []
    z_cf_list = []
    z_noise_list = []

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

            # Random noise baseline
            z_noise = torch.clamp(z + 0.15 * torch.randn_like(z), -1, 1)

            z_orig_list.append(z.cpu().numpy())
            z_cf_list.append(z_cf.cpu().numpy())
            z_noise_list.append(z_noise.cpu().numpy())

    z_orig = np.concatenate(z_orig_list, axis=0)
    z_cf = np.concatenate(z_cf_list, axis=0)
    z_noise = np.concatenate(z_noise_list, axis=0)

    print(f"Collected: {len(z_orig)} original, {len(z_cf)} CF, {len(z_noise)} noise")

    # t-SNE on combined
    n = min(500, len(z_orig))  # Limit for speed
    z_all = np.vstack([z_orig[:n], z_cf[:n], z_noise[:n]])
    labels = np.array(["Original"] * n + ["CF (RL-MCF)"] * n + ["Random perturbation"] * n)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z_all)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = {"Original": "tab:blue", "CF (RL-MCF)": "tab:red", "Random perturbation": "tab:gray"}
    alphas = {"Original": 0.5, "CF (RL-MCF)": 0.5, "Random perturbation": 0.3}
    sizes = {"Original": 20, "CF (RL-MCF)": 20, "Random perturbation": 15}

    for label in ["Random perturbation", "Original", "CF (RL-MCF)"]:
        mask = labels == label
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=colors[label], alpha=alphas[label], s=sizes[label],
                   label=label, edgecolors="none")

    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.set_title("Latent Space Visualization (t-SNE) — Original vs Counterfactual\n"
                 "ETTh1, iTransformer, TCN-AE latent dim=64", fontsize=12)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "latent_tsne_orig_vs_cf.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot] t-SNE -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Heatmap + Latent Space t-SNE")
    print("=" * 60)

    plot_heatmap()
    plot_latent_tsne()

    print("\n[OK] Done!")
