import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_provider.data_factory import data_provider
from src.models.autoencoder.ae_wrapper import build_autoencoder
from src.models.autoencoder.conv_ae_wrapper import build_conv_autoencoder
from src.utils.config import load_config
from src.utils.train_tools import get_device


@torch.no_grad()
def load_ae_and_data(config_path: str):

    cfg = load_config(config_path)
    device = get_device(cfg)

    test_data, test_loader = data_provider(cfg, "test")

    # choisir le bon AE
    if cfg.model_name == "Conv1DAutoEncoder":
        model, _ = build_conv_autoencoder(cfg, device)
    else:
        model, _ = build_autoencoder(cfg, device)

    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return cfg, device, model, test_data, test_loader


@torch.no_grad()
def reconstruct_batch(config_path: str, batch_index: int = 0):

    cfg, device, model, test_data, test_loader = load_ae_and_data(config_path)

    selected_batch = None

    for idx, batch in enumerate(test_loader):
        if idx == batch_index:
            selected_batch = batch
            break

    if selected_batch is None:
        raise IndexError("Batch index out of range")

    batch_x, _, _, _ = selected_batch

    batch_x = batch_x.float().to(device)

    x_hat = model(batch_x)

    return {
        "cfg": cfg,
        "test_data": test_data,
        "x": batch_x.detach().cpu().numpy(),
        "x_hat": x_hat.detach().cpu().numpy(),
    }


def plot_reconstruction(history, recon, save_path, title):

    plt.figure(figsize=(12,4))

    plt.plot(history, label="Original", linewidth=2)
    plt.plot(recon, label="Reconstruction", linewidth=2)

    plt.legend()
    plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

    plt.close()


def run_reconstruction_and_plot(
        config_path,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name=None
):

    outputs = reconstruct_batch(config_path, batch_index)

    cfg = outputs["cfg"]
    test_data = outputs["test_data"]

    x = outputs["x"][0]
    x_hat = outputs["x_hat"][0]

    if use_inverse:
        x = test_data.inverse_transform(x)
        x_hat = test_data.inverse_transform(x_hat)

    if feature_idx < 0:
        feature_idx = x.shape[-1] - 1

    x_series = x[:, feature_idx]
    x_hat_series = x_hat[:, feature_idx]

    os.makedirs(cfg.figures_dir, exist_ok=True)

    if save_name is None:
        save_name = f"ae_reconstruction_{cfg.dataset_name}_{cfg.seq_len}.png"

    save_path = os.path.join(cfg.figures_dir, save_name)

    plot_reconstruction(
        x_series,
        x_hat_series,
        save_path,
        f"{cfg.dataset_name} | AE reconstruction | seq_len={cfg.seq_len}"
    )

    print("Plot saved:", save_path)

    return {
        "original": x_series,
        "reconstruction": x_hat_series,
        "save_path": save_path
    }


if __name__ == "__main__":

    CONFIG_PATH = "configs/models/ae/conv_ae_etth1_96.json"

    run_reconstruction_and_plot(
        config_path=CONFIG_PATH,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name="conv_ae_etth1_96.png",
    )