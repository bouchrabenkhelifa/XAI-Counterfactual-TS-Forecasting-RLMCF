import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_provider.data_factory import data_provider
from src.models.forecaster_wrapper import build_itransformer
from src.models.autoencoder.ae_wrapper import build_autoencoder
from src.models.autoencoder.conv_ae_wrapper import build_conv_autoencoder
from src.utils.config import load_config
from src.utils.train_tools import get_device


def build_decoder_input(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    return dec_inp


@torch.no_grad()
def load_forecaster(cfg, device):
    model, _ = build_itransformer(cfg, device)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def load_autoencoder(ae_cfg, device):
    if ae_cfg.model_name == "Conv1DAutoEncoder":
        model, _ = build_conv_autoencoder(ae_cfg, device)
    else:
        model, _ = build_autoencoder(ae_cfg, device)

    checkpoint_path = os.path.join(ae_cfg.checkpoint_dir, ae_cfg.checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_forecast_consistency(
    forecaster_config_path: str,
    ae_config_path: str,
    batch_index: int = 0,
    feature_idx: int = -1,
    use_inverse: bool = True,
    save_name: str = "forecast_consistency.png",
):
    f_cfg = load_config(forecaster_config_path)
    ae_cfg = load_config(ae_config_path)

    device = get_device(f_cfg)

    test_data, test_loader = data_provider(f_cfg, "test")

    forecaster = load_forecaster(f_cfg, device)
    autoencoder = load_autoencoder(ae_cfg, device)

    selected_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == batch_index:
            selected_batch = batch
            break

    if selected_batch is None:
        raise IndexError(f"batch_index={batch_index} out of range")

    batch_x, batch_y, batch_x_mark, batch_y_mark = selected_batch

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # reconstruction
    x_hat = autoencoder(batch_x)

    # forecast on x
    dec_inp_x = build_decoder_input(batch_y, f_cfg.label_len, f_cfg.pred_len, device)
    if f_cfg.output_attention:
        pred_x = forecaster(batch_x, batch_x_mark, dec_inp_x, batch_y_mark)[0]
    else:
        pred_x = forecaster(batch_x, batch_x_mark, dec_inp_x, batch_y_mark)

    # forecast on x_hat
    dec_inp_xhat = build_decoder_input(batch_y, f_cfg.label_len, f_cfg.pred_len, device)
    if f_cfg.output_attention:
        pred_xhat = forecaster(x_hat, batch_x_mark, dec_inp_xhat, batch_y_mark)[0]
    else:
        pred_xhat = forecaster(x_hat, batch_x_mark, dec_inp_xhat, batch_y_mark)

    f_dim = -1 if f_cfg.features == "MS" else 0

    pred_x = pred_x[:, -f_cfg.pred_len:, f_dim:]
    pred_xhat = pred_xhat[:, -f_cfg.pred_len:, f_dim:]
    true_y = batch_y[:, -f_cfg.pred_len:, f_dim:]

    pred_x_np = pred_x.detach().cpu().numpy()[0]
    pred_xhat_np = pred_xhat.detach().cpu().numpy()[0]
    true_y_np = true_y.detach().cpu().numpy()[0]
    x_np = batch_x.detach().cpu().numpy()[0]
    x_hat_np = x_hat.detach().cpu().numpy()[0]

    # inverse if needed
    if use_inverse:
        if f_cfg.features == "M":
            x_np = test_data.inverse_transform(x_np)
            x_hat_np = test_data.inverse_transform(x_hat_np)
            pred_x_np = test_data.inverse_transform(pred_x_np)
            pred_xhat_np = test_data.inverse_transform(pred_xhat_np)
            true_y_np = test_data.inverse_transform(true_y_np)
        elif f_cfg.features == "S":
            x_np = test_data.inverse_transform(x_np)
            x_hat_np = test_data.inverse_transform(x_hat_np)
            pred_x_np = test_data.inverse_transform(pred_x_np)
            pred_xhat_np = test_data.inverse_transform(pred_xhat_np)
            true_y_np = test_data.inverse_transform(true_y_np)

    if f_cfg.features in ["S", "MS"]:
        feature_idx = 0
    else:
        if feature_idx < 0:
            feature_idx = x_np.shape[-1] - 1

    hist_series = x_np[:, feature_idx]
    recon_series = x_hat_np[:, feature_idx]
    pred_x_series = pred_x_np[:, feature_idx]
    pred_xhat_series = pred_xhat_np[:, feature_idx]
    true_series = true_y_np[:, feature_idx]

    mse = float(np.mean((pred_x_series - pred_xhat_series) ** 2))
    mae = float(np.mean(np.abs(pred_x_series - pred_xhat_series)))

    os.makedirs(f_cfg.figures_dir, exist_ok=True)
    save_path = os.path.join(f_cfg.figures_dir, save_name)

    plt.figure(figsize=(12, 5))
    plt.plot(
        np.concatenate([hist_series, true_series]),
        label="Real future",
        linewidth=2
    )
    plt.plot(
        np.concatenate([hist_series, pred_x_series]),
        label="Forecast on x",
        linewidth=2
    )
    plt.plot(
        np.concatenate([recon_series, pred_xhat_series]),
        label="Forecast on AE(x)",
        linewidth=2
    )
    plt.axvline(x=len(hist_series) - 1, linestyle="--")
    plt.title(f"Forecast consistency | MSE={mse:.4f}, MAE={mae:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Forecast consistency plot saved to: {save_path}")
    print(f"MSE(forecast(x), forecast(AE(x))) = {mse:.6f}")
    print(f"MAE(forecast(x), forecast(AE(x))) = {mae:.6f}")

    return {
        "mse": mse,
        "mae": mae,
        "save_path": save_path,
        "history": hist_series,
        "reconstruction": recon_series,
        "forecast_x": pred_x_series,
        "forecast_xhat": pred_xhat_series,
        "true_future": true_series,
    }


if __name__ == "__main__":
    FORECASTER_CONFIG = "configs/models/itransformer/etth1/etth1_96_96.json"
    AE_CONFIG = "configs/models/autoencoder/etth1/conv_ae_etth1_96.json"

    run_forecast_consistency(
        forecaster_config_path=FORECASTER_CONFIG,
        ae_config_path=AE_CONFIG,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name="forecast_consistency_etth1_96_96.png",
    )