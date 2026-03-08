import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data_provider.data_factory import data_provider
from src.models.forecaster_wrapper import build_itransformer
from src.utils.config import load_config
from src.utils.train_tools import get_device


def build_decoder_input(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    return dec_inp


@torch.no_grad()
def load_model_and_data(config_path: str):
    cfg = load_config(config_path)
    device = get_device(cfg)

    test_data, test_loader = data_provider(cfg, "test")

    model, _ = build_itransformer(cfg, device)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    return cfg, device, model, test_data, test_loader


@torch.no_grad()
def run_inference(config_path: str, batch_index: int = 0):
    cfg, device, model, test_data, test_loader = load_model_and_data(config_path)

    selected_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == batch_index:
            selected_batch = batch
            break

    if selected_batch is None:
        raise IndexError(f"batch_index={batch_index} out of range for test_loader")

    batch_x, batch_y, batch_x_mark, batch_y_mark = selected_batch

    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    dec_inp = build_decoder_input(
        batch_y=batch_y,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        device=device,
    )

    if cfg.output_attention:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    else:
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    f_dim = -1 if cfg.features == "MS" else 0

    preds = outputs[:, -cfg.pred_len:, f_dim:]
    trues = batch_y[:, -cfg.pred_len:, f_dim:]
    history = batch_x

    return {
        "cfg": cfg,
        "device": device,
        "test_data": test_data,
        "history": history.detach().cpu().numpy(),
        "preds": preds.detach().cpu().numpy(),
        "trues": trues.detach().cpu().numpy(),
    }


def inverse_if_needed(test_data, arr: np.ndarray, features_mode: str):
    """
    arr shape:
      - M : [T, C]
      - S : [T, 1]
      - MS: [T, 1]
    """
    if features_mode == "M":
        return test_data.inverse_transform(arr)

    if features_mode == "S":
        return test_data.inverse_transform(arr)

    # Pour MS, l'inverse_transform direct n'est pas toujours cohérent
    # si le scaler a été fit sur plusieurs variables mais qu'on ne passe qu'une colonne.
    return arr


def plot_forecast(
    history: np.ndarray,
    true_future: np.ndarray,
    pred_future: np.ndarray,
    save_path: str,
    title: str,
):
    gt = np.concatenate([history, true_future], axis=0)
    pd_ = np.concatenate([history, pred_future], axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(gt, label="Real", linewidth=2)
    plt.plot(pd_, label="Forecasted", linewidth=2)
    plt.axvline(x=len(history) - 1, linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_inference_and_plot(
    config_path: str,
    batch_index: int = 0,
    feature_idx: int = -1,
    use_inverse: bool = True,
    save_name: str = None,
):
    outputs = run_inference(config_path=config_path, batch_index=batch_index)

    cfg = outputs["cfg"]
    test_data = outputs["test_data"]

    history = outputs["history"]   # [1, seq_len, C]
    preds = outputs["preds"]       # [1, pred_len, C or 1]
    trues = outputs["trues"]       # [1, pred_len, C or 1]

    # On enlève la dimension batch
    history_0 = history[0]
    preds_0 = preds[0]
    trues_0 = trues[0]

    if use_inverse:
        history_0 = inverse_if_needed(test_data, history_0, cfg.features)
        preds_0 = inverse_if_needed(test_data, preds_0, cfg.features)
        trues_0 = inverse_if_needed(test_data, trues_0, cfg.features)

    # Choix de la feature à afficher
    if cfg.features == "S":
        feature_idx = 0
    elif cfg.features == "MS":
        feature_idx = 0
    else:
        if feature_idx < 0:
            feature_idx = history_0.shape[-1] - 1

    hist_series = history_0[:, feature_idx]
    true_series = trues_0[:, feature_idx]
    pred_series = preds_0[:, feature_idx]

    os.makedirs(cfg.figures_dir, exist_ok=True)

    if save_name is None:
        save_name = f"plot_{cfg.dataset_name.lower()}_{cfg.seq_len}_{cfg.pred_len}_batch{batch_index}.png"

    save_path = os.path.join(cfg.figures_dir, save_name)

    title = f"{cfg.dataset_name} | {cfg.model_name} | {cfg.seq_len} -> {cfg.pred_len}"

    plot_forecast(
        history=hist_series,
        true_future=true_series,
        pred_future=pred_series,
        save_path=save_path,
        title=title,
    )

    print(f"Plot saved to: {save_path}")

    return {
        "history": hist_series,
        "true_future": true_series,
        "pred_future": pred_series,
        "save_path": save_path,
    }
if __name__ == "__main__":

    CONFIG_PATH = "configs/models/itransformer/etth1_96_96.json"

    run_inference_and_plot(
        config_path=CONFIG_PATH,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name="plot_etth1_96_96.png",
    )