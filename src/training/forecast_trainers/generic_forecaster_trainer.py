"""
generic_forecaster_trainer.py
------------------------------
Trainer générique pour GRU, DLinear, TimesNet.
Génère automatiquement après l'entraînement :
  - {model}_training_curves.png
  - {model}_forecast_examples.png
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.data_provider.data_factory import data_provider
from src.utils.train_tools import EarlyStopping, adjust_learning_rate, get_device, save_history


# ─────────────────────────────────────────────────────────────────────────────
def _build_model(configs, device):
    mt = getattr(configs, "model_type", "GRU").lower()
    if mt == "gru":
        from src.models.Forecaster.GRU import Model
    elif mt == "dlinear":
        from src.models.Forecaster.DLinear import Model
    elif mt == "timesnet":
        from src.models.Forecaster.TimesNet import Model
    else:
        raise ValueError(f"Unknown model_type: '{mt}'")
    return Model(configs).float().to(device)


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, figures_dir: str, model_name: str):
    """Plot train / val / test loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train",  color="#2196F3", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val",    color="#FF9800", linewidth=2)
    ax.plot(epochs, history["test_loss"],  label="Test",   color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(figures_dir, f"{model_name.lower()}_training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig] Training curves → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast_examples(preds, targets, inputs,
                           model_name: str, figures_dir: str, n: int = 4):
    """
    Plot forecast examples from numpy arrays.
      preds   : (N, H, 1)
      targets : (N, H, 1)
      inputs  : (N, T, 1)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figures_dir, exist_ok=True)
    n_plot   = min(n, preds.shape[0])
    seq_len  = inputs.shape[1]
    pred_len = preds.shape[1]
    t_past   = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)

    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot))
    if n_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_past,   inputs[i, :, 0],  color="grey",    lw=1.2, label="Past (input)")
        ax.plot(t_future, targets[i, :, 0], color="#2196F3", lw=1.5, label="Real future")
        ax.plot(t_future, preds[i, :, 0],   color="#F44336", lw=1.5,
                linestyle="--", label="Predicted")
        ax.axvline(x=seq_len, color="black", linestyle=":", lw=1)
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} — Forecast Examples (Test Set)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, f"{model_name.lower()}_forecast_examples.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Fig] Forecast examples → {path}")


def _collect_examples(model, test_loader, configs, device, n: int = 4):
    """Run model on first batch of test_loader, return (preds, targets, inputs) as numpy."""
    model.eval()
    f_dim = -1 if getattr(configs, "features", "S") == "MS" else 0

    batch_x, batch_y, batch_x_mark, _ = next(iter(test_loader))
    batch_x      = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)

    with torch.no_grad():
        out = model(batch_x, batch_x_mark)
        if isinstance(out, tuple):
            out = out[0]

    preds   = out[:, -configs.pred_len:, f_dim:].cpu().numpy()
    targets = batch_y[:, -configs.pred_len:, f_dim:].numpy()
    inputs  = batch_x[:, :, f_dim:].cpu().numpy()
    return preds[:n], targets[:n], inputs[:n]


# ─────────────────────────────────────────────────────────────────────────────
class GenericForecasterTrainer:

    def __init__(self, configs):
        self.configs = configs
        self.device  = get_device(configs)

        self.model     = _build_model(configs, self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=configs.learning_rate,
        )

        self.train_data, self.train_loader = data_provider(configs, "train")
        self.val_data,   self.val_loader   = data_provider(configs, "val")
        self.test_data,  self.test_loader  = data_provider(configs, "test")

        self.checkpoint_path = os.path.join(configs.checkpoint_dir, configs.checkpoint_name)
        self.history_path    = os.path.join(configs.results_dir,    configs.history_name)
        self.figures_dir     = getattr(configs, "figures_dir", "assets/figures/forecaster")
        self.model_name      = getattr(configs, "model_type", "Model")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[{self.model_name}] {n_params:,} params  |  "
              f"seq={configs.seq_len}  pred={configs.pred_len}  "
              f"train={len(self.train_data)}  test={len(self.test_data)}")

    # ─────────────────────────────────────────────────────────────────────
    def _compute_loss(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x      = batch_x.float().to(self.device)
        batch_y      = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        f_dim   = -1 if getattr(self.configs, "features", "S") == "MS" else 0
        outputs = outputs[:, -self.configs.pred_len:, f_dim:]
        target  = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)

        return self.criterion(outputs, target)

    def validate(self, loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark)
                losses.append(loss.item())
        self.model.train()
        return float(np.mean(losses))

    # ─────────────────────────────────────────────────────────────────────
    def train(self):
        early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True)
        history = {"train_loss": [], "val_loss": [], "test_loss": []}

        for epoch in range(self.configs.train_epochs):
            t0 = time.time()
            train_losses = []
            self.model.train()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss   = self.validate(self.val_loader)
            test_loss  = self.validate(self.test_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)

            print(
                f"Epoch {epoch+1:3d}/{self.configs.train_epochs} | "
                f"train={train_loss:.5f}  val={val_loss:.5f}  test={test_loss:.5f} | "
                f"{time.time()-t0:.1f}s"
            )

            early_stopping(val_loss, self.model, self.checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.configs)

        # Reload best checkpoint
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        )
        save_history(history, self.history_path)
        print(f"\n[Done] Best checkpoint → {self.checkpoint_path}")

        # ── Figures ──────────────────────────────────────────────────────
        plot_training_curves(history, self.figures_dir, self.model_name)
        preds, targets, inputs = _collect_examples(
            self.model, self.test_loader, self.configs, self.device, n=4
        )
        plot_forecast_examples(preds, targets, inputs,
                               self.model_name, self.figures_dir, n=4)

        return self.model, history
