import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.training.ae_trainers.ae_trainer import AETrainer


# ── Dataset EXACT pipeline ─────────────────────────
class WindowDataset(Dataset):
    def __init__(self, series, window):
        self.series = series.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.series) - self.window + 1

    def __getitem__(self, i):
        x = self.series[i : i + self.window]
        return (torch.from_numpy(x),)  # ⚠️ IMPORTANT


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return SimpleNamespace(**json.load(f))


def build_dataloaders(cfg):
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    data = df[[cfg.target_col]].values.astype(np.float32)
    T = len(data)

    t_train = int(cfg.train_ratio * T)
    t_val = int((cfg.train_ratio + cfg.val_ratio) * T)

    scaler = StandardScaler()
    train_s = scaler.fit_transform(data[:t_train])
    val_s = scaler.transform(data[t_train:t_val])

    ds_train = WindowDataset(train_s, cfg.seq_len)
    ds_val = WindowDataset(val_s, cfg.seq_len)

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    return dl_train, dl_val


def plot_loss(history, cfg):
    os.makedirs(cfg.figures_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TCN AE Training Loss")
    plt.legend()
    plt.grid()

    save_path = os.path.join(cfg.figures_dir, "tcn_ae_loss.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Plot saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"[Run] Config loaded: {args.config}")

    # ── hyperparams (comme avant) ──
    cfg.lr = 1e-4
    cfg.lr_patience = 5
    cfg.grad_clip = 1.0
    cfg.epochs = 20
    cfg.patience = 5
    cfg.batch_size = 32

    # compatibilité trainer
    cfg.checkpoint_dir = os.path.dirname(cfg.checkpoint_path)
    cfg.checkpoint_name = os.path.basename(cfg.checkpoint_path)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ── Data ──
    dl_train, dl_val = build_dataloaders(cfg)

    # ── Train ──
    trainer = AETrainer(cfg)
    model, history = trainer.train(dl_train, dl_val)

    # ── Plot ──
    plot_loss(history, cfg)

    print("\nTraining finished.")
    print("Checkpoint:", cfg.checkpoint_path)


if __name__ == "__main__":
    main()
