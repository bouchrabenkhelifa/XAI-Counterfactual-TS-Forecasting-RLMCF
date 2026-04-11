
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from types import SimpleNamespace

from src.models.autoencoder.tcn_ae import TCNAutoEncoder


class AETrainer:
    """
    Trainer for TCNAutoEncoder.

    Args:
        cfg : SimpleNamespace with all config parameters
    """

    def __init__(self, cfg: SimpleNamespace):
        self.cfg    = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[AETrainer] device = {self.device}")

    # ── Build model ──────────────────────────────────────────
    def build_model(self) -> TCNAutoEncoder:
        cfg = self.cfg
        model = TCNAutoEncoder(
            seq_len    = cfg.seq_len,
            n_features = cfg.n_features,
            hidden_dim = cfg.hidden_dim,
            latent_dim = cfg.latent_dim,
            n_blocks   = cfg.n_blocks,
            kernel_size= cfg.kernel_size,
            dropout    = cfg.dropout,
        ).to(self.device)

        rf = model.receptive_field()
        n  = model.count_parameters()
        print(f"[AETrainer] model built | "
              f"params={n:,} | receptive_field={rf} / {cfg.seq_len}")
        return model

    # ── One epoch ────────────────────────────────────────────
    def run_epoch(self, model: TCNAutoEncoder,
                  loader: DataLoader,
                  optimizer=None,
                  train: bool = True) -> float:
        model.train(train)
        criterion = nn.MSELoss()
        total_loss = 0.0
        n          = 0

        with torch.set_grad_enabled(train):
            for x, in loader:
                x = x.to(self.device).float()

                x_recon, _ = model(x)
                loss = criterion(x_recon, x)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.grad_clip
                    )
                    optimizer.step()

                total_loss += loss.item() * x.shape[0]
                n          += x.shape[0]

        return total_loss / max(1, n)

    # ── Train ────────────────────────────────────────────────
    def train(self, dl_train: DataLoader,
              dl_val: DataLoader) -> tuple:
        """
        Full training loop with early stopping.

        Args:
            dl_train : training DataLoader
            dl_val   : validation DataLoader

        Returns:
            model   : best trained TCNAutoEncoder
            history : dict with train/val loss curves
        """
        cfg   = self.cfg
        model = self.build_model()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=cfg.lr_patience, min_lr=1e-6,
        )

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

        best_val  = float("inf")
        patience  = 0
        history   = {"train": [], "val": []}

        print(f"\n{'='*50}")
        print(f"[AETrainer] Starting training for {cfg.epochs} epochs")
        print(f"  early stopping patience = {cfg.patience}")
        print(f"{'='*50}")

        for ep in range(1, cfg.epochs + 1):

            tr_loss = self.run_epoch(model, dl_train,
                                     optimizer=optimizer, train=True)
            va_loss = self.run_epoch(model, dl_val,
                                     optimizer=None,      train=False)

            scheduler.step(va_loss)
            history["train"].append(tr_loss)
            history["val"].append(va_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {ep:03d}/{cfg.epochs} | "
                  f"train={tr_loss:.6f} | val={va_loss:.6f} | "
                  f"lr={lr_now:.2e}")

            # ── Save best checkpoint ──────────────────────────
            if va_loss < best_val:
                best_val = va_loss
                patience = 0
                torch.save({
                    "epoch":       ep,
                    "state_dict":  model.state_dict(),
                    "val_loss":    best_val,
                    "cfg": {
                        "seq_len":    cfg.seq_len,
                        "n_features": cfg.n_features,
                        "hidden_dim": cfg.hidden_dim,
                        "latent_dim": cfg.latent_dim,
                        "n_blocks":   cfg.n_blocks,
                        "kernel_size":cfg.kernel_size,
                        "dropout":    cfg.dropout,
                    },
                }, ckpt_path)
                print(f"  ✅ best checkpoint saved | val={best_val:.6f}")
            else:
                patience += 1
                if patience >= cfg.patience:
                    print(f"\n[AETrainer] Early stopping at epoch {ep}")
                    break

        # ── Load best weights ─────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location=self.device,
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n[AETrainer] Best val loss = {best_val:.6f}")
        print(f"[AETrainer] Checkpoint    = {ckpt_path}")

        return model, history