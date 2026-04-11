
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from types import SimpleNamespace

from src.models.autoencoder.nv_loss.TCN_AE import TCNAutoEncoder


# ══════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════

class DerivativeLoss(nn.Module):
    """
    L = λ_rec * MSE(x, x̂)
      + λ_d1  * L1(Δx,  Δx̂)
      + λ_d2  * L1(Δ²x, Δ²x̂)

    Δx  = x[:, 1:] - x[:, :-1]   première dérivée discrète
    Δ²x = Δx[:, 1:] - Δx[:, :-1] seconde dérivée discrète

    Args:
        lambda_rec : poids MSE          (défaut 1.0)
        lambda_d1  : poids Δx  loss     (défaut 1.0)
        lambda_d2  : poids Δ²x loss     (défaut 0.5)
    """

    def __init__(self, lambda_rec=1.0, lambda_d1=1.0, lambda_d2=0.5):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_d1  = lambda_d1
        self.lambda_d2  = lambda_d2

    @staticmethod
    def _deriv1(x: torch.Tensor) -> torch.Tensor:
        """Première dérivée discrète. x : (B, T, 1) → (B, T-1, 1)"""
        return x[:, 1:, :] - x[:, :-1, :]

    @staticmethod
    def _deriv2(x: torch.Tensor) -> torch.Tensor:
        """Seconde dérivée discrète. x : (B, T, 1) → (B, T-2, 1)"""
        d1 = x[:, 1:, :] - x[:, :-1, :]
        return d1[:, 1:, :] - d1[:, :-1, :]

    def forward(self, x_recon: torch.Tensor,
                x: torch.Tensor) -> dict:
        """
        Args:
            x_recon : (B, T, 1) — reconstruction
            x       : (B, T, 1) — original

        Returns:
            dict with total, mse, d1, d2 losses
        """
        # MSE global
        loss_mse = F.mse_loss(x_recon, x)

        # Dérivée première
        d1_orig  = self._deriv1(x)
        d1_recon = self._deriv1(x_recon)
        loss_d1  = F.l1_loss(d1_recon, d1_orig)

        # Dérivée seconde
        d2_orig  = self._deriv2(x)
        d2_recon = self._deriv2(x_recon)
        loss_d2  = F.l1_loss(d2_recon, d2_orig)

        total = (self.lambda_rec * loss_mse
                 + self.lambda_d1  * loss_d1
                 + self.lambda_d2  * loss_d2)

        return {
            "total": total,
            "mse"  : loss_mse.detach(),
            "d1"   : loss_d1.detach(),
            "d2"   : loss_d2.detach(),
        }


# ══════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════

class AETrainerDerivativeLoss:
    """
    Trainer for TCNAutoEncoder with derivative-based loss.

    Checkpoint name : ae_etth1_deriv_loss.pt
    Figures prefix  : ae_deriv_loss_*
    """

    def __init__(self, cfg: SimpleNamespace):
        self.cfg    = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AE-DerivLoss] device = {self.device}")

        self.criterion = DerivativeLoss(
            lambda_rec = cfg.lambda_rec,
            lambda_d1  = cfg.lambda_d1,
            lambda_d2  = cfg.lambda_d2,
        )
        print(f"[AE-DerivLoss] Loss = "
              f"{cfg.lambda_rec}*MSE + "
              f"{cfg.lambda_d1}*L1(Δx) + "
              f"{cfg.lambda_d2}*L1(Δ²x)")

    # ── Build model ──────────────────────────────────────────
    def build_model(self) -> TCNAutoEncoder:
        cfg   = self.cfg
        model = TCNAutoEncoder(
            seq_len     = cfg.seq_len,
            n_features  = cfg.n_features,
            hidden_dim  = cfg.hidden_dim,
            latent_dim  = cfg.latent_dim,
            n_blocks    = cfg.n_blocks,
            kernel_size = cfg.kernel_size,
            dropout     = cfg.dropout,
        ).to(self.device)

        rf = model.receptive_field()
        n  = model.count_parameters()
        print(f"[AE-DerivLoss] params={n:,} | RF={rf}/{cfg.seq_len}")
        return model

    # ── One epoch ────────────────────────────────────────────
    def run_epoch(self, model, loader, optimizer=None,
                  train=True) -> dict:
        model.train(train)
        totals = {"total": 0., "mse": 0., "d1": 0., "d2": 0.}
        n = 0

        with torch.set_grad_enabled(train):
            for x, in loader:
                x = x.to(self.device).float()

                x_recon, _ = model(x)
                losses     = self.criterion(x_recon, x)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.grad_clip)
                    optimizer.step()

                B = x.shape[0]
                for k in totals:
                    totals[k] += float(losses[k].item()) * B
                n += B

        return {k: v / max(1, n) for k, v in totals.items()}

    # ── Train ────────────────────────────────────────────────
    def train(self, dl_train: DataLoader,
              dl_val: DataLoader) -> tuple:
        cfg   = self.cfg
        model = self.build_model()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=cfg.lr_patience, min_lr=1e-6)

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.figures_dir,    exist_ok=True)

        # checkpoint nommé explicitement
        ckpt_path = os.path.join(
            cfg.checkpoint_dir, "ae_etth1_deriv_loss.pt")

        best_val = float("inf")
        patience = 0
        history  = {k: [] for k in
                    ["train_total", "train_mse", "train_d1", "train_d2",
                     "val_total",   "val_mse",   "val_d1",   "val_d2"]}

        print(f"\n{'='*55}")
        print(f"[AE-DerivLoss] Training {cfg.epochs} epochs")
        print(f"  λ_rec={cfg.lambda_rec} | "
              f"λ_d1={cfg.lambda_d1} | λ_d2={cfg.lambda_d2}")
        print(f"  early stopping patience = {cfg.patience}")
        print(f"{'='*55}")

        for ep in range(1, cfg.epochs + 1):

            tr = self.run_epoch(model, dl_train,
                                optimizer=optimizer, train=True)
            va = self.run_epoch(model, dl_val,
                                optimizer=None, train=False)

            scheduler.step(va["total"])

            for k in ["total", "mse", "d1", "d2"]:
                history[f"train_{k}"].append(tr[k])
                history[f"val_{k}"].append(va[k])

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {ep:03d}/{cfg.epochs} | "
                  f"tr={tr['total']:.5f} "
                  f"(mse={tr['mse']:.5f} d1={tr['d1']:.5f} d2={tr['d2']:.5f}) | "
                  f"va={va['total']:.5f} "
                  f"(mse={va['mse']:.5f} d1={va['d1']:.5f} d2={va['d2']:.5f}) | "
                  f"lr={lr_now:.2e}")

            if va["total"] < best_val:
                best_val = va["total"]
                patience = 0
                torch.save({
                    "epoch"      : ep,
                    "state_dict" : model.state_dict(),
                    "val_loss"   : best_val,
                    "val_mse"    : va["mse"],
                    "loss_type"  : "mse+deriv1+deriv2",
                    "lambda_rec" : cfg.lambda_rec,
                    "lambda_d1"  : cfg.lambda_d1,
                    "lambda_d2"  : cfg.lambda_d2,
                    "cfg": {
                        "seq_len"    : cfg.seq_len,
                        "n_features" : cfg.n_features,
                        "hidden_dim" : cfg.hidden_dim,
                        "latent_dim" : cfg.latent_dim,
                        "n_blocks"   : cfg.n_blocks,
                        "kernel_size": cfg.kernel_size,
                        "dropout"    : cfg.dropout,
                    },
                }, ckpt_path)
                print(f"  ✅ best saved | val_total={best_val:.5f} "
                      f"val_mse={va['mse']:.5f}")
            else:
                patience += 1
                if patience >= cfg.patience:
                    print(f"\n[AE-DerivLoss] Early stopping at epoch {ep}")
                    break

        # Load best
        ckpt = torch.load(ckpt_path, map_location=self.device,
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"\n[AE-DerivLoss] Best val_total = {best_val:.5f}")
        print(f"[AE-DerivLoss] Checkpoint     = {ckpt_path}")

        return model, history, ckpt_path

    # ── Plots ────────────────────────────────────────────────
    def plot_loss_curves(self, history: dict):
        """Plot training curves for all loss components."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        for ax, key, title, color in zip(
            axes,
            ["total", "mse", "d1", "d2"],
            ["Total loss", "MSE", "L1(Δx)", "L1(Δ²x)"],
            ["steelblue", "coral", "green", "purple"],
        ):
            ax.plot(history[f"train_{key}"],
                    color=color, lw=1.5, label="train")
            ax.plot(history[f"val_{key}"],
                    color=color, lw=1.5, ls="--", label="val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(
            f"AE Derivative Loss — ETTh1 | "
            f"λ_rec={self.cfg.lambda_rec} "
            f"λ_d1={self.cfg.lambda_d1} "
            f"λ_d2={self.cfg.lambda_d2}",
            fontsize=13
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg.figures_dir,
            "ae_deriv_loss_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[AE-DerivLoss] Loss curves → {path}")

    def plot_reconstruction(self, model, val_loader, n=6):
        """
        Plot original vs reconstructed — focus on local patterns.
        Shows original, reconstruction, and derivative error per sample.
        """
        model.eval()
        batch = next(iter(val_loader))
        x     = batch[0][:n].to(self.device).float()

        with torch.no_grad():
            x_recon, _ = model(x)

        x     = x.cpu().numpy()
        x_rec = x_recon.cpu().numpy()

        # Compute per-sample MSE for title
        mse_per = ((x - x_rec) ** 2).mean(axis=(1, 2))

        fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            # Left : reconstruction
            axes[i, 0].plot(x[i, :, 0],     color="steelblue",
                            lw=1.5, label="Original")
            axes[i, 0].plot(x_rec[i, :, 0], color="coral",
                            lw=1.5, ls="--", label="Reconstructed")
            axes[i, 0].set_title(
                f"Sample {i+1} — MSE={mse_per[i]:.5f}")
            axes[i, 0].legend(fontsize=8)
            axes[i, 0].grid(alpha=0.3)

            # Right : derivative error (highlights where peaks are missed)
            d1_orig = np.diff(x[i, :, 0])
            d1_rec  = np.diff(x_rec[i, :, 0])
            axes[i, 1].plot(d1_orig, color="steelblue",
                            lw=1.2, label="|Δx| original")
            axes[i, 1].plot(d1_rec,  color="coral",
                            lw=1.2, ls="--", label="|Δx̂| reconstructed")
            axes[i, 1].axhline(0, color="gray", lw=0.8, ls=":")
            axes[i, 1].set_title(f"Derivative Δx — sample {i+1}")
            axes[i, 1].legend(fontsize=8)
            axes[i, 1].grid(alpha=0.3)

        plt.suptitle(
            "AE Derivative Loss — Reconstruction & Derivative fidelity\n"
            "ETTh1 val set",
            fontsize=13
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg.figures_dir,
            "ae_deriv_loss_reconstruction.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[AE-DerivLoss] Reconstruction → {path}")