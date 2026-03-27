# src/models/optimization_strategy/wachter_latent.py
# ─────────────────────────────────────────────────────────────
# Counterfactual generation by latent space optimization.
#
# Strategy : Wachter latent
# ─────────────────────────
# For each input instance x, we optimize z_cf directly in the
# latent space of the pre-trained TCN AE by minimizing :
#
#   L = - r_validity(y_cf)
#       + lambda_prox  * ||z_cf - z||²
#       + lambda_plaus * plausibility_loss(z_cf)
#
# where :
#   r_validity   = sigmoid(10 * (mean_orig - mean_cf - rho*|mean_orig|))
#   proximity    = L2 distance in latent space
#   plausibility = distance to train set centroid in latent space
#
# Key difference vs RL :
#   - Instance-based : one optimization loop per input (no policy)
#   - No scalability advantage but potentially better plausibility
#   - Gradient flows directly through AE decoder + forecaster
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WachterLatentConfig:
    """Configuration for Wachter latent optimization."""

    # Objective
    rho          : float = 0.10    # target reduction (10%)

    # Loss weights
    lambda_prox  : float = 1.0    # weight for latent proximity loss
    lambda_plaus : float = 0.5    # weight for plausibility loss

    # Optimization
    n_steps      : int   = 200    # gradient steps per instance
    lr           : float = 0.05   # learning rate
    n_restarts   : int   = 3      # random restarts (keep best)

    # Latent space
    z_clip       : float = 1.0    # clip z_cf to [-z_clip, z_clip]
    noise_std    : float = 0.1    # std of random restart noise

    # Plausibility (latent centroid)
    sigma_plaus  : float = 1.0    # bandwidth for gaussian plausibility


class WachterLatent(nn.Module):
    """
    Counterfactual generator via latent space optimization.

    For each input x, optimizes z_cf to minimize a loss combining
    validity, proximity, and plausibility.

    Args:
        ae         : frozen TCNAutoEncoder
        forecaster : frozen ForecasterWrapper
        cfg        : WachterLatentConfig
    """

    def __init__(self, ae, forecaster, cfg: WachterLatentConfig):
        super().__init__()
        self.ae         = ae
        self.forecaster = forecaster
        self.cfg        = cfg

        # Latent statistics (set by fit())
        self.z_mean  : Optional[torch.Tensor] = None
        self.z_std   : Optional[torch.Tensor] = None
        self.sigma   : float = 1.0
        self.fitted_ : bool  = False

    # ──────────────────────────────────────────────
    # Fit latent statistics on train set
    # ──────────────────────────────────────────────
    def fit(self, train_loader, device):
        """
        Compute latent space statistics from training data.
        Used for the plausibility term.

        Args:
            train_loader : DataLoader yielding (batch_x, _, batch_x_mark, _)
            device       : torch.device
        """
        print("[WachterLatent] Computing latent statistics ...")
        self.ae.eval()
        zs = []

        with torch.no_grad():
            for batch in train_loader:
                batch_x = batch[0].float().to(device)
                x_ot    = batch_x[:, :, -1:]          # OT feature
                z       = self.ae.encode(x_ot)
                zs.append(z.cpu())

        Z            = torch.cat(zs, dim=0)            # (N, latent_dim)
        self.z_mean  = Z.mean(dim=0)                   # (latent_dim,)
        self.z_std   = Z.std(dim=0) + 1e-8            # (latent_dim,)
        self.sigma   = float(Z.std().item())
        self.fitted_ = True

        print(f"[WachterLatent] Fitted on {len(Z)} windows")
        print(f"  z_mean norm = {self.z_mean.norm():.4f}")
        print(f"  sigma       = {self.sigma:.4f}")

    # ──────────────────────────────────────────────
    # Loss components
    # ──────────────────────────────────────────────
    def _validity_loss(self, y_hat: torch.Tensor,
                       y_cf: torch.Tensor) -> torch.Tensor:
        """
        Soft validity loss — encourages mean(y_cf) < mean(y_hat) - rho.
        Returns scalar (mean over batch).
        """
        if y_hat.dim() == 3: y_hat = y_hat[:, :, 0]
        if y_cf.dim()  == 3: y_cf  = y_cf[:, :, 0]

        mean_orig = y_hat.mean(dim=1).detach()
        mean_cf   = y_cf.mean(dim=1)
        delta     = self.cfg.rho * mean_orig.abs()

        # sigmoid reward — negate for minimization
        r_v = torch.sigmoid(10.0 * (mean_orig - mean_cf - delta))
        return -r_v.mean()                             # minimize → maximize r_v

    def _proximity_loss(self, z: torch.Tensor,
                        z_cf: torch.Tensor) -> torch.Tensor:
        """L2 distance in latent space."""
        return F.mse_loss(z_cf, z.detach())

    def _plausibility_loss(self, z_cf: torch.Tensor) -> torch.Tensor:
        """
        Gaussian distance to training centroid in latent space.
        Minimizing this → z_cf stays close to the training distribution.
        """
        assert self.fitted_, "Call fit() first"
        z_mean = self.z_mean.to(z_cf.device)
        z_std  = self.z_std.to(z_cf.device)

        z_norm = ((z_cf - z_mean) / z_std).norm(dim=1)     # (B,)
        score  = torch.exp(
            -0.5 * (z_norm / (self.sigma + 1e-8)) ** 2
        )                                                   # (B,) ∈ (0,1]
        return -score.mean()                               # minimize → maximize score

    def _total_loss(self, z: torch.Tensor,
                    z_cf: torch.Tensor,
                    y_hat: torch.Tensor,
                    y_cf: torch.Tensor) -> dict:
        """Compute all loss components and total."""
        l_val   = self._validity_loss(y_hat, y_cf)
        l_prox  = self._proximity_loss(z, z_cf)
        l_plaus = self._plausibility_loss(z_cf)

        total = (l_val
                 + self.cfg.lambda_prox  * l_prox
                 + self.cfg.lambda_plaus * l_plaus)
        return {
            "total"   : total,
            "validity": l_val,
            "proximity": l_prox,
            "plausibility": l_plaus,
        }

    # ──────────────────────────────────────────────
    # Single instance optimization
    # ──────────────────────────────────────────────
    def _optimize_single(self, z: torch.Tensor,
                         batch_x: torch.Tensor,
                         batch_x_mark: torch.Tensor,
                         y_hat: torch.Tensor,
                         device: torch.device,
                         init_noise: Optional[torch.Tensor] = None
                         ) -> tuple:
        """
        Optimize z_cf for a single batch.

        Args:
            z            : (B, latent_dim) — encoded x_ot
            batch_x      : (B, seq_len, C) — full input
            batch_x_mark : (B, seq_len, D) — time features
            y_hat        : (B, pred_len, 1) — original forecast
            device       : torch.device
            init_noise   : optional noise for random restart

        Returns:
            z_cf_best : (B, latent_dim) — optimized latent CF
            x_cf_best : (B, seq_len, 1) — decoded CF
            y_cf_best : (B, pred_len, 1) — CF forecast
            loss_hist : list of total loss values
        """
        cfg = self.cfg

        # Initialize z_cf
        if init_noise is not None:
            z_cf_init = (z + init_noise).detach().clone()
        else:
            z_cf_init = z.detach().clone()

        z_cf_init = torch.clamp(z_cf_init, -cfg.z_clip, cfg.z_clip)
        z_cf      = nn.Parameter(z_cf_init.requires_grad_(True))
        optimizer = torch.optim.Adam([z_cf], lr=cfg.lr)

        loss_hist = []

        for step in range(cfg.n_steps):
            optimizer.zero_grad()

            # Decode z_cf → x_cf
            z_cf_clipped = torch.clamp(z_cf, -cfg.z_clip, cfg.z_clip)
            x_cf         = self.ae.decode(z_cf_clipped)   # (B, seq_len, 1)

            # Forecast from x_cf
            # Note: forecaster needs full x_full with x_cf in OT channel
            x_full_cf = batch_x.clone()
            x_full_cf[:, :, -1:] = x_cf
            y_cf = self.forecaster.predict_ot(
                x_full_cf, batch_x_mark)                  # (B, pred_len, 1)

            # Loss
            losses = self._total_loss(z, z_cf_clipped, y_hat, y_cf)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_([z_cf], 1.0)
            optimizer.step()

            loss_hist.append(float(losses["total"].item()))

        # Final decode
        with torch.no_grad():
            z_cf_final = torch.clamp(z_cf.data, -cfg.z_clip, cfg.z_clip)
            x_cf_final = self.ae.decode(z_cf_final)
            x_full_cf  = batch_x.clone()
            x_full_cf[:, :, -1:] = x_cf_final
            y_cf_final = self.forecaster.predict_ot(
                x_full_cf, batch_x_mark)

        return z_cf_final, x_cf_final, y_cf_final, loss_hist

    # ──────────────────────────────────────────────
    # Public generate method
    # ──────────────────────────────────────────────
    def generate(self, batch_x: torch.Tensor,
                 batch_x_mark: torch.Tensor,
                 device: torch.device) -> dict:
        """
        Generate counterfactuals for a batch.

        Uses n_restarts random restarts and keeps the best z_cf
        (lowest total loss).

        Args:
            batch_x      : (B, seq_len, C)
            batch_x_mark : (B, seq_len, D)
            device       : torch.device

        Returns:
            dict with x_ot, x_cf, y_hat, y_cf, z, z_cf,
                      delta_mean, success, loss_history
        """
        self.ae.eval()
        self.forecaster.model.eval()

        batch_x      = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        x_ot         = batch_x[:, :, -1:]              # (B, seq_len, 1)

        with torch.no_grad():
            z     = self.ae.encode(x_ot)               # (B, latent_dim)
            y_hat = self.forecaster.predict_ot(
                batch_x, batch_x_mark)                  # (B, pred_len, 1)

        cfg = self.cfg
        best_loss   = float("inf")
        best_result = None

        for restart in range(cfg.n_restarts):
            # Random noise for restart (except first)
            noise = None
            if restart > 0:
                noise = torch.randn_like(z) * cfg.noise_std

            z_cf, x_cf, y_cf, loss_hist = self._optimize_single(
                z, batch_x, batch_x_mark, y_hat, device, noise)

            final_loss = loss_hist[-1]
            if final_loss < best_loss:
                best_loss   = final_loss
                best_result = (z_cf, x_cf, y_cf, loss_hist)

        z_cf_best, x_cf_best, y_cf_best, loss_hist = best_result

        # Compute metrics
        with torch.no_grad():
            y_hat_mean = y_hat[:, :, 0].mean(dim=1)
            y_cf_mean  = y_cf_best[:, :, 0].mean(dim=1)
            delta_mean = y_hat_mean - y_cf_mean
            success    = (delta_mean >= cfg.rho * y_hat_mean.abs()).float()
            rel_red    = delta_mean / (y_hat_mean.abs() + 1e-8)

        return {
            "x_ot"       : x_ot.detach().cpu(),
            "x_cf"       : x_cf_best.detach().cpu(),
            "y_hat"      : y_hat.detach().cpu(),
            "y_cf"       : y_cf_best.detach().cpu(),
            "z"          : z.detach().cpu(),
            "z_cf"       : z_cf_best.detach().cpu(),
            "delta_mean" : delta_mean.detach().cpu(),
            "success"    : success.detach().cpu(),
            "rel_red"    : rel_red.detach().cpu(),
            "loss_history": loss_hist,
            "best_loss"  : best_loss,
        }