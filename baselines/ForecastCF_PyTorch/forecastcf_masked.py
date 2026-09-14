"""
ForecastCF-Masked: ForecastCF + same temporal mask as RL-MCF.
=============================================================
The temporal mask (mask_last_k, mask_ramp_k) is applied AT EVERY
optimisation step — not post-hoc — so the gradient only sees the
masked region, exactly as RL-MCF does.

Applying the mask after optimisation would be wrong: ForecastCF would
have optimised over the full horizon and the mask would merely truncate
the result without constraining the search.

Purpose (Reviewers 1213, 1223):
    Isolates the contribution of the RL policy by controlling for the
    mask.  If RL-MCF > ForecastCF-Masked, the gain comes from the
    learned policy, not from the mask alone.

Usage:
    $env:PYTHONPATH = "."
    python baselines/ForecastCF_PyTorch/run_forecastcf_masked.py \
        --dataset etth1 --model itransformer
"""

import numpy as np
import torch

from baselines.ForecastCF_PyTorch.forecastcf_pt import ForecastCFPyTorch
from src.training.RL_trainers.trainer_main import build_temporal_mask


class ForecastCFMasked(ForecastCFPyTorch):
    """
    ForecastCF with RL-MCF temporal mask applied at every optimisation step.

    Parameters
    ----------
    mask_last_k : int
        Number of fully-active timesteps at the end of the input window
        (same value used by RL-MCF, typically 12).
    mask_ramp_k : int
        Length of the linear ramp preceding the fully-active region
        (same value used by RL-MCF, typically 4).
    All other parameters are inherited from ForecastCFPyTorch.
    """

    def __init__(self, forecaster, device,
                 max_iter=100, lr=1e-4, pred_margin_weight=0.5,
                 mask_last_k=12, mask_ramp_k=4):
        super().__init__(
            forecaster=forecaster,
            device=device,
            max_iter=max_iter,
            lr=lr,
            pred_margin_weight=pred_margin_weight,
        )
        self.mask_last_k = mask_last_k
        self.mask_ramp_k = mask_ramp_k

    def transform_sample(self, x_np, alpha_np, beta_np, x_full_t, x_mark_t):
        """
        ForecastCF optimisation constrained to the masked region at every step.

        At each iteration:
            x_cf_masked = x_orig + mask * (x_cf - x_orig)
        so the forecaster only ever sees perturbations in the last k timesteps.
        The free variable x_cf is still updated globally by Adam, but the mask
        zeroes out gradients outside the active region implicitly.
        """
        x_orig  = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        alpha   = torch.tensor(alpha_np, dtype=torch.float32,
                               device=self.device).unsqueeze(0).unsqueeze(-1)
        beta    = torch.tensor(beta_np,  dtype=torch.float32,
                               device=self.device).unsqueeze(0).unsqueeze(-1)

        x_full_t = x_full_t.detach()
        x_mark_t = x_mark_t.detach()

        # Build mask once — shape [1, seq_len, 1]
        temp_mask = build_temporal_mask(
            batch_size=x_orig.shape[0],
            seq_len=x_orig.shape[1],
            channels=x_orig.shape[2],
            last_k=self.mask_last_k,
            ramp_k=self.mask_ramp_k,
            device=self.device,
        )

        # Optimisation variable — full sequence, but masked before each forward
        x_cf = x_orig.clone()

        # Adam state
        m  = torch.zeros_like(x_cf)
        v  = torch.zeros_like(x_cf)
        b1, b2, eps = 0.9, 0.999, 1e-8

        for it in range(self.max_iter):
            with torch.no_grad():
                # ── Apply mask BEFORE forecasting ─────────────────────────────
                x_cf_m = x_orig + temp_mask * (x_cf - x_orig)

                x_full_cf = torch.cat([x_full_t[:, :, :-1], x_cf_m], dim=-1)
                y_cf = self.forecaster.predict(
                    x_full_cf, x_mark_t, x_mark_t)[:, :, -1:]

                if ((y_cf >= alpha) & (y_cf <= beta)).all():
                    break

                loss_margin_curr = self._margin_mse(y_cf, alpha, beta)
                loss_prox_curr   = self._weighted_mae(x_orig, x_cf_m)
                loss_curr = (self.pred_margin_weight * loss_margin_curr
                             + (1 - self.pred_margin_weight) * loss_prox_curr)

                # SPSA gradient estimate
                epsilon = 0.01
                delta   = torch.randn_like(x_cf) * epsilon

                # Perturb only inside the masked region
                x_cf_plus   = x_cf + delta
                x_cf_plus_m = x_orig + temp_mask * (x_cf_plus - x_orig)

                x_full_cf_plus = torch.cat(
                    [x_full_t[:, :, :-1], x_cf_plus_m], dim=-1)
                y_cf_plus = self.forecaster.predict(
                    x_full_cf_plus, x_mark_t, x_mark_t)[:, :, -1:]

                loss_margin_plus = self._margin_mse(y_cf_plus, alpha, beta)
                loss_prox_plus   = self._weighted_mae(x_orig, x_cf_plus_m)
                loss_plus = (self.pred_margin_weight * loss_margin_plus
                             + (1 - self.pred_margin_weight) * loss_prox_plus)

                grad = (loss_plus - loss_curr) * delta / (epsilon ** 2)

                # Adam update
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * (grad ** 2)
                m_hat = m / (1 - b1 ** (it + 1))
                v_hat = v / (1 - b2 ** (it + 1))
                x_cf  = x_cf - self.lr * m_hat / (torch.sqrt(v_hat) + eps)

        # ── Final masked CF and forecast ──────────────────────────────────────
        with torch.no_grad():
            x_cf_m = x_orig + temp_mask * (x_cf - x_orig)
            y_cf   = self.forecaster.predict_from_ot(x_cf_m, x_full_t, x_mark_t)

        return x_cf_m.detach().cpu().numpy(), y_cf.detach().cpu().numpy()
