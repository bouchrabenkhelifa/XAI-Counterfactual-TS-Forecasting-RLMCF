# src/models/reward.py
# ─────────────────────────────────────────────────────────────
# Reward function for counterfactual RL pipeline.
#
# R = α·R_validity + β·R_proximity + γ·R_plausibility
#
# R_validity   : mean(ŷ_cf[:H]) crosses threshold S
# R_proximity  : x_cf close to x (L1 on OT channel only)
# R_plausibility : EnsemblePlausibility score of x_cf
#
# Threshold S = mean(ot_train) - 0.5 * std(ot_train)
# (computed once on train data, stored in config)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace


class CFReward(nn.Module):
    """
    Reward function for counterfactual generation.

    Args:
        threshold S    : forecast must go below this value
        alpha          : weight for validity reward
        beta           : weight for proximity reward
        gamma          : weight for plausibility reward
        plausibility   : EnsemblePlausibility (already fitted)
        horizon        : number of forecast steps to evaluate (default: all)
        ot_col_idx     : index of OT column in x (default: -1 = last)
    """

    def __init__(self,
                 threshold:    float,
                 plausibility,
                 alpha:        float = 1.0,
                 beta:         float = 0.5,
                 gamma:        float = 0.3,
                 horizon:      int   = None,
                 ot_col_idx:   int   = -1):
        super().__init__()

        self.threshold   = threshold
        self.plausibility= plausibility
        self.alpha       = alpha
        self.beta        = beta
        self.gamma       = gamma
        self.horizon     = horizon      # None = use all pred_len
        self.ot_col_idx  = ot_col_idx

    # ── Sub-rewards ───────────────────────────────────────────

    def r_validity(self, y_cf: torch.Tensor) -> torch.Tensor:
        """
        R_validity = sigmoid(10 * (S - mean(ŷ_cf[:H])))

        → 1.0 if mean forecast is well below threshold S
        → 0.0 if forecast stays above S

        Args:
            y_cf : (B, pred_len, C) or (B, pred_len, 1) or (B, pred_len)
        """
        # extract OT channel
        if y_cf.dim() == 3:
            ot = y_cf[:, :, self.ot_col_idx]   # (B, pred_len)
        else:
            ot = y_cf                           # (B, pred_len)

        # restrict to horizon
        if self.horizon is not None:
            ot = ot[:, :self.horizon]

        mean_cf = ot.mean(dim=1)               # (B,)
        return torch.sigmoid(10.0 * (self.threshold - mean_cf))

    def r_proximity(self, x: torch.Tensor,
                    x_cf: torch.Tensor) -> torch.Tensor:
        """
        R_proximity = exp(-||x_cf_OT - x_OT||_1 / seq_len)

        → 1.0 if x_cf = x  (no change)
        → decreases as x_cf moves away from x

        Args:
            x    : (B, seq_len, 1)  original OT window
            x_cf : (B, seq_len, 1)  counterfactual OT window
        """
        seq_len = x.shape[1]
        l1      = (x_cf - x).abs().mean(dim=(1, 2))   # (B,)
        return torch.exp(-l1)

    def r_plausibility(self, x_cf: torch.Tensor) -> torch.Tensor:
        """
        R_plausibility = EnsemblePlausibility.score(x_cf)

        → 1.0 if x_cf looks like real data
        → 0.0 if x_cf is unrealistic

        Args:
            x_cf : (B, seq_len, 1)
        """
        return self.plausibility.score(x_cf)   # (B,)

    # ── Total reward ──────────────────────────────────────────

    def forward(self,
                x:    torch.Tensor,
                x_cf: torch.Tensor,
                y_cf: torch.Tensor) -> dict:
        """
        Compute total reward and sub-rewards.

        Args:
            x    : (B, seq_len, 1)    original OT window
            x_cf : (B, seq_len, 1)    counterfactual OT window
            y_cf : (B, pred_len, C)   forecast of x_cf

        Returns:
            dict with keys :
              "total"       : (B,) total reward
              "validity"    : (B,) R_validity
              "proximity"   : (B,) R_proximity
              "plausibility": (B,) R_plausibility
        """
        r_v  = self.r_validity(y_cf)
        r_p  = self.r_proximity(x, x_cf)
        r_pl = self.r_plausibility(x_cf)

        total = (self.alpha * r_v
               + self.beta  * r_p
               + self.gamma * r_pl)

        return {
            "total"       : total,
            "validity"    : r_v,
            "proximity"   : r_p,
            "plausibility": r_pl,
        }

    def stats(self, reward_dict: dict) -> dict:
        """
        Compute mean stats for logging.

        Args:
            reward_dict : output of forward()

        Returns:
            dict with scalar means
        """
        return {k: float(v.mean().item())
                for k, v in reward_dict.items()}


# ── Threshold computation ─────────────────────────────────────

def compute_threshold(ot_train_scaled: np.ndarray,
                      k: float = 0.5) -> float:
    """
    Compute the CF target threshold on scaled OT series.

    S = mean(ot_train) - k * std(ot_train)

    A good CF should bring the forecast below S.

    Args:
        ot_train_scaled : np.ndarray (T, 1) scaled OT train series
        k               : severity factor (default 0.5)

    Returns:
        threshold S as float
    """
    mu  = float(ot_train_scaled.mean())
    sig = float(ot_train_scaled.std())
    S   = mu - k * sig
    print(f"[Threshold] mean={mu:.4f}  std={sig:.4f}  "
          f"k={k}  S={S:.4f}")
    return S