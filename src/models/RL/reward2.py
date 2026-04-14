import torch
import torch.nn as nn


class CFReward(nn.Module):

    def __init__(
        self,
        rho: float = 0.10,
        rho_max: float = 0.20,
        alpha: float = 2.0,
        beta: float = 0.3,
        sigmoid_scale: float = 5.0,
    ):
        super().__init__()
        self.rho = rho
        self.rho_max = rho_max
        self.alpha = alpha
        self.beta = beta
        self.sigmoid_scale = sigmoid_scale

    def r_validity(self, y_hat, y_cf):
        if y_hat.dim() == 3:
            y_hat = y_hat[..., 0]
        if y_cf.dim() == 3:
            y_cf = y_cf[..., 0]

        mean_orig = y_hat.mean(dim=1)
        mean_cf = y_cf.mean(dim=1)

        reduction = (mean_orig - mean_cf) / (mean_orig.abs() + 1e-8)

        s = self.sigmoid_scale
        r_below = torch.sigmoid(s * (reduction - self.rho))
        r_cap = torch.sigmoid(s * (self.rho_max - reduction))

        r_v = r_below * r_cap
        return r_v, reduction

    def r_proximity(self, x, x_cf):
        diff = (x_cf - x).abs().mean(dim=(1, 2))
        r_p = torch.exp(-diff)
        return r_p, diff

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        r_v, reduction = self.r_validity(y_hat, y_cf)
        r_p, _ = self.r_proximity(x, x_cf)

        total = self.alpha * r_v + self.beta * r_p

        success = ((reduction >= self.rho) & (reduction <= self.rho_max)).float()

        return {
            "total": total,
            "validity": r_v,
            "proximity": r_p,
            "reconstruction": torch.zeros_like(r_v),
            "temporal": torch.zeros_like(r_v),
            "delta_mean": reduction,
            "success": success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}
