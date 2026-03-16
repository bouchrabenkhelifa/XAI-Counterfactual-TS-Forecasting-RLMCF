import torch
import torch.nn as nn


class CFReward(nn.Module):
    def __init__(self, rho=0.10, alpha=1.0, beta=0.5):
        super().__init__()
        self.rho   = rho
        self.alpha = alpha
        self.beta  = beta

    def r_validity(self, y_hat, y_cf):
        if y_hat.dim() == 3:
            y_hat = y_hat[:, :, 0]
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        mean_orig = y_hat.mean(dim=1)
        mean_cf   = y_cf.mean(dim=1)
        delta     = self.rho * mean_orig.abs()

        # reward lisse : augmente quand mean_cf <= mean_orig - delta
        return torch.sigmoid(10.0 * (mean_orig - mean_cf - delta))

    def r_proximity(self, x, x_cf):
        # plus x_cf est proche de x, plus le reward est élevé
        return torch.exp(-(x_cf - x).abs().mean(dim=(1, 2)))

    def forward(self, x, x_cf, y_hat, y_cf):
        r_v = self.r_validity(y_hat, y_cf)
        r_p = self.r_proximity(x, x_cf)

        total = self.alpha * r_v + self.beta * r_p

        if y_hat.dim() == 3:
            _yh = y_hat[:, :, 0]
        else:
            _yh = y_hat

        if y_cf.dim() == 3:
            _yc = y_cf[:, :, 0]
        else:
            _yc = y_cf

        mean_orig  = _yh.mean(dim=1)
        mean_cf    = _yc.mean(dim=1)
        delta_mean = mean_orig - mean_cf
        success    = (delta_mean >= self.rho * mean_orig.abs()).float()

        return {
            "total": total,
            "validity": r_v,
            "proximity": r_p,
            "delta_mean": delta_mean,
            "success": success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}