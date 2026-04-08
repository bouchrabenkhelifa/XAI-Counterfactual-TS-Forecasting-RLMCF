import torch
import torch.nn as nn


class CFReward(nn.Module):
    def __init__(
        self,
        ae,
        rho: float = 0.10,
        alpha: float = 1.0,   # validity
        beta: float = 0.5,    # proximity
        gamma: float = 0.8,   # reconstruction plausibility
        tau: float = 0.4,     # temporal consistency
    ):
        super().__init__()
        self.ae = ae
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def r_validity(self, y_hat, y_cf):
        if y_hat.dim() == 3:
            y_hat = y_hat[:, :, 0]
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        mean_orig = y_hat.mean(dim=1)
        mean_cf = y_cf.mean(dim=1)
        delta = self.rho * mean_orig.abs()

        # pente modérée pour éviter la saturation
        return torch.sigmoid(4.0 * (mean_orig - mean_cf - delta))

    def r_proximity(self, x, x_cf):
        return torch.exp(-(x_cf - x).abs().mean(dim=(1, 2)))

    def r_reconstruction(self, x_cf):
        z_cf = self.ae.encode(x_cf)
        x_rec = self.ae.decode(z_cf)

        rec_err = ((x_cf - x_rec) ** 2).mean(dim=(1, 2))
        return torch.exp(-8.0 * rec_err)

    def r_temporal(self, x, x_cf):
        dx = x[:, 1:, :] - x[:, :-1, :]
        dxc = x_cf[:, 1:, :] - x_cf[:, :-1, :]
        diff = (dxc - dx).abs().mean(dim=(1, 2))
        return torch.exp(-diff)

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        r_v = self.r_validity(y_hat, y_cf)
        r_p = self.r_proximity(x, x_cf)
        r_r = self.r_reconstruction(x_cf)
        r_t = self.r_temporal(x, x_cf)

        total = (
            self.alpha * r_v
            + self.beta * r_p
            + self.gamma * r_r
            + self.tau * r_t
        )

        if y_hat.dim() == 3:
            _yh = y_hat[:, :, 0]
        else:
            _yh = y_hat

        if y_cf.dim() == 3:
            _yc = y_cf[:, :, 0]
        else:
            _yc = y_cf

        mean_orig = _yh.mean(dim=1)
        mean_cf = _yc.mean(dim=1)
        delta_mean = mean_orig - mean_cf
        success = (delta_mean >= self.rho * mean_orig.abs()).float()

        return {
            "total": total,
            "validity": r_v,
            "proximity": r_p,
            "reconstruction": r_r,
            "temporal": r_t,
            "delta_mean": delta_mean,
            "success": success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}