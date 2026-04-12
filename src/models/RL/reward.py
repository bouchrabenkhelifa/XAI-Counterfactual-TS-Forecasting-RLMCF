import torch
import torch.nn as nn


class CFReward(nn.Module):
    def __init__(
        self,
        ae=None,
        rho: float = 0.10,
        alpha: float = 1.0,   # validity
        beta: float = 0.7,    # proximity
        gamma: float = 0.8,   # reconstruction plausibility
        tau: float = 0.6,     # temporal consistency
        use_validity: bool = True,
        use_proximity: bool = True,
        use_reconstruction: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()
        self.ae = ae
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.use_validity = use_validity
        self.use_proximity = use_proximity
        self.use_reconstruction = use_reconstruction
        self.use_temporal = use_temporal

    def r_validity(self, y_hat, y_cf):
        if y_hat.dim() == 3:
            y_hat = y_hat[:, :, 0]
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        mean_orig = y_hat.mean(dim=1)
        mean_cf = y_cf.mean(dim=1)
        delta = self.rho * mean_orig.abs()

        return torch.sigmoid(2.0 * (mean_orig - mean_cf - delta))

    def r_proximity(self, x, x_cf):
        diff = (x_cf - x).abs().mean(dim=(1, 2))
        return torch.exp(-diff)

    def r_reconstruction(self, x_cf):
        if self.ae is None:
            return torch.ones(x_cf.shape[0], device=x_cf.device)

        z_cf = self.ae.encode(x_cf)
        x_rec = self.ae.decode(z_cf)
        rec_err = ((x_cf - x_rec) ** 2).mean(dim=(1, 2))
        return torch.exp(-6.0 * rec_err)

    def r_temporal(self, x, x_cf):
        dx = x[:, 1:, :] - x[:, :-1, :]
        dxc = x_cf[:, 1:, :] - x_cf[:, :-1, :]
        d1 = (dxc - dx).abs().mean(dim=(1, 2))

        ddx = dx[:, 1:, :] - dx[:, :-1, :]
        ddxc = dxc[:, 1:, :] - dxc[:, :-1, :]
        d2 = (ddxc - ddx).abs().mean(dim=(1, 2))

        return 0.6 * torch.exp(-d1) + 0.4 * torch.exp(-d2)

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        r_v = self.r_validity(y_hat, y_cf)
        r_p = self.r_proximity(x, x_cf)
        r_r = self.r_reconstruction(x_cf)
        r_t = self.r_temporal(x, x_cf)

        total = 0.0
        if self.use_validity:
            total = total + self.alpha * r_v
        if self.use_proximity:
            total = total + self.beta * r_p
        if self.use_reconstruction:
            total = total + self.gamma * r_r
        if self.use_temporal:
            total = total + self.tau * r_t

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