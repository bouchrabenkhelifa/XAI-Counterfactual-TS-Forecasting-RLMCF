import torch
import torch.nn as nn


class CFReward(nn.Module):
    def __init__(
        self,
        plausibility,
        rho: float = 0.10,
        alpha: float = 1.0,   # validity
        beta: float = 0.5,    # proximity
        gamma: float = 0.6,   # plausibility (plus fort qu'avant)
        kappa: float = 0.2,   # latent drift regularization
    ):
        super().__init__()
        self.plausibility = plausibility
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa

    def r_validity(self, y_hat, y_cf):
        if y_hat.dim() == 3:
            y_hat = y_hat[:, :, 0]
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        mean_orig = y_hat.mean(dim=1)
        mean_cf = y_cf.mean(dim=1)
        delta = self.rho * mean_orig.abs()

        # un peu moins raide que 10.0 pour éviter que la validité écrase tout trop vite
        return torch.sigmoid(8.0 * (mean_orig - mean_cf - delta))

    def r_proximity(self, x, x_cf):
        return torch.exp(-(x_cf - x).abs().mean(dim=(1, 2)))

    def r_latent_drift(self, x, x_cf):
        """
        Pénalise un déplacement latent trop fort entre x et x_cf.
        """
        ae = self.plausibility.ae
        z_x = ae.encode(x)
        z_cf = ae.encode(x_cf)

        drift = (z_cf - z_x).abs().mean(dim=1)
        return torch.exp(-drift)

    def r_plausibility(self, x_cf=None, z_cf=None):
        comp = self.plausibility.score_components(x_cf=x_cf, z_cf=z_cf)
        return comp["total"], comp["latent"], comp["recon"]

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        # 1) validité
        r_v = self.r_validity(y_hat, y_cf)

        # 2) proximité
        r_p = self.r_proximity(x, x_cf)

        # 3) plausibility sur le x_cf réel
        r_pl, r_pl_latent, r_pl_recon = self.r_plausibility(x_cf=x_cf, z_cf=z_cf)

        # 4) régularisation de dérive latente
        r_drift = self.r_latent_drift(x, x_cf)

        total = (
            self.alpha * r_v
            + self.beta * r_p
            + self.gamma * r_pl
            + self.kappa * r_drift
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
            "plausibility": r_pl,
            "plausibility_latent": r_pl_latent,
            "plausibility_recon": r_pl_recon,
            "latent_drift": r_drift,
            "delta_mean": delta_mean,
            "success": success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}