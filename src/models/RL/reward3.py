import torch
import torch.nn as nn


class CFReward(nn.Module):
    """
    Reward timestep-wise sans rho_max.
    Objectif : y_cf doit être proche de y_hat * (1 - rho),
    avec une tolérance eps de part et d'autre.
    Compatible valeurs positives ET négatives.
    """

    def __init__(
        self,
        shift: float = -0.10,
        w_validity: float = 2.0,
        w_proximity: float = 0.3,
        use_validity: bool = True,
        use_proximity: bool = True,
    ):
        super().__init__()
        self.rho = abs(shift)
        self.w_validity = w_validity
        self.w_proximity = w_proximity
        self.use_validity = use_validity
        self.use_proximity = use_proximity

    def compute_bounds(self, y_hat):
        if y_hat.dim() == 3:
            y_hat = y_hat[:, :, 0]

        target = y_hat - self.rho * y_hat.abs()  # ✅ toujours vers le bas
        eps = 0.10 * y_hat.abs()
        alpha = target - eps
        beta = target + eps

        return alpha, beta, target

    def r_validity(self, y_cf, y_target):  # ✅ signature correcte
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]
        diff = (y_cf - y_target).abs()
        return torch.exp(-diff.mean(dim=1))  # [B]

    def r_validity_hard(self, y_cf, alpha, beta):
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]
        return ((y_cf >= alpha) & (y_cf <= beta)).float().mean(dim=1)

    def r_proximity(self, x, x_cf):
        diff = (x_cf - x).abs().mean(dim=(1, 2))
        return torch.exp(-diff)

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        alpha, beta, target = self.compute_bounds(y_hat)  # ✅ on unpack target

        r_v = self.r_validity(y_cf, target)  # ✅ on passe target
        r_v_hard = self.r_validity_hard(y_cf, alpha, beta)
        r_p = self.r_proximity(x, x_cf)

        total = 0.0
        if self.use_validity:
            total = total + self.w_validity * r_v
        if self.use_proximity:
            total = total + self.w_proximity * r_p

        success = (r_v_hard >= 0.5).float()

        return {
            "total": total,
            "validity": r_v,
            "validity_hard": r_v_hard,
            "proximity": r_p,
            "reconstruction": torch.zeros_like(r_v),
            "temporal": torch.zeros_like(r_v),
            "delta_mean": r_v_hard,
            "success": success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}
