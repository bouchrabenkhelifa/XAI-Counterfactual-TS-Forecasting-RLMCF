import torch
import torch.nn as nn


class CFReward(nn.Module):
    """
    Reward aligné exactement sur ForecastCF [Wang et al. 2023].

    Bornes identiques à ForecastCF :
        sv    = median(x_ot)
        sigma = std(x_ot)
        beta  = sv * (1 + fr * sigma)
        alpha = sv * (1 - fr * sigma)
        si sv < 0 : swap alpha et beta  (pour garantir alpha <= beta)

    Ces bornes sont FIXES pour un sample donné (pas timestep-wise),
    ce qui correspond exactement à l'objectif du papier ForecastCF.

    Reward :
        R = w_validity   * r_soft    (signal continu, gradient fort)
          + w_hard_bonus * r_hard    (proportion timesteps dans [α,β])
          + w_proximity  * r_prox    (exp(-MAE(x, x_cf)))
          + w_smooth     * r_smooth  (exp(-roughness(x_cf)))
    """

    def __init__(
        self,
        fr: float = 1.0,
        rho: float = 0.1,
        w_validity: float = 4.0,
        w_proximity: float = 0.3,
        w_hard_bonus: float = 3.0,
        w_smooth: float = 0.5,
        use_validity: bool = True,
        use_proximity: bool = True,
        use_smooth: bool = True,
        direction: float = -1.0,
        global_sigma: float = None,  # fixed std from train set — eliminates train/test mismatch
    ):
        super().__init__()
        self.fr           = fr    # half-width = fr * sigma
        self.rho          = rho   # gap = rho * sigma  (separation between y_hat and beta)
        self.w_validity   = w_validity
        self.w_proximity  = w_proximity
        self.w_hard_bonus = w_hard_bonus
        self.w_smooth     = w_smooth
        self.use_validity  = use_validity
        self.use_proximity = use_proximity
        self.use_smooth    = use_smooth
        self.direction     = direction
        self.global_sigma  = global_sigma  # if set, use this instead of per-sample std

    # ─────────────────────────────────────────────────────────────────────
    def compute_bounds(self, y_hat, x_ot):
        """
        Bande STRICTEMENT en dessous de y_hat — aucune intersection garantie.

        beta  = y_hat - gap          (haut de la bande, toujours < y_hat)
        alpha = y_hat - gap - width  (bas de la bande)

        où :
            gap   = rho * std(x)   (séparation garantie entre y_hat et beta)
            width = fr  * std(x)   (largeur de la bande)

        Garantie : beta < y_hat toujours car gap > 0 (rho > 0).

        x_ot  : [B, BH, 1]
        y_hat : [B, H,  1] ou [B, H]
        retourne alpha, beta — [B, H],  target — [B, H]
        """
        x_flat = x_ot[:, :, 0] if x_ot.dim() == 3 else x_ot   # [B, BH]
        y_flat = y_hat[:, :, 0] if y_hat.dim() == 3 else y_hat  # [B, H]
        H = y_flat.shape[1]

        if self.global_sigma is not None:
            # Use fixed global sigma — same scale on train and test
            sigma = torch.full((x_flat.shape[0],), self.global_sigma,
                               dtype=x_flat.dtype, device=x_flat.device)
        else:
            sigma = x_flat.std(dim=1).clamp(min=1e-4)               # [B]

        gap   = (self.rho * sigma).unsqueeze(1).expand(-1, H)   # [B, H]
        width = (self.fr  * sigma).unsqueeze(1).expand(-1, H)   # [B, H]

        if self.direction < 0:
            beta  = y_flat - gap           # toujours < y_flat
            alpha = beta   - width
        else:
            alpha = y_flat + gap           # toujours > y_flat
            beta  = alpha  + width

        target = (alpha + beta) / 2.0
        return alpha, beta, target

    # ─────────────────────────────────────────────────────────────────────
    def r_validity_soft(self, y_cf, alpha, beta):
        """Signal continu ∈ (0,1] — gradient fort même loin des bornes."""
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        width     = (beta - alpha).clamp(min=1e-6)
        dist_low  = torch.clamp(alpha - y_cf, min=0.0)
        dist_high = torch.clamp(y_cf - beta,  min=0.0)
        dist_norm = (dist_low + dist_high) / width

        return torch.exp(-2.0 * dist_norm).mean(dim=1)  # [B]

    def r_validity_hard(self, y_cf, alpha, beta):
        """Proportion de timesteps dans [alpha, beta]."""
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]
        return ((y_cf >= alpha) & (y_cf <= beta)).float().mean(dim=1)

    def r_hard_bonus(self, y_cf, alpha, beta):
        """Fraction de timesteps valides."""
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]
        return ((y_cf >= alpha) & (y_cf <= beta)).float().mean(dim=1)

    def r_proximity(self, x, x_cf):
        """Proximité : exp(-MAE) ∈ (0,1]."""
        diff = (x_cf - x).abs().mean(dim=(1, 2))
        return torch.exp(-diff)

    def r_smooth(self, x_cf):
        """Pénalise la rugosité — encourage les transitions douces."""
        if x_cf.dim() == 3:
            x_cf = x_cf[:, :, 0]
        diff = (x_cf[:, 1:] - x_cf[:, :-1]).abs().mean(dim=1)
        return torch.exp(-5.0 * diff)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        alpha, beta, target = self.compute_bounds(y_hat, x_ot=x)

        r_v_soft = self.r_validity_soft(y_cf, alpha, beta)
        r_v_hard = self.r_validity_hard(y_cf, alpha, beta)
        r_bonus  = self.r_hard_bonus(y_cf, alpha, beta)
        r_p      = self.r_proximity(x, x_cf)
        r_s      = self.r_smooth(x_cf)

        total = torch.zeros_like(r_v_soft)
        if self.use_validity:
            total = total + self.w_validity   * r_v_soft
            total = total + self.w_hard_bonus * r_bonus
        if self.use_proximity:
            total = total + self.w_proximity  * r_p
        if self.use_smooth:
            total = total + self.w_smooth     * r_s

        success = (r_v_hard >= 0.5).float()

        return {
            "total":          total,
            "validity":       r_v_soft,
            "validity_hard":  r_v_hard,
            "proximity":      r_p,
            "smooth":         r_s,
            "reconstruction": torch.zeros_like(r_v_soft),
            "temporal":       torch.zeros_like(r_v_soft),
            "delta_mean":     r_v_hard,
            "success":        success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}
