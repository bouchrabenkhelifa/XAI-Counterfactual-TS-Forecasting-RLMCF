import torch
import torch.nn as nn


class CFReward(nn.Module):
    def __init__(
        self,
        fr: float = 1.5,
        w_validity: float = 4.0,
        w_proximity: float = 0.05,
        w_hard_bonus: float = 2.0,
        use_validity: bool = True,
        use_proximity: bool = True,
        direction: float = -1.0,
    ):
        super().__init__()
        self.fr        = fr
        self.rho       = fr
        self.w_validity   = w_validity
        self.w_proximity  = w_proximity
        self.w_hard_bonus = w_hard_bonus
        self.use_validity  = use_validity
        self.use_proximity = use_proximity
        self.direction = direction

    def compute_bounds(self, y_hat, x_ot):
        """
        Bornes relatives a y_hat — suivent la prediction point par point.

        direction=-1 : le CF doit etre SOUS y_hat
            alpha[t] = y_hat[t] - half_width
            beta[t]  = y_hat[t]

        direction=+1 : le CF doit etre AU-DESSUS de y_hat
            alpha[t] = y_hat[t]
            beta[t]  = y_hat[t] + half_width

        half_width = fr * std(x_ot) — variabilite locale du signal.

        Avantage : la bande suit exactement y_hat timestep par timestep.
        Peu importe le niveau absolu, le CF doit juste descendre/monter
        de half_width par rapport a la prediction originale.

        x_ot  : [B, back_horizon, 1]
        y_hat : [B, horizon, 1] ou [B, horizon]
        """
        if x_ot.dim() == 3:
            x_flat = x_ot[:, :, 0]
        else:
            x_flat = x_ot

        if y_hat.dim() == 3:
            y_flat = y_hat[:, :, 0]
        else:
            y_flat = y_hat

        std = x_flat.std(dim=1)              # [B]
        half_width = self.fr * std           # [B]
        hw = half_width.unsqueeze(1).expand(-1, y_flat.shape[1])  # [B, T]

        if self.direction < 0:
            # CF doit descendre sous y_hat
            alpha_out = y_flat - hw
            beta_out  = y_flat
        else:
            # CF doit monter au-dessus de y_hat
            alpha_out = y_flat
            beta_out  = y_flat + hw

        target = (alpha_out + beta_out) / 2

        return alpha_out, beta_out, target

    def r_validity_soft(self, y_cf, alpha, beta):
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        dist_low  = torch.clamp(alpha - y_cf, min=0)
        dist_high = torch.clamp(y_cf - beta,  min=0)
        width     = (beta - alpha).clamp(min=1e-6)

        dist_norm = (dist_low + dist_high) / width
        r_soft    = torch.exp(-2.0 * dist_norm)

        return r_soft.mean(dim=1)

    def r_validity_hard(self, y_cf, alpha, beta):
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]
        return ((y_cf >= alpha) & (y_cf <= beta)).float().mean(dim=1)

    def r_hard_bonus(self, y_cf, alpha, beta):
        if y_cf.dim() == 3:
            y_cf = y_cf[:, :, 0]

        B, T      = y_cf.shape
        in_bounds = (y_cf >= alpha) & (y_cf <= beta)

        consec = torch.zeros(B, device=y_cf.device)
        for t in range(T):
            still_valid = in_bounds[:, t] & (consec == t)
            consec      = consec + still_valid.float()

        return consec / T

    def r_proximity(self, x, x_cf):
        diff = (x_cf - x).abs().mean(dim=(1, 2))
        return torch.exp(-diff)

    def forward(self, x, x_cf, y_hat, y_cf, z_cf=None):
        """
        x     : [B, back_horizon, 1]
        x_cf  : [B, back_horizon, 1]
        y_hat : [B, horizon, 1]
        y_cf  : [B, horizon, 1]
        """
        alpha, beta, target = self.compute_bounds(y_hat, x_ot=x)

        r_v_soft = self.r_validity_soft(y_cf, alpha, beta)
        r_v_hard = self.r_validity_hard(y_cf, alpha, beta)
        r_bonus  = self.r_hard_bonus(y_cf, alpha, beta)
        r_p      = self.r_proximity(x, x_cf)

        total = 0.0
        if self.use_validity:
            total = total + self.w_validity   * r_v_soft
            total = total + self.w_hard_bonus * r_bonus
        if self.use_proximity:
            total = total + self.w_proximity  * r_p

        success = (r_v_hard >= 0.5).float()

        return {
            "total":          total,
            "validity":       r_v_soft,
            "validity_hard":  r_v_hard,
            "proximity":      r_p,
            "reconstruction": torch.zeros_like(r_v_soft),
            "temporal":       torch.zeros_like(r_v_soft),
            "delta_mean":     r_v_hard,
            "success":        success,
        }

    def stats(self, reward_dict):
        return {k: float(v.mean().item()) for k, v in reward_dict.items()}