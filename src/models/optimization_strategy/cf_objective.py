import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualObjective(nn.Module):
    """
    L = alpha * L_goal
      + beta_x * L_prox_x
      + beta_z * L_prox_z
      + gamma * L_plaus_gauss
    """

    def __init__(
        self,
        rho: float = 0.10,
        alpha: float = 1.0,
        beta_x: float = 1.0,
        beta_z: float = 0.1,
        gamma: float = 0.5,
        z_mean: torch.Tensor | None = None,
        z_std: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.rho = rho
        self.alpha = alpha
        self.beta_x = beta_x
        self.beta_z = beta_z
        self.gamma = gamma
        self.eps = eps

        if z_mean is not None:
            self.register_buffer("z_mean", z_mean)
        else:
            self.z_mean = None

        if z_std is not None:
            self.register_buffer("z_std", z_std)
        else:
            self.z_std = None

    @staticmethod
    def _to_univariate(y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            return y[:, :, 0]
        return y

    def goal_loss(self, y_hat: torch.Tensor, y_cf: torch.Tensor) -> torch.Tensor:
        y_hat_u = self._to_univariate(y_hat)
        y_cf_u = self._to_univariate(y_cf)

        mean_orig = y_hat_u.mean(dim=1)
        mean_cf = y_cf_u.mean(dim=1)

        target_delta = self.rho * mean_orig.abs()
        achieved_delta = mean_orig - mean_cf

        return F.relu(target_delta - achieved_delta)

    def proximity_x_loss(self, x: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        l1 = (x_cf - x).abs().mean(dim=(1, 2))
        l2 = ((x_cf - x) ** 2).mean(dim=(1, 2))
        return l1 + l2

    def proximity_z_loss(self, z: torch.Tensor, z_cf: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, z.dim()))
        return ((z_cf - z) ** 2).mean(dim=reduce_dims)

    def gaussian_plausibility_loss(self, z_cf: torch.Tensor) -> torch.Tensor:
        if self.z_mean is None or self.z_std is None:
            return torch.zeros(z_cf.shape[0], device=z_cf.device, dtype=z_cf.dtype)

        z_mean = self.z_mean.to(z_cf.device)
        z_std = self.z_std.to(z_cf.device)

        while z_mean.dim() < z_cf.dim():
            z_mean = z_mean.unsqueeze(0)
        while z_std.dim() < z_cf.dim():
            z_std = z_std.unsqueeze(0)

        z_norm = (z_cf - z_mean) / (z_std + self.eps)
        reduce_dims = tuple(range(1, z_cf.dim()))
        return (z_norm ** 2).mean(dim=reduce_dims)

    def forward(
        self,
        x: torch.Tensor,
        x_cf: torch.Tensor,
        z: torch.Tensor,
        z_cf: torch.Tensor,
        y_hat: torch.Tensor,
        y_cf: torch.Tensor,
    ):
        l_goal = self.goal_loss(y_hat, y_cf)
        l_prox_x = self.proximity_x_loss(x, x_cf)
        l_prox_z = self.proximity_z_loss(z, z_cf)
        l_plaus = self.gaussian_plausibility_loss(z_cf)

        total = (
            self.alpha * l_goal
            + self.beta_x * l_prox_x
            + self.beta_z * l_prox_z
            + self.gamma * l_plaus
        )

        y_hat_u = self._to_univariate(y_hat)
        y_cf_u = self._to_univariate(y_cf)

        mean_orig = y_hat_u.mean(dim=1)
        mean_cf = y_cf_u.mean(dim=1)
        delta_mean = mean_orig - mean_cf
        success = (delta_mean >= self.rho * mean_orig.abs()).float()

        return {
            "total": total,
            "goal": l_goal,
            "prox_x": l_prox_x,
            "prox_z": l_prox_z,
            "plaus": l_plaus,
            "delta_mean": delta_mean,
            "success": success,
        }

    @staticmethod
    def stats(loss_dict: dict) -> dict:
        out = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                out[k] = float(v.mean().item())
            else:
                out[k] = float(v)
        return out