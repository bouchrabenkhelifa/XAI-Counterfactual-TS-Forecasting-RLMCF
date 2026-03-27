import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualObjective(nn.Module):
    """
    Objectif différentiable pour optimiser directement z_cf.

    L = alpha * L_goal
      + beta_x * L_prox_x
      + beta_z * L_prox_z
      + gamma * L_plaus
    """

    def __init__(
        self,
        rho: float = 0.10,
        alpha: float = 1.0,
        beta_x: float = 1.0,
        beta_z: float = 0.1,
        gamma: float = 0.5,
        plausibility=None,
    ):
        super().__init__()
        self.rho = rho
        self.alpha = alpha
        self.beta_x = beta_x
        self.beta_z = beta_z
        self.gamma = gamma
        self.plausibility = plausibility

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
        return ((z_cf - z) ** 2).mean(dim=1)

    def plausibility_loss(self, x_cf: torch.Tensor) -> torch.Tensor:
        if self.plausibility is None:
            return torch.zeros(x_cf.shape[0], device=x_cf.device, dtype=x_cf.dtype)

        plaus_value = self.plausibility(x_cf)

        if not torch.is_tensor(plaus_value):
            plaus_value = torch.tensor(
                plaus_value, device=x_cf.device, dtype=x_cf.dtype
            )

        if plaus_value.dim() == 0:
            plaus_value = plaus_value.repeat(x_cf.shape[0])

        if plaus_value.dim() > 1:
            reduce_dims = tuple(range(1, plaus_value.dim()))
            plaus_value = plaus_value.mean(dim=reduce_dims)

        return -plaus_value

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
        l_plaus = self.plausibility_loss(x_cf)

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