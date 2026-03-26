import torch
import torch.nn as nn

from src.models.RL_up.actor import Actor
from src.models.RL_up.critic import Critic


def build_state(z: torch.Tensor, y_hat: torch.Tensor, direction: float = -1.0):
    """
    State = latent z + simple forecast summary + direction token
    """
    if y_hat.dim() == 3:
        ot = y_hat[:, :, 0]
    else:
        ot = y_hat

    f_mean = ot.mean(dim=1, keepdim=True)
    f_std = ot.std(dim=1, keepdim=True) + 1e-8
    f_min = ot.min(dim=1).values.unsqueeze(1)
    f_max = ot.max(dim=1).values.unsqueeze(1)

    T = ot.shape[1]
    t = torch.linspace(0, 1, T, device=ot.device).unsqueeze(0).expand(ot.shape[0], -1)
    t_mean = t.mean(dim=1, keepdim=True)
    ot_centered = ot - f_mean
    t_centered = t - t_mean
    slope = (ot_centered * t_centered).sum(dim=1, keepdim=True) / (
        (t_centered ** 2).sum(dim=1, keepdim=True) + 1e-8
    )

    ot_norm = ot_centered / (f_std + 1e-8)
    skew = (ot_norm ** 3).mean(dim=1, keepdim=True)

    summary = torch.cat([f_mean, f_std, f_min, f_max, slope, skew], dim=1)

    B = z.shape[0]
    dir_token = torch.full((B, 1), direction, dtype=z.dtype, device=z.device)

    return torch.cat([z, summary, dir_token], dim=1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        pred_len: int = 48,
        eta: float = 0.15,
        entropy_coef: float = 0.01,
        direction: float = -1.0,
    ):
        super().__init__()

        state_dim = latent_dim + 6 + 1
        self.actor = Actor(state_dim=state_dim, action_dim=latent_dim)
        self.critic = Critic(state_dim=state_dim)

        self.latent_dim = latent_dim
        self.pred_len = pred_len
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.direction = direction

    def build_state(self, z: torch.Tensor, y_hat: torch.Tensor):
        return build_state(z, y_hat, self.direction)

    def act(self, z: torch.Tensor, y_hat: torch.Tensor):
        s = self.build_state(z, y_hat)
        a, log_prob, _ = self.actor.sample(s)
        z_cf = torch.clamp(z + self.eta * a, -1.0, 1.0)
        return z_cf, a, log_prob, s

    def evaluate(self, s: torch.Tensor):
        return self.critic(s)

    def compute_loss(self, log_prob, reward, value, entropy=None):
        V = value.squeeze(1)
        advantage = reward - V.detach()

        if advantage.shape[0] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss_actor = -(advantage * log_prob).mean()
        loss_critic = nn.functional.mse_loss(V, reward.detach())

        if entropy is not None:
            loss_actor = loss_actor - self.entropy_coef * entropy.mean()

        return {
            "total": loss_actor + 0.5 * loss_critic,
            "actor": loss_actor,
            "critic": loss_critic,
            "advantage": float(advantage.mean().item()),
        }

    def count_parameters(self):
        return {
            "actor": self.actor.count_parameters(),
            "critic": self.critic.count_parameters(),
            "total": self.actor.count_parameters() + self.critic.count_parameters(),
        }