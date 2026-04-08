import torch
import torch.nn as nn

from src.models.RL_learned_mask.actor import ActorMask
from src.models.RL_up.critic import Critic


def build_state(z: torch.Tensor, y_hat: torch.Tensor, direction: float = -1.0):
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


class ActorCriticMask(nn.Module):
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        seq_len: int = 96,
        eta: float = 0.08,
        entropy_coef: float = 0.03,
        direction: float = -1.0,
    ):
        super().__init__()

        self.actor = ActorMask(
            state_dim=state_dim,
            action_dim=latent_dim,
            seq_len=seq_len,
        )
        self.critic = Critic(state_dim=state_dim)

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.direction = direction

    def build_state(self, z: torch.Tensor, y_hat: torch.Tensor):
        return build_state(z, y_hat, self.direction)

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