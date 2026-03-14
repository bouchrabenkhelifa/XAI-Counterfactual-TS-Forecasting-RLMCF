import torch
import torch.nn as nn

from src.models.RL.actor  import Actor
from src.models.RL.critic import Critic


def build_state(z, y_hat, direction=-1.0):
    if y_hat.dim() == 3: ot = y_hat[:, :, 0]
    else:                 ot = y_hat

    summary = torch.cat([
        ot.mean(dim=1, keepdim=True),
        ot.std(dim=1,  keepdim=True) + 1e-8,
        ot.min(dim=1).values.unsqueeze(1),
        ot.max(dim=1).values.unsqueeze(1),
    ], dim=1)

    dir_token = torch.full((z.shape[0], 1), direction,
                           dtype=z.dtype, device=z.device)
    return torch.cat([z, summary, dir_token], dim=1)


class ActorCritic(nn.Module):

    def __init__(self, latent_dim=64, pred_len=48,
                 eta=0.15, entropy_coef=0.01, direction=-1.0):
        super().__init__()
        state_dim        = latent_dim + 4 + 1
        self.actor       = Actor(state_dim=state_dim, action_dim=latent_dim)
        self.critic      = Critic(state_dim=state_dim)
        self.latent_dim  = latent_dim
        self.eta         = eta
        self.entropy_coef= entropy_coef
        self.direction   = direction

    def build_state(self, z, y_hat):
        return build_state(z, y_hat, self.direction)

    def act(self, z, y_hat):
        s              = self.build_state(z, y_hat)
        a, log_prob, _ = self.actor.sample(s)
        z_cf           = torch.clamp(z + self.eta * a, -1.0, 1.0)
        return z_cf, a, log_prob, s

    def evaluate(self, s):
        return self.critic(s)

    def compute_loss(self, log_prob, reward, value, entropy=None):
        V         = value.squeeze(1)
        advantage = reward - V.detach()
        if advantage.shape[0] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss_actor  = -(advantage * log_prob).mean()
        loss_critic = nn.functional.mse_loss(V, reward.detach())

        if entropy is not None:
            loss_actor = loss_actor - self.entropy_coef * entropy.mean()

        return {
            "total"    : loss_actor + 0.5 * loss_critic,
            "actor"    : loss_actor,
            "critic"   : loss_critic,
            "advantage": float(advantage.mean().item()),
        }

    def count_parameters(self):
        return {
            "actor" : self.actor.count_parameters(),
            "critic": self.critic.count_parameters(),
            "total" : self.actor.count_parameters() + self.critic.count_parameters(),
        }