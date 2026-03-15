import torch
import torch.nn as nn
from src.models.RL.actor  import Actor
from src.models.RL.critic import Critic


def build_state_input(x_ot, y_hat, direction=-1.0):
    if y_hat.dim() == 3: ot = y_hat[:, :, 0]
    else:                 ot = y_hat

    x_flat  = x_ot[:, :, 0]                          # (B, 96)
    summary = torch.cat([
        ot.mean(dim=1, keepdim=True),
        ot.std(dim=1,  keepdim=True) + 1e-8,
        ot.min(dim=1).values.unsqueeze(1),
        ot.max(dim=1).values.unsqueeze(1),
    ], dim=1)                                          # (B, 4)
    dir_token = torch.full((x_ot.shape[0], 1), direction,
                           dtype=x_ot.dtype, device=x_ot.device)
    return torch.cat([x_flat, summary, dir_token], dim=1)  # (B, 101)


class ActorCriticInputSpace(nn.Module):

    def __init__(self, seq_len=96, pred_len=48,
                 eta=0.05, entropy_coef=0.01, direction=-1.0):
        super().__init__()
        state_dim  = seq_len + 4 + 1   # 96 + 4 + 1 = 101
        action_dim = seq_len            # 96

        self.actor       = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic      = Critic(state_dim=state_dim)
        self.seq_len     = seq_len
        self.eta         = eta
        self.entropy_coef= entropy_coef
        self.direction   = direction

    def build_state(self, x_ot, y_hat):
        return build_state_input(x_ot, y_hat, self.direction)

    def act(self, x_ot, y_hat):
        s              = self.build_state(x_ot, y_hat)
        a, log_prob, _ = self.actor.sample(s)
        delta          = self.eta * a.unsqueeze(-1)    # (B, 96, 1)
        x_cf           = x_ot + delta
        return x_cf, a, log_prob, s

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
        a = sum(p.numel() for p in self.actor.parameters()  if p.requires_grad)
        c = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)
        return {"actor": a, "critic": c, "total": a + c}