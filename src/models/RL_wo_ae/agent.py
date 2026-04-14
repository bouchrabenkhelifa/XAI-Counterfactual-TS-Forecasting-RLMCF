import torch
import torch.nn as nn
from torch.distributions import Normal


class NoAEActor(nn.Module):
    def __init__(self, seq_len=96, pred_len=48, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.action_dim = seq_len

        in_dim = seq_len + pred_len

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, s):
        h = self.net(s)
        mu = self.mu_head(h)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        std = log_std.exp()
        dist = Normal(mu, std)
        a = dist.rsample()
        log_prob = dist.log_prob(a).sum(dim=-1)
        return a, log_prob, dist


class NoAECritic(nn.Module):
    def __init__(self, seq_len=96, pred_len=48, hidden_dim=256):
        super().__init__()
        in_dim = seq_len + pred_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


class NoAEActorCritic(nn.Module):
    def __init__(self, seq_len=96, pred_len=48, eta=0.03, entropy_coef=0.03, direction=-1.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.eta = eta
        self.entropy_coef = entropy_coef
        self.direction = direction

        self.actor = NoAEActor(seq_len=seq_len, pred_len=pred_len)
        self.critic = NoAECritic(seq_len=seq_len, pred_len=pred_len)

    def build_state(self, x_ot, y_hat):
        # x_ot: (B, seq_len, 1)
        # y_hat: (B, pred_len, 1) or compatible
        x_flat = x_ot.squeeze(-1)
        y_flat = y_hat.squeeze(-1)
        return torch.cat([x_flat, self.direction * y_flat], dim=-1)

    def evaluate(self, s):
        return self.critic(s)

    def act_deterministic(self, x_ot, y_hat):
        s = self.build_state(x_ot, y_hat)
        mu, _ = self.actor(s)
        a = mu.view(-1, self.seq_len, 1)
        return a, s

    def compute_loss(self, log_prob, reward, value, entropy):
        advantage = reward.detach() - value
        actor_loss = -(log_prob * advantage.detach()).mean() - self.entropy_coef * entropy.mean()
        critic_loss = (advantage ** 2).mean()
        total_loss = actor_loss + 0.5 * critic_loss

        return {
            "total": total_loss,
            "actor": actor_loss,
            "critic": critic_loss,
            "advantage": float(advantage.mean().item()),
        }

    def count_parameters(self):
        return {
            "actor": sum(p.numel() for p in self.actor.parameters() if p.requires_grad),
            "critic": sum(p.numel() for p in self.critic.parameters() if p.requires_grad),
        }