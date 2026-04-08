import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        init_log_std: float = -1.5,   # plus conservateur que -1.0
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, s: torch.Tensor):
        h = self.net(s)
        mu = torch.tanh(self.mu_head(h))

        # clamp pour éviter une exploration trop large / instable
        log_std = torch.clamp(self.log_std, min=-3.0, max=-0.5)
        log_std = log_std.unsqueeze(0).expand_as(mu)

        return mu, log_std

    def sample(self, s: torch.Tensor):
        mu, log_std = self(s)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1)

        return a, log_prob, mu

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)