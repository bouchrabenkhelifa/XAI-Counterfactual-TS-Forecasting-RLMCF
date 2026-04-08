import torch
import torch.nn as nn


class ActorMask(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int,
        hidden_dim: int = 256,
        init_log_std: float = -1.5,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.mask_head = nn.Linear(hidden_dim, seq_len)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std))

    def forward(self, s: torch.Tensor):
        h = self.shared(s)

        mu = torch.tanh(self.mu_head(h))

        mask_logits = self.mask_head(h)
        mask_t = torch.sigmoid(mask_logits)

        log_std = torch.clamp(self.log_std, min=-3.0, max=-0.5)
        log_std = log_std.unsqueeze(0).expand_as(mu)

        return mu, log_std, mask_t

    def sample(self, s: torch.Tensor):
        mu, log_std, mask_t = self(s)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        a = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1)

        return a, log_prob, mu, mask_t

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)