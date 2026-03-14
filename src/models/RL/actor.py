import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, state_dim=69, action_dim=64,
                 log_std_min=-4.0, log_std_max=0.5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.GELU(),
        )
        self.mu_head      = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, s):
        h       = self.net(s)
        mu      = torch.tanh(self.mu_head(h))
        log_std = torch.clamp(self.log_std_head(h),
                              self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self.forward(s)
        dist        = torch.distributions.Normal(mu, log_std.exp())
        x_t         = dist.rsample()
        a           = torch.tanh(x_t)
        log_prob    = dist.log_prob(x_t)
        log_prob   -= torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(dim=-1), mu

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)