import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor):
        return self.net(s)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)