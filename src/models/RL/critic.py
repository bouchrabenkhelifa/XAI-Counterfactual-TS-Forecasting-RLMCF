import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, state_dim=69):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128),       nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, s):
        return self.net(s)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)