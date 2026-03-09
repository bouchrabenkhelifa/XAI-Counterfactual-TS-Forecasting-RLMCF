import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        bsz = x.shape[0]
        x = x.reshape(bsz, -1)  # [B, T*C]
        z = self.net(x)         # [B, latent_dim]
        return z