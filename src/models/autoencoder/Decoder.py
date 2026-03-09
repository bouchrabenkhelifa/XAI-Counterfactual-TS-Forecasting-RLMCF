import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, enc_in: int):
        super().__init__()

        self.seq_len = seq_len
        self.enc_in = enc_in

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        x_hat = self.net(z)  # [B, output_dim]
        x_hat = x_hat.reshape(z.shape[0], self.seq_len, self.enc_in)
        return x_hat