import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, enc_in: int, seq_len: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=enc_in, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calcul de la longueur temporelle après 2 convolutions stride=2
        reduced_len = seq_len
        reduced_len = (reduced_len + 2 * 1 - 3) // 2 + 1
        reduced_len = (reduced_len + 2 * 1 - 3) // 2 + 1
        self.reduced_len = reduced_len

        self.fc = nn.Linear(hidden_dim * reduced_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.permute(0, 2, 1)      # [B, C, T]
        h = self.conv(x)            # [B, hidden_dim, T_reduced]
        h = h.reshape(h.shape[0], -1)
        z = self.fc(h)              # [B, latent_dim]
        return z