import torch
import torch.nn as nn


class Conv1DDecoder(nn.Module):
    def __init__(self, enc_in: int, seq_len: int, hidden_dim: int, latent_dim: int, reduced_len: int):
        super().__init__()

        self.enc_in = enc_in
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.reduced_len = reduced_len

        self.fc = nn.Linear(latent_dim, hidden_dim * reduced_len)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=hidden_dim,
                out_channels=enc_in,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim]
        h = self.fc(z)  # [B, hidden_dim * reduced_len]
        h = h.view(z.shape[0], self.hidden_dim, self.reduced_len)  # [B, hidden_dim, reduced_len]
        x_hat = self.deconv(h)  # [B, enc_in, T_approx]

        # Ajustement exact à seq_len
        if x_hat.shape[-1] > self.seq_len:
            x_hat = x_hat[..., :self.seq_len]
        elif x_hat.shape[-1] < self.seq_len:
            pad_len = self.seq_len - x_hat.shape[-1]
            x_hat = nn.functional.pad(x_hat, (0, pad_len))

        x_hat = x_hat.permute(0, 2, 1)  # [B, T, C]
        return x_hat