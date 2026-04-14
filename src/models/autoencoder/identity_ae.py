import torch
import torch.nn as nn


class IdentityAutoEncoder(nn.Module):
    """
    Drop-in replacement for TCNAutoEncoder when use_autoencoder=False.

    Encode simply returns x flattened to (B, seq_len) so the RL agent
    operates directly in the raw input space.
    Decode reshapes back to (B, seq_len, 1).

    The latent_dim exposed to ActorCritic must equal seq_len (96).
    """

    def __init__(self, seq_len: int = 96):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = seq_len

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, seq_len, 1)  →  z : (B, seq_len)
        return x.squeeze(-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z : (B, seq_len)  →  x : (B, seq_len, 1)
        if z.dim() == 2:
            return z.unsqueeze(-1)
        return z

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))

    @property
    def val_loss(self):
        return 0.0
