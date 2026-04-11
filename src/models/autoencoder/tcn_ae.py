# src/models/ae.py
# ─────────────────────────────────────────────────────────────
# TCN Autoencoder for time series representation learning.
#
# Receptive field avec n_blocks=6, kernel_size=3 :
#   RF = 1 + (1+2+4+8+16+32)*(3-1) = 127 > seq_len=96 ✓
#
# Compatible avec checkpoint Colab (ae_etth1.pt)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.pad  = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks,
                              dilation=dilation, padding=self.pad)
    def forward(self, x):
        return self.conv(x)[:, :, :x.shape[2]]


class TCNBlock(nn.Module):
    def __init__(self, ch, ks, dilation, dropout=0.1):
        super().__init__()
        self.c1  = CausalConv1d(ch, ch, ks, dilation)
        self.n1  = nn.LayerNorm(ch)
        self.c2  = CausalConv1d(ch, ch, ks, dilation)
        self.n2  = nn.LayerNorm(ch)
        self.dr  = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.act(self.n1(self.c1(x).transpose(1,2)).transpose(1,2))
        x = self.dr(x)
        x = self.act(self.n2(self.c2(x).transpose(1,2)).transpose(1,2))
        x = self.dr(x)
        return x + res


class TCNEncoder(nn.Module):
    def __init__(self, seq_len, n_features=1, hidden_dim=64,
                 latent_dim=64, n_blocks=6, kernel_size=3, dropout=0.1):
        super().__init__()
        self.proj   = nn.Conv1d(n_features, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.fc   = nn.Linear(hidden_dim, latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.proj(x.transpose(1, 2))
        for b in self.blocks: out = b(out)
        return self.tanh(self.fc(out.mean(2)))


class TCNDecoder(nn.Module):
    def __init__(self, seq_len, n_features=1, hidden_dim=64,
                 latent_dim=64, n_blocks=6, kernel_size=3, dropout=0.1):
        super().__init__()
        self.T = seq_len
        self.H = hidden_dim
        self.fc     = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.proj = nn.Conv1d(hidden_dim, n_features, 1)

    def forward(self, z):
        B   = z.shape[0]
        out = self.fc(z).view(B, self.H, self.T)
        for b in self.blocks: out = b(out)
        return self.proj(out).transpose(1, 2)


class TCNAutoEncoder(nn.Module):
    """
    Full TCN Autoencoder.

    Usage standard :
        ae = TCNAutoEncoder(seq_len=96, ...)
        x_recon, z = ae(x)

    Chargement depuis checkpoint :
        ae = TCNAutoEncoder.from_checkpoint("assets/checkpoints/ae/ae_etth1.pt")
    """

    def __init__(self, seq_len, n_features=1, hidden_dim=64,
                 latent_dim=64, n_blocks=6, kernel_size=3, dropout=0.1):
        super().__init__()
        kw = dict(seq_len=seq_len, n_features=n_features,
                  hidden_dim=hidden_dim, latent_dim=latent_dim,
                  n_blocks=n_blocks, kernel_size=kernel_size, dropout=dropout)
        self.encoder    = TCNEncoder(**kw)
        self.decoder    = TCNDecoder(**kw)
        self.latent_dim = latent_dim
        self.seq_len    = seq_len

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

    def receptive_field(self):
        n = len(self.encoder.blocks)
        k = self.encoder.blocks[0].c1.conv.kernel_size[0]
        return 1 + sum(2**i * (k-1) for i in range(n))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str,
                        device=None) -> "TCNAutoEncoder":
        """
        Charge le modèle depuis un checkpoint.
        Compatible Colab (n_feat) et VS Code (n_features).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt["cfg"]

        # compatibilité entre noms Colab et VS Code
        n_features = cfg.get("n_features", cfg.get("n_feat", 1))
        hidden_dim = cfg.get("hidden_dim", cfg.get("hidden", 64))
        latent_dim = cfg.get("latent_dim", cfg.get("latent", 64))
        n_blocks   = cfg.get("n_blocks",   6)
        kernel_size= cfg.get("kernel_size", cfg.get("ks", 3))
        dropout    = cfg.get("dropout",    cfg.get("drop", 0.1))

        model = cls(
            seq_len    = cfg["seq_len"],
            n_features = n_features,
            hidden_dim = hidden_dim,
            latent_dim = latent_dim,
            n_blocks   = n_blocks,
            kernel_size= kernel_size,
            dropout    = dropout,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()

        print(f"[AE] Loaded from {ckpt_path}")
        print(f"  val_loss   = {ckpt.get('val_loss', 'N/A')}")
        print(f"  seq_len    = {cfg['seq_len']}")
        print(f"  hidden_dim = {hidden_dim}")
        print(f"  latent_dim = {latent_dim}")
        print(f"  n_blocks   = {n_blocks}")
        print(f"  RF         = {model.receptive_field()}/{cfg['seq_len']} ✓")
        return model