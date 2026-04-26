import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# ─────────────────────────────────────────────────────────────────────────────
# FFT-based period detection
# ─────────────────────────────────────────────────────────────────────────────

def fft_period(x, k=2):
    """Return top-k dominant periods from FFT of x (B, T, C)."""
    xf   = torch.fft.rfft(x, dim=1)
    freq = abs(xf).mean(dim=-1).mean(dim=0)   # (T//2+1,)
    freq[0] = 0                                # remove DC
    _, top_idx = torch.topk(freq, k)
    top_idx    = top_idx.detach().cpu().numpy()
    periods    = [x.shape[1] // max(int(i), 1) for i in top_idx]
    return periods, abs(xf).mean(dim=-1)[:, top_idx]   # periods, weights (B, k)


# ─────────────────────────────────────────────────────────────────────────────
# Inception block (2D conv)
# ─────────────────────────────────────────────────────────────────────────────

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        kernels = []
        for i in range(num_kernels):
            ks = 2 * i + 1
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                                     padding=ks // 2))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = [k(x) for k in self.kernels]
        return sum(res) / len(res)


# ─────────────────────────────────────────────────────────────────────────────
# TimesBlock
# ─────────────────────────────────────────────────────────────────────────────

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len     = configs.seq_len
        self.pred_len    = configs.pred_len
        self.k           = getattr(configs, "top_k", 5)
        d_model          = getattr(configs, "d_model", 64)
        d_ff             = getattr(configs, "d_ff", 64)
        num_kernels      = getattr(configs, "num_kernels", 6)

        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels=num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N = x.shape
        periods, period_weight = fft_period(x, self.k)

        res = []
        for period in periods:
            # Pad to multiple of period
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1) * period
                padding = torch.zeros(B, length - (self.seq_len + self.pred_len), N,
                                      device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            # Reshape to 2D
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()   # (B, N, H, W)
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # Aggregate with FFT weights
        period_weight = F.softmax(period_weight, dim=1)   # (B, k)
        res = torch.stack(res, dim=-1)                    # (B, T, N, k)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)  # (B,1,1,k)
        res = (res * period_weight).sum(dim=-1)           # (B, T, N)
        return self.norm(res + x)


# ─────────────────────────────────────────────────────────────────────────────
# TimesNet model
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    TimesNet — 2D temporal variation modeling.
    Reference: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (ICLR 2023)
    Input  : (B, seq_len, enc_in)
    Output : (B, pred_len, c_out)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        enc_in        = getattr(configs, "enc_in", 1)
        c_out         = getattr(configs, "c_out", 1)
        d_model       = getattr(configs, "d_model", 64)
        e_layers      = getattr(configs, "e_layers", 2)
        dropout       = getattr(configs, "dropout", 0.1)

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.layers        = nn.ModuleList([TimesBlock(configs) for _ in range(e_layers)])
        self.norm          = nn.LayerNorm(d_model)
        self.proj          = nn.Linear(d_model, c_out)
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc : (B, seq_len, enc_in)
        B, T, _ = x_enc.shape

        # Embed + pad with zeros for prediction horizon
        x = self.enc_embedding(x_enc)                          # (B, T, d_model)
        pad = torch.zeros(B, self.pred_len, x.shape[-1],
                          device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)                        # (B, T+H, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.proj(x)                                       # (B, T+H, c_out)
        return x[:, -self.pred_len:, :]                        # (B, pred_len, c_out)
