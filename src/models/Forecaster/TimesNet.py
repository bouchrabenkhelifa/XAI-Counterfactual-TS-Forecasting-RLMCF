"""
TimesNet — faithful reimplementation of the official thuml/Time-Series-Library version.
Reference: https://raw.githubusercontent.com/thuml/Time-Series-Library/main/models/TimesNet.py

Key differences vs our previous version:
  - predict_linear projects seq_len → pred_len+seq_len BEFORE TimesBlocks (not after)
  - Non-stationary normalization (detached mean/std) instead of RevIN
  - Uses src/layers/Embed.py DataEmbedding (same as iTransformer)
  - Inception_Block_V1 from layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.Embed import DataEmbedding


# ─────────────────────────────────────────────────────────────────────────────
# Inception block (faithful replica of Conv_Blocks.Inception_Block_V1)
# ─────────────────────────────────────────────────────────────────────────────

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        kernels = []
        for i in range(num_kernels):
            ks = 2 * i + 1
            kernels.append(nn.Conv2d(in_channels, out_channels,
                                     kernel_size=ks, padding=ks // 2))
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
# FFT period detection (faithful replica)
# ─────────────────────────────────────────────────────────────────────────────

def FFT_for_Period(x, k=2):
    """x : (B, T, C) — returns top-k periods and their weights."""
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)   # (T//2+1,)
    frequency_list[0] = 0                        # remove DC
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list             # array of periods
    return period, abs(xf).mean(-1)[:, top_list]  # (B, k)


# ─────────────────────────────────────────────────────────────────────────────
# TimesBlock (faithful replica)
# ─────────────────────────────────────────────────────────────────────────────

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.k        = getattr(configs, "top_k", 5)

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=getattr(configs, "num_kernels", 6)),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=getattr(configs, "num_kernels", 6)),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(B, length - (self.seq_len + self.pred_len), N,
                                      device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2).contiguous()   # (B, N, H, W)
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)                    # (B, T, N, k)
        period_weight = F.softmax(period_weight, dim=1)   # (B, k)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, dim=-1)      # (B, T, N)
        return res + x                                    # residual


# ─────────────────────────────────────────────────────────────────────────────
# TimesNet Model (faithful replica — forecasting only)
# ─────────────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    TimesNet for long/short-term forecasting.
    Faithful replica of thuml/Time-Series-Library.

    Key design:
      1. Non-stationary normalization (detached mean/std)
      2. DataEmbedding (value + time features)
      3. predict_linear: seq_len → pred_len+seq_len BEFORE TimesBlocks
      4. De-normalization at output
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = getattr(configs, "e_layers", 2)

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            getattr(configs, "embed", "timeF"),
            getattr(configs, "freq", "h"),
            getattr(configs, "dropout", 0.1),
        )

        self.model      = nn.ModuleList([TimesBlock(configs) for _ in range(self.e_layers)])
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Project temporal dimension: seq_len → pred_len + seq_len
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection     = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # ── Non-stationary normalization ──────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()          # (B, 1, C)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / stdev

        # ── Embedding ─────────────────────────────────────────────────────
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4,
                                     device=x_enc.device)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)       # (B, T, d_model)

        # ── Temporal projection BEFORE blocks ─────────────────────────────
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # now enc_out : (B, pred_len+seq_len, d_model)

        # ── TimesBlocks ───────────────────────────────────────────────────
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))

        # ── Project to output ─────────────────────────────────────────────
        dec_out = self.projection(enc_out)                     # (B, T+H, c_out)

        # ── De-normalization ──────────────────────────────────────────────
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, self.pred_len + self.seq_len, 1)

        return dec_out[:, -self.pred_len:, :]                  # (B, pred_len, c_out)
