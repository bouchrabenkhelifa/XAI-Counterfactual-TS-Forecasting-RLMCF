"""
PatchTST — faithful adaptation of thuml/Time-Series-Library for forecasting.
Reference: https://arxiv.org/pdf/2211.14730.pdf

Key design:
  - Patches the time series into segments
  - Applies Transformer encoder on patches
  - Instance normalization (RevIN) for non-stationary series
  - Forecasting only (no task_name needed)
"""

import torch
import torch.nn as nn

from src.layers.Transformer_EncDec import Encoder, EncoderLayer
from src.layers.SelfAttention_Family import FullAttention, AttentionLayer
from src.layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten  = nn.Flatten(start_dim=-2)
        self.linear   = nn.Linear(nf, target_window)
        self.dropout  = nn.Dropout(head_dropout)

    def forward(self, x):   # x: [B, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST for univariate/multivariate forecasting.
    Input  : (B, seq_len, enc_in)
    Output : (B, pred_len, c_out)
    """

    def __init__(self, configs, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        padding       = stride

        d_model    = getattr(configs, "d_model",    128)
        n_heads    = getattr(configs, "n_heads",    8)
        e_layers   = getattr(configs, "e_layers",   3)
        d_ff       = getattr(configs, "d_ff",       256)
        dropout    = getattr(configs, "dropout",    0.1)
        activation = getattr(configs, "activation", "gelu")
        factor     = getattr(configs, "factor",     1)
        enc_in     = getattr(configs, "enc_in",     1)
        c_out      = getattr(configs, "c_out",      1)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor,
                                      attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2),
            ),
        )

        # Prediction head
        patch_num    = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = d_model * patch_num
        self.head    = FlattenHead(enc_in, self.head_nf, configs.pred_len,
                                   head_dropout=dropout)

        # Output projection (enc_in → c_out if different)
        self.proj = nn.Linear(enc_in, c_out) if enc_in != c_out else nn.Identity()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # ── Instance Normalization (RevIN) ────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()          # (B, 1, C)
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / stdev

        # ── Patch + Embed ─────────────────────────────────────────────────
        # x_enc: (B, T, C) → (B*C, T) for patch embedding
        x_enc = x_enc.permute(0, 2, 1)                        # (B, C, T)
        enc_out, n_vars = self.patch_embedding(x_enc)         # (B*C, patch_num, d_model)

        # ── Encoder ───────────────────────────────────────────────────────
        enc_out, _ = self.encoder(enc_out)                    # (B*C, patch_num, d_model)

        # Reshape: (B, C, d_model, patch_num)
        enc_out = enc_out.reshape(-1, n_vars,
                                  enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)                 # (B, C, d_model, patch_num)

        # ── Head ──────────────────────────────────────────────────────────
        dec_out = self.head(enc_out)                          # (B, C, pred_len)
        dec_out = dec_out.permute(0, 2, 1)                    # (B, pred_len, C)

        # ── De-normalize ──────────────────────────────────────────────────
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        # ── Project to c_out ──────────────────────────────────────────────
        dec_out = self.proj(dec_out)                          # (B, pred_len, c_out)
        return dec_out[:, -self.pred_len:, :]
