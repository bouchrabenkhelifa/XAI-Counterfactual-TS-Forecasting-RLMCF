import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    
def to_token(
    self,
    x_enc,
    x_mark_enc=None,
    feature_names=None,
    time_feature_names=None,
    include_time_tokens=True,
    *,
    attn=None,
    n_tokens=None,
    var_prefix="var_",
    time_prefix="time_",
):
    """
    Build token labels for attention visualizations (e.g., circuitsvis).

    Goal: return a list[str] whose length EXACTLY matches the attention matrix size.

    You can provide either:
      - attn: an attention tensor (any of [H,Q,K], [B,H,Q,K], [B,Q,K], [Q,K])
      - n_tokens: explicit token count (int)

    If neither is provided, it falls back to n_vars (+ n_time if include_time_tokens).
    """
    if x_enc.ndim != 3:
        raise ValueError(f"x_enc must be [B, L, N], got shape {tuple(x_enc.shape)}")

    _, _, n_vars = x_enc.shape  # B L N

    # --- 1) Determine target token length ---
    target = None
    if n_tokens is not None:
        target = int(n_tokens)
    elif attn is not None:
        if not torch.is_tensor(attn):
            attn = torch.tensor(attn)

        # Reduce common attention shapes to (Q,K)
        if attn.ndim == 4:      # [B,H,Q,K] or [H,B,Q,K]
            qk = attn[0, 0] if attn.shape[0] == x_enc.shape[0] else attn[0, 0]
        elif attn.ndim == 3:    # [H,Q,K] or [B,Q,K]
            qk = attn[0] if attn.shape[0] != x_enc.shape[0] else attn[0]
        elif attn.ndim == 2:    # [Q,K]
            qk = attn
        else:
            raise ValueError(f"Unexpected attn ndim={attn.ndim}, shape={tuple(attn.shape)}")

        target = int(qk.shape[-1])  # token count is K (and usually Q==K)
    else:
        # fallback assumption
        target = n_vars + (x_mark_enc.shape[-1] if (include_time_tokens and x_mark_enc is not None) else 0)

    # --- 2) Build variable token names ---
    if feature_names is None:
        var_tokens = [f"{var_prefix}{i}" for i in range(n_vars)]
    else:
        if len(feature_names) != n_vars:
            raise ValueError(f"feature_names length {len(feature_names)} != n_vars {n_vars}")
        var_tokens = list(feature_names)

    # --- 3) Optionally build time-feature token names ---
    time_tokens = []
    if include_time_tokens and x_mark_enc is not None:
        n_time = x_mark_enc.shape[-1]
        if time_feature_names is None:
            time_tokens = [f"{time_prefix}{i}" for i in range(n_time)]
        else:
            if len(time_feature_names) != n_time:
                raise ValueError(f"time_feature_names length {len(time_feature_names)} != n_time {n_time}")
            time_tokens = list(time_feature_names)

    # --- 4) Match EXACT target length (this is the key fix) ---
    tokens = var_tokens

    # Only include time tokens if they help us reach the target
    if len(tokens) < target and len(time_tokens) > 0:
        tokens = tokens + time_tokens

    # If still short, pad with synthetic tokens
    if len(tokens) < target:
        tokens = tokens + [f"tok_{i}" for i in range(len(tokens), target)]

    # If too long, truncate (e.g., embedding did not include time tokens as tokens)
    if len(tokens) > target:
        tokens = tokens[:target]

    return tokens
