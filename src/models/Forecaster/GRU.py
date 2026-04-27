import torch
import torch.nn as nn


class Model(nn.Module):
    """
    GRU forecaster with Instance Normalization (RevIN).
    Univariate (features='S').
    Input  : (B, seq_len, enc_in)
    Output : (B, pred_len, c_out)
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.hidden_dim = getattr(configs, "d_model", 256)
        self.num_layers = getattr(configs, "e_layers", 2)
        self.dropout    = getattr(configs, "dropout", 0.1)
        enc_in          = getattr(configs, "enc_in", 1)
        c_out           = getattr(configs, "c_out", 1)

        self.gru = nn.GRU(
            input_size=enc_in,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.proj  = nn.Linear(self.hidden_dim, c_out * self.pred_len)
        self.c_out = c_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # ── Instance Normalization (RevIN) ────────────────────────────────
        mean = x_enc.mean(dim=1, keepdim=True)          # (B, 1, C)
        std  = x_enc.std(dim=1, keepdim=True) + 1e-5    # (B, 1, C)
        x_enc = (x_enc - mean) / std

        # ── GRU forward ───────────────────────────────────────────────────
        out, _ = self.gru(x_enc)           # (B, seq_len, hidden)
        last    = out[:, -1, :]            # (B, hidden)
        pred    = self.proj(last)          # (B, pred_len * c_out)
        pred    = pred.view(pred.shape[0], self.pred_len, self.c_out)

        # ── Denormalize ───────────────────────────────────────────────────
        pred = pred * std + mean
        return pred
