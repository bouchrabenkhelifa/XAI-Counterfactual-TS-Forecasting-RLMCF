import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average to highlight trend."""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x : (B, T, C)
        # Pad both ends to keep length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x     = torch.cat([front, x, end], dim=1)
        x     = self.avg(x.permute(0, 2, 1))   # (B, C, T)
        return x.permute(0, 2, 1)               # (B, T, C)


class SeriesDecomp(nn.Module):
    """Series decomposition into trend + remainder."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """
    DLinear — Decomposition Linear forecaster.
    Reference: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)
    Input  : (B, seq_len, enc_in)
    Output : (B, pred_len, c_out)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len    = configs.seq_len
        self.pred_len   = configs.pred_len
        kernel_size     = getattr(configs, "moving_avg", 25)
        enc_in          = getattr(configs, "enc_in", 1)
        individual      = getattr(configs, "individual", False)

        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(enc_in)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(enc_in)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend    = nn.Linear(self.seq_len, self.pred_len)

        self.individual = individual
        self.enc_in     = enc_in

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc : (B, seq_len, enc_in)
        seasonal, trend = self.decomp(x_enc)

        if self.individual:
            seasonal_out = torch.zeros(
                x_enc.shape[0], self.pred_len, self.enc_in,
                device=x_enc.device, dtype=x_enc.dtype
            )
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.enc_in):
                seasonal_out[:, :, i] = self.linear_seasonal[i](seasonal[:, :, i])
                trend_out[:, :, i]    = self.linear_trend[i](trend[:, :, i])
        else:
            # (B, enc_in, seq_len) → linear → (B, enc_in, pred_len) → permute
            seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out    = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        return seasonal_out + trend_out   # (B, pred_len, enc_in)
