import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def make_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    month = (dt_index.month - 1) / 11.0
    day = (dt_index.day - 1) / 30.0
    weekday = dt_index.weekday / 6.0
    hour = dt_index.hour / 23.0
    return np.stack([month, day, weekday, hour], axis=1).astype(np.float32)


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        series_scaled: np.ndarray,
        dates: np.ndarray,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int = 1,
    ):
        assert series_scaled.ndim == 2 and series_scaled.shape[1] == 1
        self.X = series_scaled.astype(np.float32)
        self.dates = pd.DatetimeIndex(dates)

        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)

        total_len = self.seq_len + self.pred_len
        self.indices = np.arange(0, len(self.X) - total_len + 1, self.stride)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])

        seq_x = self.X[s:s + self.seq_len]  # [seq_len, 1]
        seq_y = self.X[s + self.seq_len - self.label_len:s + self.seq_len + self.pred_len]  # [label_len+pred_len, 1]

        dates_x = self.dates[s:s + self.seq_len]
        dates_y = self.dates[s + self.seq_len - self.label_len:s + self.seq_len + self.pred_len]

        x_mark = make_time_features(dates_x)  # [seq_len, 4]
        y_mark = make_time_features(dates_y)  # [label_len+pred_len, 4]

        return (
            torch.from_numpy(seq_x),     # x_enc
            torch.from_numpy(seq_y),     # y_full
            torch.from_numpy(x_mark),    # x_mark_enc
            torch.from_numpy(y_mark),    # y_mark_dec
        )


def load_etth1_data(
    data_path: str,
    target_col: str = "OT",
    date_col: str = "date",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    df = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    values = df[[target_col]].values.astype(np.float32)
    dates = df[date_col].values

    T = len(values)
    t_train = int(train_ratio * T)
    t_val = int((train_ratio + val_ratio) * T)

    train_values, train_dates = values[:t_train], dates[:t_train]
    val_values, val_dates = values[t_train:t_val], dates[t_train:t_val]
    test_values, test_dates = values[t_val:], dates[t_val:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values).astype(np.float32)
    val_scaled = scaler.transform(val_values).astype(np.float32)
    test_scaled = scaler.transform(test_values).astype(np.float32)

    return (
        (train_scaled, train_dates),
        (val_scaled, val_dates),
        (test_scaled, test_dates),
        scaler,
    )


def build_forecast_dataloaders(configs):
    (train_scaled, train_dates), (val_scaled, val_dates), (test_scaled, test_dates), scaler = load_etth1_data(
        data_path=configs.data_path,
        target_col=configs.target_col,
        date_col=configs.date_col,
        train_ratio=configs.train_ratio,
        val_ratio=configs.val_ratio,
    )

    train_ds = ForecastWindowDataset(
        train_scaled, train_dates,
        seq_len=configs.seq_len,
        label_len=configs.label_len,
        pred_len=configs.pred_len,
        stride=1,
    )
    val_ds = ForecastWindowDataset(
        val_scaled, val_dates,
        seq_len=configs.seq_len,
        label_len=configs.label_len,
        pred_len=configs.pred_len,
        stride=1,
    )
    test_ds = ForecastWindowDataset(
        test_scaled, test_dates,
        seq_len=configs.seq_len,
        label_len=configs.label_len,
        pred_len=configs.pred_len,
        stride=1,
    )

    train_loader = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=configs.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=configs.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scaler