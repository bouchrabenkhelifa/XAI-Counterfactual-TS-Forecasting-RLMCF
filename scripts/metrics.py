import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.forecaster_wrapper import ForecasterWrapper
from src.data_provider.data_factory import data_provider
import pandas as pd

cfg    = load_config("assets\configs\models\itransformer\etth1_96_48_S.json")
device = get_device(cfg)
fw     = ForecasterWrapper(cfg, device)

df      = pd.read_csv("assets/datasets/ETTh1.csv")
ot      = df[["OT"]].values.astype(np.float32)
t_train = int(0.70 * len(ot))
scaler  = StandardScaler()
scaler.fit(ot[:t_train])

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

_, test_loader = data_provider(cfg, "test")

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

for i, batch in enumerate(test_loader):
    if i >= 4: break
    batch_x, batch_y, batch_x_mark, _ = batch
    batch_x      = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)

    with torch.no_grad():
        y_pred = fw.predict_ot(batch_x, batch_x_mark)

    x_orig = inv(batch_x[0, :, 0].cpu().numpy())
    y_orig = inv(batch_y[0, -cfg.pred_len:, 0].cpu().numpy())
    p_orig = inv(y_pred[0, :, 0].cpu().numpy())

    t_past   = np.arange(96)
    t_future = np.arange(96, 96 + cfg.pred_len)

    rmse = float(np.sqrt(((y_orig - p_orig)**2).mean()))

    axes[i].plot(t_past,   x_orig, color="steelblue", lw=1.5, label="Passé réel")
    axes[i].plot(t_future, y_orig, color="steelblue", lw=1.5)
    axes[i].plot(t_future, p_orig, color="coral",     lw=1.5, ls="--", label="Prédit")
    axes[i].axvline(96, color="gray", ls="--", lw=1.2, label="now")
    axes[i].fill_between(t_future, y_orig, p_orig, alpha=0.15, color="coral")
    axes[i].set_title(f"Sample {i+1} — RMSE={rmse:.2f}°C")
    axes[i].set_ylabel("OT (°C)")
    axes[i].legend(fontsize=9)
    axes[i].grid(alpha=0.3)

plt.suptitle("iTransformer S pred=48 use_norm=True — échelle originale (°C)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/forecaster/test_forecast_48_norm_original.png", dpi=150)
plt.show()
print("Saved")