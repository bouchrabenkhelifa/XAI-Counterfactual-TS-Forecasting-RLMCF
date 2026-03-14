import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.forecaster_wrapper import ForecasterWrapper
from src.data_provider.data_factory import data_provider
import pandas as pd

cfg    = load_config("assets/configs/models/itransformer/etth1_96_48_S.json")
device = get_device(cfg)
fw     = ForecasterWrapper(cfg, device)

df      = pd.read_csv("assets/datasets/ETTh1.csv")
ot      = df[["OT"]].values.astype(np.float32)
t_train = int(0.70 * len(ot))
scaler  = StandardScaler()
scaler.fit(ot[:t_train])

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ── Collecter toutes les prévisions sur le test set ───────────
_, test_loader = data_provider(cfg, "test")

all_x    = []
all_true = []
all_pred = []

for batch in test_loader:
    batch_x, batch_y, batch_x_mark, _ = batch
    batch_x      = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)

    with torch.no_grad():
        y_pred = fw.predict_ot(batch_x, batch_x_mark)

    for i in range(batch_x.shape[0]):
        all_x.append(inv(batch_x[i, :, 0].cpu().numpy()))
        all_true.append(inv(batch_y[i, -cfg.pred_len:, 0].cpu().numpy()))
        all_pred.append(inv(y_pred[i, :, 0].cpu().numpy()))

print(f"Total samples : {len(all_x)}")

# ── Plot — série complète avec prévisions ─────────────────────
# On reconstruit la série en prenant chaque sample toutes les pred_len steps
stride  = cfg.pred_len   # 48 — pas de chevauchement
n_plot  = len(all_x) // stride

true_series = []
pred_series = []
x_series    = []
t_axis      = []
t           = 0

for i in range(0, len(all_x), stride):
    if i >= n_plot * stride: break
    true_series.extend(all_true[i].tolist())
    pred_series.extend(all_pred[i].tolist())
    t_axis.extend(range(t + cfg.seq_len, t + cfg.seq_len + cfg.pred_len))
    t += cfg.pred_len

true_series = np.array(true_series)
pred_series = np.array(pred_series)
t_axis      = np.array(t_axis)

fig, axes = plt.subplots(2, 1, figsize=(20, 10))

# ── Haut : série complète ─────────────────────────────────────
axes[0].plot(t_axis, true_series, color="steelblue", lw=1.0,
             label="Réel", alpha=0.9)
axes[0].plot(t_axis, pred_series, color="coral",     lw=1.0,
             ls="--", label="Forecast 48h", alpha=0.8)
axes[0].axhline(5.0, color="green", ls=":", lw=1.5,
                label="Seuil gel = 5°C")
axes[0].fill_between(t_axis, true_series, pred_series,
                     alpha=0.1, color="coral")
axes[0].set_title("iTransformer S — Prévisions 48h sur tout le test set")
axes[0].set_ylabel("OT (°C)")
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# ── Bas : erreur absolue ──────────────────────────────────────
err = np.abs(true_series - pred_series)
axes[1].fill_between(t_axis, 0, err, color="coral", alpha=0.5)
axes[1].axhline(err.mean(), color="red", ls="--", lw=1.2,
                label=f"MAE moyen = {err.mean():.2f}°C")
axes[1].set_title("Erreur absolue |réel - prédit|")
axes[1].set_ylabel("|erreur| (°C)")
axes[1].set_xlabel("Timestep")
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

rmse = float(np.sqrt(((true_series - pred_series)**2).mean()))
mae  = float(err.mean())
print(f"RMSE global = {rmse:.3f}°C")
print(f"MAE  global = {mae:.3f}°C")

plt.suptitle(f"iTransformer Univarié S — Test set complet  "
             f"RMSE={rmse:.2f}°C  MAE={mae:.2f}°C", fontsize=13)
plt.tight_layout()
plt.savefig("figures/forecaster/test.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved → figures/forecaster/test.png")