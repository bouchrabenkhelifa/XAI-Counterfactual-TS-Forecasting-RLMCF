# scripts/analyze_threshold.py
# ─────────────────────────────────────────────────────────────
# Analyse la distribution des prévisions sur le test set
# pour déterminer un seuil S pertinent pour le CF-RL.
#
# Usage :
#   python -m scripts.analyze_threshold
# ─────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))

from src.utils.config              import load_config
from src.utils.train_tools         import get_device
from src.models.forecaster_wrapper import ForecasterWrapper
from src.data_provider.data_factory import data_provider


CONFIG_PATH = "assets/configs/models/itransformer/etth1_96_48_S.json"

cfg    = load_config(CONFIG_PATH)
device = get_device(cfg)
fw     = ForecasterWrapper(cfg, device)

# ── Scaler ────────────────────────────────────────────────────
df      = pd.read_csv(os.path.join(cfg.root_path, cfg.data_path))
ot      = df[["OT"]].values.astype(np.float32)
t_train = int(0.70 * len(ot))
scaler  = StandardScaler()
scaler.fit(ot[:t_train])

def inv(arr):
    return scaler.inverse_transform(
        np.array(arr).reshape(-1, 1)
    ).flatten()

# ── Collecter toutes les prévisions sur test ──────────────────
_, test_loader = data_provider(cfg, "test")

all_mean_yhat  = []   # mean(forecast(x)) par sample
all_mean_x     = []   # mean(x) par sample
all_yhat_flat  = []   # toutes les valeurs prévues

print("Collecte des prévisions sur le test set ...")

for batch in test_loader:
    batch_x, batch_y, batch_x_mark, _ = batch
    batch_x      = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)

    with torch.no_grad():
        y_pred = fw.predict_ot(batch_x, batch_x_mark)  # (B, 48, 1)

    for b in range(y_pred.shape[0]):
        yhat = y_pred[b, :, 0].cpu().numpy()
        x_b  = batch_x[b, :, 0].cpu().numpy()
        all_mean_yhat.append(float(yhat.mean()))
        all_mean_x.append(float(x_b.mean()))
        all_yhat_flat.extend(yhat.tolist())

all_mean_yhat = np.array(all_mean_yhat)
all_mean_x    = np.array(all_mean_x)
all_yhat_flat = np.array(all_yhat_flat)

# ── Statistiques ──────────────────────────────────────────────
print(f"\n{'='*55}")
print("DISTRIBUTION DES PRÉVISIONS — test set (scaled)")
print(f"{'='*55}")
print(f"Nb samples        : {len(all_mean_yhat)}")
print(f"mean(forecast)    : {all_mean_yhat.mean():.4f}")
print(f"std(forecast)     : {all_mean_yhat.std():.4f}")
print(f"min(forecast)     : {all_mean_yhat.min():.4f}")
print(f"max(forecast)     : {all_mean_yhat.max():.4f}")
print(f"p10(forecast)     : {np.percentile(all_mean_yhat, 10):.4f}")
print(f"p25(forecast)     : {np.percentile(all_mean_yhat, 25):.4f}")
print(f"p75(forecast)     : {np.percentile(all_mean_yhat, 75):.4f}")
print(f"p90(forecast)     : {np.percentile(all_mean_yhat, 90):.4f}")

# ── Seuils candidats ──────────────────────────────────────────
mu  = all_mean_yhat.mean()
sig = all_mean_yhat.std()

S_candidates = {
    "mean - 0.5*std" : mu - 0.5 * sig,
    "mean - 1.0*std" : mu - 1.0 * sig,
    "mean - 1.5*std" : mu - 1.5 * sig,
    "p25"            : np.percentile(all_mean_yhat, 25),
    "p10"            : np.percentile(all_mean_yhat, 10),
}

print(f"\n{'='*55}")
print("SEUILS CANDIDATS")
print(f"{'='*55}")
print(f"{'Seuil':<20} {'S (scaled)':>12} {'S (°C)':>10} {'% samples > S':>15}")
print(f"{'-'*55}")
for name, S in S_candidates.items():
    pct_above = float((all_mean_yhat > S).mean() * 100)
    S_orig    = float(inv([S])[0])
    print(f"{name:<20} {S:>12.4f} {S_orig:>10.2f}°C {pct_above:>14.1f}%")

print(f"\n→ Un bon seuil : 30-50% des samples au-dessus")
print(f"  → l'agent a assez de samples pertinents")
print(f"  → pas trop facile, pas trop difficile")

# ── Recommandation ────────────────────────────────────────────
best_S_name = "mean - 0.5*std"
best_S      = S_candidates[best_S_name]
pct_above   = float((all_mean_yhat > best_S).mean() * 100)
best_S_orig = float(inv([best_S])[0])

print(f"\n{'='*55}")
print(f"RECOMMANDATION")
print(f"{'='*55}")
print(f"  S = {best_S_name} = {best_S:.4f} (scaled) = {best_S_orig:.2f}°C")
print(f"  {pct_above:.1f}% des samples ont forecast > S")
print(f"  → {pct_above:.1f}% des samples sont pertinents pour le CF")

# ── Figures ───────────────────────────────────────────────────
os.makedirs(cfg.figures_dir, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Distribution de mean(forecast)
axes[0].hist(all_mean_yhat, bins=50,
             color="steelblue", alpha=0.7, edgecolor="white")
for name, S in S_candidates.items():
    axes[0].axvline(S, ls="--", lw=1.2, label=f"{name}={S:.3f}")
axes[0].set_title("Distribution mean(forecast(x)) — test")
axes[0].set_xlabel("mean forecast (scaled)")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

# 2. mean(x) vs mean(forecast)
axes[1].scatter(all_mean_x, all_mean_yhat,
                alpha=0.3, s=5, color="steelblue")
axes[1].axhline(best_S, color="red", ls="--", lw=1.5,
                label=f"S={best_S:.3f}")
axes[1].set_xlabel("mean(x) passé")
axes[1].set_ylabel("mean(forecast) futur")
axes[1].set_title("Relation passé → futur prévu")
axes[1].legend(); axes[1].grid(alpha=0.3)

# 3. Distribution en °C
all_mean_yhat_orig = inv(all_mean_yhat)
axes[2].hist(all_mean_yhat_orig, bins=50,
             color="coral", alpha=0.7, edgecolor="white")
axes[2].axvline(best_S_orig, color="red", ls="--", lw=1.5,
                label=f"S={best_S_orig:.2f}°C")
axes[2].set_title("Distribution mean(forecast) — °C originaux")
axes[2].set_xlabel("OT (°C)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle("Analyse seuil CF — ETTh1 test set", fontsize=13)
plt.tight_layout()
path = os.path.join(cfg.figures_dir, "threshold_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n[Plot] Saved → {path}")