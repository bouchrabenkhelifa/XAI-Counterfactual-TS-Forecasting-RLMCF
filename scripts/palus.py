import torch
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.tcn_ae import TCNAutoEncoder
from src.models.plausibility import EnsemblePlausibility
from src.data_provider.data_factory import data_provider
import pandas as pd

cfg_ae = load_config("assets/configs/models/ae/tcn_ae.json")
cfg_f  = load_config("assets/configs/models/itransformer/etth1_96_48_S.json")
device = get_device(cfg_f)

ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
ae.eval()

with open("assets/checkpoints/anomaly detector/plausibility_etth1.pkl", "rb") as f:
    obj = pickle.load(f)
plausibility = obj if not isinstance(obj, dict) else (
    obj.get("ensemble") or obj.get("model") or obj.get("plausibility"))

ckpt_ae       = torch.load(cfg_ae.checkpoint_path, map_location="cpu", weights_only=False)
scaler        = StandardScaler()
scaler.mean_  = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
scaler.scale_ = np.array(ckpt_ae["scaler_std"],  dtype=np.float64)
scaler.var_   = scaler.scale_ ** 2
scaler.n_features_in_ = 1

df      = pd.read_csv("assets/datasets/ETTh1.csv")
ot      = df[["OT"]].values.astype(np.float32)
ot_test = scaler.transform(ot[int(0.80*len(ot)):]).astype(np.float32)

def make_windows(series, window=96, step=10):
    idxs = np.arange(0, len(series) - window, step)
    return np.stack([series[i:i+window] for i in idxs])

W_test = make_windows(ot_test)
t_test = torch.from_numpy(W_test).float().to(device)

# ── Test 1 : plausibilité des vraies données test ─────────────
p_real = plausibility.score(t_test)
print(f"1. Plausibilité de la série originale    : {p_real.mean():.4f}  ← référence")

# ── Test 2 : plausibilité après encode/decode SANS perturbation
with torch.no_grad():
    z     = ae.encode(t_test)
    t_rec = ae.decode(z)

p_rec = plausibility.score(t_rec)
print(f"2. Plausibilité de la série reconstruite : {p_rec.mean():.4f}  ← effet AE seul")

# ── Test 3 : plausibilité avec petite perturbation ────────────
with torch.no_grad():
    z_cf_small = torch.clamp(z + 0.025 * torch.randn_like(z), -1, 1)
    t_cf_small = ae.decode(z_cf_small)

p_small = plausibility.score(t_cf_small)
print(f"3. CF eta=0.025 (random)    : {p_small.mean():.4f}  ← AE + petite perturb")

# ── Test 4 : plausibilité avec grande perturbation ────────────
with torch.no_grad():
    z_cf_large = torch.clamp(z + 0.15 * torch.randn_like(z), -1, 1)
    t_cf_large = ae.decode(z_cf_large)

p_large = plausibility.score(t_cf_large)
print(f"4. CF eta=0.15  (random)    : {p_large.mean():.4f}  ← AE + grande perturb")

print(f"\n── Conclusion ───────────────────────────────────────────")
drop = float(p_real.mean()) - float(p_rec.mean())
print(f"  Drop dû au AE seul       : {drop:.4f}")
print(f"  Drop dû à la perturbation: {float(p_rec.mean()) - float(p_small.mean()):.4f}")

if drop > 0.2:
    print("  → Le AE est la cause principale ✓")
elif drop > 0.1:
    print("  → Le AE contribue significativement ~")
else:
    print("  → Le AE n'est pas la cause principale ✗")
    print("  → La perturbation RL est le vrai problème")