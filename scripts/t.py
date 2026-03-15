import pickle
import numpy as np

with open("assets/checkpoints/anomaly detector/plausibility_etth1.pkl", "rb") as f:
    obj = pickle.load(f)

plausibility = obj if not isinstance(obj, dict) else (
    obj.get("ensemble") or obj.get("model") or obj.get("plausibility"))

# Vérifier les features attendues
print(f"Type : {type(plausibility).__name__}")

# Tester avec univarié (96, 1)
import torch
x_uni = torch.randn(4, 96, 1)
try:
    s = plausibility.score(x_uni)
    print(f"Univarié (96,1)  : ✓  score={s.mean():.4f}")
except Exception as e:
    print(f"Univarié (96,1)  : ✗  {e}")

# Tester avec multivarié (96, 7)
x_multi = torch.randn(4, 96, 7)
try:
    s = plausibility.score(x_multi)
    print(f"Multivarié (96,7): ✓  score={s.mean():.4f}")
except Exception as e:
    print(f"Multivarié (96,7): ✗  {e}")