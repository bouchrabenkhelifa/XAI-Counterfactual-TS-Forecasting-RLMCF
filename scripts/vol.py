import torch
import numpy as np
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.tcn_ae import TCNAutoEncoder
from src.data_provider.data_factory import data_provider

cfg_ae = load_config("assets/configs/models/ae/tcn_ae.json")
cfg_f  = load_config("assets/configs/models/itransformer/etth1_96_48_S.json")
device = get_device(cfg_f)

ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
ae.eval()

_, test_loader = data_provider(cfg_f, "test")

x_list = []
for batch in test_loader:
    x_list.append(batch[0][:, :, -1:].float())
    if sum(b.shape[0] for b in x_list) >= 4:
        break

x_ot = torch.cat(x_list, dim=0)[:4].to(device)

with torch.no_grad():
    z     = ae.encode(x_ot)
    x_rec = ae.decode(z)

x_np   = x_ot.cpu().numpy()
rec_np = x_rec.detach().cpu().numpy()
n      = x_np.shape[0]

print("── Volatilité (mean|diff|) ──────────────────────────────")
print(f"{'':>10}  {'originale':>12}  {'reconstruction':>16}  {'ratio':>8}")
print("-" * 55)

for i in range(n):
    v_orig = float(np.abs(np.diff(x_np[i,:,0])).mean())
    v_rec  = float(np.abs(np.diff(rec_np[i,:,0])).mean())
    ratio  = v_rec / (v_orig + 1e-8)
    print(f"Sample {i+1}  {v_orig:>12.5f}  {v_rec:>16.5f}  {ratio:>7.3f}x")

print()
print("── MSE reconstruction ───────────────────────────────────")
mse = float(((x_ot - x_rec)**2).mean().item())
print(f"  MSE = {mse:.6f}")

print()
print("── Autocorrélation lag-1 ────────────────────────────────")
for i in range(n):
    x_i  = x_np[i,:,0]
    r_i  = rec_np[i,:,0]
    ac_x = float(np.corrcoef(x_i[:-1], x_i[1:])[0,1])
    ac_r = float(np.corrcoef(r_i[:-1], r_i[1:])[0,1])
    print(f"Sample {i+1}  ac_orig={ac_x:.3f}  ac_rec={ac_r:.3f}")

print()
print("── Conclusion ───────────────────────────────────────────")
v_orig_mean = float(np.abs(np.diff(x_np[:,:,0], axis=1)).mean())
v_rec_mean  = float(np.abs(np.diff(rec_np[:,:,0], axis=1)).mean())
ratio_mean  = v_rec_mean / (v_orig_mean + 1e-8)
print(f"  volatilité originale      = {v_orig_mean:.5f}")
print(f"  volatilité reconstruction = {v_rec_mean:.5f}")
print(f"  ratio moyen               = {ratio_mean:.3f}x")

if ratio_mean < 0.7:
    print("  → AE lisse fortement ✗  (cause de plausibilité faible)")
elif ratio_mean < 0.9:
    print("  → AE lisse modérément ~")
else:
    print("  → AE préserve la volatilité ✓")