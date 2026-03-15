import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.models.tcn_ae import TCNAutoEncoder
from src.models.RL.agent import ActorCritic
from src.models.forecaster_wrapper import ForecasterWrapper
from src.data_provider.data_factory import data_provider
import os


cfg_f  = load_config("assets/configs/models/itransformer/etth1_96_48_S.json")
cfg_ae = load_config("assets/configs/models/ae/tcn_ae.json")
cfg_rl = load_config("assets/configs/models/rl/etth1_rl_S.json")
device = get_device(cfg_f)

ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
ae.eval()
for p in ae.parameters(): p.requires_grad_(False)

forecaster = ForecasterWrapper(cfg_f, device)
for p in forecaster.model.parameters(): p.requires_grad_(False)

agent = ActorCritic(
    latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
    eta=cfg_rl.eta, entropy_coef=cfg_rl.entropy_coef, direction=-1.0,
).to(device)

ckpt = torch.load("assets/checkpoints/rl_S/rl_agent_best.pt",
                  map_location=device, weights_only=False)
agent.actor.load_state_dict(ckpt["actor_state_dict"])
agent.critic.load_state_dict(ckpt["critic_state_dict"])
agent.eval()
print("✅ Models loaded")

ckpt_ae       = torch.load(cfg_ae.checkpoint_path, map_location="cpu", weights_only=False)
scaler        = StandardScaler()
scaler.mean_  = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
scaler.scale_ = np.array(ckpt_ae["scaler_std"],  dtype=np.float64)
scaler.var_   = scaler.scale_ ** 2
scaler.n_features_in_ = 1

def inv(arr):
    return scaler.inverse_transform(arr.reshape(-1,1)).flatten()

_, test_loader = data_provider(cfg_f, "test")
examples = []

for batch in test_loader:
    if len(examples) >= 4: break
    batch_x, batch_y, batch_x_mark, _ = batch
    batch_x        = batch_x.float().to(device)
    batch_x_mark   = batch_x_mark.float().to(device)
    x_ot           = batch_x[:, :, -1:]

    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    mask = y_hat[:,:,0].mean(dim=1) >= torch.quantile(
           y_hat[:,:,0].mean(dim=1), 0.75)
    if mask.sum() == 0: continue

    x_ot_f         = x_ot[mask]
    batch_x_f      = batch_x[mask]
    batch_x_mark_f = batch_x_mark[mask]
    y_hat_f        = y_hat[mask]
    z_f            = z[mask]

    with torch.no_grad():
        s        = agent.build_state(z_f, y_hat_f)
        _, _, mu = agent.actor.sample(s)
        z_cf     = torch.clamp(z_f + cfg_rl.eta * mu, -1.0, 1.0)
        x_cf     = ae.decode(z_cf)
        y_cf     = forecaster.predict_from_ot(
            x_ot=x_cf, x_full=batch_x_f, x_mark=batch_x_mark_f)

    for i in range(min(mask.sum().item(), 4 - len(examples))):
        examples.append({
            "x_ot" : inv(x_ot_f[i,:,0].cpu().numpy()),
            "x_cf" : inv(x_cf[i,:,0].cpu().numpy()),
            "y_hat": inv(y_hat_f[i,:,0].cpu().numpy()),
            "y_cf" : inv(y_cf[i,:,0].cpu().numpy()),
        })

print(f"Samples : {len(examples)}")

n = len(examples)
fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
if n == 1: axes = [axes]

for i, ex in enumerate(examples):
    full_orig = np.concatenate([ex["x_ot"], ex["y_hat"]])
    full_cf   = np.concatenate([ex["x_cf"], ex["y_cf"]])
    t         = np.arange(len(full_orig))
    reduction = (ex["y_hat"].mean() - ex["y_cf"].mean()) \
                / (abs(ex["y_hat"].mean()) + 1e-8) * 100
    ok = "✓" if reduction >= cfg_rl.rho * 100 else "✗"

    axes[i].plot(t, full_orig, color="steelblue", lw=1.5,
                 label="x + forecast(x)")
    axes[i].plot(t, full_cf,   color="coral",     lw=1.5,
                 ls="--", label="x_cf + forecast(x_cf)")
    axes[i].axvline(96, color="gray", ls="--", lw=1.2, label="now")
    axes[i].fill_between(t, full_orig, full_cf, alpha=0.15, color="coral")
    axes[i].set_title(
        f"Sample {i+1} — réduction={reduction:+.1f}% {ok}  "
        f"({ex['y_hat'].mean():.2f}°C → {ex['y_cf'].mean():.2f}°C)",
        fontsize=11)
    axes[i].set_ylabel("OT (°C)")
    axes[i].legend(fontsize=9); axes[i].grid(alpha=0.3)
    axes[i].text(48,  axes[i].get_ylim()[0], "PASSÉ",
                 fontsize=9, color="gray", ha="center")
    axes[i].text(120, axes[i].get_ylim()[0], "FUTUR",
                 fontsize=9, color="gray", ha="center")

plt.suptitle("Counterfactual Explanations — ETTh1 (°C)", fontsize=13)
plt.tight_layout()
os.makedirs("figures/rl_S", exist_ok=True)
plt.savefig("figures/rl_S/cf_original_scale.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → figures/rl_S/cf_original_scale.png")

for i, ex in enumerate(examples):
    diff_past   = abs(ex["x_cf"]  - ex["x_ot"]).mean()
    diff_future = abs(ex["y_cf"]  - ex["y_hat"]).mean()
    reduction   = (ex["y_hat"].mean() - ex["y_cf"].mean()) \
                  / abs(ex["y_hat"].mean()) * 100
    print(f"Sample {i+1} :")
    print(f"  diff passé   = {diff_past:.3f}°C")
    print(f"  diff futur   = {diff_future:.3f}°C")
    print(f"  réduction    = {reduction:.1f}%")
    print(f"  y_hat mean   = {ex['y_hat'].mean():.2f}°C")
    print(f"  y_cf  mean   = {ex['y_cf'].mean():.2f}°C")