"""
Génère les figures CF pour la best config :
  mask_k=12, eta=0.15, rho=0.20, fr=0.40  (bande plus etroite)
Affichage en echelle originale (inverse normalisation).
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_last import RLMaskTrainer, run_episode_eval
from src.evaluation.unified_evaluator import validity_ratio, stepwise_validity_auc, proximity, compactness
from src.data_provider.data_factory import data_provider

cfg_rl = load_config('assets/configs/models/etth1_dataset/RL_ablations/config_v2.json')
cfg_f  = load_config('assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json')
cfg_ae = load_config('assets/configs/models/etth1_dataset/ae/tcn_ae.json')
device = get_device(cfg_f)
trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

ckpt = torch.load('assets/checkpoints/etth1_chpts/RL_v2/rl_cf_v2_etth1_agent_best.pt',
                  map_location=device, weights_only=False)
trainer.agent.actor.load_state_dict(ckpt['actor_state_dict'])
trainer.agent.eval()

# Best config
MASK_K = 12
ETA    = 0.15
RHO    = 0.20
FR     = 0.50   # best compromise: VR=0.980, Prox=0.577
SIGMA  = trainer.reward_fn.global_sigma
trainer.agent.eta = ETA
# Override reward bounds to match our evaluation config
trainer.reward_fn.rho = RHO
trainer.reward_fn.fr  = FR

# Get scaler from test dataset for inverse transform
test_data, _ = data_provider(cfg_f, "test")
scaler = test_data.scaler  # StandardScaler fitted on train

def inv(arr):
    """Inverse transform: arr shape (T,) or (T,1) -> original scale."""
    a = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    # scaler was fitted on all features — for univariate (S) it's 1 feature
    return scaler.inverse_transform(a).flatten()

print(f"Best config: mask_k={MASK_K}  eta={ETA}  rho={RHO}  fr={FR}")
print(f"gap={RHO*SIGMA:.4f}  width={FR*SIGMA:.4f}")

# Collect examples
all_x, all_xcf, all_yh, all_ycf = [], [], [], []
examples = []

for i, batch in enumerate(trainer.test_loader):
    if i >= 20: break
    ep = run_episode_eval(
        batch=batch, ae_arch=trainer.ae_arch, forecaster=trainer.forecaster,
        agent=trainer.agent, reward_fn=trainer.reward_fn, device=device,
        mask_last_k=MASK_K, mask_ramp_k=4,
    )
    if ep is None: continue
    all_x.append(ep['x_ot'].cpu().numpy())
    all_xcf.append(ep['x_cf'].cpu().numpy())
    all_yh.append(ep['y_hat'].cpu().numpy())
    all_ycf.append(ep['y_cf'].cpu().numpy())
    if len(examples) < 4:
        examples.append({
            'x_ot':  ep['x_ot'][0].cpu().numpy(),
            'x_cf':  ep['x_cf'][0].cpu().numpy(),
            'y_hat': ep['y_hat'][0].cpu().numpy(),
            'y_cf':  ep['y_cf'][0].cpu().numpy(),
        })

all_x   = np.concatenate(all_x)
all_xcf = np.concatenate(all_xcf)
all_yh  = np.concatenate(all_yh)
all_ycf = np.concatenate(all_ycf)

# Metrics with new fr
yh2    = all_yh[:, :, 0]
betas  = yh2 - RHO * SIGMA
alphas = betas - FR * SIGMA

vr   = validity_ratio(all_ycf, alphas, betas)
sauc = stepwise_validity_auc(all_ycf, alphas, betas)
prox = proximity(all_x, all_xcf)
comp = compactness(all_x, all_xcf)

print(f"\nMetrics (fr={FR}):")
print(f"  Validity Ratio : {vr:.4f}")
print(f"  Step AUC       : {sauc:.4f}")
print(f"  Proximity L2   : {prox:.4f}")
print(f"  Compactness    : {comp:.4f}")

# ── Plot CF examples — original scale ─────────────────────────────────────────
n = len(examples)
fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
if n == 1:
    axes = [axes]

for i, ex in enumerate(examples):
    x_ot  = ex['x_ot'][:, 0]   # (96,)
    x_cf  = ex['x_cf'][:, 0]
    y_hat = ex['y_hat'][:, 0]   # (48,)
    y_cf  = ex['y_cf'][:, 0]

    # Bounds (normalized)
    beta_s  = y_hat - RHO * SIGMA
    alpha_s = beta_s - FR * SIGMA

    # Inverse transform to original scale
    x_ot_orig  = inv(x_ot)
    x_cf_orig  = inv(x_cf)
    y_hat_orig = inv(y_hat)
    y_cf_orig  = inv(y_cf)
    alpha_orig = inv(alpha_s)
    beta_orig  = inv(beta_s)

    full_orig = np.concatenate([x_ot_orig, y_hat_orig])
    full_cf   = np.concatenate([x_cf_orig, y_cf_orig])
    t_all     = np.arange(len(full_orig))
    t_fore    = np.arange(len(x_ot_orig), len(full_orig))

    vh = float(((y_cf >= alpha_s) & (y_cf <= beta_s)).mean())

    ax = axes[i]
    ax.plot(t_all, full_orig, lw=1.5, color='#2196F3', label='Original + Forecast')
    ax.plot(t_all, full_cf,   lw=1.5, color='#FF9800', linestyle='--', label='CF + Forecast CF')
    ax.fill_between(t_fore, alpha_orig, beta_orig,
                    alpha=0.25, color='green', label='Target band')
    ax.axvline(len(x_ot_orig), linestyle='--', lw=1.2, color='gray')
    ax.axvspan(len(x_ot_orig) - MASK_K, len(x_ot_orig),
               alpha=0.06, color='orange', label=f'Mask (last {MASK_K})')
    ax.set_title(f'Sample {i+1} — validity_hard={vh:.2f}', fontsize=11)
    ax.set_ylabel('OT (°C)')
    # Y axis: start from 0, ticks every 2
    y_min = min(full_orig.min(), full_cf.min(), alpha_orig.min()) - 1
    y_max = max(full_orig.max(), full_cf.max(), beta_orig.max()) + 1
    y_min = max(0, y_min)  # start from 0
    ax.set_ylim(y_min, y_max)
    import numpy as _np
    ax.set_yticks(_np.arange(int(y_min), int(y_max) + 1, 2))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Best Config: mask_k={MASK_K}, eta={ETA}, rho={RHO}, fr={FR}\n'
    f'VR={vr:.4f}  AUC={sauc:.4f}  Prox={prox:.4f}  Comp={comp:.4f}',
    fontsize=12
)
fig.tight_layout()

os.makedirs('assets/figures/etth1_best_config', exist_ok=True)
path = 'assets/figures/etth1_best_config/best_config_cf_examples_original_scale.png'
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[Fig] Saved -> {path}")
