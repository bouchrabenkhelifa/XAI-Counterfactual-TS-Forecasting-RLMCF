"""
Test : impact de eta sur la proximity L2, sans réentraîner.
L'agent génère z_cf = z + eta * action — réduire eta réduit la perturbation.
"""
import torch
import numpy as np
from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_last import RLMaskTrainer, run_episode_eval
from src.evaluation.unified_evaluator import validity_ratio, stepwise_validity_auc, proximity, compactness

cfg_rl = load_config('assets/configs/models/etth1_dataset/RL_ablations/config_v2.json')
cfg_f  = load_config('assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json')
cfg_ae = load_config('assets/configs/models/etth1_dataset/ae/tcn_ae.json')
device = get_device(cfg_f)
trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)

ckpt = torch.load('assets/checkpoints/etth1_chpts/RL_v2/rl_cf_v2_etth1_agent_best.pt',
                  map_location=device, weights_only=False)
trainer.agent.actor.load_state_dict(ckpt['actor_state_dict'])
trainer.agent.eval()

# Bornes : rho=0.20, fr=1.00 (gap grand, bande large)
rho, fr = 0.20, 1.00
sigma = trainer.reward_fn.global_sigma

print(f"global_sigma = {sigma:.4f}")
print()
print(f"{'eta':>6} | {'VR':>8} {'AUC':>8} {'Prox L2':>10} | note")
print("-" * 55)

for eta in [0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
    trainer.agent.eta = eta  # override eta at inference

    all_x, all_xcf, all_yh, all_ycf = [], [], [], []
    for i, batch in enumerate(trainer.test_loader):
        if i >= 20: break
        ep = run_episode_eval(
            batch=batch, ae_arch=trainer.ae_arch, forecaster=trainer.forecaster,
            agent=trainer.agent, reward_fn=trainer.reward_fn, device=device,
            mask_last_k=trainer.mask_last_k, mask_ramp_k=trainer.mask_ramp_k,
        )
        if ep is None: continue
        all_x.append(ep['x_ot'].cpu().numpy())
        all_xcf.append(ep['x_cf'].cpu().numpy())
        all_yh.append(ep['y_hat'].cpu().numpy())
        all_ycf.append(ep['y_cf'].cpu().numpy())

    all_x   = np.concatenate(all_x)
    all_xcf = np.concatenate(all_xcf)
    all_yh  = np.concatenate(all_yh)
    all_ycf = np.concatenate(all_ycf)

    # Bounds with optimal rho/fr
    yh2    = all_yh[:, :, 0]
    gap    = rho * sigma
    width  = fr  * sigma
    betas  = yh2 - gap
    alphas = betas - width

    vr   = validity_ratio(all_ycf, alphas, betas)
    sauc = stepwise_validity_auc(all_ycf, alphas, betas)
    prox = proximity(all_x, all_xcf)
    comp = compactness(all_x, all_xcf)

    note = " <-- good" if vr > 0.90 and prox < 0.50 else (" <-- best" if vr > 0.95 else "")
    print(f"{eta:>6.2f} | {vr:>8.4f} {sauc:>8.4f} {prox:>10.4f} | {note}")
