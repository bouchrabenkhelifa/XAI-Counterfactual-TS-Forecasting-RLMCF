"""
Significance test on AGGREGATED validity (same metric as Table 2).
Bootstrap over batches: resample batches, compute aggregated validity each time.
"""
import os, sys, json, numpy as np, torch
from scipy import stats
sys.path.insert(0, '.')
from src.utils.config import load_config
from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.data_provider.data_factory import data_provider
from src.training.RL_trainers.trainer_main import build_temporal_mask

device = torch.device('cpu')
N_BATCHES = 50
N_BOOTSTRAP = 1000

configs = [
    ('etth1/itransformer',
     'assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json',
     'assets/configs/etth1_dataset/RL_ablations/config_v2.json',
     'assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt'),
    ('etth2/itransformer',
     'assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json',
     'assets/configs/etth2_dataset/RL/config_itransformer.json',
     'assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt'),
    ('weather/itransformer',
     'assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json',
     'assets/configs/weather_dataset/RL/config_itransformer.json',
     'assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt'),
]

results = {}
for name, f_path, ae_path, rl_path, ckpt_path in configs:
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    cfg_f = load_config(f_path)
    cfg_ae = load_config(ae_path)
    cfg_rl = load_config(rl_path)

    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    forecaster = ForecasterWrapperV2(cfg_f, device)
    forecaster.model.eval()

    agent = ActorCritic(latent_dim=cfg_ae.latent_dim, pred_len=cfg_f.pred_len,
                        eta=cfg_rl.eta, entropy_coef=0.02, direction=-1.0).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt['actor_state_dict'])
    agent.critic.load_state_dict(ckpt['critic_state_dict'])
    agent.eval()

    _, train_loader = data_provider(cfg_f, 'train')
    _stds = []
    for i, batch in enumerate(train_loader):
        if i >= 50: break
        bx, _, _, _ = batch
        _stds.append(bx[:, :, -1].numpy().std(axis=1))
    global_sigma = float(np.concatenate(_stds).mean())

    reward_fn = CFReward(fr=cfg_rl.fr, rho=cfg_rl.rho, direction=-1.0, global_sigma=global_sigma).to(device)
    _, test_loader = data_provider(cfg_f, 'test')
    lk = cfg_rl.mask_last_k
    rk = cfg_rl.mask_ramp_k

    # Collect per-batch validity matrices: list of (n_valid, n_total) per batch
    batch_valid_counts = []  # (valid_timesteps, total_timesteps) per batch
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= N_BATCHES: break
            bx, _, bx_mark, _ = batch
            bx = bx.float().to(device)
            bx_mark = bx_mark.float().to(device)
            x_ot = bx[:, :, -1:]
            z = ae.encode(x_ot)
            y_hat = forecaster.predict_ot(bx, bx_mark)
            z_cf, _, _ = agent.act_deterministic(z, y_hat)
            x_prop = ae.decode(z_cf)
            mask = build_temporal_mask(x_ot.shape[0], x_ot.shape[1], 1, lk, rk, device)
            x_cf = x_ot + mask * (x_prop - x_ot)
            y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=bx, x_mark=bx_mark)
            alpha, beta, _ = reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            
            # Count valid timesteps in this batch
            valid_mask = ((y_cf[:, :, 0] >= alpha) & (y_cf[:, :, 0] <= beta))
            n_valid = int(valid_mask.sum().item())
            n_total = valid_mask.numel()
            batch_valid_counts.append((n_valid, n_total))

    # Aggregated validity (same as Table 2)
    total_valid = sum(v for v, _ in batch_valid_counts)
    total_all = sum(t for _, t in batch_valid_counts)
    aggregated_vr = total_valid / total_all
    
    # Bootstrap: resample batches with replacement, recompute aggregated VR
    np.random.seed(42)
    n_batches = len(batch_valid_counts)
    bootstrap_vrs = []
    for _ in range(N_BOOTSTRAP):
        indices = np.random.choice(n_batches, size=n_batches, replace=True)
        boot_valid = sum(batch_valid_counts[j][0] for j in indices)
        boot_total = sum(batch_valid_counts[j][1] for j in indices)
        bootstrap_vrs.append(boot_valid / boot_total)
    
    bootstrap_vrs = np.array(bootstrap_vrs)
    ci_low = np.percentile(bootstrap_vrs, 2.5)
    ci_high = np.percentile(bootstrap_vrs, 97.5)
    std_vr = bootstrap_vrs.std()
    
    # One-sample t-test: is aggregated VR significantly > 0.5?
    t_stat, p_value = stats.ttest_1samp(bootstrap_vrs, 0.5)
    
    results[name] = {
        'mean': float(aggregated_vr),
        'std': float(std_vr),
        'n_batches': n_batches,
        'n_total_decisions': total_all,
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
        't_stat': float(t_stat),
        'p_value': float(p_value),
    }
    
    print(f'  Aggregated Validity: {aggregated_vr:.4f}')
    print(f'  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  Std: {std_vr:.4f}')
    print(f'  t-test vs 0.5: t={t_stat:.2f}, p={p_value:.2e}')
    print(f'  Total decisions: {total_all} ({n_batches} batches)')

# Save
os.makedirs('assets/results/significance', exist_ok=True)
with open('assets/results/significance/significance_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n\n{"="*60}')
print('SUMMARY (Aggregated Validity, Bootstrap CI)')
print(f'{"="*60}')
print(f'  {"Config":<25} {"VR":>8} {"95% CI":<22} {"p-value":<12}')
print(f'  {"-"*65}')
for k, v in results.items():
    print(f"  {k:<25} {v['mean']:>8.4f} [{v['ci_95_low']:.4f}, {v['ci_95_high']:.4f}]  p={v['p_value']:.2e}")
print(f'\nSaved -> assets/results/significance/significance_results.json')
