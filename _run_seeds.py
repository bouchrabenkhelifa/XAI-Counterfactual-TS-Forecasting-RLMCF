"""
Compute validity on the FULL test set (all batches) for all 5 models x 3 datasets.
Single aggregated number per config — no sampling, no variance.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, '.')
from src.utils.config import load_config
from src.training.RL_trainers.trainer_main import RLMaskTrainer, run_episode_eval

device = torch.device('cpu')

configs = [
    ('ETTh1', 'iTransformer',
     'assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_v2_etth1_agent_best.pt',
     'assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json'),
    ('ETTh1', 'PatchTST',
     'assets/checkpoints/etth1_chpts/RL/patchtst/rl_cf_patchtst_etth1_agent_best.pt',
     'assets/configs/etth1_dataset/forecasters/patchtst/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json'),
    ('ETTh1', 'TimesNet',
     'assets/checkpoints/etth1_chpts/RL/timesnet/rl_cf_timesnet_etth1_agent_best.pt',
     'assets/configs/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json'),
    ('ETTh1', 'GRU',
     'assets/checkpoints/etth1_chpts/RL/gru/rl_cf_gru_etth1_agent_best.pt',
     'assets/configs/etth1_dataset/forecasters/gru/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json'),
    ('ETTh1', 'DLinear',
     'assets/checkpoints/etth1_chpts/RL/dlinear/rl_cf_dlinear_etth1_agent_best.pt',
     'assets/configs/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json',
     'assets/configs/etth1_dataset/ae/tcn_ae.json'),
    ('ETTh2', 'iTransformer',
     'assets/checkpoints/etth2_chpts/RL/itransformer/rl_cf_itransformer_etth2_agent_best.pt',
     'assets/configs/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json'),
    ('ETTh2', 'PatchTST',
     'assets/checkpoints/etth2_chpts/RL/patchtst/rl_cf_patchtst_etth2_agent_best.pt',
     'assets/configs/etth2_dataset/forecasters/patchtst/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json'),
    ('ETTh2', 'TimesNet',
     'assets/checkpoints/etth2_chpts/RL/timesnet/rl_cf_timesnet_etth2_agent_best.pt',
     'assets/configs/etth2_dataset/forecasters/timesnet/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json'),
    ('ETTh2', 'GRU',
     'assets/checkpoints/etth2_chpts/RL/gru/rl_cf_gru_etth2_agent_best.pt',
     'assets/configs/etth2_dataset/forecasters/gru/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json'),
    ('ETTh2', 'DLinear',
     'assets/checkpoints/etth2_chpts/RL/dlinear/rl_cf_dlinear_etth2_agent_best.pt',
     'assets/configs/etth2_dataset/forecasters/dlinear/etth2_96_48_S.json',
     'assets/configs/etth2_dataset/ae/tcn_ae.json'),
    ('Weather', 'iTransformer',
     'assets/checkpoints/weather_chpts/RL/RL_itransformer/rl_cf_itransformer_weather_agent_best.pt',
     'assets/configs/weather_dataset/forecasters/itransformer/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json'),
    ('Weather', 'PatchTST',
     'assets/checkpoints/weather_chpts/RL/RL_patchtst/rl_cf_patchtst_weather_agent_best.pt',
     'assets/configs/weather_dataset/forecasters/patchtst/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json'),
    ('Weather', 'TimesNet',
     'assets/checkpoints/weather_chpts/RL/RL_timesnet/rl_cf_timesnet_weather_agent_best.pt',
     'assets/configs/weather_dataset/forecasters/timesnet/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json'),
    ('Weather', 'GRU',
     'assets/checkpoints/weather_chpts/RL/RL_gru/rl_cf_gru_weather_agent_best.pt',
     'assets/configs/weather_dataset/forecasters/gru/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json'),
    ('Weather', 'DLinear',
     'assets/checkpoints/weather_chpts/RL/RL_dlinear/rl_cf_dlinear_weather_agent_best.pt',
     'assets/configs/weather_dataset/forecasters/dlinear/weather_96_96_S.json',
     'assets/configs/weather_dataset/ae/tcn_ae.json'),
]


def eval_full_test(ckpt_path, f_cfg_path, ae_cfg_path):
    cfg_f = load_config(f_cfg_path)
    cfg_ae = load_config(ae_cfg_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    cfg_rl = ConfigObj(ckpt["cfg_rl"])
    
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.agent.eval()
    
    total_valid = 0
    total_decisions = 0
    
    with torch.no_grad():
        for batch in trainer.test_loader:
            ep = run_episode_eval(
                batch=batch,
                ae_arch=trainer.ae_arch,
                forecaster=trainer.forecaster,
                agent=trainer.agent,
                reward_fn=trainer.reward_fn,
                device=device,
                use_rl=True,
                mask_last_k=trainer.mask_last_k,
                mask_ramp_k=trainer.mask_ramp_k,
                filter_quantile=trainer.filter_quantile,
            )
            if ep is None:
                continue
            y_hat = ep["y_hat"]
            y_cf = ep["y_cf"]
            x_ot = ep["x_ot"]
            alpha, beta, _ = trainer.reward_fn.compute_bounds(y_hat, x_ot=x_ot)
            valid = ((y_cf[:, :, 0] >= alpha) & (y_cf[:, :, 0] <= beta))
            total_valid += int(valid.sum().item())
            total_decisions += valid.numel()
    
    return total_valid / total_decisions, total_decisions


print("=" * 70)
print("  RL-MCF Validity on FULL TEST SET (all batches)")
print("=" * 70)

results = {}
current_dataset = ""

for dataset, model, ckpt_path, f_cfg_path, ae_cfg_path in configs:
    if dataset != current_dataset:
        current_dataset = dataset
        print(f'\n  --- {dataset} ---')
    
    if not os.path.exists(ckpt_path):
        print(f'    {model:<14} [SKIP]')
        continue
    
    print(f'    {model:<14}', end=' ', flush=True)
    vr, n_decisions = eval_full_test(ckpt_path, f_cfg_path, ae_cfg_path)
    results[f"{dataset}/{model}"] = {'validity_full': float(vr), 'n_decisions': n_decisions}
    print(f'{vr:.4f}  (n={n_decisions})')

print(f'\n\n{"="*70}')
print(f'  {"Dataset":<10} {"Model":<14} {"Valid. (full test)"}')
print(f'  {"-"*40}')
for key, r in results.items():
    ds, mdl = key.split('/')
    print(f"  {ds:<10} {mdl:<14} {r['validity_full']:.4f}")

os.makedirs('assets/results/significance', exist_ok=True)
with open('assets/results/significance/validity_full_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved -> assets/results/significance/validity_full_test.json')
