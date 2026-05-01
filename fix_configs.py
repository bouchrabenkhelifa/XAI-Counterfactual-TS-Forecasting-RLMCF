import json
import glob
import os

# Mapping des chemins à corriger
FIXES = {
    'etth1': {
        'ae': {
            'results_dir': 'assets/results/etth1/ae',
            'figures_dir': 'assets/figures/etth1/ae',
            'checkpoint_dir': 'assets/checkpoints/etth1_chpts/ae'
        },
        'anomaly_detector': {
            'results_dir': 'assets/results/etth1/anomaly_detector',
            'figures_dir': 'assets/figures/etth1/anomaly_detector',
            'checkpoint_dir': 'assets/checkpoints/etth1_chpts/anomaly_detector'
        },
        'forecaster': {
            'results_dir': 'assets/results/etth1/forecaster',
            'checkpoint_dir': 'assets/checkpoints/etth1_chpts/forecaster'
        },
        'rl': {
            'results_dir': 'assets/results/etth1/RL',
            'figures_dir': 'assets/figures/etth1/RL',
            'checkpoint_dir': 'assets/checkpoints/etth1_chpts/RL'
        }
    },
    'etth2': {
        'ae': {
            'results_dir': 'assets/results/etth2/ae',
            'figures_dir': 'assets/figures/etth2/ae',
            'checkpoint_dir': 'assets/checkpoints/etth2_chpts/ae'
        },
        'forecaster': {
            'results_dir': 'assets/results/etth2/forecaster',
            'checkpoint_dir': 'assets/checkpoints/etth2_chpts/forecaster'
        },
        'rl': {
            'results_dir': 'assets/results/etth2/RL',
            'figures_dir': 'assets/figures/etth2/RL',
            'checkpoint_dir': 'assets/checkpoints/etth2_chpts/RL'
        }
    }
}

configs = glob.glob('assets/configs/models/**/*.json', recursive=True)
updated = 0

for cfg_path in configs:
    with open(cfg_path) as f:
        data = json.load(f)
    
    dataset = 'etth1' if 'etth1' in cfg_path.lower() else 'etth2'
    cfg_lower = cfg_path.lower()
    
    # Déterminer le type (ae, forecaster, rl, anomaly_detector)
    config_type = None
    if 'ae' in cfg_lower and 'anomaly' not in cfg_lower:
        config_type = 'ae'
    elif 'anomaly' in cfg_lower:
        config_type = 'anomaly_detector'
    elif 'forecaster' in cfg_lower:
        config_type = 'forecaster'
    elif 'rl' in cfg_lower:
        config_type = 'rl'
    
    if config_type and config_type in FIXES[dataset]:
        fixes = FIXES[dataset][config_type]
        changed = False
        
        for field, new_value in fixes.items():
            if field in data and data[field] != new_value:
                old_value = data[field]
                data[field] = new_value
                changed = True
                print(f'  {field}: {old_value} -> {new_value}')
        
        if changed:
            with open(cfg_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f'✅ Updated: {os.path.basename(cfg_path)}')
            updated += 1

print(f'\n📊 Total configs updated: {updated}')
