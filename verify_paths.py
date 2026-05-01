import json
import glob
import os

configs = glob.glob('assets/configs/models/**/*.json', recursive=True)
issues = []

for cfg_path in configs:
    with open(cfg_path) as f:
        data = json.load(f)
    
    dataset = 'etth1' if 'etth1' in cfg_path.lower() else 'etth2'
    cfg_lower = cfg_path.lower()
    
    # Déterminer le type
    config_type = None
    if 'ae' in cfg_lower and 'anomaly' not in cfg_lower:
        config_type = 'ae'
    elif 'anomaly' in cfg_lower:
        config_type = 'anomaly_detector'
    elif 'forecaster' in cfg_lower:
        config_type = 'forecaster'
    elif 'rl' in cfg_lower:
        config_type = 'rl'
    
    # Vérifier checkpoint_dir
    if 'checkpoint_dir' in data:
        expected_base = f'assets/checkpoints/{dataset}_chpts'
        if not data['checkpoint_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'checkpoint_dir',
                'current': data['checkpoint_dir'],
                'expected_base': expected_base
            })
    
    # Vérifier results_dir
    if 'results_dir' in data:
        expected_base = f'assets/results/{dataset}'
        if not data['results_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'results_dir',
                'current': data['results_dir'],
                'expected_base': expected_base
            })
    
    # Vérifier figures_dir
    if 'figures_dir' in data:
        expected_base = f'assets/figures/{dataset}'
        if not data['figures_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'figures_dir',
                'current': data['figures_dir'],
                'expected_base': expected_base
            })

print(f'Total configs: {len(configs)}')
print(f'Problemes: {len(issues)}')

if issues:
    print('\n❌ Chemins mal alignés:')
    for issue in issues:
        print(f'  {issue["file"]}: {issue["field"]}')
        print(f'    Current: {issue["current"]}')
        print(f'    Expected base: {issue["expected_base"]}')
else:
    print('\n✅ Tous les chemins sont alignés!')
