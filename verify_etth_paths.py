import json
import glob
import os

configs = glob.glob('assets/configs/models/**/*.json', recursive=True)
etth_configs = [c for c in configs if 'weather' not in c.lower()]
issues = []

for cfg_path in etth_configs:
    with open(cfg_path) as f:
        data = json.load(f)
    
    dataset = 'etth1' if 'etth1' in cfg_path.lower() else 'etth2'
    
    # Vérifier checkpoint_dir
    if 'checkpoint_dir' in data:
        expected_base = f'assets/checkpoints/{dataset}_chpts'
        if not data['checkpoint_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'checkpoint_dir',
                'current': data['checkpoint_dir'],
                'expected': expected_base
            })
    
    # Vérifier results_dir
    if 'results_dir' in data:
        expected_base = f'assets/results/{dataset}'
        if not data['results_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'results_dir',
                'current': data['results_dir'],
                'expected': expected_base
            })
    
    # Vérifier figures_dir
    if 'figures_dir' in data:
        expected_base = f'assets/figures/{dataset}'
        if not data['figures_dir'].startswith(expected_base):
            issues.append({
                'file': os.path.basename(cfg_path),
                'field': 'figures_dir',
                'current': data['figures_dir'],
                'expected': expected_base
            })

print(f'ETTh1 + ETTh2 configs: {len(etth_configs)}')
print(f'Problemes: {len(issues)}')

if issues:
    print('\nProblemes:')
    for issue in issues:
        print(f'  {issue["file"]}: {issue["field"]}')
        print(f'    Current: {issue["current"]}')
        print(f'    Expected: {issue["expected"]}')
else:
    print('\n✅ Tous les chemins ETTh1/ETTh2 sont alignés!')
