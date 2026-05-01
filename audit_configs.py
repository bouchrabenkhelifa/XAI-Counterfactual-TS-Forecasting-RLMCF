import json
import glob
import os

configs = glob.glob('assets/configs/models/**/*.json', recursive=True)
issues = []

for cfg_path in configs:
    with open(cfg_path) as f:
        data = json.load(f)
    
    dataset = 'etth1' if 'etth1' in cfg_path.lower() else 'etth2'
    
    # Vérifier checkpoint_dir
    if 'checkpoint_dir' in data:
        expected = f'assets/checkpoints/{dataset}_chpts'
        if not data['checkpoint_dir'].startswith(expected):
            issues.append({
                'file': os.path.basename(cfg_path),
                'path': cfg_path,
                'field': 'checkpoint_dir',
                'current': data['checkpoint_dir'],
                'expected': expected
            })
    
    # Vérifier results_dir
    if 'results_dir' in data:
        expected = f'assets/results/{dataset}'
        if not data['results_dir'].startswith(expected):
            issues.append({
                'file': os.path.basename(cfg_path),
                'path': cfg_path,
                'field': 'results_dir',
                'current': data['results_dir'],
                'expected': expected
            })
    
    # Vérifier figures_dir
    if 'figures_dir' in data:
        expected = f'assets/figures/{dataset}'
        if not data['figures_dir'].startswith(expected):
            issues.append({
                'file': os.path.basename(cfg_path),
                'path': cfg_path,
                'field': 'figures_dir',
                'current': data['figures_dir'],
                'expected': expected
            })

print(f'Total configs: {len(configs)}')
print(f'Problemes trouves: {len(issues)}')

if issues:
    print('\nPremiers problemes:')
    for issue in issues[:10]:
        print(f'  {issue["file"]}: {issue["field"]}')
        print(f'    Current: {issue["current"]}')
        print(f'    Expected: {issue["expected"]}')
else:
    print('\nTous les chemins sont corrects!')
