import json
import glob

configs = glob.glob('assets/configs/models/etth2_dataset/RL/*.json', recursive=True)

for cfg_path in configs:
    with open(cfg_path) as f:
        data = json.load(f)
    
    # Extraire le modèle du nom
    name = data.get('name', '')
    model = None
    for m in ['dlinear', 'gru', 'itransformer', 'patchtst', 'timesnet']:
        if m in name.lower():
            model = m
            break
    
    if model:
        # Corriger les chemins
        old_checkpoint = data.get('checkpoint_dir_lp', '')
        data['checkpoint_dir_lp'] = f'assets/checkpoints/etth2_chpts/RL/RL_{model}'
        data['figures_dir_lp'] = f'assets/figures/etth2/forecaster/{model}'
        data['results_dir_lp'] = f'assets/results/etth2/RL_{model}'
        
        with open(cfg_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f'Updated: {cfg_path}')
        print(f'  Old: {old_checkpoint}')
        print(f'  New: {data["checkpoint_dir_lp"]}')
