"""
Script pour générer des exemples qualitatifs de counterfactuals
Pour l'annexe ICONIP - Section E
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ajouter le projet au path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.training.RL_trainers.trainer_main import RLMaskTrainer
from baselines.common.bounds import compute_bounds_np


def generate_qualitative_examples(
    dataset="etth1",
    model="itransformer",
    n_examples=4,
    batch_idx=0
):
    """
    Génère des exemples qualitatifs de counterfactuals.
    
    Args:
        dataset: nom du dataset
        model: nom du modèle
        n_examples: nombre d'exemples à générer
        batch_idx: index du batch à utiliser
    
    Returns:
        dict avec les données pour plotting
    """
    
    # Charger configs
    seq_config = "96_96" if dataset == "weather" else "96_48"
    cfg_f_path = f"assets/configs/{dataset}_dataset/forecasters/{model}/{dataset}_{seq_config}_S.json"
    cfg_ae_path = f"assets/configs/{dataset}_dataset/ae/tcn_ae.json"
    
    # Chemin du config RL selon le dataset
    if dataset == "etth1":
        # ETTh1 utilise RL_ablations
        if model == "itransformer":
            rl_config_path = f"assets/configs/{dataset}_dataset/RL_ablations/config_itransformer_best.json"
        else:
            rl_config_path = f"assets/configs/{dataset}_dataset/RL_ablations/config_{model}.json"
    else:
        # ETTh2 et Weather utilisent RL
        rl_config_path = f"assets/configs/{dataset}_dataset/RL/config_{model}.json"
    
    cfg_f = load_config(cfg_f_path)
    cfg_ae = load_config(cfg_ae_path)
    cfg_rl = load_config(rl_config_path)
    device = get_device(cfg_f)
    
    print(f"\n{'='*60}")
    print(f"Generating Qualitative Examples")
    print(f"Dataset: {dataset} | Model: {model}")
    print(f"{'='*60}\n")
    
    # Créer trainer
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    
    # Charger checkpoint
    ckpt_path = os.path.join(
        cfg_rl.checkpoint_dir_lp,
        f"{cfg_rl.name}_agent_best.pt"
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
        trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"[✓] Loaded checkpoint: {ckpt_path}\n")
    else:
        print(f"[!] Checkpoint not found: {ckpt_path}\n")
    
    trainer.agent.eval()
    
    # Récupérer un batch
    for i, batch in enumerate(trainer.test_loader):
        if i == batch_idx:
            break
    
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    
    # Limiter au nombre d'exemples demandés
    batch_x = batch_x[:n_examples]
    batch_x_mark = batch_x_mark[:n_examples]
    
    # Prédiction originale
    with torch.no_grad():
        y_hat = trainer.forecaster.predict_ot(batch_x, batch_x_mark)
    
    # Calculer les bornes
    x_ot = batch_x[:, :, -1:].cpu().numpy()
    y_hat_np = y_hat.cpu().numpy()
    
    rho = getattr(cfg_rl, 'rho', 0.2)
    fr = getattr(cfg_rl, 'fr', 0.5)
    
    alphas, betas = compute_bounds_np(
        x_ot=x_ot,
        y_hat=y_hat_np,
        rho=rho,
        fr=fr,
        direction=-1.0,
        global_sigma=None
    )
    
    # Générer counterfactuals
    with torch.no_grad():
        z = trainer.ae_arch.encode(batch_x)
        action, _, _ = trainer.agent.actor(z)
        z_cf = z + action
        x_cf = trainer.ae_arch.decode(z_cf)
        y_cf = trainer.forecaster.predict_ot_from_modified(
            x_full=batch_x,
            x_ot_modified=x_cf,
            x_mark=batch_x_mark
        )
    
    # Préparer les données pour plotting
    results = {
        'x_orig': batch_x[:, :, -1].cpu().numpy(),  # (n, seq_len)
        'x_cf': x_cf[:, :, 0].cpu().numpy(),        # (n, seq_len)
        'y_hat': y_hat_np[:, :, 0],                 # (n, pred_len)
        'y_cf': y_cf.cpu().numpy()[:, :, 0],        # (n, pred_len)
        'alphas': alphas[:, :, 0],                  # (n, pred_len)
        'betas': betas[:, :, 0],                    # (n, pred_len)
        'seq_len': cfg_f.seq_len,
        'pred_len': cfg_f.pred_len,
        'mask_last_k': getattr(cfg_rl, 'mask_last_k', 24),
        'dataset': dataset,
        'model': model
    }
    
    return results


def plot_qualitative_examples(results, output_path):
    """
    Génère la figure avec les exemples qualitatifs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_examples = len(results['x_orig'])
    seq_len = results['seq_len']
    pred_len = results['pred_len']
    mask_k = results['mask_last_k']
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    t_past = np.arange(seq_len)
    t_future = np.arange(seq_len, seq_len + pred_len)
    
    for i, ax in enumerate(axes):
        # Zone de masque temporel
        mask_start = seq_len - mask_k
        ax.axvspan(mask_start, seq_len, alpha=0.15, color='orange', 
                   label='Temporal Mask Region')
        
        # Séries temporelles
        ax.plot(t_past, results['x_orig'][i], 
                color='gray', linewidth=1.5, label='Original Input', alpha=0.7)
        ax.plot(t_past, results['x_cf'][i], 
                color='red', linewidth=1.5, label='Counterfactual Input', 
                linestyle='--', alpha=0.8)
        
        # Forecasts
        ax.plot(t_future, results['y_hat'][i], 
                color='blue', linewidth=2, label='Original Forecast')
        ax.plot(t_future, results['y_cf'][i], 
                color='green', linewidth=2, label='Counterfactual Forecast', 
                linestyle='--')
        
        # Target band
        ax.fill_between(t_future, 
                       results['alphas'][i], 
                       results['betas'][i],
                       alpha=0.2, color='green', label='Target Band')
        
        # Ligne de séparation
        ax.axvline(x=seq_len, color='black', linestyle=':', linewidth=1)
        
        ax.set_title(f'Sample {i+1}', fontsize=11)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    dataset_name = results['dataset'].upper()
    model_name = results['model'].capitalize()
    fig.suptitle(f'Qualitative Examples: {dataset_name} - {model_name}', 
                 fontsize=14, y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Figure saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate qualitative examples")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, default="itransformer")
    parser.add_argument("--n_examples", type=int, default=4)
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    # Générer les exemples
    results = generate_qualitative_examples(
        dataset=args.dataset,
        model=args.model,
        n_examples=args.n_examples,
        batch_idx=args.batch_idx
    )
    
    # Générer la figure
    plot_qualitative_examples(results, args.output)
    
    print(f"\n{'='*60}")
    print(f"Qualitative Examples Complete: {args.dataset}")
    print(f"{'='*60}\n")
