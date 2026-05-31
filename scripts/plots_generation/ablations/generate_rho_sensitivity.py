"""
Script pour générer la courbe de sensibilité au paramètre rho (ρ)
Pour l'annexe ICONIP - Section D.2
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


def evaluate_rho_sensitivity(
    dataset="etth1",
    model="itransformer",
    rho_values=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    fr=0.5,
    n_batches=20
):
    """
    Évalue la validité pour différentes valeurs de rho.
    
    Args:
        dataset: nom du dataset (etth1, etth2, weather)
        model: nom du modèle (itransformer, gru, etc.)
        rho_values: liste des valeurs de rho à tester
        fr: valeur fixe de fr
        n_batches: nombre de batches à évaluer
    
    Returns:
        dict avec rho_values et validity_scores
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
    print(f"Rho Sensitivity Analysis")
    print(f"Dataset: {dataset} | Model: {model} | fr={fr}")
    print(f"Testing rho values: {rho_values}")
    print(f"{'='*60}\n")
    
    # Créer trainer pour charger les données
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    
    # Charger le meilleur checkpoint
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
        print(f"[!] Checkpoint not found: {ckpt_path}")
        print("[!] Using random weights - results may be poor\n")
    
    trainer.agent.eval()
    
    results = {
        'rho_values': rho_values,
        'validity_scores': [],
        'success_rates': []
    }
    
    # Tester chaque valeur de rho
    for rho in rho_values:
        print(f"Testing rho={rho:.2f}...")
        
        valid_count = 0
        total_count = 0
        
        for i, batch in enumerate(trainer.test_loader):
            if i >= n_batches:
                break
            
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            
            # Prédiction originale
            with torch.no_grad():
                y_hat = trainer.forecaster.predict_ot(batch_x, batch_x_mark)
            
            # Calculer les bornes avec ce rho
            x_ot = batch_x[:, :, -1:].cpu().numpy()
            y_hat_np = y_hat.cpu().numpy()
            
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
                # Encoder
                z = trainer.ae_arch.encode(batch_x)
                
                # Action de l'agent
                action, _, _ = trainer.agent.actor(z)
                z_cf = z + action
                
                # Decoder
                x_cf = trainer.ae_arch.decode(z_cf)
                
                # Prédiction counterfactuelle
                y_cf = trainer.forecaster.predict_ot_from_modified(
                    x_full=batch_x,
                    x_ot_modified=x_cf,
                    x_mark=batch_x_mark
                )
            
            # Vérifier validité
            y_cf_np = y_cf.cpu().numpy()
            alphas_t = torch.from_numpy(alphas).float().to(device)
            betas_t = torch.from_numpy(betas).float().to(device)
            
            valid = (y_cf >= alphas_t) & (y_cf <= betas_t)
            valid_per_sample = valid.all(dim=1).all(dim=1)
            
            valid_count += valid_per_sample.sum().item()
            total_count += len(batch_x)
        
        validity = valid_count / total_count if total_count > 0 else 0.0
        results['validity_scores'].append(validity)
        results['success_rates'].append(validity * 100)
        
        print(f"  → Validity: {validity:.4f} ({validity*100:.1f}%)\n")
    
    return results


def plot_rho_sensitivity(results, output_path="scripts/appendix/figures/rho_sensitivity.png"):
    """
    Génère la figure de sensibilité à rho.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(results['rho_values'], results['validity_scores'], 
            marker='o', linewidth=2, markersize=8, color='#2196F3')
    
    ax.set_xlabel(r'Gap Parameter $\rho$', fontsize=12)
    ax.set_ylabel('Validity Score', fontsize=12)
    ax.set_title(r'Sensitivity to Gap Parameter $\rho$ (fixed $f_r=0.5$)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Annoter les valeurs
    for rho, val in zip(results['rho_values'], results['validity_scores']):
        ax.annotate(f'{val:.3f}', 
                   xy=(rho, val), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] Figure saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate rho sensitivity analysis")
    parser.add_argument("--dataset", type=str, default="etth2", choices=["etth1", "etth2", "weather"])
    parser.add_argument("--model", type=str, default="itransformer")
    parser.add_argument("--fr", type=float, default=0.5)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--output", type=str, default="scripts/appendix/figures/rho_sensitivity.png")
    
    args = parser.parse_args()
    
    # Valeurs de rho à tester
    rho_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Évaluer
    results = evaluate_rho_sensitivity(
        dataset=args.dataset,
        model=args.model,
        rho_values=rho_values,
        fr=args.fr,
        n_batches=args.n_batches
    )
    
    # Générer la figure
    plot_rho_sensitivity(results, args.output)
    
    print("\n" + "="*60)
    print("Rho Sensitivity Analysis Complete")
    print("="*60)
