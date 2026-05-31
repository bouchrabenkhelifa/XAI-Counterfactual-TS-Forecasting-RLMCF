"""
Génère des exemples qualitatifs pour ETTh2 et Weather
Style identique à ETTh1 (série continue avec ligne de séparation)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.data_provider.data_factory import data_provider


def plot_qualitative_examples(dataset="etth2", n_examples=3):
    """
    Génère des exemples qualitatifs de séries temporelles.
    Style: série continue (historique + futur) avec ligne de séparation
    
    Args:
        dataset: nom du dataset (etth2 ou weather)
        n_examples: nombre d'exemples à afficher
    """
    
    print(f"\n{'='*60}")
    print(f"Generating Qualitative Examples: {dataset.upper()}")
    print(f"{'='*60}\n")
    
    # Charger config
    seq_config = "96_96" if dataset == "weather" else "96_48"
    model = "itransformer" if dataset == "etth2" else "timesnet"
    cfg_f_path = f"assets/configs/{dataset}_dataset/forecasters/{model}/{dataset}_{seq_config}_S.json"
    
    cfg_f = load_config(cfg_f_path)
    
    # Charger les données
    _, test_loader = data_provider(cfg_f, "test")
    
    # Prendre plusieurs batches pour avoir n_examples
    examples = []
    for i, batch in enumerate(test_loader):
        if len(examples) >= n_examples:
            break
        
        batch_x, batch_y, _, _ = batch
        
        # Extraire tous les exemples du batch
        batch_size = batch_x.shape[0]
        for j in range(batch_size):
            if len(examples) >= n_examples:
                break
            x_vals = batch_x[j, :, -1].numpy()  # Historique (dernière feature)
            y_vals = batch_y[j, :, -1].numpy()  # Futur (dernière feature)
            examples.append((x_vals, y_vals))
    
    print(f"Extracted {len(examples)} examples from test data")
    
    # Créer la figure (style ETTh1)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx, (x_vals, y_vals) in enumerate(examples):
        ax = axes[idx]
        
        seq_len = len(x_vals)
        pred_len = len(y_vals)
        
        print(f"  Example {idx+1}: seq_len={seq_len}, pred_len={pred_len}")
        
        # Créer les indices temporels
        time_hist = np.arange(seq_len)
        time_fut = np.arange(seq_len, seq_len + pred_len)
        
        # Tracer la série continue
        # Historique (bleu)
        ax.plot(time_hist, x_vals, color='#2196F3', linewidth=2.5, 
                label='Historical', alpha=0.8)
        
        # Futur (vert)
        ax.plot(time_fut, y_vals, color='#4CAF50', linewidth=2.5, 
                label='Future (Ground Truth)', alpha=0.8)
        
        # Ligne verticale de séparation (grise pointillée)
        ax.axvline(seq_len - 0.5, color='gray', linestyle='--', 
                  linewidth=1.5, alpha=0.6, label='Forecast Horizon')
        
        # Styling
        ax.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Example {idx+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # Limites Y adaptatives
        all_vals = np.concatenate([x_vals, y_vals])
        y_min, y_max = all_vals.min(), all_vals.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])
    
    # Titre global
    dataset_name = dataset.upper()
    fig.suptitle(f'Qualitative Examples: {dataset_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = f"scripts/appendix/figures/qualitative_{dataset}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] Figure saved: {output_path}\n")
    print(f"{'='*60}")
    print(f"Qualitative Examples Complete: {dataset_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate qualitative examples")
    parser.add_argument("--dataset", type=str, default="etth2", 
                       choices=["etth2", "weather"],
                       help="Dataset to generate examples for")
    parser.add_argument("--n_examples", type=int, default=3,
                       help="Number of examples to show")
    
    args = parser.parse_args()
    
    plot_qualitative_examples(
        dataset=args.dataset,
        n_examples=args.n_examples
    )
