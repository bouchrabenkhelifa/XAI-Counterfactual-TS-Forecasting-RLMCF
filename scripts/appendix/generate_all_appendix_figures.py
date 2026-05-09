"""
Script Python pour générer toutes les figures de l'annexe ICONIP
Alternative au script bash pour Windows
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Execute une commande et affiche le résultat."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"[!] Erreur lors de l'exécution de: {description}")
        return False
    
    print(f"[✓] {description} - Terminé")
    return True


def main():
    print("\n" + "="*60)
    print("Generating Appendix Figures for ICONIP")
    print("="*60)
    print("\nNote: Using ETTh2 only due to dimension mismatch in ETTh1 checkpoints")
    print("(ETTh1 checkpoints were trained with different architecture)")
    
    # Créer le dossier de sortie
    os.makedirs("scripts/appendix/figures", exist_ok=True)
    
    commands = [
        # D.2 - Sensitivity to rho (ETTh2)
        {
            'cmd': 'python scripts/appendix/generate_rho_sensitivity.py '
                   '--dataset etth2 --model itransformer --fr 0.5 --n_batches 10 '
                   '--output scripts/appendix/figures/rho_sensitivity_etth2.png',
            'desc': '[1/2] Generating rho sensitivity curve (ETTh2)'
        },
        
        # E - Qualitative Examples ETTh2
        {
            'cmd': 'python scripts/appendix/generate_qualitative_examples.py '
                   '--dataset etth2 --model itransformer --n_examples 4 --batch_idx 0 '
                   '--output scripts/appendix/figures/qualitative_etth2.png',
            'desc': '[2/2] Generating ETTh2 qualitative examples'
        }
    ]
    
    success_count = 0
    for cmd_info in commands:
        if run_command(cmd_info['cmd'], cmd_info['desc']):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"Completed: {success_count}/{len(commands)} figures generated")
    print("="*60)
    print("\nOutput directory: scripts/appendix/figures/")
    print("\nFiles created:")
    print("  - rho_sensitivity_etth2.png")
    print("  - qualitative_etth2.png")
    print("\nNotes:")
    print("  - ETTh1 skipped (dimension mismatch in checkpoints)")
    print("  - Weather skipped (checkpoint not trained yet)")
    print()


if __name__ == "__main__":
    main()
