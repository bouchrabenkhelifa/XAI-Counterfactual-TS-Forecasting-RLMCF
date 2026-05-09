"""
Script de vérification rapide pour confirmer que toutes les figures
de l'appendix ICONIP sont prêtes.
"""

import os
from pathlib import Path

def verify_figures():
    """Vérifie que toutes les figures requises existent."""
    
    figures_dir = Path("scripts/appendix/figures")
    
    required_figures = [
        "sensitivity_fr.png",
        "rho_tradeoff.png",
        "plausibility_vs_rho.png",
        "qualitative_etth1.png",
        "qualitative_etth2.png",
        "qualitative_weather.png",
    ]
    
    print("\n" + "="*60)
    print("VÉRIFICATION DES FIGURES APPENDIX ICONIP")
    print("="*60)
    print(f"\nDossier: {figures_dir}")
    print(f"Figures requises: {len(required_figures)}")
    print()
    
    all_present = True
    total_size = 0
    
    for i, fig_name in enumerate(required_figures, 1):
        fig_path = figures_dir / fig_name
        
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            total_size += size_kb
            status = "✅"
            print(f"{status} [{i}/6] {fig_name:<30} ({size_kb:>7.1f} KB)")
        else:
            status = "❌"
            all_present = False
            print(f"{status} [{i}/6] {fig_name:<30} MANQUANT")
    
    print()
    print("="*60)
    
    if all_present:
        print(f"✅ TOUTES LES FIGURES SONT PRÊTES ({total_size/1024:.2f} MB)")
        print()
        print("📝 Prochaines étapes:")
        print("   1. Consulter COMPLETE_FINAL.md pour le code LaTeX")
        print("   2. Copier les figures dans le dossier LaTeX")
        print("   3. Compiler et vérifier le rendu")
        print("   4. Soumettre avant le 10 Mai 2026")
    else:
        print("❌ CERTAINES FIGURES MANQUENT")
        print()
        print("🔧 Actions requises:")
        print("   - Exécuter les scripts de génération manquants")
    
    print("="*60)
    print()
    
    return all_present


if __name__ == "__main__":
    verify_figures()
