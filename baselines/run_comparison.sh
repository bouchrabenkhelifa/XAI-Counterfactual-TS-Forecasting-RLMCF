#!/bin/bash
# Script complet pour comparer votre méthode RL avec ForecastCF baseline
# sur iTransformer + ETTh1

set -e  # Exit on error

echo "=========================================="
echo "Comparaison RL vs ForecastCF"
echo "Modèle: iTransformer | Dataset: ETTh1"
echo "=========================================="
echo

# Chemins
RL_RESULTS="assets/results/rl_cf_etth1_itransformer.csv"  # À adapter selon votre structure
FCF_RESULTS="baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv"
COMPARISON_DIR="baselines/comparison_plots"

# Étape 1: Lancer ForecastCF baseline
echo "Étape 1/3: Exécution de ForecastCF baseline..."
echo "----------------------------------------------"
cd baselines/ForecastCF
bash run_etth1_itransformer.sh
cd ../..
echo "✓ ForecastCF terminé"
echo

# Étape 2: Vérifier que les résultats RL existent
echo "Étape 2/3: Vérification des résultats RL..."
echo "----------------------------------------------"
if [ ! -f "$RL_RESULTS" ]; then
    echo "⚠ Fichier de résultats RL non trouvé: $RL_RESULTS"
    echo "Veuillez d'abord exécuter votre méthode RL et sauvegarder les résultats."
    echo
    echo "Exemple de commande pour lancer votre RL:"
    echo "  python -m src.experiments.rl_cf.run --config <votre_config>"
    echo
    exit 1
fi
echo "✓ Résultats RL trouvés: $RL_RESULTS"
echo

# Étape 3: Comparer les résultats
echo "Étape 3/3: Comparaison des résultats..."
echo "----------------------------------------------"
python baselines/compare_results.py \
    --rl-results "$RL_RESULTS" \
    --fcf-results "$FCF_RESULTS" \
    --output-dir "$COMPARISON_DIR"

echo
echo "=========================================="
echo "✓ Comparaison terminée!"
echo "=========================================="
echo
echo "Résultats sauvegardés dans: $COMPARISON_DIR"
echo "  - comparison_barplot.png"
echo "  - comparison_boxplot.png"
echo "  - comparison_radar.png"
echo
