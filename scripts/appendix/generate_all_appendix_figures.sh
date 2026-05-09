#!/bin/bash

# Script pour générer toutes les figures de l'annexe ICONIP
# Usage: bash scripts/appendix/generate_all_appendix_figures.sh

echo "=========================================="
echo "Generating Appendix Figures for ICONIP"
echo "=========================================="

# Créer le dossier de sortie
mkdir -p scripts/appendix/figures

# ============================================
# D.2 - Sensitivity to rho
# ============================================
echo ""
echo "[1/3] Generating rho sensitivity curve..."
python scripts/appendix/generate_rho_sensitivity.py \
    --dataset etth1 \
    --model itransformer \
    --fr 0.5 \
    --n_batches 20 \
    --output scripts/appendix/figures/rho_sensitivity.png

# ============================================
# E.1 - Qualitative Examples ETTh2
# ============================================
echo ""
echo "[2/3] Generating ETTh2 qualitative examples..."
python scripts/appendix/generate_qualitative_examples.py \
    --dataset etth2 \
    --model itransformer \
    --n_examples 4 \
    --batch_idx 0 \
    --output scripts/appendix/figures/qualitative_etth2.png

# ============================================
# E.2 - Qualitative Examples Weather
# ============================================
echo ""
echo "[3/3] Generating Weather qualitative examples..."
python scripts/appendix/generate_qualitative_examples.py \
    --dataset weather \
    --model itransformer \
    --n_examples 4 \
    --batch_idx 0 \
    --output scripts/appendix/figures/qualitative_weather.png

echo ""
echo "=========================================="
echo "All appendix figures generated!"
echo "Output directory: scripts/appendix/figures/"
echo "=========================================="
echo ""
echo "Files created:"
echo "  - rho_sensitivity.png"
echo "  - qualitative_etth2.png"
echo "  - qualitative_weather.png"
