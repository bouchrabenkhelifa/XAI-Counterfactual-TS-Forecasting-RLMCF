# Comparaison avec ForecastCF Baseline

Ce dossier contient l'adaptation de ForecastCF pour comparer avec votre méthode RL de génération de contrefactuels.

## Structure

```
baselines/
├── ForecastCF/                          # Code ForecastCF original + adaptation PyTorch
│   ├── src/
│   │   ├── forecastcf.py               # Code TensorFlow original (non modifié)
│   │   ├── cf_search.py                # Script TensorFlow original (non modifié)
│   │   ├── pytorch_adapter.py          # ✨ NOUVEAU: Wrapper PyTorch
│   │   └── cf_search_pytorch.py        # ✨ NOUVEAU: Script pour modèles PyTorch
│   ├── run_etth1.sh                    # Script original TensorFlow
│   ├── run_etth1_itransformer.sh       # ✨ NOUVEAU: Script pour iTransformer
│   └── README_PYTORCH.md               # ✨ NOUVEAU: Documentation adaptation
├── compare_results.py                   # ✨ NOUVEAU: Script de comparaison
└── run_comparison.sh                    # ✨ NOUVEAU: Pipeline complet

```

## Quick Start

### Option 1: Pipeline complet automatique

```bash
# Depuis la racine du projet
bash baselines/run_comparison.sh
```

Ce script va :
1. Lancer ForecastCF avec iTransformer sur ETTh1
2. Vérifier que vos résultats RL existent
3. Comparer les deux méthodes et générer des graphiques

### Option 2: Étape par étape

#### 1. Lancer ForecastCF baseline

```bash
cd baselines/ForecastCF
bash run_etth1_itransformer.sh
```

Résultats sauvegardés dans: `baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv`

#### 2. Lancer votre méthode RL

```bash
# Depuis la racine du projet
python -m src.experiments.rl_cf.run --config <votre_config>
```

Assurez-vous que les résultats sont sauvegardés dans un CSV avec les colonnes :
- `random_seed`
- `validity_ratio`
- `proximity`
- `compactness`
- `step_validity_auc`

#### 3. Comparer les résultats

```bash
python baselines/compare_results.py \
    --rl-results <chemin_vers_vos_resultats_rl.csv> \
    --fcf-results baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv \
    --output-dir baselines/comparison_plots
```

## Configuration

### Paramètres alignés pour comparaison équitable

Les deux méthodes utilisent les mêmes paramètres :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Dataset | ETTh1 | Electricity Transformer Temperature |
| Modèle | iTransformer | Transformer inversé |
| Look-back (L) | 96 | Fenêtre d'entrée |
| Horizon (H) | 48 | Horizon de prévision |
| Features | Univarié (S) | Une seule variable (OT) |
| Center | median | Point de départ des bornes |
| Desired change | -0.1 | Tendance de -10% |
| Fraction std | 1.0 | Largeur des bornes |
| Seeds | 1, 9, 30, 33, 39 | Pour la reproductibilité |

### Adapter pour d'autres modèles

Pour tester avec d'autres modèles PyTorch (TimesNet, DLinear, GRU, PatchTST), modifiez le script :

```bash
# Exemple pour TimesNet
python baselines/ForecastCF/src/cf_search_pytorch.py \
    --model-path assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_timesnet_S.pth \
    --model-type timesnet \
    --config-path assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json \
    --dataset etth1 \
    --data-path assets/datasets/ETTh1.csv \
    --horizon 48 \
    --back-horizon 96 \
    --center median \
    --desired-change -0.1 \
    --fraction-std 1.0 \
    --random-seed 39 \
    --output baselines/ForecastCF/results/forecastcf_etth1_timesnet.csv
```

## Métriques de comparaison

Les deux méthodes sont évaluées sur :

1. **Validity Ratio** (↑) : Proportion de prédictions dans les bornes cibles
2. **Step Validity AUC** (↑) : AUC de la courbe de validité cumulative
3. **Proximity** (↓) : Distance L2 entre contrefactuel et original
4. **Compactness** (↑) : Proportion de timesteps avec perturbation < ε

## Résultats attendus

Le script `compare_results.py` génère :

1. **Tableau de comparaison** (console) :
   ```
   Metric                    Your RL              ForecastCF           Winner
   --------------------------------------------------------------------------------
   Validity Ratio            0.8523 ± 0.0234      0.7891 ± 0.0312      Your RL ✓
   ...
   ```

2. **Graphiques** (dans `baselines/comparison_plots/`) :
   - `comparison_barplot.png` : Barres avec barres d'erreur
   - `comparison_boxplot.png` : Distributions complètes
   - `comparison_radar.png` : Vue d'ensemble radar

## Avantages de votre méthode RL

Votre méthode devrait montrer :

✅ **Meilleure validité** grâce à l'optimisation RL dans l'espace latent
✅ **Meilleure proximité** grâce au masque temporel
✅ **Meilleure compacité** grâce aux perturbations localisées
✅ **Génération plus rapide** : forward pass unique vs optimisation itérative
✅ **Politique globale** : généralise à tous les échantillons

## Troubleshooting

### Erreur: "Module not found"

```bash
# Assurez-vous d'être dans le bon environnement
cd baselines/ForecastCF
export PYTHONPATH="${PYTHONPATH}:../../.."
```

### Erreur: "CUDA out of memory"

```bash
# Réduire le nombre d'échantillons de test
python src/cf_search_pytorch.py ... --test-samples 500
```

### Erreur: "Checkpoint not found"

Vérifiez que le modèle iTransformer est bien entraîné :

```bash
# Entraîner iTransformer si nécessaire
bash scripts/etth1_runs/run_itransformer.sh
```

## Citation

Si vous utilisez cette adaptation dans vos travaux :

```bibtex
@inproceedings{wang2023forecastcf,
  title={Counterfactual Explanations for Time Series Forecasting},
  author={Wang, Zhendong and others},
  booktitle={ICDM},
  year={2023}
}
```

## Contact

Pour toute question sur cette adaptation, référez-vous à :
- `baselines/ForecastCF/README_PYTORCH.md` : Documentation détaillée de l'adaptation
- `baselines/ForecastCF/README.md` : Documentation ForecastCF originale
