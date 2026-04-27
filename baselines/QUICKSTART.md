# Guide de démarrage rapide - Comparaison RL vs ForecastCF

## 🎯 Objectif

Comparer votre méthode RL de génération de contrefactuels avec la baseline ForecastCF sur iTransformer + ETTh1.

## 📋 Prérequis

1. ✅ Modèle iTransformer entraîné sur ETTh1 (96→48)
   - Checkpoint: `assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_S.pth`
   - Config: `assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json`

2. ✅ Résultats de votre méthode RL sauvegardés en CSV

3. ✅ Dépendances installées:
   ```bash
   pip install torch numpy pandas matplotlib seaborn
   ```

## 🚀 Lancement rapide

### Méthode 1: Tout automatique (recommandé)

```bash
bash baselines/run_comparison.sh
```

### Méthode 2: Étape par étape

#### Étape 1: Lancer ForecastCF

```bash
cd baselines/ForecastCF
bash run_etth1_itransformer.sh
cd ../..
```

⏱️ Temps estimé: ~10-30 minutes (selon GPU)

#### Étape 2: Comparer les résultats

```bash
python baselines/compare_results.py \
    --rl-results <CHEMIN_VERS_VOS_RESULTATS_RL> \
    --fcf-results baselines/ForecastCF/results/forecastcf_etth1_itransformer.csv \
    --output-dir baselines/comparison_plots
```

## 📊 Résultats

### Console

```
==============================================================================
COMPARAISON: Votre méthode RL vs ForecastCF Baseline
==============================================================================

Metric                    Your RL              ForecastCF           Winner
------------------------------------------------------------------------------
Validity Ratio            0.8523 ± 0.0234      0.7891 ± 0.0312      Your RL ✓
  ↑ higher is better | Improvement: +8.01%

Step Validity AUC         0.9123 ± 0.0156      0.8456 ± 0.0289      Your RL ✓
  ↑ higher is better | Improvement: +7.89%

Proximity (L2)            0.1234 ± 0.0089      0.1567 ± 0.0123      Your RL ✓
  ↓ lower is better | Improvement: +21.25%

Compactness               0.7845 ± 0.0234      0.6234 ± 0.0345      Your RL ✓
  ↑ higher is better | Improvement: +25.84%
```

### Graphiques

Dans `baselines/comparison_plots/` :

1. **comparison_barplot.png** : Comparaison par métrique avec barres d'erreur
2. **comparison_boxplot.png** : Distribution complète des résultats
3. **comparison_radar.png** : Vue d'ensemble radar

## 🔧 Personnalisation

### Changer le nombre d'échantillons de test

Éditez `baselines/ForecastCF/run_etth1_itransformer.sh` :

```bash
--test-samples 500  # Au lieu de 1000
```

### Tester d'autres modèles

```bash
# TimesNet
python baselines/ForecastCF/src/cf_search_pytorch.py \
    --model-path assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_timesnet_S.pth \
    --model-type timesnet \
    --config-path assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json \
    ...

# DLinear
python baselines/ForecastCF/src/cf_search_pytorch.py \
    --model-path assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_dlinear_S.pth \
    --model-type dlinear \
    --config-path assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json \
    ...
```

### Changer les paramètres de génération

Éditez `baselines/ForecastCF/run_etth1_itransformer.sh` :

```bash
--desired-change -0.2    # Tendance de -20% au lieu de -10%
--fraction-std 1.5       # Bornes plus larges
--center mean            # Utiliser la moyenne au lieu de la médiane
```

## ❓ FAQ

### Q: Le script prend trop de temps

**R:** Réduisez le nombre d'échantillons de test :
```bash
--test-samples 100  # Au lieu de 1000
```

### Q: Erreur "CUDA out of memory"

**R:** Utilisez le CPU ou réduisez le batch size :
```bash
--device cpu
```

### Q: Le checkpoint iTransformer n'existe pas

**R:** Entraînez d'abord le modèle :
```bash
bash scripts/etth1_runs/run_itransformer.sh
```

### Q: Mes résultats RL ne sont pas au bon format

**R:** Assurez-vous que votre CSV contient ces colonnes :
- `random_seed`
- `validity_ratio`
- `proximity`
- `compactness`
- `step_validity_auc`

### Q: Je veux comparer sur un autre dataset

**R:** Adaptez les chemins dans le script :
```bash
--dataset weather \
--data-path assets/datasets/weather.csv \
--model-path assets/checkpoints/weather_chpts/forecaster/chpt_weather_96_48_S.pth \
--config-path assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_48_S.json
```

## 📝 Notes importantes

1. **Seeds identiques** : Les deux méthodes utilisent les mêmes seeds (1, 9, 30, 33, 39) pour une comparaison équitable

2. **Paramètres alignés** : 
   - Look-back: 96
   - Horizon: 48
   - Desired change: -0.1
   - Fraction std: 1.0
   - Center: median

3. **Même modèle** : Les deux méthodes utilisent le même checkpoint iTransformer

4. **Même dataset** : Même split de test ETTh1

## 📚 Documentation complète

- `baselines/README.md` : Vue d'ensemble complète
- `baselines/ForecastCF/README_PYTORCH.md` : Détails de l'adaptation PyTorch
- `baselines/ForecastCF/README.md` : Documentation ForecastCF originale

## 🎉 Prochaines étapes

Après avoir obtenu les résultats :

1. ✅ Analysez les graphiques de comparaison
2. ✅ Identifiez les forces de votre méthode RL
3. ✅ Testez sur d'autres modèles (TimesNet, DLinear, GRU)
4. ✅ Testez sur d'autres datasets (Weather, Traffic, Tourism)
5. ✅ Incluez les résultats dans votre papier/rapport

Bonne chance ! 🚀
