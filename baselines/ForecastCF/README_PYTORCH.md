# Adaptation ForecastCF pour PyTorch

Cette adaptation permet d'utiliser ForecastCF avec des modèles PyTorch (iTransformer, TimesNet, DLinear, GRU, PatchTST) sans modifier le code TensorFlow original.

## Fichiers ajoutés

- `src/pytorch_adapter.py` : Wrapper pour rendre les modèles PyTorch compatibles avec l'interface TensorFlow de ForecastCF
- `src/cf_search_pytorch.py` : Script d'exécution adapté pour PyTorch
- `run_etth1_itransformer.sh` : Script bash pour lancer ForecastCF avec iTransformer sur ETTh1

## Installation

Les dépendances PyTorch sont déjà installées dans votre projet principal. Assurez-vous d'avoir :

```bash
pip install torch numpy pandas
```

## Usage

### 1. Avec le script bash (recommandé)

```bash
cd baselines/ForecastCF
bash run_etth1_itransformer.sh
```

Ce script lance ForecastCF avec iTransformer sur ETTh1 pour 5 seeds différents (1, 9, 30, 33, 39).

### 2. Manuellement

```bash
cd baselines/ForecastCF

python src/cf_search_pytorch.py \
    --model-path ../../../assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_S.pth \
    --model-type itransformer \
    --config-path ../../../assets/configs/models/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json \
    --dataset etth1 \
    --data-path ../../../assets/datasets/ETTh1.csv \
    --horizon 48 \
    --back-horizon 96 \
    --center median \
    --desired-shift 0 \
    --desired-change -0.1 \
    --poly-order 1 \
    --fraction-std 1.0 \
    --random-seed 39 \
    --output results/forecastcf_etth1_itransformer.csv \
    --device cuda \
    --test-samples 1000
```

## Paramètres

- `--model-path` : Chemin vers le checkpoint PyTorch (.pth)
- `--model-type` : Type de modèle (itransformer, timesnet, dlinear, gru, patchtst)
- `--config-path` : Chemin vers le fichier de configuration JSON
- `--dataset` : Nom du dataset (etth1, weather, traffic, etc.)
- `--data-path` : Chemin vers le fichier CSV du dataset
- `--horizon` : Horizon de prévision (H)
- `--back-horizon` : Fenêtre look-back (L)
- `--center` : Point de départ pour les bornes (median, mean, last, min, max)
- `--desired-shift` : Décalage par rapport au centre (ex: 0.2 pour 120% du centre)
- `--desired-change` : Changement de tendance désiré (ex: -0.1 pour -10%)
- `--poly-order` : Ordre du polynôme pour la tendance (1 = linéaire)
- `--fraction-std` : Fraction de l'écart-type pour la largeur des bornes
- `--random-seed` : Seed aléatoire
- `--output` : Fichier CSV de sortie
- `--device` : Device PyTorch (cuda ou cpu)
- `--test-samples` : Nombre d'échantillons de test à utiliser (None = tous)

## Autres modèles

Pour utiliser d'autres modèles PyTorch, changez simplement les paramètres :

### TimesNet
```bash
python src/cf_search_pytorch.py \
    --model-path ../../../assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_timesnet_S.pth \
    --model-type timesnet \
    --config-path ../../../assets/configs/models/etth1_dataset/forecasters/timesnet/etth1_96_48_S.json \
    ...
```

### DLinear
```bash
python src/cf_search_pytorch.py \
    --model-path ../../../assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_dlinear_S.pth \
    --model-type dlinear \
    --config-path ../../../assets/configs/models/etth1_dataset/forecasters/dlinear/etth1_96_48_S.json \
    ...
```

### GRU
```bash
python src/cf_search_pytorch.py \
    --model-path ../../../assets/checkpoints/etth1_chpts/forecaster/chpt_etth1_96_48_gru_S.pth \
    --model-type gru \
    --config-path ../../../assets/configs/models/etth1_dataset/forecasters/gru/etth1_96_48_S.json \
    ...
```

## Résultats

Les résultats sont sauvegardés dans un fichier CSV avec les colonnes :
- `random_seed` : Seed utilisé
- `method_name` : Type de modèle (itransformer, etc.)
- `cf_method_name` : Méthode CF (ForecastCF)
- `horizon` : Horizon de prévision
- `desired_change` : Changement désiré
- `fraction_std` : Fraction std utilisée
- `validity_ratio` : Ratio de validité
- `proximity` : Distance de proximité
- `compactness` : Compacité des perturbations
- `step_validity_auc` : AUC de validité par étape

## Comparaison avec votre méthode RL

Pour comparer avec votre méthode RL, vous pouvez :

1. Lancer ForecastCF avec ce script
2. Lancer votre méthode RL avec les mêmes paramètres
3. Comparer les métriques dans les fichiers CSV de résultats

Les paramètres sont alignés pour une comparaison équitable :
- Même dataset (ETTh1)
- Même modèle (iTransformer)
- Même configuration (96→48, univarié)
- Mêmes bornes (median, desired_change=-0.1, fraction_std=1.0)
- Mêmes seeds (1, 9, 30, 33, 39)
