# Fix Weather Forecasters (GRU, TimesNet, DLinear) - Bugfix Design

## Overview

Ce document détaille la stratégie technique pour corriger les trois forecasters (GRU, TimesNet, DLinear) qui ne fonctionnent pas correctement sur le dataset Weather. Les problèmes identifiés sont :

1. **TimesNet** : Crash au démarrage dû à l'attribut `embed` manquant dans la configuration
2. **GRU** : Entraînement extrêmement lent (timeout sans progression) dû à des hyperparamètres inadaptés
3. **DLinear** : Entraînement lent (1 epoch en 30s) dû à un `moving_avg` trop petit

La stratégie de correction consiste à :
- Ajouter l'attribut `embed: "timeF"` à la configuration TimesNet
- Optimiser les hyperparamètres de GRU (réduire d_model, ajuster learning_rate, réduire train_epochs)
- Optimiser les hyperparamètres de DLinear (augmenter moving_avg à 25, ajuster learning_rate)

Ces corrections permettront aux trois modèles de s'entraîner rapidement comme iTransformer et PatchTST, débloquant ainsi leur entraînement RL.

## Glossary

- **Bug_Condition (C)**: La condition qui déclenche le bug - configurations Weather de TimesNet/GRU/DLinear avec paramètres problématiques
- **Property (P)**: Le comportement désiré - entraînement rapide et sans erreur pour tous les forecasters Weather
- **Preservation**: Les configurations ETTh1/ETTh2 et les forecasters Weather fonctionnels (iTransformer, PatchTST) doivent rester inchangés
- **embed**: Paramètre de configuration qui détermine le type d'encodage temporel (`"timeF"` pour time features, autre valeur pour encodage positionnel)
- **d_model**: Dimension du modèle (hidden size pour GRU, embedding dimension pour transformers)
- **moving_avg**: Taille de la fenêtre de moyenne mobile pour la décomposition série temporelle dans DLinear
- **data_factory.py**: Module qui charge les datasets et accède à `args.embed` (ligne 25)
- **GenericForecasterTrainer**: Classe d'entraînement générique dans `src/training/forecast_trainers/generic_forecaster_trainer.py`

## Bug Details

### Bug Condition

Le bug se manifeste lorsque les configurations Weather de TimesNet, GRU ou DLinear sont utilisées pour l'entraînement. Les trois modèles présentent des problèmes différents mais tous empêchent l'entraînement RL :

1. **TimesNet** : `data_factory.py` ligne 25 tente d'accéder à `args.embed` qui n'existe pas dans la configuration
2. **GRU** : Le modèle démarre avec 161,760 paramètres (d_model=128, e_layers=2) mais ne progresse pas et timeout après 60s
3. **DLinear** : Le modèle progresse mais très lentement (1 epoch en 30s) avec moving_avg=13

**Formal Specification:**
```
FUNCTION isBugCondition(config)
  INPUT: config of type ForecasterConfig
  OUTPUT: boolean
  
  RETURN (config.model_type = "TimesNet" 
          AND config.dataset_name = "Weather" 
          AND NOT hasAttribute(config, "embed"))
         OR (config.model_type = "GRU" 
             AND config.dataset_name = "Weather" 
             AND config.d_model = 128 
             AND config.train_epochs = 50)
         OR (config.model_type = "DLinear" 
             AND config.dataset_name = "Weather" 
             AND config.moving_avg = 13 
             AND config.learning_rate = 0.005)
END FUNCTION
```

### Examples

- **TimesNet Crash** : Lancement avec `weather_96_96_timesnet_S.json` → `AttributeError: 'types.SimpleNamespace' object has no attribute 'embed'` à la ligne 25 de `data_factory.py`
- **GRU Timeout** : Lancement avec `weather_96_96_gru_S.json` → Affiche "161,760 params" puis timeout après 60s sans compléter d'epoch
- **DLinear Lenteur** : Lancement avec `weather_96_96_dlinear_S.json` → Complète 1 epoch en 30s puis timeout (trop lent pour 50 epochs)
- **iTransformer OK** : Lancement avec `weather_96_96_itransformer_S.json` → S'entraîne rapidement et complète plusieurs epochs sans problème

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Les configurations ETTh1 de GRU, DLinear et TimesNet doivent continuer à fonctionner exactement comme avant
- Les configurations ETTh2 de GRU, DLinear et TimesNet doivent continuer à fonctionner exactement comme avant
- Les configurations Weather de iTransformer et PatchTST doivent continuer à fonctionner exactement comme avant
- Le code de `data_factory.py` ligne 25 (`timeenc = 0 if args.embed != 'timeF' else 1`) doit continuer à fonctionner
- Le code de `GenericForecasterTrainer` doit continuer à fonctionner pour tous les modèles
- Les chemins de sauvegarde des checkpoints et historiques doivent rester inchangés

**Scope:**
Toutes les configurations qui ne correspondent PAS aux trois fichiers Weather problématiques (TimesNet, GRU, DLinear) doivent être complètement inaffectées par cette correction. Cela inclut :
- Toutes les configurations ETTh1 et ETTh2
- Les configurations Weather de iTransformer et PatchTST
- Tous les autres datasets (Traffic, Solar, PEMS, etc.)

## Hypothesized Root Cause

Basé sur l'analyse du code et des configurations, les causes les plus probables sont :

1. **TimesNet - Attribut Manquant** : La configuration `weather_96_96_timesnet_S.json` ne contient pas l'attribut `embed`
   - `data_factory.py` ligne 25 accède à `args.embed` sans vérification
   - Les configurations ETTh1 de TimesNet contiennent `"embed": "timeF"`
   - Les configurations Weather de iTransformer et PatchTST contiennent aussi `"embed": "timeF"`
   - Solution : Ajouter `"embed": "timeF"` à la configuration TimesNet Weather

2. **GRU - Hyperparamètres Inadaptés** : Le modèle GRU avec d_model=128 et e_layers=2 génère 161,760 paramètres
   - Calcul : `params = (enc_in * d_model + d_model * d_model * 3) * e_layers + d_model * c_out * pred_len`
   - Pour Weather (dataset plus large que ETTh1), ces paramètres causent une lenteur excessive
   - Les configurations ETTh1 de GRU utilisent d_model=128 mais Weather nécessite des paramètres plus légers
   - Solution : Réduire d_model à 64, réduire train_epochs à 20, ajuster learning_rate à 0.005

3. **DLinear - Moving Average Trop Petit** : La configuration utilise moving_avg=13
   - Les configurations ETTh1 de DLinear utilisent moving_avg=25
   - Les configurations Weather de iTransformer utilisent moving_avg=25
   - Un moving_avg plus petit (13) nécessite plus de calculs de décomposition
   - Solution : Augmenter moving_avg à 25, ajuster learning_rate à 0.001

4. **Dataset Weather Plus Large** : Le dataset Weather est significativement plus large que ETTh1
   - Weather nécessite des hyperparamètres optimisés pour la vitesse
   - iTransformer et PatchTST utilisent déjà des configurations optimisées (learning_rate=0.0001, train_epochs=50)
   - GRU et DLinear nécessitent des ajustements similaires

## Correctness Properties

Property 1: Bug Condition - Configuration Completeness and Training Performance

_For any_ forecaster configuration where the bug condition holds (TimesNet Weather without embed, GRU Weather with d_model=128, DLinear Weather with moving_avg=13), the fixed configuration SHALL include all required attributes (embed="timeF" for TimesNet) and optimized hyperparameters (d_model=64 for GRU, moving_avg=25 for DLinear), enabling successful training without crashes or timeouts.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

Property 2: Preservation - Non-Buggy Configuration Behavior

_For any_ forecaster configuration where the bug condition does NOT hold (all ETTh1/ETTh2 configs, Weather iTransformer/PatchTST), the fixed code SHALL produce exactly the same training behavior as the original code, preserving all existing functionality including checkpoint paths, training curves, and model performance.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Basé sur l'analyse des causes racines, les modifications suivantes sont nécessaires :

**File**: `assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_96_S.json`

**Function**: Configuration JSON

**Specific Changes**:
1. **Ajouter l'attribut embed** : Ajouter `"embed": "timeF"` dans la configuration
   - Placer après l'attribut `"freq": "t"` pour cohérence avec les autres configs
   - Utiliser la valeur `"timeF"` (time features) comme iTransformer et PatchTST

**File**: `assets/configs/models/weather_dataset/forecasters/gru/weather_96_96_S.json`

**Function**: Configuration JSON

**Specific Changes**:
1. **Réduire d_model** : Changer `"d_model": 128` → `"d_model": 64`
   - Réduit le nombre de paramètres de 161,760 à ~40,000
   - Accélère significativement l'entraînement

2. **Ajuster learning_rate** : Changer `"learning_rate": 0.001` → `"learning_rate": 0.005`
   - Compense la réduction de capacité du modèle
   - Accélère la convergence

3. **Réduire train_epochs** : Changer `"train_epochs": 50` → `"train_epochs": 20`
   - Réduit le temps d'entraînement total
   - Cohérent avec un modèle plus léger

**File**: `assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json`

**Function**: Configuration JSON

**Specific Changes**:
1. **Augmenter moving_avg** : Changer `"moving_avg": 13` → `"moving_avg": 25`
   - Cohérent avec les configurations ETTh1 et iTransformer Weather
   - Réduit le coût de calcul de la décomposition série temporelle

2. **Ajuster learning_rate** : Changer `"learning_rate": 0.005` → `"learning_rate": 0.001`
   - Cohérent avec les configurations ETTh1 de DLinear
   - Améliore la stabilité de l'entraînement

3. **Réduire train_epochs** : Changer `"train_epochs": 50` → `"train_epochs": 20`
   - Réduit le temps d'entraînement total
   - Cohérent avec GRU optimisé

## Testing Strategy

### Validation Approach

La stratégie de test suit une approche en deux phases : d'abord, démontrer les bugs sur le code non corrigé (exploratory bug condition checking), puis vérifier que les corrections fonctionnent et préservent le comportement existant (fix checking et preservation checking).

### Exploratory Bug Condition Checking

**Goal**: Démontrer les bugs AVANT d'implémenter les corrections. Confirmer ou réfuter l'analyse des causes racines. Si réfutation, ré-hypothèse nécessaire.

**Test Plan**: Exécuter les scripts d'entraînement avec les configurations non corrigées et observer les échecs. Documenter les erreurs exactes et les temps d'exécution.

**Test Cases**:
1. **TimesNet Crash Test** : Lancer `python scripts/train_forecaster.py --config assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_96_S.json` (échouera avec AttributeError sur unfixed code)
2. **GRU Timeout Test** : Lancer `python scripts/train_forecaster.py --config assets/configs/models/weather_dataset/forecasters/gru/weather_96_96_S.json` avec timeout de 60s (échouera avec 0 epochs sur unfixed code)
3. **DLinear Slow Test** : Lancer `python scripts/train_forecaster.py --config assets/configs/models/weather_dataset/forecasters/dlinear/weather_96_96_S.json` avec timeout de 30s (échouera avec seulement 1 epoch sur unfixed code)
4. **iTransformer Baseline Test** : Lancer `python scripts/train_forecaster.py --config assets/configs/models/weather_dataset/forecasters/itransformer/weather_96_96_S.json` avec timeout de 30s (réussira avec plusieurs epochs sur unfixed code)

**Expected Counterexamples**:
- TimesNet : `AttributeError: 'types.SimpleNamespace' object has no attribute 'embed'` à la ligne 25 de `data_factory.py`
- GRU : Affiche "161,760 params" puis timeout sans progression
- DLinear : Complète 1 epoch en ~30s puis timeout
- Causes possibles : attribut manquant (TimesNet), hyperparamètres inadaptés (GRU, DLinear), dataset trop large

### Fix Checking

**Goal**: Vérifier que pour toutes les configurations où la condition de bug est vraie, les configurations corrigées produisent le comportement attendu.

**Pseudocode:**
```
FOR ALL config WHERE isBugCondition(config) DO
  config_fixed := applyFix(config)
  result := trainForecaster(config_fixed)
  ASSERT result.no_crash AND result.epochs_completed > 0 AND result.training_time < TIMEOUT
END FOR
```

**Test Plan**: Exécuter les scripts d'entraînement avec les configurations corrigées et vérifier qu'ils s'entraînent rapidement sans erreur.

**Test Cases**:
1. **TimesNet Fixed Test** : Lancer avec config corrigée (embed="timeF") → doit démarrer sans erreur et compléter plusieurs epochs
2. **GRU Fixed Test** : Lancer avec config corrigée (d_model=64, lr=0.005, epochs=20) → doit compléter plusieurs epochs rapidement
3. **DLinear Fixed Test** : Lancer avec config corrigée (moving_avg=25, lr=0.001, epochs=20) → doit compléter plusieurs epochs rapidement
4. **Performance Comparison** : Comparer les temps d'entraînement de GRU/DLinear/TimesNet corrigés avec iTransformer/PatchTST → doivent être comparables

### Preservation Checking

**Goal**: Vérifier que pour toutes les configurations où la condition de bug est fausse, les configurations corrigées produisent exactement le même résultat que les configurations originales.

**Pseudocode:**
```
FOR ALL config WHERE NOT isBugCondition(config) DO
  ASSERT trainForecaster_original(config).behavior = trainForecaster_fixed(config).behavior
END FOR
```

**Testing Approach**: Les tests basés sur les propriétés (property-based testing) sont recommandés pour la vérification de préservation car :
- Ils génèrent automatiquement de nombreux cas de test à travers le domaine d'entrée
- Ils détectent les cas limites que les tests unitaires manuels pourraient manquer
- Ils fournissent de fortes garanties que le comportement est inchangé pour toutes les entrées non bugguées

**Test Plan**: Observer le comportement sur le code NON CORRIGÉ d'abord pour les configurations ETTh1/ETTh2 et Weather iTransformer/PatchTST, puis écrire des tests basés sur les propriétés capturant ce comportement.

**Test Cases**:
1. **ETTh1 GRU Preservation** : Observer que GRU ETTh1 s'entraîne correctement sur unfixed code, puis vérifier que le comportement continue après fix
2. **ETTh1 DLinear Preservation** : Observer que DLinear ETTh1 s'entraîne correctement sur unfixed code, puis vérifier que le comportement continue après fix
3. **ETTh1 TimesNet Preservation** : Observer que TimesNet ETTh1 s'entraîne correctement sur unfixed code, puis vérifier que le comportement continue après fix
4. **Weather iTransformer Preservation** : Observer que iTransformer Weather s'entraîne correctement sur unfixed code, puis vérifier que le comportement continue après fix
5. **Weather PatchTST Preservation** : Observer que PatchTST Weather s'entraîne correctement sur unfixed code, puis vérifier que le comportement continue après fix
6. **Checkpoint Path Preservation** : Vérifier que les chemins de sauvegarde des checkpoints restent inchangés pour toutes les configurations
7. **History Path Preservation** : Vérifier que les chemins de sauvegarde des historiques restent inchangés pour toutes les configurations

### Unit Tests

- Tester le chargement des configurations corrigées avec `json.load()` et vérifier que tous les attributs requis sont présents
- Tester l'accès à `args.embed` dans `data_factory.py` avec les configurations corrigées
- Tester l'instanciation des modèles GRU, DLinear, TimesNet avec les configurations corrigées
- Tester le calcul du nombre de paramètres pour GRU avec d_model=64 (doit être ~40,000)
- Tester la création de `SeriesDecomp` avec moving_avg=25 pour DLinear
- Tester la création de `DataEmbedding` avec embed="timeF" pour TimesNet

### Property-Based Tests

- Générer des configurations aléatoires de forecasters et vérifier que celles avec embed="timeF" fonctionnent correctement
- Générer des valeurs aléatoires de d_model pour GRU et vérifier que les valeurs plus petites s'entraînent plus rapidement
- Générer des valeurs aléatoires de moving_avg pour DLinear et vérifier que les valeurs plus grandes (25) s'entraînent plus rapidement que les petites (13)
- Tester que toutes les configurations ETTh1/ETTh2 continuent à fonctionner à travers de nombreux scénarios

### Integration Tests

- Tester le pipeline complet d'entraînement Weather avec les trois forecasters corrigés (GRU, DLinear, TimesNet)
- Tester que les checkpoints sont sauvegardés correctement dans `assets/checkpoints/weather_chpts/forecaster/`
- Tester que les historiques sont sauvegardés correctement dans `assets/results/weather/forecaster/`
- Tester que les figures sont générées correctement dans `assets/figures/weather/forecaster/{model}/`
- Tester le pipeline complet d'entraînement ETTh1 avec les trois forecasters pour vérifier la préservation
- Tester que l'entraînement RL peut démarrer après l'entraînement des forecasters corrigés
