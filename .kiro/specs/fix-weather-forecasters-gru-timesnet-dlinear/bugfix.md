# Bugfix Requirements Document

## Introduction

Ce document décrit les corrections nécessaires pour les forecasters GRU, TimesNet et DLinear sur le dataset Weather. Ces trois modèles présentent des problèmes qui empêchent leur entraînement RL, alors que iTransformer et PatchTST fonctionnent parfaitement. Les problèmes identifiés sont :

1. **TimesNet** : Ne démarre pas du tout à cause d'un attribut `embed` manquant dans la configuration
2. **GRU et DLinear** : Démarrent mais sont extrêmement lents (timeout sans progression visible)

L'objectif est de corriger ces trois forecasters pour qu'ils s'entraînent correctement et rapidement comme iTransformer et PatchTST, permettant ainsi de lancer leur entraînement RL.

## Bug Analysis

### Current Behavior (Defect)

#### 1. TimesNet - Erreur au démarrage

1.1 WHEN le script d'entraînement est lancé avec la config `assets/configs/models/weather_dataset/forecasters/timesnet/weather_96_96_S.json` THEN le système crash avec l'erreur `AttributeError: 'types.SimpleNamespace' object has no attribute 'embed'`

1.2 WHEN `data_factory.py` ligne 25 tente d'accéder à `args.embed` THEN l'attribut n'existe pas dans la configuration TimesNet pour Weather

#### 2. GRU - Timeout sans progression

1.3 WHEN le modèle GRU démarre l'entraînement avec la config Weather THEN le système affiche "161,760 params" mais ne progresse pas et timeout après 60 secondes sans afficher d'epoch

1.4 WHEN le modèle GRU s'entraîne sur Weather (dataset plus large que ETTh1) THEN les hyperparamètres (d_model=128, e_layers=2, learning_rate=0.001) causent une lenteur excessive

#### 3. DLinear - Lenteur excessive

1.5 WHEN le modèle DLinear démarre l'entraînement avec la config Weather THEN le système progresse mais prend beaucoup trop de temps et timeout après 30 secondes avec seulement une epoch complétée

1.6 WHEN le modèle DLinear s'entraîne sur Weather avec moving_avg=13 THEN la performance est significativement plus lente que les configurations ETTh1 qui utilisent moving_avg=25

### Expected Behavior (Correct)

#### 1. TimesNet - Démarrage correct

2.1 WHEN le script d'entraînement est lancé avec la config TimesNet pour Weather THEN le système SHALL démarrer sans erreur en utilisant l'attribut `embed` défini dans la configuration

2.2 WHEN `data_factory.py` accède à `args.embed` THEN l'attribut SHALL exister dans toutes les configurations de forecasters (GRU, DLinear, TimesNet, PatchTST, iTransformer)

#### 2. GRU - Performance optimisée

2.3 WHEN le modèle GRU s'entraîne sur Weather THEN le système SHALL compléter les epochs dans un temps raisonnable comparable à iTransformer et PatchTST

2.4 WHEN le modèle GRU utilise des hyperparamètres optimisés pour Weather THEN le système SHALL utiliser des valeurs adaptées à la taille du dataset (batch_size, learning_rate, train_epochs ajustés)

#### 3. DLinear - Performance optimisée

2.5 WHEN le modèle DLinear s'entraîne sur Weather THEN le système SHALL compléter les epochs dans un temps raisonnable comparable à iTransformer et PatchTST

2.6 WHEN le modèle DLinear utilise des hyperparamètres optimisés THEN le système SHALL utiliser des valeurs cohérentes avec les configurations ETTh1 qui fonctionnent (moving_avg=25, batch_size optimisé)

### Unchanged Behavior (Regression Prevention)

3.1 WHEN iTransformer et PatchTST s'entraînent sur Weather THEN le système SHALL CONTINUE TO fonctionner correctement avec leurs configurations actuelles

3.2 WHEN les forecasters GRU, DLinear et TimesNet s'entraînent sur ETTh1 THEN le système SHALL CONTINUE TO fonctionner correctement avec leurs configurations existantes

3.3 WHEN `data_factory.py` traite les configurations avec l'attribut `embed` THEN le système SHALL CONTINUE TO utiliser `timeenc = 0 if args.embed != 'timeF' else 1`

3.4 WHEN les modèles sont entraînés avec leurs checkpoints et historiques THEN le système SHALL CONTINUE TO sauvegarder les résultats dans les chemins configurés

3.5 WHEN le pipeline Weather est exécuté THEN le système SHALL CONTINUE TO entraîner tous les forecasters dans l'ordre défini dans `FORECASTER_CONFIGS`

## Bug Condition and Property Specification

### Bug Condition Function

```pascal
FUNCTION isBugCondition(config)
  INPUT: config of type ForecasterConfig
  OUTPUT: boolean
  
  // Returns true when the bug condition is met
  RETURN (config.model_type = "TimesNet" AND config.dataset_name = "Weather" AND NOT hasAttribute(config, "embed"))
         OR (config.model_type = "GRU" AND config.dataset_name = "Weather" AND config.d_model = 128)
         OR (config.model_type = "DLinear" AND config.dataset_name = "Weather" AND config.moving_avg = 13)
END FUNCTION
```

### Property: Fix Checking

```pascal
// Property 1: TimesNet Configuration Completeness
FOR ALL config WHERE config.model_type = "TimesNet" AND config.dataset_name = "Weather" DO
  result ← loadConfig(config)
  ASSERT hasAttribute(result, "embed") AND result.embed = "timeF"
END FOR

// Property 2: GRU Performance Optimization
FOR ALL config WHERE config.model_type = "GRU" AND config.dataset_name = "Weather" DO
  training_time ← trainModel(config)
  ASSERT training_time < TIMEOUT_THRESHOLD AND epochs_completed > 0
END FOR

// Property 3: DLinear Performance Optimization
FOR ALL config WHERE config.model_type = "DLinear" AND config.dataset_name = "Weather" DO
  training_time ← trainModel(config)
  ASSERT training_time < TIMEOUT_THRESHOLD AND epochs_per_minute > MIN_EPOCH_RATE
END FOR
```

### Property: Preservation Checking

```pascal
// Property: Preservation of Working Configurations
FOR ALL config WHERE NOT isBugCondition(config) DO
  result_before ← trainModel_original(config)
  result_after ← trainModel_fixed(config)
  ASSERT result_before.behavior = result_after.behavior
END FOR
```

**Key Definitions:**
- **F**: Original configuration loading and training logic
- **F'**: Fixed configuration loading and training logic with optimized hyperparameters
- **isBugCondition(X)**: Returns true for Weather configs of GRU/DLinear/TimesNet with problematic settings
- **¬isBugCondition(X)**: All other configurations (ETTh1, ETTh2, or working Weather configs like iTransformer/PatchTST)

## Concrete Counterexamples

### Example 1: TimesNet Missing Attribute
```json
// Input: weather_96_96_S.json for TimesNet (BEFORE FIX)
{
  "model_type": "TimesNet",
  "dataset_name": "Weather",
  // ... other params ...
  // MISSING: "embed": "timeF"
}

// Error: AttributeError: 'types.SimpleNamespace' object has no attribute 'embed'
// Expected: Should load successfully with embed="timeF"
```

### Example 2: GRU Slow Training
```json
// Input: weather_96_96_S.json for GRU (BEFORE FIX)
{
  "model_type": "GRU",
  "d_model": 128,
  "e_layers": 2,
  "learning_rate": 0.001,
  "train_epochs": 50
}

// Result: 161,760 params, timeout after 60s, 0 epochs completed
// Expected: Should complete multiple epochs within reasonable time
```

### Example 3: DLinear Slow Training
```json
// Input: weather_96_96_S.json for DLinear (BEFORE FIX)
{
  "model_type": "DLinear",
  "moving_avg": 13,
  "learning_rate": 0.005,
  "train_epochs": 50
}

// Result: Timeout after 30s, only 1 epoch completed
// Expected: Should complete multiple epochs within reasonable time
```
