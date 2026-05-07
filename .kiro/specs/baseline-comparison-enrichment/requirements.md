# Requirements Document

## Introduction

Ce document spécifie les exigences pour enrichir la section expérimentale du papier de recherche sur **RL-MCF** (Reinforcement Learning for Masked Counterfactual Forecasting). L'objectif est de construire une comparaison complète et rigoureuse entre RL-MCF et les baselines de référence, entièrement en PyTorch, sur les datasets et modèles de forecasting déjà disponibles dans le projet.

**Contexte :** Le papier compare RL-MCF à ForecastCF (Wang et al., ICDM 2023). La méthode de comparaison principale est désormais **ForecastCF-PyTorch** — une réimplémentation native PyTorch de l'algorithme original (sans dépendance TensorFlow), qui permet une comparaison directe et honnête avec la méthode de référence du papier. Les baselines triviales (BaseNN, BaseShift, BaseGrad) complètent la comparaison pour contextualiser les résultats.

**Contrainte principale :** ForecastCF original est en TensorFlow, les modèles de forecasting du projet sont en PyTorch → toute la comparaison doit rester en PyTorch pur, en réutilisant l'infrastructure existante (`CounterfactualEvaluator`, `ForecasterWrapperV2`, `prepare_rl_data`). L'approche bridge TF↔PyTorch (`pytorch_adapter.py`) est conservée comme fallback mais n'est pas prioritaire.

**Deadline : 10 mai 2026.**

---

## Glossaire

- **RL-MCF** : La méthode proposée — agent Actor-Critic qui génère des counterfactuals en espace latent avec masque temporel.
- **ForecastCF-PyTorch** : Réimplémentation native PyTorch de l'algorithme ForecastCF (Wang et al., ICDM 2023) — descente de gradient Adam sur `x_cf` avec loss `margin_mse + weighted_mae`, lr=0.0001, max_iter=100, sans dépendance TensorFlow. Méthode de comparaison principale. **Utilise les bornes RL-MCF** (voir définition ci-dessous).
- **ForecastCF-Bridge** : L'ancienne approche via `pytorch_adapter.py` (bridge TF↔PyTorch) — conservée comme fallback mais non prioritaire.
- **BaseNN** : Baseline 1-nearest-neighbour — pour chaque sample test, retourne le sample d'entraînement dont le forecast est le plus proche du centre de la bande cible [α, β]. **Utilise les bornes RL-MCF.**
- **BaseShift** : Baseline de décalage constant — applique un shift uniforme sur toute la série d'entrée pour atteindre la cible. **Utilise les bornes RL-MCF.**
- **BaseGrad** : Baseline de descente de gradient directe — optimise `x_cf` par gradient descent sur la loss `relu hinge + L1 proximity`, lr=0.01, max_iter=300, sans autoencoder ni RL. **Utilise les bornes RL-MCF.**
- **Bornes RL-MCF** (`compute_bounds`) : Définition des bornes [α, β] propre au projet — **ancrées sur le forecast y_hat** (pas sur la médiane de l'input comme ForecastCF original) :
  ```
  sigma = global_sigma  (std fixe du train set, ou std(x_ot) si non disponible)
  gap   = rho * sigma   (séparation garantie entre y_hat et la bande)
  width = fr  * sigma   (largeur de la bande)
  
  direction < 0 (baisse) :  beta  = y_hat - gap
                             alpha = beta  - width
  direction > 0 (hausse) :  alpha = y_hat + gap
                             beta  = alpha + width
  ```
  Implémentation de référence : `src/models/RL/reward_last.py::CFReward.compute_bounds()` et `src/training/RL_trainers/trainer_last.py::RLMaskTrainer._compute_bounds_np()`.
- **Bornes ForecastCF originales** (NE PAS UTILISER pour la comparaison) : ancrées sur `median(x_ot)` avec tendance polynomiale — incompatibles avec l'objectif du projet.
- **margin_mse** : Composante de loss ForecastCF — MSE entre la prédiction et la borne la plus proche, masquée sur les timesteps hors-bande [α, β] uniquement.
- **weighted_mae** : Composante de loss ForecastCF — MAE pondérée par `step_weights` (ones par défaut → MAE simple) entre `x_orig` et `x_cf`.
- **Forecaster** : Modèle de prévision gelé (iTransformer, PatchTST, DLinear, GRU, TimesNet).
- **Counterfactual_Evaluator** : Module `src/evaluation/unified_evaluator.py` — calcule les 8 métriques standardisées.
- **ForecastCF_Evaluator** : Module `baselines/ForecastCF/src/forecastcf_evaluator.py` — calcule les 4 métriques exactes du papier ForecastCF (validity_ratio, stepwise_auc, proximity_l2, compactness avec atol=0.01).
- **Pipeline_Baseline** : Script orchestrateur qui exécute une baseline sur un (dataset, forecaster) et sauvegarde les résultats JSON.
- **Comparison_Table_Builder** : Module qui agrège les résultats JSON de toutes les méthodes et génère le tableau LaTeX final.
- **Dataset** : ETTh1, ETTh2, ou Weather — datasets disponibles avec checkpoints entraînés.
- **Seed** : Graine aléatoire pour la reproductibilité (seeds = [1, 9, 30]).

---

## Requirements

### Requirement 0 : Bornes partagées — toutes les méthodes utilisent `compute_bounds` de RL-MCF

**User Story :** En tant que chercheur, je veux que toutes les méthodes de comparaison (ForecastCF-PyTorch, BaseNN, BaseShift, BaseGrad) utilisent exactement les mêmes bornes [α, β] que RL-MCF, afin que la comparaison soit équitable et que les métriques de validity soient calculées sur le même objectif.

#### Acceptance Criteria

1. THE `Pipeline_Baseline` SHALL calculer les bornes [α, β] en appelant `_compute_bounds_np(x_ot, y_hat)` (ou son équivalent inline) pour chaque sample, avec les paramètres `rho`, `fr`, `direction`, et `global_sigma` lus depuis la config RL correspondante (`assets/configs/models/{dataset}_dataset/RL_ablations/config_{model}.json`).
2. THE bornes SHALL être calculées selon la formule :
   ```
   sigma = global_sigma  (depuis la config RL, ou std(x_ot) si absent)
   gap   = rho * sigma
   width = fr  * sigma
   direction < 0 :  beta = y_hat - gap ;  alpha = beta - width
   direction > 0 :  alpha = y_hat + gap ;  beta = alpha + width
   ```
3. THE `Pipeline_Baseline` SHALL NEVER utiliser les bornes polynomiales de ForecastCF original (`sv * (1 + poly_trend ± fr * std)`) — ces bornes sont incompatibles avec l'objectif du projet.
4. WHEN `global_sigma` est absent de la config RL, THE `Pipeline_Baseline` SHALL calculer `sigma = std(x_ot, ddof=1).clip(min=1e-4)` par sample, identique à `_compute_bounds_np`.
5. THE bornes calculées par `Pipeline_Baseline` SHALL être numériquement identiques à celles utilisées par RL-MCF pour les mêmes samples, afin de garantir la comparabilité des métriques de validity.

---

### Requirement 1 : ForecastCF-PyTorch natif (méthode de comparaison principale)

**User Story :** En tant que chercheur, je veux une réimplémentation native PyTorch de l'algorithme ForecastCF original, afin de comparer RL-MCF directement avec la méthode de référence du papier sans dépendance TensorFlow.

#### Acceptance Criteria

1. THE `ForecastCF-PyTorch` SHALL implémenter `margin_mse` en PyTorch pur : calculer le MSE entre la prédiction `ŷ_cf` et la borne la plus proche (`alpha` ou `beta`), en masquant uniquement les timesteps hors-bande (où `ŷ_cf < α` ou `ŷ_cf > β`).
2. THE `ForecastCF-PyTorch` SHALL implémenter `weighted_mae` en PyTorch pur : calculer `mean(step_weights * |x_orig - x_cf|)`, avec `step_weights = ones` par défaut (équivalent à une MAE simple).
3. THE `ForecastCF-PyTorch` SHALL calculer la loss totale comme `L = pred_margin_weight * margin_mse(ŷ_cf, α, β) + (1 - pred_margin_weight) * weighted_mae(x_orig, x_cf, step_weights)`, avec `pred_margin_weight = 0.5` par défaut.
4. WHEN l'optimisation est lancée, THE `ForecastCF-PyTorch` SHALL utiliser Adam avec `lr=0.0001` et un maximum de `max_iter=100` itérations, en initialisant `x_cf = x_orig` (variable optimisée).
5. WHEN tous les timesteps de `ŷ_cf` sont dans la bande [α, β], THE `ForecastCF-PyTorch` SHALL arrêter l'optimisation (early stopping) avant d'atteindre `max_iter`.
6. THE `ForecastCF-PyTorch` SHALL utiliser le forecaster PyTorch directement (`.eval()`, `requires_grad=False` sur les paramètres du forecaster), sans wrapper TensorFlow ni bridge.
7. THE bornes [α, β] utilisées dans la loss et l'évaluation SHALL être calculées par `_compute_bounds_np` (Requirement 0) — **pas** les bornes polynomiales de ForecastCF original.
8. WHEN les counterfactuals sont générés, THE `ForecastCF_Evaluator` SHALL calculer les 4 métriques exactes du papier ForecastCF : `validity_ratio`, `stepwise_auc`, `proximity_l2`, `compactness` (avec `atol=0.01`).
9. WHEN les counterfactuals sont générés, THE `Counterfactual_Evaluator` SHALL également calculer les 8 métriques étendues pour permettre la comparaison complète avec RL-MCF.
10. THE `Pipeline_Baseline` SHALL sauvegarder les résultats ForecastCF-PyTorch dans `baselines/ForecastCF_PyTorch/results/{dataset}_{model}.json` avec la structure `{"method", "dataset", "model", "n_samples", "runtime_seconds", "avg_metrics", "forecastcf_metrics", "all_seeds"}`.
11. WHEN les seeds [1, 9, 30] sont utilisées, THE `Pipeline_Baseline` SHALL calculer la moyenne et l'écart-type inter-seeds pour chaque métrique.

---

### Requirement 2 : Baseline BaseNN généralisée

**User Story :** En tant que chercheur, je veux exécuter BaseNN sur tous les (dataset, forecaster) disponibles, afin de disposer d'une baseline de référence complète pour la comparaison.

#### Acceptance Criteria

1. THE `Pipeline_Baseline` SHALL accepter en paramètres : le nom du dataset (`etth1`, `etth2`, `weather`), le nom du forecaster (`itransformer`, `patchtst`, `dlinear`, `gru`, `timesnet`), et le répertoire de sortie.
2. WHEN `Pipeline_Baseline` est exécuté pour BaseNN, THE `Pipeline_Baseline` SHALL charger le checkpoint forecaster correspondant depuis `assets/checkpoints/{dataset}_chpts/forecaster/` et le config JSON depuis `assets/configs/models/{dataset}_dataset/forecasters/{model}/`.
3. WHEN le forecaster est chargé, THE `Pipeline_Baseline` SHALL collecter l'intégralité du train set en mémoire pour construire la base de recherche de BaseNN.
4. WHEN les counterfactuals BaseNN sont générés, THE `Counterfactual_Evaluator` SHALL calculer les 8 métriques : `validity_ratio`, `stepwise_auc`, `proximity_l2`, `compactness`, `roughness_ratio`, `temporal_consistency`, `relative_reduction`, `plausibility_ensemble`.
5. THE `Pipeline_Baseline` SHALL sauvegarder les résultats dans `baselines/BaseNN/results/{baseline}_{dataset}_{model}.json` avec la structure `{"method", "dataset", "model", "n_samples", "runtime_seconds", "avg_metrics", "all_seeds"}`.
6. WHEN les seeds [1, 9, 30] sont utilisées, THE `Pipeline_Baseline` SHALL calculer la moyenne et l'écart-type inter-seeds pour chaque métrique.

---

### Requirement 3 : Baseline BaseShift

**User Story :** En tant que chercheur, je veux une baseline de décalage constant (BaseShift) qui applique un shift uniforme sur la série d'entrée, afin d'avoir une référence triviale qui maximise la validity au prix d'une mauvaise compactness.

#### Acceptance Criteria

1. THE `BaseShift` SHALL, pour chaque sample test, calculer le shift scalaire minimal `δ` tel que le forecast de `x + δ` tombe dans la bande cible [α, β] calculée par `_compute_bounds_np` (Requirement 0).
2. WHEN le shift `δ` est calculé, THE `BaseShift` SHALL appliquer `x_cf = x + δ` uniformément sur tous les timesteps de la fenêtre d'entrée.
3. IF aucun shift scalaire ne permet d'atteindre la bande cible après 100 itérations de recherche binaire, THEN THE `BaseShift` SHALL retourner `x_cf = x` (pas de modification) et logger un avertissement.
4. THE `Counterfactual_Evaluator` SHALL évaluer BaseShift avec les mêmes 8 métriques que BaseNN.
5. THE `Pipeline_Baseline` SHALL sauvegarder les résultats BaseShift dans `baselines/BaseShift/results/{baseline}_{dataset}_{model}.json`.

---

### Requirement 4 : Baseline BaseGrad (descente de gradient directe)

**User Story :** En tant que chercheur, je veux une baseline de gradient direct (BaseGrad) qui optimise la série d'entrée par descente de gradient sans autoencoder, afin de montrer l'apport de l'espace latent et du masque temporel de RL-MCF et de distinguer clairement BaseGrad de ForecastCF-PyTorch.

#### Acceptance Criteria

1. THE `BaseGrad` SHALL optimiser `x_cf` par descente de gradient Adam sur la loss `L = w_v · L_validity + w_p · L_proximity`, où `L_validity = mean(relu(α - ŷ_cf) + relu(ŷ_cf - β))` (hinge loss sur tous les timesteps) et `L_proximity = mean(|x_cf - x|)` (L1).
2. WHEN l'optimisation est lancée, THE `BaseGrad` SHALL utiliser un learning rate de 0.01, un maximum de 300 itérations, et les poids `w_v = 1.0`, `w_p = 0.5` par défaut.
3. WHILE l'optimisation est en cours, THE `BaseGrad` SHALL clipper `x_cf` dans l'intervalle `[x.min() - 3σ, x.max() + 3σ]` à chaque itération pour éviter les valeurs aberrantes.
4. THE `BaseGrad` SHALL utiliser le même forecaster gelé (`.eval()`, `requires_grad=False` sur les paramètres) que les autres baselines.
5. THE `Counterfactual_Evaluator` SHALL évaluer BaseGrad avec les mêmes 8 métriques.
6. THE `Pipeline_Baseline` SHALL sauvegarder les résultats BaseGrad dans `baselines/BaseGrad/results/{baseline}_{dataset}_{model}.json`.

---

### Requirement 5 : Couverture multi-datasets et multi-modèles

**User Story :** En tant que chercheur, je veux que toutes les méthodes (ForecastCF-PyTorch, BaseNN, BaseShift, BaseGrad) soient évaluées sur les combinaisons (dataset, forecaster) pour lesquelles des checkpoints RL-MCF existent, afin que la comparaison soit équitable et directement comparable.

#### Acceptance Criteria

1. THE `Pipeline_Baseline` SHALL supporter les combinaisons (dataset, forecaster) suivantes, qui correspondent aux checkpoints disponibles :
   - ETTh1 : iTransformer, PatchTST, DLinear, GRU, TimesNet
   - ETTh2 : iTransformer, PatchTST, DLinear, GRU, TimesNet
   - Weather : GRU, iTransformer, PatchTST
2. WHEN un checkpoint forecaster est absent pour une combinaison donnée, THE `Pipeline_Baseline` SHALL logger un message d'erreur explicite et passer à la combinaison suivante sans interrompre le pipeline.
3. THE `Pipeline_Baseline` SHALL utiliser les mêmes paramètres de bandes cibles (ρ, fr, direction) que ceux définis dans les configs RL correspondantes (`assets/configs/models/{dataset}_dataset/RL*/config_{model}.json`).
4. WHEN toutes les combinaisons sont évaluées, THE `Pipeline_Baseline` SHALL produire un fichier de résumé `baselines/results_summary.json` listant toutes les combinaisons traitées avec leur statut (`success`, `skipped`, `error`).

---

### Requirement 6 : Tableau de comparaison agrégé

**User Story :** En tant que chercheur, je veux un tableau de comparaison agrégé (RL-MCF vs ForecastCF-PyTorch vs BaseNN vs BaseShift vs BaseGrad) prêt à être inséré dans le papier, afin de gagner du temps sur la rédaction.

#### Acceptance Criteria

1. THE `Comparison_Table_Builder` SHALL charger les résultats JSON de RL-MCF (depuis `assets/results/{dataset}/RL/{model}/`), de ForecastCF-PyTorch (depuis `baselines/ForecastCF_PyTorch/results/`), et des trois baselines triviales (depuis `baselines/{Baseline}/results/`).
2. THE `Comparison_Table_Builder` SHALL générer un tableau LaTeX avec les colonnes : Méthode, Validity↑, AUC↑, Proximity↓, Compactness↑, T-Consistency↑, Plausibility↓, et les lignes : RL-MCF, ForecastCF-PyTorch, BaseNN, BaseShift, BaseGrad.
3. WHEN une valeur est la meilleure de sa colonne, THE `Comparison_Table_Builder` SHALL la mettre en gras (`\textbf{...}`) dans le LaTeX.
4. THE `Comparison_Table_Builder` SHALL afficher les valeurs au format `mean ± std` avec 3 décimales.
5. THE `Comparison_Table_Builder` SHALL sauvegarder le tableau LaTeX dans `assets/results/comparison/comparison_table.tex` et le tableau JSON dans `assets/results/comparison/comparison_table.json`.
6. WHEN les résultats sont agrégés sur plusieurs (dataset, forecaster), THE `Comparison_Table_Builder` SHALL calculer la moyenne macro sur toutes les combinaisons disponibles pour chaque méthode.

---

### Requirement 7 : Script d'orchestration global

**User Story :** En tant que chercheur avec 3 jours avant la deadline, je veux un script unique qui lance toutes les méthodes sur toutes les combinaisons et génère le tableau final, afin de minimiser les interventions manuelles.

#### Acceptance Criteria

1. THE `Orchestrator` SHALL accepter les arguments `--datasets`, `--models`, `--baselines`, `--seeds`, et `--output_dir` pour permettre une exécution partielle, avec `forecastcf_pytorch` comme valeur valide pour `--baselines`.
2. WHEN `--datasets etth1 etth2` est spécifié, THE `Orchestrator` SHALL exécuter uniquement les combinaisons impliquant ETTh1 et ETTh2.
3. THE `Orchestrator` SHALL exécuter les méthodes séquentiellement (pas de parallélisme) pour éviter les conflits de ressources GPU/CPU.
4. WHEN une méthode échoue sur une combinaison, THE `Orchestrator` SHALL logger l'erreur, continuer avec les combinaisons suivantes, et inclure un rapport d'erreurs dans le résumé final.
5. WHEN toutes les méthodes sont terminées, THE `Orchestrator` SHALL appeler automatiquement `Comparison_Table_Builder` pour générer le tableau final.
6. THE `Orchestrator` SHALL afficher une barre de progression indiquant le nombre de combinaisons traitées sur le total.

---

### Requirement 8 : Reproductibilité et traçabilité

**User Story :** En tant que chercheur soumettant un papier, je veux que tous les résultats soient reproductibles et traçables, afin de pouvoir répondre aux reviewers.

#### Acceptance Criteria

1. THE `Pipeline_Baseline` SHALL fixer les seeds NumPy et PyTorch (`np.random.seed(seed)`, `torch.manual_seed(seed)`) avant chaque évaluation.
2. THE `Pipeline_Baseline` SHALL enregistrer dans chaque fichier JSON de résultats : la version Python, la version PyTorch, le nom du checkpoint utilisé, le hash MD5 du checkpoint, et la date d'exécution.
3. THE `Pipeline_Baseline` SHALL utiliser `device="cpu"` par défaut pour garantir la reproductibilité numérique entre machines.
4. WHEN un fichier de résultats existe déjà, THE `Pipeline_Baseline` SHALL vérifier si le checkpoint MD5 correspond et, si oui, sauter le recalcul en affichant un message `[Cache hit] {path}`.

---

### Requirement 9 : Visualisations pour le papier

**User Story :** En tant que chercheur, je veux des figures de comparaison publication-ready (300 DPI, format PDF+PNG), afin de les insérer directement dans le papier.

#### Acceptance Criteria

1. THE `Comparison_Table_Builder` SHALL générer un barplot groupé (une barre par méthode, un groupe par métrique) pour les 6 métriques principales, sauvegardé en PDF et PNG à 300 DPI dans `assets/figures/comparison/`.
2. THE `Comparison_Table_Builder` SHALL générer un radar chart comparant RL-MCF, ForecastCF-PyTorch, BaseNN, BaseShift, BaseGrad sur les métriques normalisées (validity, AUC, compactness, temporal consistency, plausibility inversée).
3. WHEN des exemples de counterfactuals sont disponibles, THE `Comparison_Table_Builder` SHALL générer une figure de 5 exemples côte-à-côte (x original, x_cf de chaque méthode, forecast, bande cible) pour le dataset ETTh1 / GRU.
4. THE `Comparison_Table_Builder` SHALL utiliser une palette de couleurs cohérente : RL-MCF en bleu (`#2196F3`), ForecastCF-PyTorch en rouge (`#F44336`), BaseNN en orange (`#FF5722`), BaseShift en vert (`#4CAF50`), BaseGrad en violet (`#9C27B0`).
