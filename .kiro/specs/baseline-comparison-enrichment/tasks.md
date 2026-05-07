# Implementation Tasks: baseline-comparison-enrichment

## Task List

- [x] 1. Module commun `baselines/common/`
  - [x] 1.1 Créer `baselines/common/__init__.py` (vide)
  - [x] 1.2 Créer `baselines/common/bounds.py` — `compute_bounds_np()` et `load_bounds_params()` (réplique exacte de `_compute_bounds_np` de `trainer_last.py`)
  - [x] 1.3 Créer `baselines/common/result_io.py` — `save_results()`, `load_results()`, `check_cache()` (MD5), `aggregate_seeds()`
  - [x] 1.4 Créer `baselines/common/evaluator_wrapper.py` — `run_evaluation()` qui appelle `CounterfactualEvaluator` (8 métriques) + `evaluate_forecastcf_metrics` (4 métriques)

- [-] 2. Méthode ForecastCF-PyTorch
  - [x] 2.1 Créer `baselines/ForecastCF_PyTorch/__init__.py` (vide)
  - [x] 2.2 Créer `baselines/ForecastCF_PyTorch/forecastcf_pt.py` — classe `ForecastCFPyTorch` avec `_margin_mse()`, `_weighted_mae()`, `transform_sample()`, `transform()` (Adam lr=1e-4, max_iter=100, early stopping, bornes RL-MCF)
  - [ ] 2.3 Créer `baselines/ForecastCF_PyTorch/run_forecastcf_pt.py` — runner CLI (args: dataset, model, seeds, device, output_dir) qui charge ForecasterWrapperV2, appelle `prepare_rl_data()`, calcule les bornes via `compute_bounds_np()`, génère les CFs, évalue et sauvegarde JSON
  - [ ] 2.4 Valider sur ETTh1/GRU : lancer `run_forecastcf_pt.py --dataset etth1 --model gru` et vérifier que le JSON de sortie contient les 8 métriques étendues + 4 métriques ForecastCF

- [-] 3. Baseline BaseNN généralisée
  - [x] 3.1 Créer `baselines/BaseNN/__init__.py` (vide)
  - [x] 3.2 Créer `baselines/BaseNN/basenn.py` — classe `BaseNN` avec `fit(X_train, Y_hat_train)` et `transform(X_test, alphas, betas)` (1-NN sur les forecasts, target = (alpha+beta)/2)
  - [x] 3.3 Créer `baselines/BaseNN/run_basenn.py` — runner CLI généralisé (même pattern que `run_forecastcf_pt.py`) qui collecte le train set complet, appelle `BaseNN.fit()` puis `transform()`, évalue et sauvegarde JSON
  - [ ] 3.4 Valider sur ETTh1/GRU : vérifier que les résultats sont cohérents avec `run_basenn_etth1_gru.py` existant

- [-] 4. Baseline BaseShift
  - [x] 4.1 Créer `baselines/BaseShift/__init__.py` (vide)
  - [x] 4.2 Créer `baselines/BaseShift/baseshift.py` — classe `BaseShift` avec `transform_sample()` (recherche binaire sur delta ∈ [-5σ, +5σ], 100 itérations, fallback x_orig si hors-bande) et `transform()`
  - [ ] 4.3 Créer `baselines/BaseShift/run_baseshift.py` — runner CLI (même pattern)
  - [ ] 4.4 Valider sur ETTh1/GRU : vérifier validity_ratio élevée (BaseShift doit avoir une validity proche de 1.0 sur des cas simples)

- [ ] 5. Baseline BaseGrad
  - [ ] 5.1 Créer `baselines/BaseGrad/__init__.py` (vide)
  - [ ] 5.2 Créer `baselines/BaseGrad/basegrad.py` — classe `BaseGrad` avec `transform_sample()` (Adam lr=0.01, max_iter=300, hinge loss sur tous les timesteps, clipping x_cf dans [min-3σ, max+3σ]) et `transform()`
  - [ ] 5.3 Créer `baselines/BaseGrad/run_basegrad.py` — runner CLI (même pattern)
  - [ ] 5.4 Valider sur ETTh1/GRU : vérifier que BaseGrad converge et que la loss diminue

- [ ] 6. Généralisation multi-datasets et multi-modèles
  - [ ] 6.1 Vérifier que les 4 runners acceptent tous les datasets (etth1, etth2, weather) et modèles (itransformer, patchtst, dlinear, gru, timesnet) via les args CLI
  - [ ] 6.2 Implémenter le mapping des chemins de configs RL par dataset dans `bounds.py::load_bounds_params()` (etth1 → `RL_ablations/`, etth2 → `RL/`, weather → `RL/`)
  - [ ] 6.3 Tester sur ETTh2/GRU et Weather/GRU pour valider la généralisation

- [ ] 7. Orchestrateur global
  - [ ] 7.1 Créer `scripts/run_all_baselines.py` — args `--datasets`, `--models`, `--baselines`, `--seeds`, `--device`, `--output_dir`
  - [ ] 7.2 Implémenter la boucle séquentielle sur toutes les combinaisons (dataset, model, baseline) avec vérification de cache MD5, gestion des erreurs (log + continue), et barre de progression
  - [ ] 7.3 Implémenter la génération du fichier `baselines/results_summary.json` avec statut (success/skipped/error) par combinaison
  - [ ] 7.4 Appel automatique de `build_comparison_table.py` à la fin de l'orchestration

- [ ] 8. Tableau de comparaison et visualisations
  - [ ] 8.1 Étendre `src/experiments/comparison/build_comparison_table.py` pour charger les résultats JSON de ForecastCF-PyTorch et des 3 baselines triviales, en plus des résultats RL-MCF existants
  - [ ] 8.2 Générer le tableau LaTeX (5 méthodes × 6 métriques, format `mean±std`, meilleures valeurs en `\textbf{}`) et le sauvegarder dans `assets/results/comparison/comparison_table.tex`
  - [ ] 8.3 Créer `src/experiments/comparison/plot_comparison.py` — barplot groupé (6 métriques, 5 méthodes, errorbars) + radar chart (5 axes normalisés), palette de couleurs définie, sauvegarde PDF+PNG 300 DPI dans `assets/figures/comparison/`
  - [ ] 8.4 Générer la figure d'exemples CF côte-à-côte (5 méthodes, ETTh1/GRU, x_orig + x_cf + forecast + bande cible)
