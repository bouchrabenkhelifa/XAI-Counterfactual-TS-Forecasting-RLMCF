# Paper Finalization Tasks
# Counterfactual Explanations for Time Series Forecasting via RL
# Deadline: 4 jours

---

## Task 1 — Anomaly Detection sur ETTh2

- [ ] 1.1 Créer la config `assets/configs/models/etth2_dataset/anomaly_detector/plausibility.json`
        (copie de `assets/configs/models/etth1_dataset/anomaly_detector/plausibility.json` avec data_path=ETTh2.csv et noms adaptés)
- [ ] 1.2 Créer les dossiers de sortie manquants :
        `assets/checkpoints/etth2_chpts/anomaly_detector/`
        `assets/results/etth2/anomaly_detector/`
        `assets/figures/etth2/anomaly_detector/`
- [ ] 1.3 Lancer l'entraînement :
        `python src/experiments/anomaly_detection/run.py --config assets/configs/models/etth2_dataset/anomaly_detector/plausibility.json`
- [ ] 1.4 Vérifier la figure sanity check générée dans `assets/figures/etth2/anomaly_detector/plausibility_sanity_ETTh2.png`

---

## Task 2 — Figure qualitative : RLCF vs RLP vs RL-wo-mask (Ablation)

**Objectif** : une seule figure avec 3 colonnes (une par méthode) × N lignes (N samples identiques),
montrant pour chaque méthode sur le **même sample** :
- `x_original` (série d'entrée, en bleu)
- `x_cf` (série contrefactuelle, en rouge pointillé)
- `y_hat` (prévision originale, en bleu tiret)
- `y_cf` (prévision CF, en rouge tiret)
- bande α/β (zone verte transparente)
- ligne verticale séparant lookback / horizon

**Layout** : 3 colonnes × 3 lignes = 9 sous-figures
- Colonne 1 : RLCF (iTransformer, ETTh1)
- Colonne 2 : RLP (Random Latent Perturbation, même samples)
- Colonne 3 : RL-wo-mask (sans masque temporel, même samples)
- Titre de chaque colonne en haut
- Annotation validity_hard sur chaque sous-figure

**Script à créer** : `scripts/generate_ablation_series_figure.py`

- [ ] 2.1 Créer `scripts/generate_ablation_series_figure.py` :
        - Charger les 3 modèles (RLCF, RLP, wo_mask) avec leurs checkpoints
        - Fixer un seed et extraire les mêmes N=3 samples du test set ETTh1
        - Pour chaque sample, générer x_cf via les 3 méthodes
        - Tracer la figure 3×3 côte-à-côte, style publication (fond blanc, police 11pt)
        - Sauvegarder dans `assets/figures/comparison/ablation_series_3methods.pdf` et `.png`
- [ ] 2.2 Lancer le script et vérifier la figure

---

## Task 3 — Figure qualitative : RLCF vs ForecastCF (Baseline)

**Objectif** : une figure avec 2 colonnes (RLCF | ForecastCF) × N lignes (N samples identiques),
montrant pour chaque méthode sur le **même sample** :
- `x_original` (série d'entrée, en bleu)
- `x_cf` (série contrefactuelle, en rouge pointillé)
- `y_hat` (prévision originale, en bleu tiret)
- `y_cf` (prévision CF, en rouge tiret)
- bande α/β (zone verte transparente)
- ligne verticale séparant lookback / horizon

**Layout** : 2 colonnes × 3 lignes = 6 sous-figures
- Colonne 1 : RLCF (GRU, ETTh1) — checkpoint `assets/checkpoints/etth1_chpts/RL/gru/rl_cf_gru_etth1_agent_best.pt`
- Colonne 2 : ForecastCF (GRU, ETTh1) — utiliser `baselines/ForecastCF/run_etth1_gru.py` pour générer les CFs
- Annotation validity_hard sur chaque sous-figure

**Script à créer** : `scripts/generate_baseline_series_figure.py`

- [ ] 3.1 Créer `scripts/generate_baseline_series_figure.py` :
        - Charger RLCF (GRU) et générer x_cf pour N=3 samples fixes du test set ETTh1
        - Charger ForecastCF et générer x_cf pour les **mêmes** N=3 samples
        - Tracer la figure 2×3 côte-à-côte, style publication
        - Sauvegarder dans `assets/figures/comparison/rlcf_vs_forecastcf_series.pdf` et `.png`
- [ ] 3.2 Lancer le script et vérifier la figure

---

## Task 4 — Tableau LaTeX pour le papier

- [ ] 4.1 Créer `scripts/generate_latex_table.py` :
        - Tableau principal : RLCF sur 5 modèles × ETTh1 + ETTh2
          (validity_ratio, stepwise_auc, proximity_l2, compactness, temporal_consistency)
        - Tableau ablation : RLCF vs RLP vs wo_mask (ETTh1, iTransformer)
        - Tableau comparaison : RLCF+GRU vs ForecastCF+GRU (ETTh1)
        - `\textbf{}` sur les meilleures valeurs par colonne
        - Sauvegarder dans `assets/results/latex_tables/`
- [ ] 4.2 Vérifier les fichiers `.tex` générés

---

## Résumé des fichiers à créer

| Fichier | Description |
|---|---|
| `assets/configs/models/etth2_dataset/anomaly_detector/plausibility.json` | Config anomaly detector ETTh2 |
| `scripts/generate_ablation_series_figure.py` | Figure séries RLCF vs RLP vs wo_mask |
| `scripts/generate_baseline_series_figure.py` | Figure séries RLCF vs ForecastCF |
| `scripts/generate_latex_table.py` | Tableaux LaTeX |
| `assets/figures/comparison/ablation_series_3methods.png` | Figure ablation (sortie) |
| `assets/figures/comparison/rlcf_vs_forecastcf_series.png` | Figure baseline (sortie) |
| `assets/results/latex_tables/*.tex` | Tableaux LaTeX (sortie) |

---

## Checkpoints disponibles

| Méthode | Checkpoint |
|---|---|
| RLCF iTransformer ETTh1 | `assets/checkpoints/etth1_chpts/RL/itransformer/rl_cf_itransformer_best_etth1_agent_best.pt` |
| RLCF GRU ETTh1 | `assets/checkpoints/etth1_chpts/RL/gru/rl_cf_gru_etth1_agent_best.pt` |
| wo_mask ETTh1 | `assets/checkpoints/etth1_chpts/RL/wo_mask/rl_cf_wo_mask_etth1_agent_best.pt` |
| AE ETTh1 | `assets/checkpoints/etth1_chpts/ae/ae_etth1.pt` |
