# Project Structure

## Overview

```
counterfactual-forecasting-rl/
├── src/                          # Source code
├── scripts/                      # Pipelines, evaluation, plot generation
├── baselines/                    # Baseline implementations
├── assets/                       # Data, configs, checkpoints, results, figures
├── README.md                     # Project introduction and quick start
├── RESULTS_AND_DISCUSSION.md     # Full results, discussion, future directions
├── STRUCTURE.md                  # This file
├── run_full.py                   # Full evaluation pipeline (tables + figures)
└── requirements.txt              # Python dependencies
```

---

## `src/` — Source Code

```
src/
├── data_provider/
│   ├── data_factory.py           # Factory: dataset_name → DataLoader
│   └── data_loader.py            # Dataset classes (ETT, Weather, Custom)
├── models/
│   ├── Forecaster/
│   │   ├── iTransformer.py       # iTransformer architecture
│   │   ├── PatchTST.py           # PatchTST architecture
│   │   ├── TimesNet.py           # TimesNet architecture
│   │   ├── GRU.py                # GRU architecture
│   │   ├── DLinear.py            # DLinear architecture
│   │   ├── forecaster_wrapper.py     # Original wrapper (iTransformer)
│   │   └── forecaster_wrapper_v2.py  # Generic wrapper (all 5 models)
│   ├── autoencoder/
│   │   ├── tcn_ae.py             # TCN Autoencoder (encoder + decoder)
│   │   └── identity_ae.py       # Identity AE (ablation)
│   ├── RL/
│   │   ├── agent.py             # ActorCritic class
│   │   ├── actor.py             # Actor network (MLP + tanh)
│   │   ├── critic.py            # Critic network (MLP → scalar)
│   │   └── reward_last.py       # Reward function (validity + proximity)
│   └── anomaly_detector/
│       └── plausibility.py      # Ensemble detector (IForest + LOF + OC-SVM)
├── training/
│   ├── RL_trainers/
│   │   ├── trainer_main.py      # Main RL trainer (temporal mask, eval)
│   │   ├── trainer_last.py      # Legacy trainer (used by baselines/ablations)
│   │   └── trainer_wo_mask.py   # Ablation: no temporal mask
│   ├── forecast_trainers/
│   │   ├── itransformer_trainer.py
│   │   └── generic_forecaster_trainer.py
│   ├── ae_trainers/
│   │   └── ae_trainer.py
│   └── ad_trainers/
│       └── ad_trainer.py
├── experiments/
│   ├── rl_cf/run.py             # Train RL agents
│   ├── forecasting/run.py       # Train forecasters
│   ├── autoencoder/run.py       # Train AE
│   ├── anomaly_detection/run.py # Train anomaly detector
│   ├── comparison/              # ForecastCF comparison scripts
│   └── ablations/               # wo_mask, RLP ablations
├── evaluation/
│   └── unified_evaluator.py    # All 7 metrics
├── layers/                      # Transformer layers (Embed, Attention, EncDec)
└── utils/
    ├── config.py                # JSON → SimpleNamespace loader
    ├── train_tools.py           # Device selection, EarlyStopping
    ├── metrics.py               # MSE, MAE
    ├── masking.py               # Causal/Prob masks
    └── timefeatures.py          # Time feature engineering
```

---

## `scripts/` — Orchestration & Analysis

```
scripts/
├── pipelines/
│   ├── pipeline_etth1.py        # Full ETTh1: train → eval
│   ├── pipeline_etth2.py        # Full ETTh2: train → eval
│   └── pipeline_weather.py      # Full Weather: train → eval
├── evals/
│   ├── eval_etth1.py            # Evaluate 5 models on ETTh1
│   ├── eval_etth2.py            # Evaluate 5 models on ETTh2
│   ├── eval_weather.py          # Evaluate 5 models on Weather
│   ├── eval_forecasters.py      # Evaluate forecaster accuracy
│   ├── eval_with_significance.py # Statistical significance tests
│   ├── reeval_baselines_etth1.py
│   ├── run_all_baselines_etth1.py
│   └── run_all_baselines_weather_parallel.py
├── plots_generation/
│   ├── baselines/               # Comparison plots (scatter, barplots, tables)
│   ├── cf_examples/             # Counterfactual visualization
│   ├── model_analysis/          # Radars, heatmaps, t-SNE, barplots
│   ├── ablations/               # Ablation & sensitivity figures
│   ├── training/                # Training curves, AE reconstruction
│   └── datasets_inspection/     # Dataset exploration & periodicity
└── all_eval.py                  # Top-level eval orchestrator
```

---

## `baselines/` — Comparison Methods

```
baselines/
├── ForecastCF/                  # Original TensorFlow implementation (reference)
├── ForecastCF_PyTorch/          # PyTorch re-implementation
├── BaseGrad/                    # SPSA gradient-free optimization
├── BaseNN/                      # 1-Nearest-Neighbour baseline
└── common/                      # Shared: bounds, evaluator_wrapper, result_io
```

---

## `assets/` — Data, Configs, Outputs

```
assets/
├── datasets/                    # ETTh1.csv, ETTh2.csv, weather.csv
├── configs/{dataset}_dataset/
│   ├── ae/tcn_ae.json
│   ├── anomaly_detector/plausibility.json
│   ├── forecasters/{model}/     # Per-model forecaster configs
│   └── RL/config_{model}.json   # Per-model RL configs
├── checkpoints/{dataset}_chpts/
│   ├── ae/                      # ae_{dataset}.pt
│   ├── forecaster/              # chpt_{dataset}_{seq}_{pred}_{model}_S.pth
│   ├── RL/{model}/              # rl_cf_{model}_{dataset}_agent_best.pt
│   └── anomaly_detector/        # plausibility_{dataset}.pkl
├── results/{dataset}/
│   ├── RL/{model}/              # Per-model evaluation JSONs
│   ├── summary/                 # Aggregated results + plots
│   └── forecaster/              # Forecaster training histories
├── figures/
│   ├── global_analysis/         # Cross-dataset figures (radars, scatter, heatmaps, t-SNE)
│   ├── datasets_overview/       # Dataset decomposition plots
│   ├── etth1/                   # ETTh1-specific (ae, forecaster, RL per model)
│   ├── etth2/                   # ETTh2-specific
│   └── weather/                 # Weather-specific
└── results/significance/        # Statistical significance results
```

---

## Naming Conventions

| Type | Pattern | Example |
|------|---------|---------|
| Forecaster checkpoint | `chpt_{dataset}_{seq}_{pred}_{model}_S.pth` | `chpt_etth1_96_48_S.pth` |
| RL agent checkpoint | `rl_cf_{model}_{dataset}_agent_best.pt` | `rl_cf_itransformer_etth1_agent_best.pt` |
| AE checkpoint | `ae_{dataset}.pt` | `ae_etth1.pt` |
| Training history | `history_{dataset}_{seq}_{pred}_{model}_S.json` | `history_etth1_96_48_S.json` |
| RL config | `config_{model}.json` | `config_itransformer.json` |
