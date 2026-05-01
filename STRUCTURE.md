# Project Structure

## Quick Start

```bash
# Full pipeline for a dataset (train forecasters → AE → RL → eval)
python scripts/pipelines/pipeline_etth1.py

# Eval only (checkpoints already trained)
python scripts/pipelines/pipeline_etth1.py --eval_only

# Skip specific steps
python scripts/pipelines/pipeline_etth2.py --skip_forecasters
python scripts/pipelines/pipeline_weather.py --skip_rl
```

---

## Directory Layout

```
├── assets/
│   ├── checkpoints/
│   │   ├── etth1_chpts/          # AE, forecasters, RL agents for ETTh1
│   │   │   ├── ae/
│   │   │   ├── forecaster/
│   │   │   ├── RL_itransformer_best/
│   │   │   ├── RL_patchtst/
│   │   │   ├── RL_timesnet/
│   │   │   ├── RL_gru/
│   │   │   └── RL_dlinear/
│   │   ├── etth2_chpts/          # ETTh2 (same structure)
│   │   └── weather_chpts/        # Weather (same structure)
│   │
│   ├── configs/
│   │   └── models/
│   │       ├── etth1_dataset/
│   │       │   ├── ae/               # TCN-AE config
│   │       │   ├── forecasters/      # One config per model
│   │       │   └── RL_ablations/     # RL configs (config_itransformer_best, config_gru, ...)
│   │       ├── etth2_dataset/
│   │       │   ├── ae/
│   │       │   ├── forecasters/
│   │       │   └── RL/
│   │       └── weather_dataset/
│   │           ├── ae/
│   │           ├── forecasters/
│   │           └── RL/
│   │
│   ├── datasets/                 # Raw CSV files
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   └── weather.csv
│   │
│   ├── figures/                  # Training curves, CF examples, eval plots
│   └── results/                  # Evaluation JSONs, summaries
│
├── baselines/
│   ├── ForecastCF/               # ForecastCF (Wang et al., ICDM 2023)
│   │   ├── src/                  # Original + PyTorch adapter
│   │   ├── results/
│   │   └── run_etth1_gru.py      # Launch ForecastCF on ETTh1/GRU
│   └── BaseNN/                   # 1-NN baseline
│       └── run_basenn_etth1_gru.py
│
├── scripts/
│   ├── pipelines/
│   │   ├── pipeline_etth1.py     # Full ETTh1 pipeline
│   │   ├── pipeline_etth2.py     # Full ETTh2 pipeline
│   │   └── pipeline_weather.py   # Full Weather pipeline
│   ├── evals/
│   │   ├── eval_etth1_all_models.py    # Eval 5 models → table + radar + barplot
│   │   ├── eval_etth2_all_models.py
│   │   ├── eval_weather_all_models.py
│   │   ├── eval_forecasters.py         # Forecaster quality (MSE/MAE + plots)
│   │   ├── build_results_table.py      # Aggregate results across datasets
│   │   └── search_bounds.py            # Hyperparameter search for ρ, fr
│   └── generate_latex_table.py         # Generate LaTeX table for paper
│
└── src/
    ├── data_provider/
    │   ├── data_factory.py       # DataLoader factory
    │   └── data_loader.py        # Dataset_ETT_hour, Dataset_Custom, ...
    │
    ├── evaluation/
    │   ├── unified_evaluator.py  # Main evaluator (all 7 metrics)
    │   ├── validity_metrics.py
    │   ├── proximity_metrics.py
    │   ├── plausibility_metrics.py
    │   └── ...
    │
    ├── experiments/
    │   ├── ablations/
    │   │   ├── RLP/              # Random Latent Perturbation
    │   │   │   ├── run_rlp.py
    │   │   │   ├── plot_rlp.py
    │   │   │   └── results/
    │   │   └── wo_mask/          # Without temporal mask
    │   │       └── run_wo_mask.py
    │   ├── autoencoder/
    │   │   └── run.py            # Train TCN-AE
    │   ├── forecasting/
    │   │   └── run.py            # Train any forecaster
    │   └── rl_cf/
    │       └── run_last.py       # Train RL agent (main)
    │
    ├── models/
    │   ├── autoencoder/
    │   │   └── tcn_ae.py         # TCN Autoencoder
    │   ├── Forecaster/
    │   │   ├── iTransformer.py
    │   │   ├── PatchTST.py
    │   │   ├── TimesNet.py
    │   │   ├── GRU.py
    │   │   ├── DLinear.py
    │   │   ├── forecaster_wrapper.py    # iTransformer only
    │   │   └── forecaster_wrapper_v2.py # All models (used by pipelines)
    │   └── RL/
    │       ├── actor.py
    │       ├── critic.py
    │       ├── agent.py          # ActorCritic + build_state
    │       └── reward_last.py    # CFReward (validity + proximity)
    │
    ├── training/
    │   ├── ae_trainers/
    │   ├── forecast_trainers/
    │   │   ├── generic_forecaster_trainer.py  # GRU, DLinear, TimesNet, PatchTST
    │   │   └── itransformer_trainer.py
    │   └── RL_trainers/
    │       ├── trainer_last.py        # Main RL trainer
    │       └── trainer_wo_mask.py     # Ablation: without temporal mask
    │
    └── utils/
        ├── config.py             # load_config (JSON → SimpleNamespace)
        └── train_tools.py        # EarlyStopping, get_device, ...
```

---

## Datasets

| Dataset | Domain | L | H | Loader |
|---|---|---|---|---|
| ETTh1 | Energy (oil temperature) | 96 | 48 | `Dataset_ETT_hour` |
| ETTh2 | Energy (oil temperature) | 96 | 48 | `Dataset_ETT_hour` |
| Weather | Meteorology (T degC) | 96 | 48 | `Dataset_Custom` |

---

## Evaluation Metrics

| Metric | Description | ↑/↓ |
|---|---|---|
| Validity Ratio | Proportion of CF forecast steps within [α, β] | ↑ |
| Stepwise AUC | AUC of cumulative per-instance validity curve | ↑ |
| Proximity L2 | L2 distance between CF and original input | ↓ |
| Compactness | Proportion of unchanged timesteps (ε=1e-3) | ↑ |
| Roughness Ratio | Roughness(CF) / Roughness(original) | ≈1 |
| Temporal Consistency | Pearson correlation of first-order differences | ↑ |
| Plausibility | Ensemble anomaly score (IF + LOF + OC-SVM) | ↓ |

---

## Ablations

| Variant | Script | Description |
|---|---|---|
| RLP | `src/experiments/ablations/RLP/run_rlp.py` | Random action instead of learned policy |
| w/o Mask | `src/experiments/ablations/wo_mask/run_wo_mask.py` | No temporal masking |

---

## Baselines

| Method | Script | Description |
|---|---|---|
| ForecastCF | `baselines/ForecastCF/run_etth1_gru.py` | Wang et al., ICDM 2023 |
| BaseNN | `baselines/BaseNN/run_basenn_etth1_gru.py` | 1-nearest-neighbour from train set |
