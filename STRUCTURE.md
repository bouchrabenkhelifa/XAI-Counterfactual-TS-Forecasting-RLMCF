# Project Structure

## Overview

```
counterfactual-forecasting-rl/
├── assets/
│   ├── checkpoints/          # Model checkpoints
│   ├── configs/              # Configuration files
│   ├── datasets/             # Time series datasets
│   ├── figures/              # Generated visualizations
│   └── results/              # Training histories & metrics
├── src/                      # Source code
├── scripts/                  # Utility scripts & pipelines
├── baselines/                # Baseline implementations
└── README.md
```

---

## Detailed Structure

### `assets/checkpoints/`

Organized by dataset and component:

```
assets/checkpoints/
├── etth1_chpts/
│   ├── ae/                   # AutoEncoder checkpoint
│   │   └── ae_etth1.pt
│   ├── forecaster/           # Forecaster models
│   │   ├── chpt_etth1_96_48_S.pth
│   │   ├── chpt_etth1_96_96.pth
│   │   └── ...
│   └── RL/                   # RL agents by model
│       ├── dlinear/
│       ├── gru/
│       ├── itransformer/
│       ├── patchtst/
│       ├── timesnet/
│       └── wo_mask/
├── etth2_chpts/
│   ├── ae/
│   ├── forecaster/
│   └── RL/
├── traffic_chpts/            # (Optional)
└── weather_chpts/            # (Optional)
```

### `assets/configs/`

Configuration files for models and experiments:

```
assets/configs/
└── models/
    ├── etth1_dataset/
    │   ├── ae/
    │   │   └── tcn_ae.json
    │   ├── anomaly_detector/
    │   │   └── plausibility.json
    │   ├── forecasters/
    │   │   ├── dlinear/
    │   │   ├── gru/
    │   │   ├── itransformer/
    │   │   ├── patchtst/
    │   │   └── timesnet/
    │   └── RL_ablations/
    │       ├── config_dlinear.json
    │       ├── config_final.json
    │       └── ...
    ├── etth2_dataset/
    │   ├── ae/
    │   ├── forecasters/
    │   └── RL/
    └── weather_dataset/
        ├── forecasters/
        └── RL/
```

### `assets/figures/`

Visualizations organized by dataset and model:

```
assets/figures/
├── Architecture.png          # System architecture diagram
├── etth1/
│   ├── ae/
│   │   └── tcn_ae_loss.png
│   ├── forecaster/
│   │   ├── dlinear/
│   │   │   ├── dlinear_training_curves.png
│   │   │   ├── dlinear_forecast_examples.png
│   │   │   └── ...
│   │   ├── gru/
│   │   ├── itransformer/
│   │   ├── patchtst/
│   │   └── timesnet/
│   └── RL/
│       ├── dlinear/
│       ├── gru/
│       ├── itransformer/
│       ├── patchtst/
│       ├── timesnet/
│       └── wo_mask/
├── etth2/
│   ├── ae/
│   └── forecaster/
│       ├── dlinear/
│       ├── gru/
│       ├── itransformer/
│       ├── patchtst/
│       └── timesnet/
└── anomaly_detector/
    └── plausibility_sanity_ETTh1.png
```

### `assets/results/`

Training histories and evaluation metrics:

```
assets/results/
├── etth1/
│   ├── ae/
│   │   └── history_tcn_ae.json
│   ├── forecaster/
│   │   ├── history_etth1_96_48_S.json
│   │   ├── history_etth1_96_96.json
│   │   └── ...
│   ├── RL/
│   │   ├── dlinear/
│   │   ├── gru/
│   │   ├── itransformer/
│   │   ├── patchtst/
│   │   ├── timesnet/
│   │   └── wo_mask/
│   └── anomaly_detector/
├── etth2/
│   ├── ae/
│   ├── forecaster/
│   └── RL/
└── comparison/
    └── (Baseline comparison results)
```

### `src/`

Source code organized by functionality:

```
src/
├── data_provider/            # Data loading & preprocessing
│   ├── data_factory.py
│   └── data_loader.py
├── models/
│   ├── Forecaster/           # Forecasting models
│   │   ├── iTransformer.py
│   │   ├── GRU.py
│   │   ├── DLinear.py
│   │   ├── PatchTST.py
│   │   ├── TimesNet.py
│   │   └── forecaster_wrapper.py
│   ├── AE/                   # AutoEncoder
│   │   └── TCN_AE.py
│   ├── RL/                   # RL agents
│   │   ├── agent.py
│   │   └── environment.py
│   └── layers/               # Reusable layers
├── training/
│   ├── forecast_trainers/
│   │   ├── itransformer_trainer.py
│   │   ├── generic_forecaster_trainer.py
│   │   └── ae_trainer.py
│   └── rl_trainers/
├── experiments/
│   ├── forecasting/
│   │   └── run.py            # Train forecasters
│   ├── rl_cf/
│   │   └── run.py            # Train RL agents
│   └── comparison/
│       └── run_forecastcf.py
├── utils/
│   ├── config.py
│   ├── train_tools.py
│   └── metrics.py
└── evaluation/
    └── evaluator.py
```

### `scripts/`

Utility scripts and pipelines:

```
scripts/
├── pipelines/
│   ├── pipeline_etth1.py     # Full ETTh1 pipeline
│   ├── pipeline_etth2.py     # Full ETTh2 pipeline
│   └── pipeline_weather.py
├── evals/
│   ├── eval_forecasters.py
│   └── eval_etth2_all_models.py
└── generate_latex_table.py
```

---

## Configuration Files

Each config JSON contains:

```json
{
  "model_name": "iTransformer",
  "dataset_name": "ETTh2",
  
  "root_path": "assets/datasets",
  "data_path": "ETTh2.csv",
  
  "seq_len": 96,
  "label_len": 48,
  "pred_len": 48,
  
  "checkpoint_dir": "assets/checkpoints/etth2_chpts/forecaster",
  "checkpoint_name": "chpt_etth2_96_48_S.pth",
  "history_name": "history_etth2_96_48_S.json",
  "results_dir": "assets/results/etth2/forecaster",
  "figures_dir": "assets/figures/etth2/forecaster/itransformer",
  
  "plot_training_curves": true,
  "plot_forecast_examples": true
}
```

---

## Key Naming Conventions

### Checkpoints
- Format: `chpt_{dataset}_{seq}_{pred}_{model}_{variant}.pth`
- Example: `chpt_etth1_96_48_S.pth` (ETTh1, seq=96, pred=48, univariate)

### Results (History)
- Format: `history_{dataset}_{seq}_{pred}_{model}_{variant}.json`
- Example: `history_etth2_96_48_S.json`

### Figures
- Format: `{model}_{type}.png`
- Types: `training_curves`, `forecast_examples`, `error_distribution`
- Example: `itransformer_training_curves.png`

### RL Agents
- Format: `rl_cf_{variant}_{dataset}_agent_{type}.pt`
- Types: `best`, `final`
- Example: `rl_cf_v2_etth1_agent_best.pt`

---

## Dataset Organization

```
assets/datasets/
├── ETTh1.csv                 # 17,420 samples, 7 features
├── ETTh2.csv                 # 17,420 samples, 7 features
├── electricity.csv           # Traffic dataset
└── weather.csv               # Weather dataset
```

---

## Notes

- **Univariate (S)**: Single target variable (OT - Oil Temperature)
- **Multivariate (MS)**: All 7 features used
- **Variants**: `S` = univariate, `MS` = multivariate
- **Seq/Pred**: `96_48` = 96-step lookback, 48-step forecast
- **Models**: iTransformer, GRU, DLinear, PatchTST, TimesNet
