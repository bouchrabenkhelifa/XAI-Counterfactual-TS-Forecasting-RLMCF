# RL-MCF: Reinforcement Learning for Masked Counterfactual Forecasting

A reinforcement learning framework for generating counterfactual explanations in time series forecasting. An Actor-Critic agent learns a global policy in the latent space of a pre-trained autoencoder, enabling efficient counterfactual generation in a single forward pass.

---

## Architecture

```
x → TCN-AE Encoder → z → Actor-Critic Agent → z_cf → TCN-AE Decoder → x̃
                                                                          ↓
x_cf = x + mask ⊙ (x̃ − x)  →  Forecaster f  →  ŷ_cf  →  Reward
```

**Three frozen components + one trainable agent:**
- **TCN-AE** — encodes input into latent space, constrains perturbations to data manifold
- **Forecasting model** — any black-box forecaster (iTransformer, PatchTST, TimesNet, GRU, DLinear)
- **Temporal mask** — localizes perturbations to the most recent k timesteps
- **Actor-Critic agent** — learns to navigate the latent space toward valid counterfactuals

---

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

## Project Structure

```
├── assets/
│   ├── checkpoints/          # Trained model checkpoints
│   │   ├── etth1_chpts/      # AE, forecasters, RL agents for ETTh1
│   │   ├── etth2_chpts/      # ETTh2
│   │   └── weather_chpts/    # Weather
│   ├── configs/
│   │   └── models/
│   │       ├── etth1_dataset/    # Forecaster + AE + RL configs for ETTh1
│   │       ├── etth2_dataset/    # ETTh2
│   │       └── weather_dataset/  # Weather
│   ├── datasets/             # Raw CSV datasets (ETTh1.csv, ETTh2.csv, weather.csv)
│   ├── figures/              # Training curves, CF examples
│   └── results/              # Evaluation JSONs, summary tables
│
├── baselines/
│   ├── ForecastCF/           # ForecastCF baseline (Wang et al., ICDM 2023)
│   ├── BaseShift/            # BaseShift naive baseline
│   └── BaseNN/               # BaseNN nearest-neighbour baseline
│
├── scripts/
│   ├── pipelines/
│   │   ├── pipeline_etth1.py     # Full ETTh1 pipeline
│   │   ├── pipeline_etth2.py     # Full ETTh2 pipeline
│   │   └── pipeline_weather.py   # Full Weather pipeline
│   ├── evals/
│   │   ├── eval_etth1_all_models.py   # Eval 5 models on ETTh1
│   │   ├── eval_etth2_all_models.py   # Eval 5 models on ETTh2
│   │   ├── eval_weather_all_models.py # Eval 5 models on Weather
│   │   ├── eval_forecasters.py        # Forecaster quality (MSE/MAE + plots)
│   │   └── build_results_table.py     # Build comparison table
│   └── generate_latex_table.py        # Generate LaTeX table for paper
│
├── src/
│   ├── data_provider/        # Dataset loaders (ETT, Custom)
│   ├── evaluation/           # Metrics (validity, proximity, compactness, ...)
│   ├── experiments/
│   │   ├── ablations/
│   │   │   └── RLP/          # Random Latent Perturbation ablation
│   │   │   └── wo_mask/      # Without temporal mask ablation
│   │   ├── autoencoder/      # AE training
│   │   ├── forecasting/      # Forecaster training
│   │   └── rl_cf/            # RL counterfactual training
│   ├── models/
│   │   ├── autoencoder/      # TCN-AE
│   │   ├── Forecaster/       # iTransformer, PatchTST, TimesNet, GRU, DLinear
│   │   └── RL/               # Actor, Critic, Agent, Reward
│   ├── training/
│   │   ├── ae_trainers/
│   │   ├── forecast_trainers/
│   │   └── RL_trainers/      # trainer_last.py (main), trainer_wo_mask.py (ablation)
│   └── utils/
│
└── requirements.txt
```

---

## Datasets

| Dataset | Domain | L | H | Loader |
|---|---|---|---|---|
| ETTh1 | Energy (oil temperature) | 96 | 48 | `Dataset_ETT_hour` |
| ETTh2 | Energy (oil temperature) | 96 | 48 | `Dataset_ETT_hour` |
| Weather | Meteorology (temperature) | 96 | 48 | `Dataset_Custom` |

---

## Evaluation Metrics

| Metric | Description | ↑/↓ |
|---|---|---|
| Validity Ratio | Proportion of CF forecast steps within target bounds | ↑ |
| Stepwise AUC | AUC of cumulative per-instance validity curve | ↑ |
| Proximity L2 | L2 distance between CF and original input | ↓ |
| Compactness | Proportion of unchanged timesteps (ε=1e-3) | ↑ |
| Roughness Ratio | Roughness(CF) / Roughness(original) | ≈1 |
| Temporal Consistency | Pearson correlation of first-order differences | ↑ |
| Plausibility | Ensemble anomaly score (IF + LOF + OC-SVM) | ↓ |

---

## Ablations

| Variant | Description |
|---|---|
| **RLP** | Random Latent Perturbation — same pipeline, random action instead of learned policy |
| **w/o Mask** | Temporal masking disabled — full sequence modified |

```bash
# RLP ablation
python src/experiments/ablations/RLP/run_rlp.py

# Without mask ablation
python src/experiments/ablations/wo_mask/run_wo_mask.py
```

---

## Baselines

```bash
# ForecastCF (Wang et al., ICDM 2023) — GRU on ETTh1
python baselines/ForecastCF/run_etth1_gru.py

# BaseShift
python baselines/BaseShift/run_baseshift_etth1_gru.py

# BaseNN
python baselines/BaseNN/run_basenn_etth1_gru.py
```

---

## References

- Wang et al., *Counterfactual Explanations for Time Series Forecasting*, ICDM 2023
- Li et al., *Counterfactual Explanations for Time Series Data via Reinforcement Learning*, 2026
