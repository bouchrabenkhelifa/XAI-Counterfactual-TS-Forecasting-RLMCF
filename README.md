# Counterfactual Explanations for Time Series Forecasting via Reinforcement Learning

An **agnostic XAI framework** that generates **counterfactual explanations** for time series forecasting models using **reinforcement learning** in latent space.

---

## Motivation

Deep learning forecasting models achieve strong performance but remain black boxes. Existing XAI methods explain *why* a prediction was made, but not *how to change it*. Counterfactual explanations answer "what-if" questions and provide actionable insights.

Most counterfactual methods target classification. For time series forecasting, the main prior work (ForecastCF) relies on per-instance gradient optimization — computationally expensive and prone to temporally inconsistent perturbations.

We propose a **global RL policy** that generalizes across all instances: once trained, counterfactuals are generated in a **single forward pass**.

---

## Architecture

![Architecture](assets/figures/Architecture.png)

The pipeline has three **frozen** components and one **trainable** agent:

1. Input `X` is encoded into latent `z` via a pre-trained **TCN Autoencoder**
2. **Actor-Critic agent** perturbs `z → z_cf` in latent space
3. Decoder reconstructs `X_cf = ψ(z_cf)`
4. A **temporal mask** `m` localizes perturbations to the most recent timesteps: `X_cf = X + m ⊙ (X̃ − X)`
5. Frozen forecaster evaluates `Ŷ_cf = f(X_cf)`
6. Reward is computed from **validity** and **proximity**

---

## Key Contributions

**Forecast-anchored validity bounds** — bounds are anchored on the original forecast `ŷ`, not on the input median:
```
β = ŷ − ρ·σx,   α = β − fr·σx
```
This guarantees `ŷ ∉ [α, β]` by construction, ensuring a non-trivial objective.

**Latent-space optimization** — the agent operates in the AE latent space, implicitly constraining perturbations to the data manifold without an explicit plausibility term.

**Temporal masking** — a ramp mask enforces sparse, localized perturbations on the `k` most recent timesteps, improving compactness and temporal consistency.

**Model-agnostic** — compatible with any frozen PyTorch forecaster. Validated on five architectures: iTransformer, PatchTST, TimesNet, GRU, DLinear.

---

## Project Structure

```
counterfactual-forecasting-rl/
│
├── assets/
│   ├── checkpoints/          # Saved model weights
│   │   ├── etth1_chpts/      #   ae/, anomaly_detector/, forecaster/, RL/
│   │   ├── etth2_chpts/
│   │   └── weather_chpts/
│   ├── configs/              # JSON configs per dataset
│   │   ├── etth1_dataset/    #   ae/, anomaly_detector/, forecasters/, RL/, RL_ablations/
│   │   ├── etth2_dataset/
│   │   └── weather_dataset/
│   ├── datasets/             # ETTh1.csv, ETTh2.csv, weather.csv
│   ├── figures/              # Generated plots
│   └── results/              # Evaluation JSONs and summaries
│
├── baselines/
│   ├── ForecastCF/           # Original TF baseline (reference)
│   ├── ForecastCF_PyTorch/   # Our PyTorch re-implementation
│   ├── BaseGrad/             # Gradient-based baseline
│   ├── BaseNN/               # Neural network baseline
│   └── common/               # Shared evaluation utilities
│
├── scripts/
│   ├── evals/                # eval_etth1.py, eval_etth2.py, eval_weather.py
│   ├── pipelines/            # pipeline_etth1.py, pipeline_etth2.py, pipeline_weather.py
│   ├── analysis/             # CF visualization scripts
│   └── appendix/             # Sensitivity and ablation figures
│
└── src/
    ├── data_provider/        # Dataset loaders
    ├── evaluation/           # unified_evaluator.py
    ├── experiments/          # Entry points: rl_cf/, forecasting/, autoencoder/, ...
    ├── layers/               # Transformer layers
    ├── models/
    │   ├── Forecaster/       # iTransformer, PatchTST, TimesNet, GRU, DLinear
    │   ├── autoencoder/      # TCN-AE
    │   ├── RL/               # Actor-Critic agent, reward function
    │   └── anomaly_detector/
    ├── training/
    │   ├── RL_trainers/      # trainer_main.py (main), trainer_last.py, trainer_wo_mask.py
    │   ├── forecast_trainers/
    │   ├── ae_trainers/
    │   └── ad_trainers/
    └── utils/
```

---

## Quick Start

See [ENV_SETUP.md](ENV_SETUP.md) for full installation and usage instructions.

```bash
# 1. Install dependencies
pip install -r requirements.txt
export PYTHONPATH=.   # or: $env:PYTHONPATH = "." on Windows PowerShell

# 2. Run full pipeline (train + eval)
python scripts/pipelines/pipeline_etth1.py

# 3. Evaluate only (checkpoints already trained)
python scripts/evals/eval_etth1.py
```

---

## Datasets

| Dataset | Series | Train | Test | Freq |
|---------|--------|-------|------|------|
| ETTh1   | OT (oil temperature) | 8497 | 2833 | Hourly |
| ETTh2   | OT (oil temperature) | 8497 | 2833 | Hourly |
| Weather | T (°C) | 36696 | 10444 | 10-min |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity Ratio ↑ | Fraction of CF forecasts inside target band `[α, β]` |
| Stepwise AUC ↑ | Area under cumulative validity curve |
| Proximity L2 ↓ | Mean L2 distance between `X` and `X_cf` |
| Compactness ↑ | Fraction of unchanged input timesteps |
| Roughness Ratio ↓ | Smoothness of CF vs original |
| Temporal Consistency ↑ | Pearson correlation between `X` and `X_cf` |
| Plausibility ↓ | Ensemble anomaly score (IForest + LOF + OC-SVM) |

---

## References

- Wang et al., *Counterfactual Explanations for Time Series Forecasting*, ICDM 2023
- Li et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024
