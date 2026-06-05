# RL-MCF : Reinforcement Learning for Counterfactual Explanations in Time Series Forecasting

A **model-agnostic XAI framework** that generates **counterfactual explanations** for time series forecasting models using **reinforcement learning** in latent space. A single trained actor-critic policy produces counterfactuals in one forward pass (~2 ms), achieving ×3000 speedup over instance-specific optimization baselines.

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

- **Forecast-anchored validity bounds** — bounds anchored on the original forecast `ŷ`, guaranteeing `ŷ ∉ [α, β]` by construction (non-trivial objective).
- **Latent-space optimization** — perturbations constrained to the data manifold via the autoencoder, implicitly enforcing plausibility.
- **Temporal masking** — ramp mask enforces sparse, localized perturbations on the `k` most recent timesteps.
- **Model-agnostic** — validated on 5 architectures (iTransformer, PatchTST, TimesNet, GRU, DLinear) × 3 datasets.
- **Single forward-pass inference** — ×3000 faster than ForecastCF at inference.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validity Ratio ↑ | Fraction of CF forecasts inside target band `[α, β]` |
| Stepwise AUC ↑ | Area under cumulative validity curve |
| Proximity L2 ↓ | Mean L2 distance between `X` and `X_cf` |
| Compactness ↑ | Fraction of unchanged input timesteps |
| Temporal Consistency ↑ | Pearson correlation of first-order differences |
| Plausibility ↓ | Ensemble anomaly score (IForest + LOF + OC-SVM) |

---

## Quick Start

```bash
pip install -r requirements.txt
export PYTHONPATH=.   # Windows PowerShell: $env:PYTHONPATH = "."
```

---

## Commands

### Full pipeline (train all components + evaluate + generate figures)

Trains forecasters, autoencoder, RL agents, then evaluates and generates all plots for one dataset.

```bash
python scripts/pipelines/pipeline_etth1.py
python scripts/pipelines/pipeline_etth2.py
python scripts/pipelines/pipeline_weather.py
```

Options: `--eval_only`, `--skip_forecasters`, `--skip_ae`, `--skip_rl`

### Full evaluation (tables + all figures)

Generates the 3 comparison tables (baselines, architectures, ablation) and all associated figures in one command. No training required — uses existing checkpoints and result JSONs.

```bash
python run_full.py
```

### Per-dataset evaluation

Evaluates 5 RL agents on one dataset, recalculates plausibility, and generates per-dataset radar + barplot.

```bash
python scripts/evals/eval_etth1.py
python scripts/evals/eval_etth2.py
python scripts/evals/eval_weather.py
```

### Train individual components

```bash
# Train a forecaster
python src/experiments/forecasting/run.py --config assets/configs/{dataset}_dataset/forecasters/{model}/{config}.json

# Train the autoencoder
python src/experiments/autoencoder/run.py --config assets/configs/{dataset}_dataset/ae/tcn_ae.json

# Train an RL agent
python src/experiments/rl_cf/run.py --config assets/configs/{dataset}_dataset/RL/config_{model}.json --ae_config assets/configs/{dataset}_dataset/ae/tcn_ae.json
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[RESULTS_AND_DISCUSSION.md](RESULTS_AND_DISCUSSION.md)** | Full results, discussion, ablation study, and future directions |
| **[STRUCTURE.md](STRUCTURE.md)** | Detailed project structure and naming conventions |

---

## Datasets

| Dataset | Target | Train | Test | Freq |
|---------|--------|-------|------|------|
| ETTh1 | OT (oil temperature) | 8497 | 2833 | Hourly |
| ETTh2 | OT (oil temperature) | 8497 | 2833 | Hourly |
| Weather | T (°C) | 36696 | 10444 | 10-min |

---

## References

- Wang et al., *Counterfactual Explanations for Time Series Forecasting*, ICDM 2023
- Li et al., *Counterfactual Explanations for Time Series Data via Reinforcement Learning*, ICLR 2026 (under review)
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024

---

## License

This project is for academic research purposes.
