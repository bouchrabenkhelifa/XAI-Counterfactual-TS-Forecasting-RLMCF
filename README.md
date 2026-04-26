# Counterfactual Explanations for Time Series Forecasting via Reinforcement Learning

This work proposes an **agnostic XAI framework** to generate **counterfactual explanations** for time series forecasting models using **reinforcement learning (RL)** in latent space.

---

## Motivation

Deep learning models, especially Transformers, have significantly improved time series forecasting performance in high-stakes domains such as healthcare, finance, and energy. However, their black-box nature limits interpretability, which is critical in real-world decision-making.

Existing XAI methods mainly focus on feature attribution — explaining *why* a prediction was made, but not *how to change it*. As a result, they lack actionability.

Counterfactual explanations address this by answering "what-if" questions, providing actionable insights. However, most existing work focuses on **classification tasks**, while counterfactual generation for **time series forecasting** remains largely underexplored. The main existing approach, ForecastCF, relies on instance-specific gradient-based optimization that is computationally expensive and may produce temporally inconsistent perturbations.

---

## Architecture

![Architecture](./figures/Architecture.png)

The pipeline consists of three **frozen** components and one **trainable** agent:

1. Input time series `X` is encoded into latent representation `z` via a pre-trained **Temporal Convolutional Autoencoder (TCN-AE)**
2. **Actor-Critic RL agent** perturbs `z → z_cf` in latent space
3. Decoder reconstructs counterfactual time series `X_cf = ψ(z_cf)`
4. A **temporal mask** `m` is applied to localize perturbations to the most recent timesteps: `X_cf = X + m ⊙ (X̃ − X)`
5. Forecasting model evaluates `Ŷ_cf = f(X_cf)`
6. Reward is computed based on **validity** and **proximity**

---

## Contributions

### 1. RL Framework for Counterfactual Forecasting

We introduce the first reinforcement learning framework for counterfactual explanations in time series forecasting. An Actor-Critic agent learns a **global policy** that generalizes across all instances — once trained, counterfactuals are generated in a **single forward pass**, unlike per-instance gradient optimization.

### 2. Range-Based Validity Objective

We adopt the same range-based objective as ForecastCF, requiring forecasted values to lie within user-defined bounds `[αt, βt]` derived from the look-back window:

```
β = sv · (1 + fr · σx),   α = sv · (1 − fr · σx)
```

where `sv = median(x)` and `σx = std(x)`.

### 3. Latent-Space Optimization

Instead of modifying raw time series directly, the agent operates in the **latent space** of a pre-trained autoencoder:

```
a ~ π_θ(s),   z_cf = clip(z + η · a, −1, 1)
```

This implicitly constrains perturbations to the data manifold, enforcing **plausibility** without an explicit plausibility term in the reward.

### 4. Temporal Masking Mechanism

A ramp-based temporal mask encourages **sparse and localized** perturbations, modifying only the `k` most recent timesteps:

```
mt = 0               if t < L − k − r
     (t−(L−k−r))/r  if L − k − r ≤ t < L − k
     1               if t ≥ L − k
```

This directly improves compactness and temporal consistency.

### 5. Reward Function

The reward combines validity and proximity:

```
R = wv · Validity + wp · Proximity
```

- **Validity**: `r_valid = (1/H) Σ exp(−2 · dt / (βt − αt))` — penalizes forecasts outside target bounds
- **Proximity**: `r_prox = exp(−(1/L) Σ |x_cf_t − x_t|)` — penalizes large input perturbations

### 6. Model-Agnostic Design

The framework is compatible with any black-box forecasting architecture. We validate across four representative model families:

- **Itransformer** (Transformer-based)
- **TimesNet** (CNN-based)
- **GRU** (Recurrent)
- **DLinear** (Linear decomposition)

---

## Evaluation Metrics

We evaluate counterfactual quality along five axes:

| Metric | Description | Direction |
|---|---|---|
| **Validity Ratio** | Proportion of predicted steps within target bounds | ↑ |
| **Stepwise Validity AUC** | AUC of cumulative per-instance validity curve | ↑ |
| **Proximity (ℓ2)** | Distance between counterfactual and original | ↓ |
| **Compactness** | Proportion of timesteps with perturbation below threshold ε | ↑ |
| **Roughness Ratio** | Ratio of temporal roughness: `ρ = Roughness(X_cf) / Roughness(X)` (best ≈ 1) | ↓ |
| **Temporal Consistency** | Pearson correlation of first-order differences, mapped to [0,1] | ↑ |
| **Plausibility** | Ensemble anomaly score (Isolation Forest + LOF + OC-SVM) | ↓ |

---

## Experiments

### Datasets

| Dataset | Domain | Look-back (L) | Horizon (H) |
|---|---|---|---|
| ETTh1 | Energy (oil temperature) | 96 | 24 |
| M4 Finance | Finance | 48 | 12 |
| Weather | Meteorology | 96 | 24 |
| MIMIC-III | Healthcare (ICU heart rate) | 48 | 12 |

### Results (Summary)

Our method consistently outperforms ForecastCF and the RLP baseline across all datasets and forecasting architectures:

- **Higher validity** and stepwise AUC
- **Lower proximity** — more minimal perturbations
- **Higher compactness** — more localized changes (driven by temporal masking)
- **Roughness ratio closer to 1** — smoother, more temporally coherent counterfactuals
- **Higher temporal consistency**
- **Lower plausibility score** — counterfactuals closer to the training distribution

The ablation (Ours w/o Mask) confirms that the temporal masking mechanism is responsible for significant gains in compactness and temporal quality.

---

## References

- Wang et al., *Counterfactual Explanations for Time Series Forecasting*, ICDM 2023 (ForecastCF)
- Li et al., *Counterfactual Explanations for Time Series Data via Reinforcement Learning*, 2026