# Counterfactual Explanations for Time Series Forecasting via Reinforcement Learning

This work proposes an **agnostic XAI framework** to generate **counterfactual explanations** for time series forecasting models using **reinforcement learning (RL)** in latent space.

> See [STRUCTURE.md](STRUCTURE.md) for the full project structure, dataset details, and usage guide.

---

## Motivation

Deep learning models, especially Transformers, have significantly improved time series forecasting performance in high-stakes domains such as healthcare, finance, and energy. However, their black-box nature limits interpretability, which is critical in real-world decision-making.

Existing XAI methods mainly focus on feature attribution — explaining *why* a prediction was made, but not *how to change it*. As a result, they lack actionability.

Counterfactual explanations address this by answering "what-if" questions, providing actionable insights. However, most existing work focuses on **classification tasks**, while counterfactual generation for **time series forecasting** remains largely underexplored. The main existing approach, ForecastCF, relies on instance-specific gradient-based optimization that is computationally expensive and may produce temporally inconsistent perturbations.

---

## Architecture

![Architecture](assets/figures/Architecture.png)

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

### 2. Forecast-Anchored Validity Bounds

Unlike ForecastCF which centers bounds on `median(x)`, we anchor bounds directly on the original forecast `ŷ`:

```
β = ŷ − ρ · σx,   α = β − fr · σx
```

This guarantees by construction that `ŷ ∉ [α, β]`, ensuring a non-trivial counterfactual objective.

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

- **iTransformer** (Transformer-based)
- **PatchTST** (Patch-based Transformer)
- **TimesNet** (CNN-based)
- **GRU** (Recurrent)
- **DLinear** (Linear decomposition)

---

## References

- Wang et al., *Counterfactual Explanations for Time Series Forecasting*, ICDM 2023
- Li et al., *Counterfactual Explanations for Time Series Data via Reinforcement Learning*, 2026
