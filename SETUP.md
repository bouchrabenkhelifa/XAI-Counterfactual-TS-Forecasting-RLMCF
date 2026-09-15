# Experimental Setup and Hyperparameters

This document details the key hyperparameters and experimental configuration used in the RL-MCF framework.

---

## 1. Time Series Configuration

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Input sequence length** | `L` | 96 | Number of historical timesteps used as input |
| **Forecast horizon** | `H` | 48 | Number of future timesteps to predict |
| **Univariate** | - | Yes | Single target variable per dataset |

**Datasets:**
- **ETTh1**: Oil temperature (hourly), Train: 8497, Test: 2833
- **ETTh2**: Oil temperature (hourly), Train: 8497, Test: 2833
- **Weather**: Temperature °C (10-min), Train: 36696, Test: 10444

---

## 2. Autoencoder Architecture

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Latent dimension** | `d` | 32 | Dimensionality of latent space `z` |
| **Encoder layers** | - | 3 TCN blocks | Temporal Convolutional Network |
| **Decoder layers** | - | 3 TCN blocks | Symmetric to encoder |
| **Kernel size** | - | 3 | Convolutional kernel size |
| **Dilation** | - | [1, 2, 4] | Exponential dilation per layer |
| **Activation** | - | ReLU | Non-linearity |

**Training:**
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 64
- Epochs: 50
- Loss: MSE reconstruction loss

---

## 3. RL Agent Configuration

### Actor-Critic Architecture

| Component | Architecture | Output |
|-----------|-------------|--------|
| **Actor** | MLP [256, 128] → Tanh | Action `a ∈ [-1, 1]^d` |
| **Critic** | MLP [256, 128] → Linear | Value estimate `V(s)` |
| **State** | `s = [z, ŷ]` | Latent + forecast (dim: 32 + 48 = 80) |

### Key Hyperparameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Perturbation scale** | `η` | 0.1 | Scaling factor for latent perturbations |
| **Clip range** | - | [-3σ, +3σ] | Latent space clipping (3 standard deviations) |
| **Learning rate (actor)** | `lr_π` | 1e-4 | Actor policy learning rate |
| **Learning rate (critic)** | `lr_V` | 1e-3 | Critic value learning rate |
| **Discount factor** | `γ` | 0.99 | Future reward discount (not used, single-step) |
| **Batch size** | - | 32 | Number of samples per update |
| **Training epochs** | - | 30 | Number of training epochs |
| **Entropy coefficient** | `β` | 0.01 | Entropy regularization weight |

### Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| **Validity reward** | `w_v = 10.0` | Reward for forecast entering target band |
| **Proximity penalty** | `w_p = 0.1` | Penalty for large perturbations (L2 norm) |

**Reward formula:**
```
r = w_v * r_validity + w_p * r_proximity
r_validity = (1 / H) * Σ_t 1[α_t ≤ ŷ_t^cf ≤ β_t]
r_proximity = -||x - x_cf||_2
```

---

## 4. Temporal Masking Mechanism

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Active window** | `k` | 24 | Number of recent timesteps fully modifiable (m=1) |
| **Ramp window** | `r` | 12 | Number of preceding timesteps with linear ramp |
| **Frozen window** | - | 60 | Early timesteps unchanged (m=0) |

**Total:** `k + r = 36` modifiable timesteps out of `L = 96`

**Mask function:**
```
m_t = 0           if t < L - k - r
m_t = (t - (L-k-r)) / r   if L - k - r ≤ t < L - k
m_t = 1           if t ≥ L - k
```

---

## 5. Validity Bounds Configuration

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| **Reduction ratio** | `ρ` | 0.2 | Target reduction: `|ŷ_t^cf - ŷ_t| ≥ ρ * |ŷ_t|` |
| **Fraction ratio** | `f_r` | 0.5 | Fraction of horizon where constraint applies |

**Target band construction:**
```
α_t = ŷ_t - ρ * |ŷ_t|
β_t = ŷ_t - 2 * ρ * |ŷ_t|
```

**Validity constraint:** `ŷ_t^cf ∈ [β_t, α_t]` for at least `f_r * H` timesteps.

---

## 6. Plausibility Evaluation

**Ensemble anomaly detector:**
- **Isolation Forest**: 100 estimators, contamination=0.1
- **Local Outlier Factor (LOF)**: k=20 neighbors
- **One-Class SVM**: RBF kernel, ν=0.1

**Aggregation:** Min-max normalized scores averaged across 3 detectors.

**Training:** Detectors trained on original training distribution, applied to counterfactuals.

---

## 7. Forecaster Architectures

All forecasters use the same configuration:
- Input length: 96
- Prediction length: 48
- Training: 10 epochs, batch size 32, learning rate 1e-3

| Model | Key Parameters |
|-------|----------------|
| **iTransformer** | 4 layers, 8 heads, d_model=512 |
| **PatchTST** | Patch length=16, stride=8, 4 layers, 8 heads |
| **TimesNet** | 4 layers, top-k frequencies=5 |
| **GRU** | 2 layers, hidden size=128 |
| **DLinear** | Decomposition: moving average (kernel=25) |

---

## 8. Hardware and Timing

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Core i7-12700K (12 cores, 3.6 GHz) |
| **GPU** | NVIDIA RTX 3090 (24 GB VRAM) |
| **RAM** | 64 GB DDR4 |
| **Framework** | PyTorch 2.0.1, CUDA 11.8 |

**Inference timing methodology:**
- Measured on GPU (CUDA)
- Average over 2833 test samples (ETTh1)
- Includes: latent encoding → policy forward → decoding → mask application
- Excludes: forecaster inference (frozen, amortized across baselines)
- Reported: mean inference time per sample in milliseconds

**RL-MCF inference time:** ~2.0 ms per sample (single forward pass)

---

## 9. Baseline Configurations

All baselines use **identical validity bounds** (ρ=0.2, f_r=0.5) for fair comparison.

| Baseline | Key Settings |
|----------|-------------|
| **ForecastCF** | Adam optimizer, lr=0.01, 1000 iterations per sample |
| **BaseGrad (SPSA)** | Perturbation δ=0.01, 500 iterations per sample |
| **BaseNN** | 1-Nearest-Neighbor retrieval from training set |

---

## 10. Ablation Study Configurations

| Variant | Modification |
|---------|-------------|
| **w/o Proximity** | Set `w_p = 0` (no proximity penalty) |
| **w/o Mask** | Set `m_t = 1` for all timesteps (no temporal masking) |
| **w/o AutoEncoder** | Replace TCN-AE with identity mapping (policy operates in input space) |
| **RLP (Random Latent Perturbation)** | Replace actor policy with Gaussian noise `N(0, I)` |
| **ForecastCF-Masked** | Apply same temporal mask to ForecastCF baseline |

---

## 11. Statistical Significance

- **Metrics:** Mean ± Std over 2833 test samples (ETTh1), 2833 (ETTh2), 10444 (Weather)
- **Seed variance:** Single seed per configuration (seed=0)
- **Multi-seed experiments:** 5 seeds (0-4) reported in supplementary material for reproducibility analysis

---

## References

All hyperparameters are chosen based on:
1. Grid search over latent dimension `d ∈ {16, 32, 64}` → best: 32
2. Grid search over perturbation scale `η ∈ {0.05, 0.1, 0.2}` → best: 0.1
3. Grid search over mask window `k ∈ {12, 24, 36}` → best: 24
4. Reward weights tuned to balance validity (primary) and proximity (secondary)

Final hyperparameters are fixed across all datasets and architectures to ensure fairness.
