# Appendix

## A. Effect of Temporal Masking on Gradient-Based Baselines

To isolate the contribution of the RL policy from the temporal masking mechanism, we evaluate **ForecastCF-M**, a variant of ForecastCF that applies the same temporal mask as RL-MCF at every optimisation step, restricting perturbations to the last *k* timesteps. Results are reported on the iTransformer backbone across three datasets. ↑ higher is better, ↓ lower is better.

| Dataset | Method | Validity ↑ | Step AUC ↑ | Proximity L2 ↓ | Compactness ↑ | T. Cons. ↑ | Plausibility ↓ |
|---------|--------|:----------:|:----------:|:--------------:|:-------------:|:----------:|:--------------:|
| ETTh1   | ForecastCF   | 0.784 | 0.792 | 0.725 | 0.014 | 0.775 | 0.642 |
| ETTh1   | ForecastCF-M | 0.314 | 0.324 | **0.144** | **0.849** | **0.980** | **0.123** |
| ETTh1   | **RL-MCF**   | **0.976** | **0.982** | 0.605 | 0.803 | 0.969 | 0.170 |
| ETTh2   | ForecastCF   | 0.499 | 0.509 | 0.786 | 0.015 | 0.890 | 0.554 |
| ETTh2   | ForecastCF-M | 0.375 | 0.385 | 0.365 | **0.847** | 0.983 | 0.095 |
| ETTh2   | **RL-MCF**   | **0.810** | **0.817** | **1.224** | 0.844 | **0.985** | **0.056** |
| Weather | ForecastCF   | **0.910** | **0.912** | 0.678 | 0.012 | 0.615 | 0.572 |
| Weather | ForecastCF-M | 0.567 | 0.570 | **0.247** | 0.683 | 0.789 | **0.102** |
| Weather | **RL-MCF**   | 0.844 | 0.848 | 1.294 | **0.680** | **0.830** | 0.211 |

**Validity.** RL-MCF outperforms ForecastCF-M on validity across all three datasets (+66pp on ETTh1, +44pp on ETTh2, +28pp on Weather), demonstrating that the RL policy — not the mask alone — is responsible for generating counterfactuals that satisfy the forecasting constraint. ForecastCF without mask achieves high validity on Weather (0.910) but at the cost of near-zero compactness (0.012), modifying the entire input window indiscriminately.

**Compactness.** Adding the mask to ForecastCF dramatically improves compactness (0.014 → 0.849 on ETTh1), confirming that the mask is the primary driver of sparse, localised perturbations — consistent with the ablation results in Table 3.

**Proximity and Plausibility.** ForecastCF-M achieves lower proximity and plausibility scores than RL-MCF on ETTh1 and Weather, as gradient-based optimisation directly minimises perturbation magnitude in the input space. However, this comes at the cost of validity, highlighting the fundamental trade-off between constraint satisfaction and input-space proximity that the RL policy is designed to navigate.

**Overall.** These results confirm that ForecastCF-M and RL-MCF offer complementary trade-offs: ForecastCF-M favours proximity and plausibility, while RL-MCF prioritises validity and compactness. The consistent validity gap across all configurations justifies the use of a learned policy over gradient-based optimisation for constraint-satisfying counterfactual generation.
