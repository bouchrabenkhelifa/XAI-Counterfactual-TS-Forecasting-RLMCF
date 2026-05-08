# Requirements Document: Baseline Extension for ETTh2 Dataset

## 1. Functional Requirements

### 1.1 BaseNN Extension

**1.1.1** The system SHALL support BaseNN counterfactual generation for ETTh2 dataset with iTransformer model

**1.1.2** The system SHALL support BaseNN counterfactual generation for ETTh2 dataset with GRU model

**1.1.3** BaseNN SHALL use the same algorithm as ETTh1 implementation:
- Store training set forecasts during fit()
- Compute target = (alpha + beta) / 2 for each test sample
- Find nearest neighbor in training forecasts using L2 distance
- Return corresponding training input as counterfactual

**1.1.4** BaseNN SHALL use the same hyperparameters as RLCF:
- back_horizon = 96
- horizon = 48
- desired_change = -0.1 (via RL-MCF bounds)

### 1.2 BaseGrad Extension

**1.2.1** The system SHALL support BaseGrad counterfactual generation for ETTh2 dataset with iTransformer model

**1.2.2** The system SHALL support BaseGrad counterfactual generation for ETTh2 dataset with GRU model

**1.2.3** BaseGrad SHALL extend ETTh1/GRU support (currently missing)

**1.2.4** BaseGrad SHALL use gradient descent optimization with:
- Learning rate = 0.01
- Max iterations = 300
- Validity weight = 1.0
- Proximity weight = 0.5
- SPSA gradient approximation for non-differentiable models

**1.2.5** BaseGrad SHALL clip counterfactuals to [x_min - 3σ, x_max + 3σ]

### 1.3 ForecastCF Extension

**1.3.1** The system SHALL support ForecastCF counterfactual generation for ETTh2 dataset with iTransformer model

**1.3.2** The system SHALL support ForecastCF counterfactual generation for ETTh2 dataset with GRU model

**1.3.3** ForecastCF SHALL use the PyTorch adapter for model compatibility

**1.3.4** ForecastCF SHALL use masked gradient optimization:
- Max iterations = 300
- Adam optimizer with learning rate = 0.01
- Prediction margin weight = 0.5
- Temporal masking for violating timesteps only

### 1.4 Bounds Calculation

**1.4.1** All baseline methods SHALL use RL-MCF bounds methodology:
- β = ŷ - ρ · σ_global
- α = β - fr · σ_global

**1.4.2** Bounds parameters SHALL be loaded from RL config files:
- `assets/configs/models/etth2_dataset/RL/config_gru.json`
- `assets/configs/models/etth2_dataset/RL/config_itransformer.json`

**1.4.3** The system SHALL verify that ŷ ∉ [α, β] for all samples

### 1.5 Data Loading

**1.5.1** The system SHALL load ETTh2 dataset from `assets/datasets/ETTh2.csv`

**1.5.2** The system SHALL use 80/20 train/test split

**1.5.3** The system SHALL normalize data using training set statistics

**1.5.4** The system SHALL create sliding window sequences with:
- seq_len = 96
- pred_len = 48
- univariate mode (OT channel only)

### 1.6 Model Loading

**1.6.1** The system SHALL load forecaster checkpoints from:
- `assets/checkpoints/etth2_chpts/forecaster/chpt_etth2_96_48_gru_S.pth`
- `assets/checkpoints/etth2_chpts/forecaster/chpt_etth2_96_48_S.pth` (iTransformer)

**1.6.2** The system SHALL load autoencoder checkpoint from:
- `assets/checkpoints/etth2_chpts/ae/ae_etth2.pt`

**1.6.3** The system SHALL set forecaster models to eval mode with frozen parameters

### 1.7 Evaluation Metrics

**1.7.1** The system SHALL compute the following metrics for all baseline methods:
- Validity Ratio: Fraction of samples where all forecast timesteps fall within [α, β]
- Stepwise AUC: Area under cumulative validity curve across forecast horizon
- Proximity L2: L2 distance between original and counterfactual inputs
- Compactness: Fraction of timesteps modified
- Temporal Consistency: Smoothness of perturbations (gradient variance)
- Plausibility: Distance to nearest training sample

**1.7.2** The system SHALL run experiments with 3 random seeds: [1, 9, 30]

**1.7.3** The system SHALL aggregate metrics across seeds using mean and standard deviation

### 1.8 Result Output

**1.8.1** The system SHALL save results in JSON format with structure:
```json
{
  "method": "string",
  "dataset": "string",
  "model": "string",
  "n_samples": "integer",
  "runtime_seconds": "float",
  "metrics": {
    "validity_ratio": {"mean": "float", "std": "float"},
    "stepwise_auc": {"mean": "float", "std": "float"},
    "proximity_l2": {"mean": "float", "std": "float"},
    "compactness": {"mean": "float", "std": "float"},
    "temporal_consistency": {"mean": "float", "std": "float"}
  },
  "seeds": ["array of integers"],
  "bounds_params": {"rho": "float", "fr": "float", "direction": "string"}
}
```

**1.8.2** The system SHALL save results to:
- `baselines/BaseNN/results/basenn_etth2_gru.json`
- `baselines/BaseNN/results/basenn_etth2_itransformer.json`
- `baselines/BaseGrad/results/basegrad_etth2_gru.json`
- `baselines/BaseGrad/results/basegrad_etth2_itransformer.json`
- `baselines/ForecastCF/results/forecastcf_etth2_gru.csv`
- `baselines/ForecastCF/results/forecastcf_etth2_itransformer.csv`

**1.8.3** The system SHALL generate CSV results compatible with LaTeX table generation

**1.8.4** The system SHALL generate visualization figures for sample counterfactuals

### 1.9 Runner Scripts

**1.9.1** The system SHALL provide runner scripts for ETTh2 experiments:
- `baselines/BaseNN/run_basenn_etth2.py`
- `baselines/BaseGrad/run_basegrad_etth2.py`
- `baselines/ForecastCF/run_etth2_gru.py`
- `baselines/ForecastCF/run_etth2_itransformer.py`

**1.9.2** Runner scripts SHALL accept command-line arguments:
- `--dataset`: Dataset name (etth1, etth2, weather)
- `--model`: Model name (gru, itransformer, patchtst, dlinear, timesnet)
- `--seeds`: List of random seeds (default: [1, 9, 30])
- `--device`: Compute device (cpu, cuda)
- `--n_batches`: Number of test batches to process (default: 20)
- `--output_dir`: Output directory for results

**1.9.3** Runner scripts SHALL implement caching to skip re-computation if results exist

## 2. Non-Functional Requirements

### 2.1 Performance

**2.1.1** BaseNN SHALL generate counterfactuals in < 20ms per sample

**2.1.2** BaseGrad SHALL generate counterfactuals in < 1000ms per sample

**2.1.3** ForecastCF SHALL generate counterfactuals in < 1500ms per sample

**2.1.4** Total experiment time for one method/dataset/model combination SHALL be < 10 minutes

**2.1.5** Memory usage SHALL not exceed 2GB per experiment

### 2.2 Reliability

**2.2.1** The system SHALL produce reproducible results for fixed random seeds

**2.2.2** The system SHALL handle missing checkpoints gracefully with clear error messages

**2.2.3** The system SHALL handle CUDA out-of-memory errors by falling back to CPU

**2.2.4** The system SHALL validate bounds to ensure α < β for all samples

**2.2.5** The system SHALL skip invalid samples and log warnings

### 2.3 Maintainability

**2.3.1** Runner scripts SHALL follow the same structure as existing ETTh1 scripts

**2.3.2** Code SHALL reuse existing components from `baselines/common/`

**2.3.3** Configuration files SHALL follow the same format as ETTh1 configs

**2.3.4** Results SHALL be saved in the same format as RLCF results for consistency

### 2.4 Usability

**2.4.1** Runner scripts SHALL provide clear progress messages during execution

**2.4.2** Runner scripts SHALL display final metrics summary after completion

**2.4.3** Error messages SHALL include file paths and suggested recovery actions

**2.4.4** Visualizations SHALL be saved with descriptive filenames including method, dataset, model

### 2.5 Compatibility

**2.5.1** The system SHALL be compatible with Python 3.11+

**2.5.2** The system SHALL be compatible with PyTorch 2.0+

**2.5.3** The system SHALL be compatible with TensorFlow 2.13+ (for ForecastCF optimizer)

**2.5.4** The system SHALL work on both CPU and CUDA devices

## 3. Data Requirements

### 3.1 Input Data

**3.1.1** ETTh2 dataset SHALL contain 17,420 samples with 7 features

**3.1.2** Test set SHALL contain approximately 3,484 samples (20% of total)

**3.1.3** Each sample SHALL have:
- Input sequence: 96 timesteps × 1 feature (OT channel)
- Forecast: 48 timesteps × 1 feature (OT channel)

### 3.2 Model Checkpoints

**3.2.1** iTransformer checkpoint SHALL be trained on ETTh2 with seq_len=96, pred_len=48

**3.2.2** GRU checkpoint SHALL be trained on ETTh2 with seq_len=96, pred_len=48

**3.2.3** Autoencoder checkpoint SHALL be trained on ETTh2 for latent space representation

### 3.3 Configuration Files

**3.3.1** Forecaster configs SHALL specify:
- model_name, dataset_name
- seq_len, pred_len
- checkpoint_dir, checkpoint_name
- Model-specific hyperparameters

**3.3.2** RL configs SHALL specify bounds parameters:
- rho: Offset from forecast (typically 0.5)
- fr: Bound width factor (typically 1.0)
- direction: "decrease" or "increase"
- global_sigma: Global standard deviation of training data

## 4. Interface Requirements

### 4.1 Command-Line Interface

**4.1.1** Runner scripts SHALL be executable from project root directory

**4.1.2** Runner scripts SHALL accept arguments via argparse

**4.1.3** Runner scripts SHALL return exit code 0 on success, non-zero on failure

### 4.2 File System Interface

**4.2.1** The system SHALL read from:
- `assets/datasets/ETTh2.csv`
- `assets/checkpoints/etth2_chpts/`
- `assets/configs/models/etth2_dataset/`

**4.2.2** The system SHALL write to:
- `baselines/{Method}/results/`
- `baselines/{Method}/figures/`

**4.2.3** The system SHALL create output directories if they do not exist

### 4.3 Python API

**4.3.1** Baseline methods SHALL implement consistent interfaces:
- `fit(X_train, Y_hat_train)` for BaseNN
- `transform(X_test, alphas, betas, ...)` for all methods

**4.3.2** Evaluator SHALL accept numpy arrays and return dict of metrics

**4.3.3** Result I/O SHALL accept dict and save to JSON/CSV

## 5. Constraint Requirements

### 5.1 Technical Constraints

**5.1.1** The system SHALL use existing forecaster checkpoints (no retraining)

**5.1.2** The system SHALL use existing autoencoder checkpoints (no retraining)

**5.1.3** The system SHALL use RL-MCF bounds (no alternative bound methods)

**5.1.4** The system SHALL process test samples in batches to fit in memory

### 5.2 Time Constraints

**5.2.1** Implementation SHALL be completed within 2 days

**5.2.2** All experiments SHALL be completed within 2 hours total runtime

**5.2.3** Results SHALL be ready for paper submission deadline

### 5.3 Resource Constraints

**5.3.1** The system SHALL run on a single GPU or CPU

**5.3.2** The system SHALL not require more than 16GB RAM

**5.3.3** The system SHALL not require more than 10GB disk space for results

## 6. Quality Requirements

### 6.1 Correctness

**6.1.1** Bounds SHALL satisfy α < β for all samples

**6.1.2** Forecasts SHALL be excluded from bounds: ŷ ∉ [α, β]

**6.1.3** Metrics SHALL be in valid ranges:
- validity_ratio ∈ [0, 1]
- stepwise_auc ∈ [0, 1]
- proximity_l2 ≥ 0
- compactness ∈ [0, 1]
- temporal_consistency ∈ [0, 1]

### 6.2 Consistency

**6.2.1** ETTh2 results SHALL be comparable to ETTh1 results (within 0.2 for validity_ratio)

**6.2.2** Same hyperparameters SHALL be used across all datasets and models

**6.2.3** Same evaluation metrics SHALL be used for all baseline methods

### 6.3 Completeness

**6.3.1** All 3 baseline methods SHALL be tested on ETTh2

**6.3.2** Both iTransformer and GRU models SHALL be tested

**6.3.3** All 6 evaluation metrics SHALL be computed for each experiment

**6.3.4** Results SHALL be saved in both JSON and CSV formats

## 7. Acceptance Criteria

### 7.1 Functional Acceptance

- [ ] BaseNN runs successfully on ETTh2/iTransformer with seeds [1, 9, 30]
- [ ] BaseNN runs successfully on ETTh2/GRU with seeds [1, 9, 30]
- [ ] BaseGrad runs successfully on ETTh2/iTransformer with seeds [1, 9, 30]
- [ ] BaseGrad runs successfully on ETTh2/GRU with seeds [1, 9, 30]
- [ ] ForecastCF runs successfully on ETTh2/iTransformer with seeds [1, 9, 30]
- [ ] ForecastCF runs successfully on ETTh2/GRU with seeds [1, 9, 30]

### 7.2 Quality Acceptance

- [ ] Validity ratio > 0.5 for all methods (indicates bounds are achievable)
- [ ] Proximity L2 < 5.0 for all methods (indicates reasonable perturbations)
- [ ] Standard deviation < 0.1 across seeds for all metrics (indicates stability)
- [ ] Runtime < 10 minutes per method/dataset/model combination

### 7.3 Output Acceptance

- [ ] JSON results saved for BaseNN (2 files: gru, itransformer)
- [ ] JSON results saved for BaseGrad (2 files: gru, itransformer)
- [ ] CSV results saved for ForecastCF (2 files: gru, itransformer)
- [ ] Visualization figures generated for all methods
- [ ] Comparison CSV generated for LaTeX table

### 7.4 Documentation Acceptance

- [ ] README updated with ETTh2 experiment instructions
- [ ] Results table generated for paper
- [ ] Comparison with RLCF results documented

## 8. Traceability Matrix

| Requirement ID | Design Component | Implementation File |
|---------------|------------------|---------------------|
| 1.1.1, 1.1.2 | BaseNN Extension | `baselines/BaseNN/run_basenn_etth2.py` |
| 1.2.1, 1.2.2 | BaseGrad Extension | `baselines/BaseGrad/run_basegrad_etth2.py` |
| 1.3.1, 1.3.2 | ForecastCF Extension | `baselines/ForecastCF/run_etth2_*.py` |
| 1.4.1, 1.4.2 | Bounds Calculator | `baselines/common/bounds.py` |
| 1.5.1-1.5.4 | Data Pipeline | `src/training/RL_trainers/trainer_last.py` |
| 1.6.1-1.6.3 | Model Loading | `src/models/Forecaster/forecaster_wrapper_v2.py` |
| 1.7.1-1.7.3 | Evaluator | `baselines/common/evaluator_wrapper.py` |
| 1.8.1-1.8.4 | Result I/O | `baselines/common/result_io.py` |
| 2.1.1-2.1.5 | Performance | All runner scripts |
| 2.2.1-2.2.5 | Reliability | Error handling in all components |
| 2.3.1-2.3.4 | Maintainability | Code structure and reuse |
| 2.4.1-2.4.4 | Usability | Logging and visualization |
