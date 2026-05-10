# Environment Setup & Usage Guide

## 1. Clone the Repository

```bash
git clone https://github.com/bouchrabenkhelifa/counterfactual-forecasting-rl
cd counterfactual-forecasting-rl
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch separately before the other dependencies:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Set Python Path

```bash
# Linux / macOS
export PYTHONPATH=.

# Windows — PowerShell
$env:PYTHONPATH = "."

# Windows — CMD
set PYTHONPATH=.
```

## 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__, '| GPU:', torch.cuda.is_available())"
```

---

## Running the Pipeline

The pipeline automates the full workflow: train forecasters → train AE → train RL agents → evaluate.

```bash
python scripts/pipelines/pipeline_etth1.py    # ETTh1
python scripts/pipelines/pipeline_etth2.py    # ETTh2
python scripts/pipelines/pipeline_weather.py  # Weather
```

**Options:**

| Flag | Description |
|------|-------------|
| `--eval_only` | Skip all training, only evaluate |
| `--skip_forecasters` | Skip forecaster training |
| `--skip_ae` | Skip AutoEncoder training |
| `--skip_rl` | Skip RL agent training |

---

## Running Individual Steps

### Train a Forecaster

```bash
python src/experiments/forecasting/run.py \
  --config assets/configs/{DATASET}_dataset/forecasters/{MODEL}/{DATASET}_96_48_S.json
```

`{DATASET}`: `etth1`, `etth2`, `weather`  
`{MODEL}`: `itransformer`, `patchtst`, `timesnet`, `gru`, `dlinear`

### Train the AutoEncoder

```bash
python src/experiments/autoencoder/run.py \
  --config assets/configs/{DATASET}_dataset/ae/tcn_ae.json
```

### Train an RL Agent

```bash
python src/experiments/rl_cf/run.py \
  --config assets/configs/{DATASET}_dataset/RL/config_{MODEL}.json \
  --ae_config assets/configs/{DATASET}_dataset/ae/tcn_ae.json
```

### Evaluate (Colab metrics + local plausibility)

```bash
python scripts/evals/eval_etth1.py
python scripts/evals/eval_etth2.py
python scripts/evals/eval_weather.py
```

### Generate CF Visualizations

```bash
python scripts/analysis/plot_cf_itransformer_3datasets.py
```

---

## Output Locations

| Type | Path |
|------|------|
| Checkpoints | `assets/checkpoints/{dataset}_chpts/` |
| Eval results | `assets/results/{dataset}/` |
| Summary JSON + plots | `assets/results/{dataset}/summary/` |
| Figures | `assets/figures/{dataset}/` |
| CF visualizations | `assets/figures/analysis/` |
