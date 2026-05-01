# Quick Start Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/bouchrabenkhelifa/counterfactual-forecasting-rl
cd counterfactual-forecasting-rl
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support (CUDA 11.8 or 12.1), install PyTorch separately:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

## Step 4: Set Python Path

```bash
# Linux/macOS
export PYTHONPATH=.

# Windows (PowerShell)
$env:PYTHONPATH = "."

# Windows (CMD)
set PYTHONPATH=.
```

## Step 5: Run Commands

### Train a Single Forecaster

```bash
python src/experiments/forecasting/run.py --config assets/configs/models/{DATASET}_dataset/forecasters/{MODEL}/{DATASET}_96_48_S.json
```

**Replace**: `{DATASET}` with `etth1`, `etth2`, or `weather` — `{MODEL}` with `itransformer`, `gru`, `dlinear`, `patchtst`, or `timesnet`

### Train AutoEncoder

```bash
python src/experiments/ae/run.py --config assets/configs/models/{DATASET}_dataset/ae/tcn_ae.json
```

**Replace**: `{DATASET}` with `etth1`, `etth2`, or `weather`

### Train Single RL Agent

```bash
python src/experiments/rl_cf/run.py --config assets/configs/models/{DATASET}_dataset/RL/config_{MODEL}.json --ae_config assets/configs/models/{DATASET}_dataset/ae/tcn_ae.json
```

**Replace**: `{DATASET}` with `etth1`, `etth2`, or `weather` — `{MODEL}` with `itransformer`, `gru`, `dlinear`, `patchtst`, or `timesnet`

## Pipeline

The pipeline automates the full workflow: train forecasters → train AE → train RL agents → evaluate.

```bash
python scripts/pipelines/pipeline_{DATASET}.py
```

**Replace**: `{DATASET}` with `etth1`, `etth2`, or `weather`

**Options**:
- `--eval_only` - Skip training, only evaluate RL agents
- `--skip_forecasters` - Skip forecaster training
- `--skip_ae` - Skip AutoEncoder training
- `--skip_rl` - Skip RL agent training

## Results

- **Checkpoints**: `assets/checkpoints/{etth1,etth2,weather}_chpts/`
- **Figures**: `assets/figures/{etth1,etth2,weather}/`
- **Results**: `assets/results/{etth1,etth2,weather}/`

## Documentation

- `README.md` - Project overview
- `STRUCTURE.md` - Project structure details
