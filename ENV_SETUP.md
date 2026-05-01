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

### Train Forecasters

| Dataset | Command |
|---------|---------|
| ETTh1 | `python src/experiments/forecasting/run.py --config assets/configs/models/etth1_dataset/forecasters/{MODEL}/etth1_96_48_S.json` |
| ETTh2 | `python src/experiments/forecasting/run.py --config assets/configs/models/etth2_dataset/forecasters/{MODEL}/etth2_96_48_S.json` |
| Weather | `python src/experiments/forecasting/run.py --config assets/configs/models/weather_dataset/forecasters/{MODEL}/weather_96_48_S.json` |

**Models**: Replace `{MODEL}` with `itransformer`, `gru`, `dlinear`, `patchtst`, or `timesnet`

### Train AutoEncoder

| Dataset | Command |
|---------|---------|
| ETTh1 | `python src/experiments/ae/run.py --config assets/configs/models/etth1_dataset/ae/tcn_ae.json` |
| ETTh2 | `python src/experiments/ae/run.py --config assets/configs/models/etth2_dataset/ae/tcn_ae.json` |
| Weather | `python src/experiments/ae/run.py --config assets/configs/models/weather_dataset/ae/tcn_ae.json` |

### Train All RL Agents

| Dataset | Command |
|---------|---------|
| ETTh1 | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/etth1_dataset/RL_ablations --ae_config assets/configs/models/etth1_dataset/ae/tcn_ae.json` |
| ETTh2 | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/etth2_dataset/RL --ae_config assets/configs/models/etth2_dataset/ae/tcn_ae.json` |
| Weather | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/weather_dataset/RL --ae_config assets/configs/models/weather_dataset/ae/tcn_ae.json` |

### Train Single RL Agent

| Dataset | Command |
|---------|---------|
| ETTh1 | `python src/experiments/rl_cf/run_last_v2.py --config assets/configs/models/etth1_dataset/RL_ablations/config_final.json --ae_config assets/configs/models/etth1_dataset/ae/tcn_ae.json` |
| ETTh2 | `python src/experiments/rl_cf/run_last_v2.py --config assets/configs/models/etth2_dataset/RL/config_{MODEL}.json --ae_config assets/configs/models/etth2_dataset/ae/tcn_ae.json` |
| Weather | `python src/experiments/rl_cf/run_last_v2.py --config assets/configs/models/weather_dataset/RL/config_{MODEL}.json --ae_config assets/configs/models/weather_dataset/ae/tcn_ae.json` |

**Models**: Replace `{MODEL}` with `itransformer`, `gru`, `dlinear`, `patchtst`, or `timesnet`

### Evaluate RL Agents

| Dataset | Command |
|---------|---------|
| ETTh1 | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/etth1_dataset/RL_ablations --ae_config assets/configs/models/etth1_dataset/ae/tcn_ae.json --eval_only --eval_batches 50` |
| ETTh2 | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/etth2_dataset/RL --ae_config assets/configs/models/etth2_dataset/ae/tcn_ae.json --eval_only --eval_batches 50` |
| Weather | `python src/experiments/rl_cf/run_last_v2.py --config_dir assets/configs/models/weather_dataset/RL --ae_config assets/configs/models/weather_dataset/ae/tcn_ae.json --eval_only --eval_batches 50` |

## Results

- **Checkpoints**: `assets/checkpoints/{etth1,etth2,weather}_chpts/`
- **Figures**: `assets/figures/{etth1,etth2,weather}/`
- **Results**: `assets/results/{etth1,etth2,weather}/`

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Set PYTHONPATH before running:

```bash
export PYTHONPATH=.
```

### `CUDA out of memory`

Reduce batch size in config JSON or use CPU.

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with correct CUDA version.

## Documentation

- `README.md` - Project overview
- `STRUCTURE.md` - Project structure details
