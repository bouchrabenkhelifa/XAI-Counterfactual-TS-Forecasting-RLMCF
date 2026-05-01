# Environment Setup Guide

## Prerequisites

- **Python**: 3.9+
- **CUDA**: 11.8+ (for GPU support, optional)
- **Git**: For version control

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/counterfactual-forecasting-rl.git
cd counterfactual-forecasting-rl
```

### 2. Create Virtual Environment

#### Using `venv` (Recommended)

```bash
python -m venv venv

# Activate
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Using `conda`

```bash
conda create -n cf-rl python=3.10
conda activate cf-rl
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Key Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
tensorboard>=2.11.0
```

### 4. GPU Support (Optional)

For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Project Structure Setup

The project structure is automatically created when you run experiments. Key directories:

```
assets/
├── checkpoints/              # Model weights (auto-created)
├── configs/                  # Configuration files (included)
├── datasets/                 # Time series data (download required)
├── figures/                  # Visualizations (auto-created)
└── results/                  # Training histories (auto-created)

src/                          # Source code (included)
scripts/                      # Utility scripts (included)
```

---

## Dataset Setup

### Download Datasets

The project uses ETTh1, ETTh2, and optionally Weather/Traffic datasets.

#### Option 1: Automatic Download (if available)

```bash
python scripts/download_datasets.py
```

#### Option 2: Manual Download

1. Download from [ETDataset](https://github.com/zhouhaoyi/ETDataset)
2. Place in `assets/datasets/`:
   - `ETTh1.csv`
   - `ETTh2.csv`
   - `electricity.csv` (optional)
   - `weather.csv` (optional)

#### Dataset Format

Each CSV should have:
- **Index**: Timestamp (hourly)
- **Columns**: Features (e.g., OT, HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Target**: OT (Oil Temperature) for univariate, all features for multivariate

---

## Configuration

### Environment Variables

Create a `.env` file (optional):

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO

# Paths (auto-detected, override if needed)
DATA_PATH=assets/datasets
CHECKPOINT_PATH=assets/checkpoints
RESULTS_PATH=assets/results
```

### Configuration Files

All model configs are in `assets/configs/models/`:

```
assets/configs/models/
├── etth1_dataset/
│   ├── forecasters/
│   │   ├── itransformer/etth1_96_48_S.json
│   │   ├── gru/etth1_96_48_S.json
│   │   └── ...
│   └── RL_ablations/
│       └── config_final.json
└── etth2_dataset/
    ├── forecasters/
    └── RL/
```

---

## Verification

### 1. Check Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### 2. Check GPU (if applicable)

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

### 3. Check Project Structure

```bash
ls -la assets/datasets/
ls -la assets/configs/models/
```

### 4. Test a Simple Training Run

```bash
# Set Python path
export PYTHONPATH=.

# Train a forecaster (ETTh2, iTransformer)
python src/experiments/forecasting/run.py \
  --config assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json
```

Expected output:
```
[iTransformer] 1,234,567 params | seq=96 pred=48 train=8497 test=2833
Epoch: 1, Steps: 265 | Train Loss: 0.2301 Vali Loss: 0.1694 Test Loss: 0.1145
...
[Done] Best checkpoint → assets/checkpoints/etth2_chpts/forecaster/chpt_etth2_96_48_S.pth
[Fig] Training curves → assets/figures/etth2/forecaster/itransformer/itransformer_training_curves.png
```

---

## Common Issues & Solutions

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**: Set PYTHONPATH before running:

```bash
export PYTHONPATH=.
python src/experiments/forecasting/run.py --config ...
```

Or on Windows:

```powershell
$env:PYTHONPATH = "."
python src/experiments/forecasting/run.py --config ...
```

### Issue: `CUDA out of memory`

**Solution**: Reduce batch size in config:

```json
{
  "batch_size": 16  // Reduce from 32
}
```

### Issue: Dataset not found

**Solution**: Verify dataset location:

```bash
ls assets/datasets/ETTh1.csv
ls assets/datasets/ETTh2.csv
```

If missing, download from [ETDataset](https://github.com/zhouhaoyi/ETDataset).

### Issue: Figures not generated

**Solution**: Ensure figures directory exists:

```bash
mkdir -p assets/figures/etth2/forecaster/itransformer
```

---

## Running Experiments

### Train a Single Forecaster

```bash
export PYTHONPATH=.

# ETTh2, iTransformer
python src/experiments/forecasting/run.py \
  --config assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json
```

### Run Full Pipeline (ETTh2)

```bash
python scripts/pipelines/pipeline_etth2.py
```

Options:
```bash
python scripts/pipelines/pipeline_etth2.py --skip_ae --skip_rl
```

### Train RL Agent

```bash
python src/experiments/rl_cf/run.py \
  --rl_config assets/configs/models/etth2_dataset/RL/config_final.json \
  --forecast_config assets/configs/models/etth2_dataset/forecasters/itransformer/etth2_96_48_S.json \
  --ae_config assets/configs/models/etth2_dataset/ae/tcn_ae.json
```

---

## Development

### Code Style

```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/

# Type checking
mypy src/
```

### Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

---

## Troubleshooting

### Check Logs

Training logs are printed to console. For persistent logging:

```bash
python src/experiments/forecasting/run.py ... 2>&1 | tee training.log
```

### Debug Mode

Add debug prints in config:

```json
{
  "debug": true,
  "verbose": true
}
```

### GPU Debugging

```bash
# Monitor GPU usage
nvidia-smi -l 1  # Update every 1 second

# Or in Python
import torch
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
```

---

## Next Steps

1. **Verify setup**: Run the test command above
2. **Explore configs**: Check `assets/configs/models/`
3. **Train a model**: Start with ETTh2 iTransformer
4. **Check results**: View figures in `assets/figures/`
5. **Read documentation**: See `README.md` and `STRUCTURE.md`

---

## Support

For issues or questions:
1. Check this guide
2. Review `README.md`
3. Check existing GitHub issues
4. Create a new issue with:
   - Python version
   - PyTorch version
   - CUDA version (if applicable)
   - Error message & traceback
   - Steps to reproduce
