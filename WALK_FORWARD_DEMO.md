# Walk-Forward Training Demo

## Overview

This project trains a **118 million parameter transformer model** to predict relative stock performance across the US equity market. The goal is not to predict absolute returns, but to **rank stocks** against each other—identifying which stocks will outperform or underperform their peers over the next 1-5 trading days.

### The Approach

- **Model**: 16-layer transformer encoder, processing sequences of 750 trading days
- **Training**: Walk-forward validation that respects temporal ordering—we never train on future data
- **Loss Function**: Pairwise ranking loss that directly optimizes the model's ability to rank stocks correctly, rather than predict exact returns
- **Data**: 20+ years of daily data (2001-2026) covering 750 (4000+ capable) US stocks, including price/volume features, technical indicators, and market context

### The Data

The dataset (~26GB) is aggregated from multiple financial APIs:
- **yfinance**: Price and volume data (OHLCV, splits, dividends)
- **Alpha Vantage**: Financial news articles
- **Financial Modeling Prep (FMP)**: Fundamental data and company financials

The processed features include:
- **News Embeddings**: Transformer embeddings of financial news articles, capturing market sentiment and company-specific events
- **Price & Volume**: OHLCV data, returns at multiple horizons, volume profiles
- **Technical Features**: Moving averages, RSI, MACD, Bollinger bands, etc.
- **Cross-sectional Features**: Relative strength vs market, sector, and peers
- **Temporal Encoding**: Day of week, month, market regime indicators

### Normalization Strategy

Features are normalized differently based on their type:

- **Cross-sectional normalization** (across stocks on each day): Fundamental ratios (P/E, ROE, debt/equity), financial metrics, and valuation multiples. This preserves relative rankings—a stock with low P/E stays "cheap" relative to peers.

- **Temporal normalization** (per-stock over time): Price returns, volume ratios, technical indicators (RSI, MACD, Bollinger Bands). These are computed relative to each stock's own history.

- **Percentile ranks**: Many features are converted to cross-sectional percentiles (0-100), making them comparable across different market regimes and eliminating outlier sensitivity.

### Walk-Forward Training

Unlike standard train/test splits, walk-forward validation simulates real trading:
1. Train on data from 2003-2014
2. Test on 2014-2020 (data the model has never seen)
3. Retrain on 2003-2020
4. Test on 2020-2026

This ensures all reported metrics reflect true out-of-sample performance.

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, ~24G-96B+ VRAM for large comparitive batch sizes)
- ~30GB disk space for data

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/JamesBedichek/Stock-Prediction.git
cd Stock-Prediction
```

### 2. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data from Hugging Face

```bash
# Download all data and checkpoints (~26GB)
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data

# Or download only the data files (no pre-trained checkpoints)
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data --data-only
```

For private repositories, authenticate first:
```bash
pip install huggingface_hub
huggingface-cli login
```

### 4. Run Walk-Forward Training

Basic training run:
```bash
python -m training.walk_forward_training \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices.h5 \
    --num-folds 5 \
    --epochs-per-fold 3 \
    --checkpoint-dir checkpoints/my_walk_forward
```

## Training Options

### Quick Demo Run (for testing)

Small run to verify everything works:
```bash
python -m training.walk_forward_training --ddp --no-preload --data all_complete_dataset.h5  --prices actual_prices_clean.h5 --no-compile --data-fraction 0.02 --incremental --incremental-epochs 1 --incremental-data-fraction 0.15 --ranking-only --ranking-margin 0.01  -ranking-loss-type pairwise  --seed 3
```

or for a quick run with DDP multi-node setup:
```bash
torchrun --nproc_per_node={num_gpus} -m training.walk_forward_training --ddp --no-preload --data all_complete_dataset.h5  --prices actual_prices_clean.h5 --no-compile --data-fraction 0.025 --incremental --incremental-epochs 1 --incremental-data-fraction 0.15 --ranking-only --ranking-margin 0.01  -ranking-loss-type pairwise  --seed 3
```


## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to HDF5 dataset |
| `--prices` | None | Path to prices file for backtesting |
| `--num-folds` | 10 | Number of temporal folds |
| `--mode` | expanding | `expanding` (growing window) or `sliding` (fixed window) |
| `--epochs-per-fold` | 1 | Training epochs per fold |
| `--batch-size` | 128 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--hidden-dim` | 768 | Transformer hidden dimension |
| `--num-layers` | 16 | Number of transformer layers |
| `--checkpoint-dir` | checkpoints/walk_forward | Where to save model checkpoints |
| `--data-fraction` | 1.0 | Fraction of data to use (for quick tests) |
| `--no-monte-carlo` | False | Skip Monte Carlo validation |
| `--device` | cuda | Device to train on |

## Output

After training completes, you'll find:

```
checkpoints/walk_forward/
|
├── monte_carlo/      # Monte carlo simulation results for each fold
├── fold_0_best.pt      # Best checkpoint for fold 0
├── ...
└── training_config.json  # Training configuration
```

Results JSON file (default: `walk_forward_training_results.json`):
- Per-fold metrics and statistics
- Aggregated performance across all folds
- Monte Carlo validation results (if enabled)

Look in the monte_carlo/ folder



### Download Fails
Verify the repository exists and you have access:
```bash
huggingface-cli login
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data
```


## Docker

Pre-built image available on Docker Hub - no build required:

```bash
# Pull the image (~9GB)
docker pull jamesbedichek/stock-prediction:latest

# Create directories
mkdir -p data checkpoints

# Download data
docker run --rm \
    -v $(pwd)/data:/app/data \
    jamesbedichek/stock-prediction:latest \
    python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data --data-only

# Run training
docker run --gpus all --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    jamesbedichek/stock-prediction:latest \
    python -m training.walk_forward_training \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --num-folds 2 \
    --epochs-per-fold 1 \
    --checkpoint-dir checkpoints/my_model
```

### GPU Not Detected?

Install NVIDIA Container Toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Building Locally

If you need to modify the code and rebuild:
```bash
docker build -t stock-prediction .
```

The image uses PyTorch 2.10.0 with CUDA 12.8, supporting modern GPUs including NVIDIA Blackwell.
