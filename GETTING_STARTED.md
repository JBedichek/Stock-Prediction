# Getting Started

This guide explains how to set up and run the stock prediction models.

## Quick Start (Docker)

The easiest way to get started is using Docker:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Stock-Prediction.git
cd Stock-Prediction

# 2. Build the Docker image
docker build -t stock-prediction .

# 3. Download the data (replace with actual repo ID)
docker run -it -v $(pwd)/data:/app/data \
    stock-prediction python scripts/download_data.py \
    --repo-id JamesBedichek/stock-prediction-data

# 4. Run evaluation
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints \
    stock-prediction python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --sweep
```

Or use Docker Compose:

```bash
# Download data
HF_REPO_ID=JamesBedichek/stock-prediction-data docker compose run download-data

# Run evaluation
docker compose run evaluate

# Run training
docker compose run train
```

## Manual Setup (No Docker)

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~30GB disk space for data

### Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/Stock-Prediction.git
cd Stock-Prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install huggingface_hub

# 4. Download data
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data
```

### Running

```bash
# Evaluate pre-trained models
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --sweep

# Train new model
python -m training.walk_forward_training \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --num-folds 6
```

## Data Files

| File | Size | Description |
|------|------|-------------|
| `all_complete_dataset.h5` | ~26GB | Main features dataset (4000+ stocks, 20+ years) |
| `actual_prices_clean.h5` | ~80MB | **Cleaned** split-adjusted prices (3,424 validated tickers) |
| `actual_prices_raw.h5` | ~86MB | Raw prices (may have data quality issues, for reference only) |
| `checkpoints/walk_forward/fold_*_best.pt` | ~650MB total | Pre-trained model checkpoints |

**Note:** Always use `actual_prices_clean.h5` for backtesting. The raw file contains 155 tickers with data quality issues (yfinance bugs, untracked corporate actions).

## Project Structure

```
Stock-Prediction/
├── training/                 # Training scripts
│   ├── walk_forward_training.py  # Main training script
│   └── train_new_format.py       # Model architecture
├── inference/                # Evaluation and inference
│   ├── principled_evaluation.py  # Comprehensive evaluation
│   └── backtest_simulation.py    # Trading simulation
├── data_scraping/            # Data collection (not needed if using HF data)
├── scripts/
│   ├── download_data.py      # Download from Hugging Face
│   └── upload_to_hf.py       # Upload to Hugging Face
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## For Data Providers

If you want to host your own copy of the data:

```bash
# 1. Install Hugging Face CLI
pip install huggingface_hub
huggingface-cli login

# 2. Upload data
python scripts/upload_to_hf.py --repo-id JamesBedichek/stock-prediction-data

# 3. Update download script default
# Edit scripts/download_data.py and change DEFAULT_REPO_ID
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python -m training.walk_forward_training --batch-size 32
```

### Download Fails
For private repos, authenticate first:
```bash
huggingface-cli login
```

### Missing Data Files
Verify data downloaded correctly:
```bash
ls -la data/
# Should show all_complete_dataset.h5 and actual_prices_clean.h5
```
