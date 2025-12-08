# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock-Prediction is a deep learning-based stock price prediction system using transformer models. The model predicts probability distributions over future stock price movements (next 4 days) using 3 years of historical price data and company metadata.

**Key Features:**
- Predicts discrete probability distributions over 320 bins (not single point predictions)
- Incorporates 3 years (600 market days) of historical data per prediction
- Includes market-wide movement data in addition to individual stock data
- Uses RoBERTa embeddings of company sector/industry/business summaries

## Core Architecture

### Model Architecture (models.py)

**t_Dist_Pred** - Main prediction model:
- Mean-pooling transformer encoder with 72 layers
- Stochastic depth regularization (dropout prob increases linearly from 0 to 0.4 across layers)
- Input: (batch, 604, 218) after combining price sequence + summary embeddings
  - 600-day historical price sequences (5 features: open/high/low/close/volume)
  - Company summary: 218*4 features reshaped to (4, 218) and concatenated
- Output: (batch, 320, 4) - probability distributions for 4 future days over 320 price change bins
- Learnable positional embeddings with dropout
- Residual connections updated every ~10 layers
- Temperature-scaled softmax applied only during inference
- Mean pooling splits final activations in half before classification

**Other Models:**
- **Dist_Pred**: Earlier, less sophisticated variant
- **L1_Dist_Pred/L2_Dist_Pred**: Meta-models that take base predictions as input
- **meta_model**: Predicts confidence/error scores for filtering predictions
- **RobertaEncoder** (models.py): Encodes company text summaries to 768-dim vectors

### Data Pipeline

**1. Data Scraping (Stock.py - stock_info class)**
```python
stock_info(ticker_dict, dataset_name)
```
- `ticker_dict`: {ticker_symbol: company_name} from predefined dictionaries (a-z_lot_of_stocks)
- Available tickers: ~4000+ companies across all exchanges
- Key methods:
  - `get_prices(interval, start, end)`: Scrapes historical OHLCV data via yfinance
  - `get_company_text_info()`: Generates RoBERTa embeddings of sector + industry + business summary
  - `get_bulk_price_data_series(period)`: Bulk download for date ranges
  - `get_company_sectors()`: Sector/industry classification
- Outputs:
  - `unnorm_price_series_{dataset_name}_{period}_{date}.pickle`: Raw price tensors indexed by {company: {date: tensor}}
  - `{dataset_name}_summary_embs.pickle`: RoBERTa embeddings {company: tensor(768)}
  - `{dataset_name}_sector_dict`: Sector/industry labels

**2. Data Dictionary Generation (training.py - GenerateDataDict)**
- Converts raw scraped data into efficient date -> company -> data structure
- Each entry contains 600-day rolling windows
- Applies Gaussian normalization to price sequences
- Handles missing data by skipping companies without complete sequences
- Output: `DataDict_{date}.pt` files

**3. Dataset Creation (training.py - QTrainingData)**
- Generates training/inference datasets from DataDict
- Creates binned target distributions (not single values)
- Bin edges stored in 'Bin_Edges_300' file
- Supports both training mode and inference mode
- Handles NaN/inf values using `set_nan_inf()` from utils.py

**4. Training (training.py)**
- `Train_Dist_Direct_Predictor`: Standard training loop with KL divergence loss
- `Train_Dist_Direct_Predictor_SAM`: Training with Sharpness Aware Minimization
- Supports distributed training (DDP)
- Custom loss can include entropy regularization

**5. Inference & Trading Simulation (inference.py)**
- `generate_current_day_dataset()`: Creates inference dataset for current/specific date
- `stock_inference` class: Main trading simulation engine
  - `get_top_q_buy()`: Selects top N companies based on predictions
  - `run_trading_sim()`: Runs backtesting simulation over time period
  - `run_random_trading_sim()`: Baseline random selection for comparison
- Noise-averaged inference across multiple model checkpoints
- Entropy-based confidence filtering
- Optional meta-model scoring for prediction reliability

## Common Commands

### Data Scraping

```python
# In Stock.py or python shell
from Stock import stock_info, all_stocks, s_lot_of_stocks

# Scrape all available stocks (warning: ~4000+ companies, takes hours)
scraper = stock_info(all_stocks, 'complete_dataset')
scraper.get_bulk_price_data_series('-10y')  # Last 10 years

# Scrape subset (S-companies, ~200 tickers)
scraper = stock_info(s_lot_of_stocks, 's_only')
scraper.get_bulk_price_data_series('-5y')

# Generate company summaries
summaries = scraper.get_company_text_info()
# Saved as '{dataset_name}_summary_embs.pickle'

# Get sector information
scraper.get_company_sectors()
# Saved as '{dataset_name}_sector_dict'
```

### Training

```python
python training.py
```

Requirements before training:
1. Pre-generated DataDict files (from GenerateDataDict)
2. Summary embeddings pickle
3. Bulk prices pickle
4. Sector dictionary (optional)

Training outputs model checkpoints: `DistPred_m_E{epoch}`

### Running Trading Simulations

```python
python inference.py
```

Key parameters in `main()` function (bottom of inference.py):
- `model_pths`: List of model checkpoint paths (ensemble predictions)
- `c_prune`: Proportion of companies to evaluate (0.0-1.0, default 0.75 = 75% of ~4000)
- `low`: True for shorting stocks, False for buying
- `entropy`: Select company with lowest entropy (highest confidence) from top_n
- `top_n`: Number of top predictions to consider (default 10)
- `generate`: Generate new inference dataset vs. load existing
- `use_meta_model`: Use meta-model for confidence scoring
- `mm_prop`: Keep top X proportion when using meta-model (default 0.25)
- `track_wandb`: Log results to Weights & Biases
- `ignore_thr`: Disable threshold-based trade filtering

Default starting capital: $80,000

## File Structure

- **Stock.py**: Data scraping via yfinance, RoBERTa encoding, ~4000 ticker dictionaries
- **models.py**: Neural network definitions (t_Dist_Pred, meta_model, RobertaEncoder)
- **training.py**: Dataset generation (GenerateDataDict, QTrainingData) and training loops
- **inference.py**: Trading simulation engine and backtesting framework
- **utils.py**: Helper functions (normalization, SAM optimizer, entropy calculation)
- **contrastive_pretraining.py**: Only `gauss_normalize()` function is used (rest ineffective)
- **Models.py** (capitalized): Appears to be duplicate/older version
- **Training.py** (capitalized): Appears to be duplicate/older version
- **unused_historical_code.py**: Archive of deprecated code

## Important Technical Details

### Stock Ticker Dictionaries
The codebase includes comprehensive ticker lists organized alphabetically:
- `a_lot_of_stocks` through `z_lot_of_stocks`: Detailed {ticker: company_name} dictionaries
- `all_stocks`: Combined dictionary of all ~4000+ companies
- `inference_stocks`: Subset for inference (currently S-companies)
- `test_stock_tickers`: Small test set (MSFT, AAPL, TSLA)

### Data Shapes & Formats
- **Price data**: 5 features per day (Open, High, Low, Close, Volume)
- **Model input**: (batch, 604, 218) = 600 days + 4 summary chunks, 218 dims per chunk
- **Model output**: (batch, 320, 4) = 320 bins × 4 future days
- **Bin edges**: Stored in 'Bin_Edges_300', defines price change ranges for distribution
- **Company embeddings**: 768-dim RoBERTa vectors (sector + industry + summary)

### Data Normalization
- **Gaussian normalization**: `(x - mean) / std` applied to price sequences
- **Cube normalization**: Projects to [-1, 1] range for some features
- **NaN/inf handling**: Replaced with 1.0 using `set_nan_inf()` from utils.py

### Training Details
- **Loss**: KL divergence between predicted and target distributions
- **Optimizer**: AdamW or Lion, optional SAM (Sharpness Aware Minimization)
- **Regularization**: Entropy regularization, stochastic depth, dropout
- **Compilation**: Models compiled with `torch.compile(mode='max-autotune')` for speed
- **Distributed**: Supports DDP for multi-GPU training

### Inference Strategy
- **Ensemble**: Average predictions across multiple model checkpoints
- **Confidence filtering**: Use entropy to select high-confidence predictions
- **Meta-model**: Optional secondary model to score prediction reliability
- **Date correction**: Predictions adjusted by comparing adjacent days: `prices[n-2] - prices[n-3]`

### Trading Simulation
- Simulates buying/shorting stocks based on model predictions
- Supports:
  - Long positions (buying stocks expected to rise)
  - Short positions (selling stocks expected to fall)
  - Top-N selection with entropy filtering
  - Threshold-based trade filtering
  - Position sizing (equal weight across selected stocks)
- Tracks:
  - Point profit (per trade)
  - Cumulative returns
  - Win rate (probability of profit)
  - Account value over time

## Dependencies

Core dependencies (no requirements.txt provided):
- **torch**: PyTorch with CUDA support
- **transformers**: HuggingFace (for RoBERTa)
- **yfinance**: Yahoo Finance API for price data
- **gnews**: Google News scraping
- **pandas**: Data manipulation
- **tqdm**: Progress bars
- **wandb**: Experiment tracking (optional)
- **lion_pytorch**: Lion optimizer

## Notes from README

The original README states:
> "This code was not made with the intention of being easy or intuitive to read by others, so it may appear somewhat disorganized."

The codebase prioritizes functionality over code organization. Some files have capitalized duplicates (Models.py, Training.py) which may be older versions.

## Getting Started

1. **Scrape data**: Use Stock.py to download price data and generate embeddings
2. **Generate DataDict**: Use GenerateDataDict in training.py to create efficient data structure
3. **Train model**: Run training.py with appropriate dataset paths
4. **Run simulations**: Use inference.py to backtest trading strategies

Or use pre-existing pickle files if available in the directory.
