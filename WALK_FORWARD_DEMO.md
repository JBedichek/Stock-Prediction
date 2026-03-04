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
    --prices data/actual_prices_clean.h5 \
    --num-folds 5 \
    --epochs-per-fold 3 \
    --checkpoint-dir checkpoints/my_walk_forward
```

After training, check `checkpoints/my_walk_forward_seed42/stats.log` for detailed IC, IR, and Rank IC metrics.

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

### Loss Function Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pred-mode` | classification | `classification` (bins) or `regression` (direct) |
| `--ranking-only` | False | Train with ONLY ranking loss (no CE/MSE) |
| `--ranking-loss-type` | pairwise | `pairwise` (margin-based) or `listnet` (distribution) |
| `--ranking-margin` | 0.01 | Margin for pairwise ranking loss |
| `--ranking-loss-weight` | 0.0 | Weight for ranking loss when combined with standard loss |
| `--transaction-cost-bps` | 10 | Transaction cost in basis points for evaluation |

---

## Loss Functions and Objective

The model can be trained with different loss functions depending on the goal:

### Prediction Modes

**Classification Mode** (`--pred-mode classification`, default):
- Model outputs logits over discrete return bins: shape `(batch, num_bins, num_horizons)`
- Bins are adaptively computed from training data (e.g., 50 bins covering the return distribution)
- Standard loss: Cross-entropy between predicted bin probabilities and true bin labels

**Regression Mode** (`--pred-mode regression`):
- Model outputs direct return predictions: shape `(batch, num_horizons)`
- Standard loss: Mean squared error between predictions and actual returns

### Ranking Loss (`--ranking-only`)

When `--ranking-only` is enabled, the model is trained **exclusively** to rank stocks correctly relative to each other, rather than predict exact returns. This is often more aligned with the actual trading objective.

#### Computing Prediction Scores

Before applying ranking loss, we need a scalar "score" for each stock:

**Classification mode:**
```
score_i = Σ_b P(bin_b | stock_i) × midpoint(bin_b)
```
The score is the **expected return** - a probability-weighted average of bin midpoints.

**Regression mode:**
```
score_i = prediction_i
```
The model's direct output is used as the score.

---

### Pairwise Ranking Loss (`--ranking-loss-type pairwise`, default)

**Intuition:** For every pair of stocks where stock A actually outperformed stock B, we want the model to predict a higher score for A than B.

**Mathematical Formulation:**

Given a batch of N stocks with prediction scores `s` and actual returns `r`:

```
L_pairwise = (1/|P|) × Σ_{(i,j) ∈ P} max(0, margin - (s_i - s_j))

where P = {(i,j) : r_i > r_j}  (all pairs where stock i beat stock j)
```

**Components:**
- `s_i - s_j`: How much higher the model scores stock i vs stock j
- `margin`: Minimum desired score difference (default: 0.01, set via `--ranking-margin`)
- `max(0, ...)`: Hinge loss - zero if prediction gap exceeds margin, positive otherwise
- `|P|`: Number of valid pairs (normalization)

**Example:**
```
Stocks:     [AAPL, MSFT, GOOG]
Returns:    [+2%,  +1%,  -1%]  → AAPL > MSFT > GOOG
Predictions:[0.8,  0.6,  0.3]  → Model ranks correctly ✓

Pairs where r_i > r_j:
  (AAPL, MSFT): margin - (0.8 - 0.6) = 0.01 - 0.2 = -0.19 → max(0, -0.19) = 0 ✓
  (AAPL, GOOG): margin - (0.8 - 0.3) = 0.01 - 0.5 = -0.49 → max(0, -0.49) = 0 ✓
  (MSFT, GOOG): margin - (0.6 - 0.3) = 0.01 - 0.3 = -0.29 → max(0, -0.29) = 0 ✓

Loss = 0 (model ranks all pairs correctly with sufficient margin)
```

**Incorrect ranking example:**
```
Predictions:[0.3,  0.6,  0.8]  → Model ranks GOOG > MSFT > AAPL (inverted!)

Pairs:
  (AAPL, MSFT): margin - (0.3 - 0.6) = 0.01 - (-0.3) = 0.31 → loss = 0.31 ✗
  (AAPL, GOOG): margin - (0.3 - 0.8) = 0.01 - (-0.5) = 0.51 → loss = 0.51 ✗
  (MSFT, GOOG): margin - (0.6 - 0.8) = 0.01 - (-0.2) = 0.21 → loss = 0.21 ✗

Loss = (0.31 + 0.51 + 0.21) / 3 = 0.343 (high loss for incorrect ranking)
```

**Code location:** `training/walk_forward_training.py:434-445`

---

### ListNet Loss (`--ranking-loss-type listnet`)

**Intuition:** Treat ranking as a probability distribution matching problem. Convert both predictions and targets into probability distributions over stocks, then minimize their cross-entropy.

**Mathematical Formulation:**

```
L_listnet = -Σ_i P_target(i) × log(P_pred(i))

where:
  P_pred(i)   = exp(s_i / τ) / Σ_j exp(s_j / τ)    (softmax over prediction scores)
  P_target(i) = exp(r_i / τ) / Σ_j exp(r_j / τ)    (softmax over actual returns)
  τ = temperature (default: 1.0)
```

**Interpretation:**
- `P_target(i)`: Probability that stock i should be ranked highest (based on actual returns)
- `P_pred(i)`: Probability the model assigns to stock i being ranked highest
- Cross-entropy measures how well the predicted distribution matches the target distribution

**Example:**
```
Stocks:     [AAPL, MSFT, GOOG]
Returns:    [+2%,  +1%,  -1%]
Predictions:[0.8,  0.6,  0.3]

P_target = softmax([0.02, 0.01, -0.01]) = [0.340, 0.333, 0.327]
P_pred   = softmax([0.8, 0.6, 0.3])     = [0.387, 0.317, 0.296]

L = -(0.340×log(0.387) + 0.333×log(0.317) + 0.327×log(0.296))
  = -(0.340×(-0.95) + 0.333×(-1.15) + 0.327×(-1.22))
  = 1.10
```

**Code location:** `training/walk_forward_training.py:447-451`

---

### Hybrid Loss (`--ranking-loss-weight`)

Instead of `--ranking-only`, you can combine ranking loss with the standard loss:

```
L_total = L_standard + λ × L_ranking

where λ = --ranking-loss-weight (default: 0.0)
```

This allows the model to learn both accurate return predictions AND correct rankings.

**Example usage:**
```bash
# 50% standard loss, 50% ranking loss
python -m training.walk_forward_training \
    --data data/all_complete_dataset.h5 \
    --ranking-loss-weight 1.0 \
    --ranking-loss-type pairwise
```

---

### Which Loss to Use?

| Scenario | Recommended Loss | Why |
|----------|------------------|-----|
| Stock selection (pick top K) | `--ranking-only --ranking-loss-type pairwise` | Directly optimizes ranking accuracy |
| Return forecasting | Standard (no ranking flags) | Optimizes prediction accuracy |
| Balanced approach | `--ranking-loss-weight 0.5` | Combines both objectives |
| Large batches (128+) | `pairwise` | More stable with many pairs |
| Small batches (<32) | `listnet` | Works better with few samples |

**Typical training command for ranking:**
```bash
python -m training.walk_forward_training \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --ranking-only \
    --ranking-loss-type pairwise \
    --ranking-margin 0.01 \
    --batch-size 128
```

### Critical: Cross-Sectional Batch Sampling

**Why standard shuffling breaks ranking loss:**

Ranking losses (pairwise, ListNet) compare stocks *within each batch* to learn relative rankings. With standard shuffling, batches mix samples from **different trading days**:

| Sample | Stock | Date | Return |
|--------|-------|------|--------|
| 1 | AAPL | 2023-01-15 | +5% |
| 2 | MSFT | 2023-03-22 | +3% |
| 3 | GOOG | 2023-02-01 | -2% |

Comparing these stocks is **meaningless** because:
- Stock returns depend on market conditions that day
- A +3% return on a down day may outperform a +5% return on an up day
- Cross-sectional ranking requires comparing stocks on the **same date**

**The Fix: CrossSectionalBatchSampler**

When `--ranking-only` or `--ranking-loss-weight > 0` is specified, the training automatically uses `CrossSectionalBatchSampler`:

```
    Using cross-sectional batch sampling for ranking loss
    CrossSectionalBatchSampler: 487/512 dates have >= 10 samples
```

Each batch now contains stocks from a **single trading day**, ensuring valid cross-sectional comparisons:

| Sample | Stock | Date | Return |
|--------|-------|------|--------|
| 1 | AAPL | 2023-01-15 | +5% |
| 2 | MSFT | 2023-01-15 | +3% |
| 3 | GOOG | 2023-01-15 | -2% |

Now the ranking loss can correctly learn: AAPL (+5%) > MSFT (+3%) > GOOG (-2%) on this date.

**Diagnosing Negative IC:**

If your model shows consistently negative IC (predictions inversely correlated with returns), check:
1. Are you using `--ranking-only` or `--ranking-loss-weight > 0`?
2. Is the "cross-sectional batch sampling" message appearing in the training log?
3. Multi-GPU (DDP) is fully supported with `DistributedCrossSectionalBatchSampler`

**Multi-GPU Training with Ranking Loss:**

```bash
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --ddp \
    --ranking-only \
    --ranking-loss-type pairwise \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5
```

You should see:
```
    Using distributed cross-sectional batch sampling for ranking loss
    DistributedCrossSectionalBatchSampler:
      Total valid dates: 2500
      Dates per GPU: ~625
      Num GPUs: 4
```

The distributed sampler divides **dates** (not samples) across GPUs, ensuring each GPU processes complete cross-sections for valid pairwise comparisons.

---

## Output

After training completes, you'll find:

```
checkpoints/walk_forward_seed{N}/
│
├── stats.log               # ⭐ KEY METRICS LOG - IC, IR, Rank IC, returns
├── monte_carlo/            # Monte Carlo simulation results per fold
├── fold_0_best.pt          # Best checkpoint for fold 0
├── fold_1_best.pt          # Best checkpoint for fold 1
├── ...
└── training_config.json    # Training configuration
```

### stats.log - Key Model Statistics

The `stats.log` file in your checkpoint directory contains **detailed evaluation metrics** for each fold. This is the primary place to assess model quality:

```
================================================================================
FOLD 1/5 EVALUATION RESULTS
================================================================================
Training Period:   2003-01-02 to 2014-06-30
Test Period:       2014-07-01 to 2018-12-31
Eval Samples:      892 non-overlapping periods
Stock Universe:    1000 stocks evaluated per period
Horizon:           1 day(s)
Top-K Selection:   10 stocks
Transaction Cost:  10 bps (0.10%)
--------------------------------------------------------------------------------

📊 INFORMATION COEFFICIENT METRICS (Primary)
   Mean IC:          +0.0234
   IC Std Dev:       0.0512
   Information Ratio:+0.457  (IC / IC_std)
   Mean Rank IC:     +0.0198  (Spearman correlation)
   Pct IC > 0:       58.2%
   IC T-statistic:   +4.31  (p=0.0001 ***)

📈 QUANTILE ANALYSIS
   Top Decile Ret:   +0.082% per period
   Bottom Decile:    -0.041% per period
   Long-Short Spread:+0.123%

📊 BASELINE COMPARISONS (Gross Returns)
   Model Return:     +0.065% per period
   Momentum Return:  +0.032% per period
   Random Return:    +0.021% per period
   Excess vs Random: +0.044%  (p=0.008)
   Excess vs Momentum:+0.033%  (p=0.042)
```

**Key Metrics to Look For:**

| Metric | What It Means | Good Values |
|--------|---------------|-------------|
| **Mean IC** | Correlation between predictions and actual returns | > 0.02 useful, > 0.05 good |
| **Information Ratio** | Risk-adjusted IC (like Sharpe for predictions) | > 0.5 consistent |
| **Mean Rank IC** | Spearman rank correlation (robust to outliers) | > 0.02 useful |
| **Pct IC > 0** | Percentage of days with positive correlation | > 55% |
| **Long-Short Spread** | Top decile return minus bottom decile | > 0% (model ranks correctly) |
| **Excess vs Random** | Model return minus random baseline | Significantly positive |

Results JSON file (default: `walk_forward_training_results.json`):
- Per-fold metrics and statistics
- Aggregated performance across all folds
- Monte Carlo validation results (if enabled)

The `monte_carlo/` folder contains detailed statistical comparisons against random and momentum baselines



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
