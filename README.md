# Stock Prediction

A transformer-based stock prediction system using walk-forward validation for realistic backtesting. The system predicts cross-sectional stock rankings and evaluates performance using Information Coefficient (IC), Information Ratio (IR), and simulated portfolio returns.

## Features

- **Walk-Forward Training**: Time-series cross-validation that prevents look-ahead bias
- **Transformer Architecture**: Attention-based model for temporal feature processing
- **Multi-Horizon Predictions**: Predicts returns for 1, 5, 10, and 20-day horizons
- **Comprehensive Evaluation**: IC, IR, Sharpe ratio, turnover, and transaction cost modeling
- **Ablation Framework**: Systematic hyperparameter testing with statistical significance
- **Baseline Comparisons**: Compare against Ridge, LightGBM, and MLP models
- **Multi-GPU Support**: Distributed training with PyTorch DDP

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions.

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Hugging Face
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data

# Train a model
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --num-folds 10

# Evaluate checkpoints
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5
```

## Project Structure

```
Stock-Prediction/
├── training/                    # Model training
├── inference/                   # Evaluation and backtesting
├── data_scraping/               # Data collection pipeline
├── dataset_creation/            # Feature engineering and HDF5 conversion
├── sanity_checks/               # Ablation studies and validation
├── rl/                          # Reinforcement learning (experimental)
├── cluster/                     # Stock clustering (experimental)
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── tests/                       # Test files
└── utils/                       # Shared utilities
```

## Documentation

### Core Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation and quick start guide |
| [docs/walk_forward_training.md](docs/walk_forward_training.md) | Complete training guide with all arguments |
| [docs/eval.md](docs/eval.md) | Evaluation methodology and metrics |
| [docs/features.md](docs/features.md) | Feature engineering and dataset structure |

### Technical Deep Dives

| Document | Description |
|----------|-------------|
| [docs/listnet_loss.md](docs/listnet_loss.md) | ListNet ranking loss implementation |
| [docs/transaction_costs.md](docs/transaction_costs.md) | Transaction cost and turnover modeling |
| [docs/statistical_background.md](docs/statistical_background.md) | Statistical methods and significance testing |
| [docs/differentiable_portfolio_selection.md](docs/differentiable_portfolio_selection.md) | Differentiable top-k selection |

### Experimental Features

| Document | Description |
|----------|-------------|
| [rl/docs/README.md](rl/docs/README.md) | Reinforcement learning for trading |
| [docs/cluster_top_k_filtering.md](docs/cluster_top_k_filtering.md) | Cluster-based stock filtering |
| [docs/coverage_based_cluster_selection.md](docs/coverage_based_cluster_selection.md) | Diversified cluster selection |

---

## Training

The main training script is `training/walk_forward_training.py`. It implements walk-forward validation where training data always precedes test data chronologically.

### Key Scripts

| Script | Description |
|--------|-------------|
| `training/walk_forward_training.py` | Main training script with walk-forward validation |
| `training/model.py` | Transformer model architecture (`SimpleTransformerPredictor`) |
| `training/baseline_models.py` | Ridge, LightGBM, MLP baselines for comparison |
| `training/hdf5_data_loader.py` | Efficient HDF5 data loading with caching |

### Basic Training

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --num-folds 10 \
    --epochs-per-fold 5 \
    --hidden-dim 512 \
    --num-layers 4 \
    --batch-size 128
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --ddp
```

### With Baseline Comparison

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --compare-models
```

### With Ranking Loss

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --ranking-loss-type listnet \
    --ranking-loss-weight 0.1
```

See [docs/walk_forward_training.md](docs/walk_forward_training.md) for all 50+ arguments.

---

## Inference & Evaluation

### Key Scripts

| Script | Description |
|--------|-------------|
| `inference/principled_evaluation.py` | Comprehensive checkpoint evaluation with IC, IR, returns |
| `inference/backtest_simulation.py` | Trading simulation with transaction costs |
| `inference/evaluate_walk_forward_checkpoints.py` | Evaluate all folds from walk-forward training |
| `inference/walk_forward_validation.py` | Out-of-sample validation framework |
| `inference/interactive_tc_analysis.py` | Interactive transaction cost analysis |

### Evaluate Checkpoints

```bash
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --sweep  # Test multiple top-k values
```

### Run Backtest

```bash
python -m inference.backtest_simulation \
    --checkpoint checkpoints/walk_forward/fold_0_best.pt \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --top-k 25 \
    --transaction-cost-bps 10
```

See [docs/eval.md](docs/eval.md) for evaluation methodology details.

---

## Data Pipeline

The data pipeline collects, processes, and converts financial data into training-ready HDF5 format.

### Data Collection Scripts

| Script | Description |
|--------|-------------|
| `data_scraping/fmp_comprehensive_scraper.py` | Financial Modeling Prep API scraper (fundamentals, ratios) |
| `data_scraping/yfinance_price_scraper.py` | Yahoo Finance price data |
| `data_scraping/news_scraper.py` | News article collection |
| `data_scraping/news_embedder.py` | Generate news embeddings with Nomic AI |
| `data_scraping/market_indices_scraper.py` | S&P 500, VIX, sector ETFs |
| `data_scraping/derived_features_calculator.py` | Technical indicators and derived features |

### Dataset Generation

| Script | Description |
|--------|-------------|
| `dataset_creation/generate_full_dataset.py` | End-to-end dataset generation |
| `dataset_creation/convert_to_hdf5.py` | Convert processed data to HDF5 format |
| `dataset_creation/extract_prices_to_hdf5.py` | Extract price data for backtesting |
| `dataset_creation/validate_price_data.py` | Validate price data quality |

### Generate Dataset from Scratch

```bash
# 1. Scrape all data sources
python -m data_scraping.fmp_comprehensive_scraper
python -m data_scraping.yfinance_price_scraper
python -m data_scraping.news_scraper
python -m data_scraping.news_embedder

# 2. Generate full dataset
python -m dataset_creation.generate_full_dataset

# 3. Convert to HDF5
python -m dataset_creation.convert_to_hdf5

# 4. Extract prices
python -m dataset_creation.extract_prices_to_hdf5
```

See [docs/features.md](docs/features.md) for feature descriptions.

---

## Sanity Checks & Ablation Studies

The `sanity_checks/` directory contains tools for validating data quality and running hyperparameter ablations.

### Key Scripts

| Script | Description |
|--------|-------------|
| `sanity_checks/ablate.sh` | Generic ablation launcher script |
| `sanity_checks/analyze_ablation_metrics.py` | Analyze and compare ablation runs |
| `sanity_checks/check_dataset_quality.py` | Validate dataset integrity |
| `sanity_checks/check_predictive_signal.py` | Test for predictive signal in features |
| `sanity_checks/feature_ir_analysis.py` | Calculate feature-level Information Ratios |

### Run an Ablation Study

```bash
# Ablate hidden dimension across 3 GPUs
./sanity_checks/ablate.sh hidden-dim "256 512 768 1024" "0 1 2"

# Ablate learning rate
./sanity_checks/ablate.sh learning-rate "1e-5 5e-5 1e-4 5e-4" "0 1"
```

### Analyze Ablation Results

```bash
python sanity_checks/analyze_ablation_metrics.py \
    --runs checkpoints/ablation_hidden_dim/hidden-dim_256__listnet_seed4 \
           checkpoints/ablation_hidden_dim/hidden-dim_512__listnet_seed4 \
           checkpoints/ablation_hidden_dim/hidden-dim_768__listnet_seed4 \
    --labels "256" "512" "768" \
    --output results/ablation_hidden_dim_analysis
```

### Feature Analysis

```bash
# Check dataset quality
python sanity_checks/check_dataset_quality.py \
    --data all_complete_dataset.h5

# Analyze feature predictive power
python sanity_checks/feature_ir_analysis.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5
```

---

## Reinforcement Learning (Experimental)

The `rl/` directory contains experimental reinforcement learning approaches for trading, organized into submodules.

### Structure

```
rl/
├── actor_critic/    # Actor-Critic approach
├── dqn/             # Deep Q-Network approach
├── core/            # Shared components (environment, networks)
├── utils/           # Utility functions
├── evaluation/      # Evaluation metrics and tools
└── docs/            # General RL documentation
```

### Key Scripts

| Script | Description |
|--------|-------------|
| `rl/actor_critic/train_actor_critic.py` | Actor-Critic training for portfolio management |
| `rl/actor_critic/inference_actor_critic.py` | Run inference with trained AC agent |
| `rl/dqn/train_dqn_simple.py` | Simplified DQN training |
| `rl/core/rl_environment.py` | Trading environment implementation |
| `rl/core/rl_components.py` | Neural network architectures |

### Train Actor-Critic

```bash
python -m rl.actor_critic.train_actor_critic \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --predictor-checkpoint checkpoints/walk_forward/fold_0_best.pt
```

### Train DQN

```bash
python -m rl.dqn.train_dqn_simple \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5
```

See [rl/docs/README.md](rl/docs/README.md) for detailed RL documentation.

---

## Clustering (Experimental)

The `cluster/` directory contains stock clustering functionality for diversification.

### Key Scripts

| Script | Description |
|--------|-------------|
| `cluster/create_clusters.py` | Generate stock clusters from embeddings |
| `cluster/compute_embeddings.py` | Compute stock embeddings for clustering |
| `cluster/dynamic_cluster_filter.py` | Filter stocks by cluster for diversification |
| `cluster/analyze_clusters.py` | Analyze cluster composition |

### Create Clusters

```bash
python -m cluster.create_clusters \
    --data all_complete_dataset.h5 \
    --n-clusters 50
```

---

## Utility Scripts

### scripts/

| Script | Description |
|--------|-------------|
| `scripts/download_data.py` | Download data from Hugging Face Hub |
| `scripts/upload_to_hf.py` | Upload data to Hugging Face Hub |

### Daily Predictions

| Script | Description |
|--------|-------------|
| `daily_predictions/auto_daily_predictions_enhanced.py` | Generate daily stock predictions |

---

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **IC (Information Coefficient)** | Correlation between predictions and actual returns | > 0.02 |
| **IR (Information Ratio)** | IC / std(IC), measures consistency | > 0.5 |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Turnover** | Fraction of portfolio changed per period | < 50% |
| **Win Rate** | Percentage of profitable trades | > 52% |

---

## Data Files

| File | Size | Description |
|------|------|-------------|
| `all_complete_dataset.h5` | ~30GB | Main features (1123 features × 748 stocks × 9000+ days) |
| `actual_prices_clean.h5` | ~80MB | Split-adjusted prices for backtesting |

### Dataset Structure

```
all_complete_dataset.h5
├── {ticker}/
│   ├── features    # (num_days, 1123) float32
│   └── dates       # (num_days,) string dates
```

Features include:
- **Base features (0-354)**: Technical indicators, fundamentals, market data
- **News embeddings (355-1122)**: 768-dim Nomic AI embeddings

---

## Potentially Deprecated Scripts

The following scripts may be outdated or experimental. Contact maintainers before using:

| Script | Status | Notes |
|--------|--------|-------|
| `training/selection_aware_training.py` | Experimental | Alternative training strategy |
| `training/new_data_loader.py` | Legacy | May be superseded by `hdf5_data_loader.py` |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- ~32GB RAM for full dataset
- ~30GB disk space for data

See [requirements.txt](requirements.txt) for full dependencies.

---

## License

[Add license information]

## Citation

[Add citation information if applicable]
