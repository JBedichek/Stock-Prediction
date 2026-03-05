# Stock Prediction

A transformer-based stock prediction system using walk-forward validation for realistic backtesting. The system predicts cross-sectional stock rankings and evaluates performance using Information Coefficient (IC), Information Ratio (IR), and simulated portfolio returns.

## Features

- **Walk-Forward Training**: Time-series cross-validation that prevents look-ahead bias by ensuring training data always precedes test data chronologically
- **Transformer Architecture**: Attention-based model (`SimpleTransformerPredictor`) for temporal feature processing with configurable depth and width
- **Multi-Horizon Predictions**: Predicts returns for 1, 5, 10, and 20-day horizons simultaneously
- **Ranking Loss**: ListNet and pairwise ranking losses for learning relative stock orderings rather than absolute returns
- **Comprehensive Evaluation**: IC, IR, Sharpe ratio, turnover, win rate, and transaction cost modeling
- **Ablation Framework**: Systematic hyperparameter testing with deterministic seeding and statistical significance testing
- **Baseline Comparisons**: Compare transformer against Ridge regression, LightGBM, and MLP models
- **Multi-GPU Support**: Distributed training with PyTorch DDP for large-scale experiments
- **Monte Carlo Validation**: Robust evaluation using random stock sampling across multiple trials

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions and [WALK_FORWARD_DEMO.md](WALK_FORWARD_DEMO.md) for a walkthrough.

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Hugging Face
python scripts/download_data.py --repo-id JamesBedichek/stock-prediction-data

# Train a model with walk-forward validation
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

---

## Project Structure

```
Stock-Prediction/
├── training/                    # Model training and architectures
│   ├── walk_forward_training.py # Main training script
│   ├── model.py                 # Transformer architecture
│   ├── baseline_models.py       # Ridge, LightGBM, MLP baselines
│   ├── hdf5_data_loader.py      # Efficient data loading
│   └── docs/                    # Training-specific documentation
├── inference/                   # Evaluation and backtesting
│   ├── principled_evaluation.py # Comprehensive evaluation
│   ├── backtest_simulation.py   # Trading simulation
│   └── *.py                     # Various evaluation scripts
├── data_scraping/               # Data collection pipeline
│   ├── fmp_comprehensive_scraper.py
│   ├── yfinance_price_scraper.py
│   ├── news_scraper.py
│   └── news_embedder.py
├── dataset_creation/            # Feature engineering and HDF5 conversion
│   ├── generate_full_dataset.py
│   ├── convert_to_hdf5.py
│   └── extract_prices_to_hdf5.py
├── sanity_checks/               # Ablation studies and validation
│   ├── ablate.sh                # Ablation launcher
│   ├── analyze_ablation_metrics.py
│   ├── feature_ir_analysis.py
│   └── check_*.py               # Various validation scripts
├── rl/                          # Reinforcement learning (experimental)
│   ├── actor_critic/            # Actor-Critic approach
│   ├── dqn/                     # Deep Q-Network approach
│   ├── core/                    # Shared components
│   └── docs/                    # RL documentation
├── cluster/                     # Stock clustering (experimental)
│   ├── create_clusters.py
│   ├── dynamic_cluster_filter.py
│   └── docs/                    # Clustering documentation
├── docs/                        # Main documentation
├── scripts/                     # Utility scripts
├── tests/                       # Test files
└── utils/                       # Shared utilities
```

---

## Documentation Index

### Getting Started

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Installation, dependencies, and initial setup |
| [WALK_FORWARD_DEMO.md](WALK_FORWARD_DEMO.md) | Step-by-step walkthrough of training and evaluation |

### Core Documentation (docs/)

| Document | Description |
|----------|-------------|
| [docs/walk_forward_training.md](docs/walk_forward_training.md) | Complete training guide with all 50+ arguments, examples, and best practices |
| [docs/eval.md](docs/eval.md) | Evaluation methodology: IC, IR, Sharpe ratio calculations and interpretation |
| [docs/features.md](docs/features.md) | Feature engineering: 1123 features including technicals, fundamentals, and news embeddings |
| [docs/listnet_loss.md](docs/listnet_loss.md) | ListNet ranking loss: mathematical formulation and implementation details |
| [docs/transaction_costs.md](docs/transaction_costs.md) | Turnover calculation, transaction cost modeling, and gross vs net returns |
| [docs/statistical_background.md](docs/statistical_background.md) | Statistical methods: significance testing, confidence intervals, bootstrap methods |
| [docs/differentiable_portfolio_selection.md](docs/differentiable_portfolio_selection.md) | Differentiable top-k selection for end-to-end portfolio optimization |

### Training Documentation (training/docs/)

| Document | Description |
|----------|-------------|
| [training/docs/BINNING_EXPLAINED.md](training/docs/BINNING_EXPLAINED.md) | Return binning for classification: adaptive vs fixed bins |
| [training/docs/NEW_FORMAT_README.md](training/docs/NEW_FORMAT_README.md) | HDF5 dataset format specification and data loading |
| [training/docs/SELECTION_AWARE_TRAINING_README.md](training/docs/SELECTION_AWARE_TRAINING_README.md) | Selection-aware training that accounts for portfolio construction |

### Inference Documentation (inference/)

| Document | Description |
|----------|-------------|
| [inference/BACKTEST_README.md](inference/BACKTEST_README.md) | Backtesting methodology and simulation details |
| [inference/PREDICT_README.md](inference/PREDICT_README.md) | Running predictions on new data |
| [inference/STATISTICAL_COMPARISON_README.md](inference/STATISTICAL_COMPARISON_README.md) | Comparing models with statistical significance |

### Reinforcement Learning Documentation (rl/docs/)

| Document | Description |
|----------|-------------|
| [rl/docs/README.md](rl/docs/README.md) | RL overview: Actor-Critic and DQN approaches |
| [rl/docs/EVALUATION_GUIDE.md](rl/docs/EVALUATION_GUIDE.md) | Evaluating RL trading agents |
| [rl/docs/PERFORMANCE_OPTIMIZATION.md](rl/docs/PERFORMANCE_OPTIMIZATION.md) | Speed optimization for RL training |
| [rl/docs/PRECOMPUTE_OPTIMIZATION.md](rl/docs/PRECOMPUTE_OPTIMIZATION.md) | Pre-computing state representations |
| [rl/docs/README_PRECOMPUTE.md](rl/docs/README_PRECOMPUTE.md) | Precomputation pipeline details |
| [rl/docs/PROFILING_GUIDE.md](rl/docs/PROFILING_GUIDE.md) | Profiling RL training bottlenecks |
| [rl/docs/TOP_K_FILTERING.md](rl/docs/TOP_K_FILTERING.md) | Filtering stocks for RL environment |
| [rl/docs/VECTORIZATION_SUMMARY.md](rl/docs/VECTORIZATION_SUMMARY.md) | Vectorized environment implementation |
| [rl/actor_critic/docs/ACTOR_CRITIC_GUIDE.md](rl/actor_critic/docs/ACTOR_CRITIC_GUIDE.md) | Actor-Critic architecture and training |
| [rl/actor_critic/docs/QUICK_START.md](rl/actor_critic/docs/QUICK_START.md) | Quick start for Actor-Critic training |
| [rl/evaluation/README.md](rl/evaluation/README.md) | RL evaluation metrics and tools |

### Clustering Documentation (cluster/docs/)

| Document | Description |
|----------|-------------|
| [cluster/README.md](cluster/README.md) | Cluster-based stock filtering overview |
| [cluster/docs/STATIC_VS_DYNAMIC.md](cluster/docs/STATIC_VS_DYNAMIC.md) | Static vs dynamic cluster assignment comparison |
| [cluster/docs/INFERENCE_INTEGRATION.md](cluster/docs/INFERENCE_INTEGRATION.md) | Integrating cluster filtering into inference |
| [cluster/docs/CACHING_GUIDE.md](cluster/docs/CACHING_GUIDE.md) | Pre-computing embeddings for fast filtering |

### Dataset Creation Documentation

| Document | Description |
|----------|-------------|
| [dataset_creation/CREATE_INFERENCE_DATASET.md](dataset_creation/CREATE_INFERENCE_DATASET.md) | Creating datasets for inference/production |

---

## Training

The main training script is `training/walk_forward_training.py`. It implements walk-forward validation where training data always precedes test data chronologically, preventing look-ahead bias.

### How Walk-Forward Validation Works

```
Timeline: ════════════════════════════════════════════════════════════►

Fold 1:   [████ Train ████][Gap][▓▓ Test ▓▓]
Fold 2:   [██████ Train ██████][Gap][▓▓ Test ▓▓]
Fold 3:   [████████ Train ████████][Gap][▓▓ Test ▓▓]
...
Fold N:   [██████████████ Train ██████████████][Gap][▓▓ Test ▓▓]
```

Each fold trains on all data up to a cutoff point, then tests on subsequent data with a gap to prevent leakage.

### Key Training Scripts

| Script | Description |
|--------|-------------|
| `training/walk_forward_training.py` | Main training with walk-forward validation, 50+ configurable arguments |
| `training/model.py` | `SimpleTransformerPredictor` architecture with configurable layers, heads, dropout |
| `training/baseline_models.py` | Ridge regression, LightGBM, MLP baselines for comparison |
| `training/hdf5_data_loader.py` | Efficient HDF5 data loading with optional preloading and caching |
| `training/selection_aware_training.py` | Experimental: training that accounts for top-k selection |

### Basic Training

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --num-folds 10 \
    --epochs-per-fold 5 \
    --hidden-dim 512 \
    --num-layers 4 \
    --batch-size 128 \
    --lr 1e-4 \
    --weight-decay 0.1 \
    --seed 42
```

### Training with Ranking Loss (Recommended)

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --ranking-loss-type listnet \
    --ranking-only \
    --ranking-margin 0.01
```

See [docs/listnet_loss.md](docs/listnet_loss.md) for details on ranking loss.

### Multi-GPU Training with DDP

```bash
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --ddp \
    --batch-size 256
```

### Fast Experimentation Mode

For quick iteration, use data subsampling:

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --data-fraction 0.1 \
    --incremental \
    --incremental-epochs 1 \
    --no-preload
```

### Training with Baseline Comparison

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --compare-models
```

This trains Ridge, LightGBM, and MLP baselines alongside the transformer for comparison.

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden-dim` | 512 | Transformer hidden dimension |
| `--num-layers` | 4 | Number of transformer layers |
| `--num-heads` | 8 | Number of attention heads |
| `--dropout` | 0.1 | Dropout rate |
| `--lr` | 1e-4 | Learning rate |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--batch-size` | 128 | Training batch size |
| `--seq-len` | 1536 | Input sequence length |
| `--epochs-per-fold` | 10 | Epochs per walk-forward fold |
| `--ranking-loss-type` | none | `listnet` or `pairwise` |
| `--seed` | 42 | Random seed for reproducibility |

See [docs/walk_forward_training.md](docs/walk_forward_training.md) for all 50+ arguments.

---

## Inference & Evaluation

### Key Evaluation Scripts

| Script | Description |
|--------|-------------|
| `inference/principled_evaluation.py` | Comprehensive evaluation with IC, IR, returns, turnover |
| `inference/backtest_simulation.py` | Full trading simulation with transaction costs |
| `inference/evaluate_walk_forward_checkpoints.py` | Evaluate all folds from walk-forward training |
| `inference/walk_forward_validation.py` | Out-of-sample validation framework |
| `inference/interactive_tc_analysis.py` | Interactive transaction cost analysis |
| `inference/statistical_comparison.py` | Compare models with significance testing |

### Evaluate Checkpoints

```bash
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --top-k 25 \
    --horizon-idx 1 \
    --sweep  # Test multiple top-k values
```

### Run Full Backtest

```bash
python -m inference.backtest_simulation \
    --checkpoint checkpoints/walk_forward/fold_0_best.pt \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --top-k 25 \
    --transaction-cost-bps 10 \
    --initial-capital 100000
```

See [inference/BACKTEST_README.md](inference/BACKTEST_README.md) for backtest methodology.

### Key Evaluation Metrics

| Metric | Formula | Good Value | Description |
|--------|---------|------------|-------------|
| **IC** | `corr(pred, actual)` | > 0.02 | Daily cross-sectional correlation |
| **IR** | `mean(IC) / std(IC)` | > 0.5 | Risk-adjusted IC (consistency) |
| **Sharpe** | `mean(ret) / std(ret) * √252` | > 1.0 | Annualized risk-adjusted return |
| **Turnover** | `changed_positions / total` | < 50% | Portfolio churn per rebalance |
| **Win Rate** | `profitable_trades / total` | > 52% | Percentage of winning trades |

See [docs/eval.md](docs/eval.md) for detailed metric explanations.

### Comparing Models

```bash
python inference/statistical_comparison.py \
    --baseline checkpoints/baseline \
    --treatment checkpoints/treatment \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5
```

See [inference/STATISTICAL_COMPARISON_README.md](inference/STATISTICAL_COMPARISON_README.md) for significance testing.

---

## Data Pipeline

The data pipeline collects financial data from multiple sources, engineers features, and creates HDF5 datasets for training.

### Data Collection Scripts

| Script | Description |
|--------|-------------|
| `data_scraping/fmp_comprehensive_scraper.py` | Financial Modeling Prep API: fundamentals, ratios, earnings |
| `data_scraping/yfinance_price_scraper.py` | Yahoo Finance: OHLCV price data |
| `data_scraping/news_scraper.py` | News article collection from multiple sources |
| `data_scraping/news_embedder.py` | Generate 768-dim news embeddings with Nomic AI |
| `data_scraping/market_indices_scraper.py` | S&P 500, VIX, sector ETFs, market breadth |
| `data_scraping/alphavantage_news_scraper.py` | Alpha Vantage news sentiment |
| `data_scraping/derived_features_calculator.py` | Technical indicators: RSI, MACD, Bollinger, etc. |

### Dataset Generation Scripts

| Script | Description |
|--------|-------------|
| `dataset_creation/generate_full_dataset.py` | End-to-end dataset generation pipeline |
| `dataset_creation/convert_to_hdf5.py` | Convert processed data to HDF5 format |
| `dataset_creation/extract_prices_to_hdf5.py` | Extract clean price data for backtesting |
| `dataset_creation/validate_price_data.py` | Validate price data quality and detect issues |

See [dataset_creation/CREATE_INFERENCE_DATASET.md](dataset_creation/CREATE_INFERENCE_DATASET.md) for creating inference datasets.

### Generate Dataset from Scratch

```bash
# 1. Scrape all data sources
python -m data_scraping.fmp_comprehensive_scraper
python -m data_scraping.yfinance_price_scraper
python -m data_scraping.news_scraper
python -m data_scraping.news_embedder

# 2. Calculate derived features
python -m data_scraping.derived_features_calculator

# 3. Generate full dataset
python -m dataset_creation.generate_full_dataset

# 4. Convert to HDF5
python -m dataset_creation.convert_to_hdf5

# 5. Extract clean prices
python -m dataset_creation.extract_prices_to_hdf5

# 6. Validate
python -m dataset_creation.validate_price_data
```

### Dataset Structure

```
all_complete_dataset.h5
├── AAPL/
│   ├── features    # (num_days, 1123) float32
│   └── dates       # (num_days,) string dates "YYYY-MM-DD"
├── GOOGL/
│   ├── features
│   └── dates
└── ... (748 tickers)

actual_prices_clean.h5
├── AAPL/
│   ├── prices      # (num_days,) float32 - split-adjusted close
│   └── dates       # (num_days,) string dates
└── ...
```

### Feature Groups (1123 total)

| Range | Count | Description |
|-------|-------|-------------|
| 0-99 | 100 | Price-derived: returns, volatility, momentum |
| 100-199 | 100 | Technical indicators: RSI, MACD, Bollinger, ATR |
| 200-299 | 100 | Fundamental ratios: P/E, P/B, ROE, debt ratios |
| 300-354 | 55 | Market features: sector returns, VIX, breadth |
| 355-1122 | 768 | News embeddings: Nomic AI text embeddings |

See [docs/features.md](docs/features.md) for complete feature descriptions.

---

## Sanity Checks & Ablation Studies

The `sanity_checks/` directory contains tools for validating data quality and running controlled hyperparameter experiments.

### Validation Scripts

| Script | Description |
|--------|-------------|
| `sanity_checks/check_dataset_quality.py` | Check for NaN, Inf, data ranges, missing values |
| `sanity_checks/check_predictive_signal.py` | Verify features have predictive signal (not noise) |
| `sanity_checks/feature_ir_analysis.py` | Calculate per-feature Information Ratios |
| `sanity_checks/validate_metrics.py` | Validate evaluation metric calculations |

### Ablation Framework

| Script | Description |
|--------|-------------|
| `sanity_checks/ablate.sh` | Generic ablation launcher (parallel GPU execution) |
| `sanity_checks/analyze_ablation_metrics.py` | Analyze and compare ablation runs with significance |
| `sanity_checks/hyperparameter_ablation.py` | Programmatic hyperparameter search |

### Run an Ablation Study

The `ablate.sh` script runs experiments in parallel across GPUs with deterministic seeding:

```bash
# Ablate hidden dimension across 3 GPUs
./sanity_checks/ablate.sh hidden-dim "256 512 768" "0 1 2"

# Ablate learning rate on 2 GPUs
./sanity_checks/ablate.sh lr "1e-5 5e-5 1e-4 5e-4" "0 1"

# Ablate weight decay
./sanity_checks/ablate.sh weight-decay "0.01 0.1 0.3" "0 1 2"

# Ablate sequence length
./sanity_checks/ablate.sh seq-len "256 512 1024" "0 1 2"
```

### Analyze Ablation Results

```bash
python sanity_checks/analyze_ablation_metrics.py \
    --runs checkpoints/ablation_hidden_dim/hidden-dim_256_* \
           checkpoints/ablation_hidden_dim/hidden-dim_512_* \
           checkpoints/ablation_hidden_dim/hidden-dim_768_* \
    --labels "256" "512" "768" \
    --output results/ablation_analysis
```

Outputs include:
- Statistical comparison with confidence intervals
- Performance plots across configurations
- Best configuration recommendation

### Feature Quality Analysis

```bash
# Check dataset for issues
python sanity_checks/check_dataset_quality.py \
    --data all_complete_dataset.h5

# Verify predictive signal exists
python sanity_checks/check_predictive_signal.py \
    --dataset all_complete_dataset.h5

# Analyze per-feature predictive power
python sanity_checks/feature_ir_analysis.py \
    --dataset all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --horizon-days 5 \
    --output results/feature_ir
```

---

## Reinforcement Learning (Experimental)

The `rl/` directory contains experimental RL approaches for portfolio management. Two main approaches are implemented:

### Directory Structure

```
rl/
├── actor_critic/           # Actor-Critic approach
│   ├── train_actor_critic.py
│   ├── inference_actor_critic.py
│   └── docs/
│       ├── ACTOR_CRITIC_GUIDE.md
│       └── QUICK_START.md
├── dqn/                    # Deep Q-Network approach
│   └── train_dqn_simple.py
├── core/                   # Shared components
│   ├── rl_environment.py   # Trading environment
│   ├── rl_components.py    # Neural network architectures
│   └── backtest.py         # RL-specific backtesting
├── utils/                  # Utility functions
├── evaluation/             # Evaluation tools
│   └── README.md
└── docs/                   # General RL documentation
    ├── README.md
    ├── EVALUATION_GUIDE.md
    ├── PERFORMANCE_OPTIMIZATION.md
    ├── PRECOMPUTE_OPTIMIZATION.md
    ├── README_PRECOMPUTE.md
    ├── PROFILING_GUIDE.md
    ├── TOP_K_FILTERING.md
    └── VECTORIZATION_SUMMARY.md
```

### Actor-Critic Training

```bash
python -m rl.actor_critic.train_actor_critic \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --predictor-checkpoint checkpoints/walk_forward/fold_0_best.pt \
    --episodes 1000 \
    --top-k 25
```

See [rl/actor_critic/docs/QUICK_START.md](rl/actor_critic/docs/QUICK_START.md) for quick start.

### DQN Training

```bash
python -m rl.dqn.train_dqn_simple \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --episodes 500
```

### RL Documentation

| Document | Description |
|----------|-------------|
| [rl/docs/README.md](rl/docs/README.md) | RL overview and approach comparison |
| [rl/docs/EVALUATION_GUIDE.md](rl/docs/EVALUATION_GUIDE.md) | Evaluating trained RL agents |
| [rl/docs/PERFORMANCE_OPTIMIZATION.md](rl/docs/PERFORMANCE_OPTIMIZATION.md) | Speed optimization techniques |
| [rl/docs/VECTORIZATION_SUMMARY.md](rl/docs/VECTORIZATION_SUMMARY.md) | Vectorized environment details |
| [rl/actor_critic/docs/ACTOR_CRITIC_GUIDE.md](rl/actor_critic/docs/ACTOR_CRITIC_GUIDE.md) | Actor-Critic architecture guide |

---

## Clustering (Experimental)

The `cluster/` module uses transformer embeddings to group stocks by learned representations, then filters to trade only stocks from profitable clusters.

### How It Works

1. **Encode**: Mean-pool transformer activations to get stock embeddings
2. **Cluster**: K-means clustering on embeddings
3. **Analyze**: Compute return/win rate per cluster
4. **Filter**: Trade only stocks from best-performing clusters

### Key Scripts

| Script | Description |
|--------|-------------|
| `cluster/create_clusters.py` | Encode dataset and create clusters (all-in-one) |
| `cluster/analyze_clusters.py` | Compute per-cluster performance metrics |
| `cluster/cluster_filter.py` | Static filtering (fixed cluster assignments) |
| `cluster/dynamic_cluster_filter.py` | Dynamic filtering (daily reassignment) |
| `cluster/cache_embeddings.py` | Pre-compute embeddings for fast filtering |
| `cluster/gpu_kmeans.py` | GPU-accelerated K-means |

### Create and Analyze Clusters

```bash
# Step 1: Create clusters
python -m cluster.create_clusters \
    --model-path checkpoints/fold_0_best.pt \
    --dataset-path all_complete_dataset.h5 \
    --n-clusters 50 \
    --output-dir cluster_results

# Step 2: Analyze cluster performance
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path all_complete_dataset.h5 \
    --prices-path actual_prices_clean.h5

# Step 3: Use in backtesting
python -m inference.backtest_simulation \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt
```

### Clustering Documentation

| Document | Description |
|----------|-------------|
| [cluster/README.md](cluster/README.md) | Clustering overview and pipeline |
| [cluster/docs/STATIC_VS_DYNAMIC.md](cluster/docs/STATIC_VS_DYNAMIC.md) | Static vs dynamic filtering comparison |
| [cluster/docs/INFERENCE_INTEGRATION.md](cluster/docs/INFERENCE_INTEGRATION.md) | Integrating filtering into inference |
| [cluster/docs/CACHING_GUIDE.md](cluster/docs/CACHING_GUIDE.md) | Pre-computing embeddings for speed |

---

## Utility Scripts

### scripts/

| Script | Description |
|--------|-------------|
| `scripts/download_data.py` | Download datasets from Hugging Face Hub |
| `scripts/upload_to_hf.py` | Upload datasets to Hugging Face Hub |

### Daily Predictions

| Script | Description |
|--------|-------------|
| `daily_predictions/auto_daily_predictions_enhanced.py` | Generate daily stock predictions for production |

---

## Key Metrics Reference

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| **IC** | `corr(predicted_rank, actual_return)` | > 0.02 | Higher = better ranking predictions |
| **IR** | `mean(IC) / std(IC)` | > 0.5 | Higher = more consistent signal |
| **Sharpe** | `(mean_return - rf) / std_return * √252` | > 1.0 | Risk-adjusted annual return |
| **Turnover** | `Σ|w_new - w_old| / 2` | < 0.5 | Portfolio churn (lower = cheaper) |
| **Win Rate** | `profitable_trades / total_trades` | > 52% | Percentage of winning trades |
| **Max Drawdown** | `max(peak - trough) / peak` | < 20% | Worst peak-to-trough decline |

---

## Data Files

| File | Size | Description |
|------|------|-------------|
| `all_complete_dataset.h5` | ~30GB | Features: 1123 features × 748 stocks × 9000+ days |
| `actual_prices_clean.h5` | ~80MB | Split-adjusted close prices for backtesting |
| `adaptive_bin_edges.pt` | ~1KB | Learned bin edges for return classification |

---

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+ (with CUDA support recommended)
- **GPU**: CUDA-capable GPU with 8GB+ VRAM
- **RAM**: 32GB+ for full dataset loading
- **Disk**: 35GB+ for data files

### Key Dependencies

```
torch>=2.0.0
numpy
pandas
h5py
scipy
scikit-learn
lightgbm
matplotlib
tqdm
```

See [requirements.txt](requirements.txt) for complete list.

---

## Reproducibility

All experiments use deterministic seeding for reproducibility:

```bash
python -m training.walk_forward_training \
    --seed 42 \
    ...
```

The training script sets:
- `random.seed()`
- `np.random.seed()`
- `torch.manual_seed()`
- `torch.cuda.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`

---

## License

[Add license information]

## Citation

[Add citation information if applicable]
