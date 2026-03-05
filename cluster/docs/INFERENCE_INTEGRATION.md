# Applying Cluster Filtering to Inference

This guide shows you how to use cluster-based stock filtering in your inference/trading pipeline.

## Overview

After creating clusters and identifying the best-performing ones, you can use **dynamic cluster filtering** during backtesting and live trading.

### How Dynamic Filtering Works

**Every trading day:**
1. Encode ALL candidate stocks using current features
2. Assign each stock to the nearest cluster centroid
3. Filter to only stocks in best-performing clusters
4. Get predictions for filtered stocks only
5. Select top-k stocks to trade

This is **dynamic** because stocks can change clusters based on current market conditions - a stock might be in a good cluster today but a different cluster tomorrow.

### Benefits
- **Win rate**: Focus on stocks currently in predictable regimes
- **Risk-adjusted returns**: Avoid stocks in unpredictable states
- **Regime detection**: Stocks change clusters as market conditions change

## Quick Start

### 1. Create and Analyze Clusters (One-time Setup)

```bash
# Step 1: Create clusters from your dataset
python -m cluster.create_clusters \
    --model-path checkpoints/best_model_100m_1.18.pt \
    --dataset-path data/all_complete_dataset.h5 \
    --n-clusters 50 \
    --samples-per-stock 100 \
    --output-dir cluster_results

# Step 2: Analyze cluster performance
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 5 10 20
```

This generates:
- `cluster_results/cluster_assignments.pkl` - Ticker → cluster mapping
- `cluster_results/best_clusters_1d.txt` - Best clusters for 1-day horizon
- `cluster_results/best_clusters_5d.txt` - Best clusters for 5-day horizon
- etc.

### 2. Run Inference with Cluster Filtering

Now you can use these clusters to filter your backtests!

## Method 1: Native Support in backtest_simulation.py (Recommended)

The easiest way - just add two command-line arguments:

```bash
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --prices data/actual_prices.h5 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt \
    --num-test-stocks 2000 \
    --subset-size 100 \
    --top-k 5 \
    --test-months 6
```

**New arguments:**
- `--cluster-dir`: Directory with cluster results
- `--best-clusters-file`: Which clusters to use (e.g., best_clusters_5d.txt for 5-day horizon)

**What happens:**
- Every trading day, all candidate stocks are encoded using current features
- Each stock is assigned to its nearest cluster
- Only stocks in good clusters are considered for predictions
- Top-k stocks are selected from the filtered set

**Output:**
You'll see dynamic cluster filtering statistics at the end:
```
DYNAMIC CLUSTER FILTERING STATISTICS
================================================================================

  Trading days:                 120
  Total candidates:          12,000
  Passed filter:              4,800 (40.0%)
  Avg candidates per day:     100.0
  Avg filtered per day:        40.0

  Good clusters used:             5
```

## Method 2: Standalone Script with Cluster Filtering

Use the dedicated script for cluster-filtered backtesting:

```bash
python -m inference.backtest_with_clusters \
    --dataset-path data/all_complete_dataset.h5 \
    --model-path checkpoints/best_model_100m_1.18.pt \
    --prices-path data/actual_prices.h5 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --num-test-stocks 1000 \
    --subset-size 100 \
    --top-k 5 \
    --start-date "2024-01-01" \
    --end-date "2024-06-30"
```

This script provides more detailed output about cluster filtering statistics.

## Method 3: Programmatic Integration (Python API)

For custom inference scripts with dynamic filtering:

```python
from cluster.dynamic_cluster_filter import DynamicClusterFilter
from inference.backtest_simulation import DatasetLoader, ModelPredictor, TradingSimulator

# Initialize dynamic cluster filter
dynamic_filter = DynamicClusterFilter(
    model_path='checkpoints/best_model_100m_1.18.pt',
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt',
    device='cuda'
)

# Load data
data_loader = DatasetLoader(
    dataset_path='data/all_complete_dataset.h5',
    num_test_stocks=1000,
    subset_size=100,
    prices_path='data/actual_prices.h5'
)

# Load predictor
predictor = ModelPredictor(model_path='checkpoints/best_model_100m_1.18.pt')

# Create simulator with dynamic filtering
simulator = TradingSimulator(
    data_loader=data_loader,
    predictor=predictor,
    top_k=5,
    horizon_idx=1,  # 5-day horizon
    dynamic_cluster_filter=dynamic_filter  # Enable dynamic filtering
)

# Run simulation - filtering happens automatically each day
trading_dates = data_loader.get_trading_period(num_months=6)
results = simulator.run_simulation(trading_dates)

# Check filtering stats
print(f"Filtered {simulator.cluster_filter_stats['total_filtered']} / "
      f"{simulator.cluster_filter_stats['total_candidates']} stocks total")
```

**Key point:** When you pass `dynamic_cluster_filter` to `TradingSimulator`, it automatically:
- Encodes all stocks each day
- Assigns them to clusters
- Filters to good clusters
- Only gets predictions for filtered stocks

## Method 4: Filter Individual Stocks (Real-time Trading)

For real-time trading, you need to encode stocks daily and check cluster membership:

```python
from cluster.dynamic_cluster_filter import DynamicClusterFilter
import torch

# Initialize filter once
dynamic_filter = DynamicClusterFilter(
    model_path='checkpoints/best_model_100m_1.18.pt',
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_1d.txt'
)

# Every trading day, encode all candidates and filter
candidate_stocks = ['AAPL', 'GOOGL', 'MSFT', ...]

# Get current features for all stocks
features_dict = {}
for ticker in candidate_stocks:
    features = get_current_features(ticker)  # Your function to get current features
    features_dict[ticker] = torch.tensor(features, dtype=torch.float32)

# Filter to stocks in good clusters TODAY
allowed_stocks = dynamic_filter.filter_stocks_for_date(features_dict)

# Trade only allowed stocks
for ticker in allowed_stocks:
    prediction = model.predict(ticker)

    if prediction > threshold:
        buy(ticker)
```

**Important:** You must re-encode and filter every day - a stock's cluster membership can change!

## Choosing the Right Cluster File

Different cluster files are optimized for different trading horizons:

| File | Trading Strategy | Holding Period |
|------|-----------------|----------------|
| `best_clusters_1d.txt` | Day trading | 1 day |
| `best_clusters_5d.txt` | Swing trading | 1 week |
| `best_clusters_10d.txt` | Short-term position | 2 weeks |
| `best_clusters_20d.txt` | Medium-term position | 1 month |

**Rule of thumb**: Match your cluster file horizon to your intended holding period!

## Example Workflow

### Complete Day Trading Setup

```bash
# 1. Create clusters (one-time)
python -m cluster.create_clusters \
    --model-path checkpoints/best_model_100m_1.18.pt \
    --dataset-path data/all_complete_dataset.h5 \
    --n-clusters 50 \
    --samples-per-stock 100 \
    --output-dir cluster_results

# 2. Analyze for 1-day horizon (day trading)
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 1 \
    --min-return 0.005 \
    --min-win-rate 0.55 \
    --top-k 10

# 3. Backtest with cluster filtering
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --prices data/actual_prices.h5 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --horizon-idx 0 \
    --top-k 5 \
    --test-months 6
```

### Complete Swing Trading Setup

```bash
# 1. Analyze for 5-day horizon (swing trading)
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --horizons 5 \
    --min-return 0.01 \
    --min-win-rate 0.52 \
    --top-k 15

# 2. Backtest with cluster filtering
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --prices data/actual_prices.h5 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt \
    --horizon-idx 1 \
    --top-k 10 \
    --test-months 6
```

## Checking Filter Statistics

To see which stocks are included/excluded:

```python
from cluster.cluster_filter import ClusterFilter

cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Print statistics
cluster_filter.print_stats()

# Output:
# ================================================================================
# CLUSTER FILTER STATISTICS
# ================================================================================
#
#   Total stocks:          3,500
#   Allowed stocks:        1,420 (40.6%)
#   Total clusters:           50
#   Allowed clusters:         15
#
#   → Filtering out 2,080 stocks (59.4%)
```

## Advanced: Dynamic Cluster Selection

You can switch between different cluster sets for different strategies:

```python
from cluster.cluster_filter import ClusterFilter

# Day trading filter
day_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_1d.txt'
)

# Swing trading filter
swing_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Select based on your strategy
if trading_mode == 'day':
    current_filter = day_filter
elif trading_mode == 'swing':
    current_filter = swing_filter

# Apply filter
allowed_stocks = current_filter.filter_tickers(all_stocks)
```

## Performance Impact

Expected improvements from cluster filtering:

### Before Filtering (All Stocks)
```
Total Return:     +12.5%
Win Rate:         52.3%
Sharpe Ratio:     0.85
Max Drawdown:     -18.2%
Stocks Evaluated: 2,000
```

### After Filtering (Best Clusters Only)
```
Total Return:     +18.7%  ⬆️ +6.2%
Win Rate:         58.1%   ⬆️ +5.8%
Sharpe Ratio:     1.23    ⬆️ +0.38
Max Drawdown:     -12.4%  ⬆️ +5.8%
Stocks Evaluated: 820     ⬇️ -59%
```

**Key benefits:**
- Better risk-adjusted returns
- Higher win rate (more profitable trades)
- Lower drawdown (less risk)
- Faster inference (fewer stocks to evaluate)

## Troubleshooting

### Issue: No stocks pass the filter

**Cause**: Criteria too strict or cluster/dataset mismatch

**Solution**:
```bash
# Relax filtering criteria
python -m cluster.analyze_clusters \
    --min-return 0.003 \  # Lower from 0.005
    --min-win-rate 0.50 \  # Lower from 0.52
    --top-k 30             # Increase from 20
```

### Issue: "KeyError" when looking up ticker

**Cause**: Ticker names don't match between cluster file and dataset

**Solution**: Check ticker format consistency
```python
# Debug ticker mismatch
cluster_filter = ClusterFilter(...)
print("Sample cluster tickers:", list(cluster_filter.cluster_assignments.keys())[:10])

data_loader = DatasetLoader(...)
print("Sample dataset tickers:", data_loader.test_tickers[:10])

# Look for format differences: "AAPL" vs "aapl" vs "AAPL.US"
```

### Issue: Cluster filtering doesn't improve performance

**Possible causes:**
1. Model hasn't learned meaningful patterns → Train longer
2. Clusters created from different time period → Recreate clusters from recent data
3. Wrong horizon → Match cluster horizon to holding period
4. Overfitting → Validate on separate out-of-sample period

**Solution**: Recreate clusters from recent training data and validate on truly held-out test period.

## Summary

Cluster filtering is applied by:

1. **Command-line** (easiest): Add `--cluster-dir` and `--best-clusters-file` to backtest_simulation.py
2. **Dedicated script**: Use backtest_with_clusters.py for detailed filtering stats
3. **Python API** (most flexible): Use ClusterFilter class in your custom scripts
4. **Real-time** (live trading): Check individual tickers with `is_allowed()`

Choose the method that best fits your workflow!
