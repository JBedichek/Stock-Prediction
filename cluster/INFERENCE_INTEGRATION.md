# Applying Cluster Filtering to Inference

This guide shows you how to use cluster-based stock filtering in your inference/trading pipeline.

## Overview

After creating clusters and identifying the best-performing ones, you can filter your inference to only consider stocks from those clusters. This improves:
- **Win rate**: Focus on stocks the model predicts well
- **Risk-adjusted returns**: Avoid unpredictable stocks
- **Efficiency**: Smaller search space = faster inference

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

## Method 1: Native Support in backtest_simulation.py

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

The script will automatically filter the stocks before backtesting!

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

For custom inference scripts:

```python
from cluster.cluster_filter import ClusterFilter
from inference.backtest_simulation import DatasetLoader, ModelPredictor, BacktestSimulator

# Initialize cluster filter
cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Load your data
data_loader = DatasetLoader(
    dataset_path='data/all_complete_dataset.h5',
    num_test_stocks=1000,
    subset_size=100,
    prices_path='data/actual_prices.h5'
)

# Apply cluster filtering
original_count = len(data_loader.test_tickers)
filtered_tickers = cluster_filter.filter_tickers(data_loader.test_tickers)

# Update data loader with filtered tickers
data_loader.test_tickers = filtered_tickers
data_loader.full_pool = filtered_tickers.copy()

print(f"Filtered: {original_count} → {len(filtered_tickers)} stocks")

# Now use data_loader as normal
predictor = ModelPredictor(model_path='checkpoints/best_model.pt')
simulator = BacktestSimulator(data_loader, predictor, top_k=5)
results = simulator.run_backtest(start_date='2024-01-01', end_date='2024-06-30')
```

## Method 4: Filter Individual Stocks (Real-time Trading)

For real-time trading where you check stocks one by one:

```python
from cluster.cluster_filter import ClusterFilter

# Initialize filter once
cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_1d.txt'
)

# Check individual stocks
for ticker in candidate_stocks:
    if cluster_filter.is_allowed(ticker):
        # This stock is in a good cluster - consider trading it
        prediction = model.predict(ticker)

        if prediction > threshold:
            buy(ticker)
    else:
        # This stock is NOT in a good cluster - skip it
        continue
```

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
