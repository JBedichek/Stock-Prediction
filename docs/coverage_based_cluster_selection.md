# Coverage-Based Cluster Selection

## Overview

The cluster analysis system now supports **coverage-based selection** as an alternative to threshold-based filtering. Instead of using arbitrary thresholds (minimum return, win rate, Sharpe ratio), you can now select the top-performing clusters that collectively cover a target percentage of your total dataset.

## Motivation

**Threshold-Based Selection (Original):**
- Requires tuning multiple hyperparameters (min return, min win rate, min Sharpe)
- Number of selected clusters varies unpredictably
- Difficult to know what threshold values to use
- Can select too few or too many clusters depending on thresholds

**Coverage-Based Selection (New, Recommended):**
- Simply specify target coverage percentage (e.g., "top 30% of stocks")
- Rank clusters by performance metric (mean return, Sharpe ratio, etc.)
- Select top clusters until target coverage is reached
- Consistent, data-driven approach with one intuitive parameter

## Usage

### Command Line

Use the `--top-k-percent-coverage` flag with `analyze_clusters.py`:

```bash
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --top-k-percent-coverage 0.3 \  # Select top 30% of stocks
    --ranking-metric mean_return \   # Rank by mean return
    --output-dir cluster_results
```

**Parameters:**
- `--top-k-percent-coverage`: Float between 0.0 and 1.0
  - `0.1` = select clusters covering top 10% of stocks
  - `0.3` = select clusters covering top 30% of stocks (recommended)
  - `0.5` = select clusters covering top 50% of stocks
  - `None` (omit flag) = use threshold-based selection (legacy)

- `--ranking-metric`: Metric to rank clusters by
  - `mean_return` (default) - average return
  - `sharpe` - risk-adjusted return
  - `median_return` - median return
  - `win_rate` - probability of profit

### Python API

```python
from cluster.analyze_clusters import ClusterAnalyzer

analyzer = ClusterAnalyzer(cluster_dir='cluster_results')
cluster_stats = analyzer.compute_returns(dataset_path, prices_path)

# Coverage-based selection (recommended)
best_clusters = analyzer.identify_best_clusters(
    cluster_stats,
    horizon=5,
    top_k_percent_coverage=0.3,  # Top 30% coverage
    ranking_metric='mean_return'
)

# Threshold-based selection (legacy)
best_clusters = analyzer.identify_best_clusters(
    cluster_stats,
    horizon=5,
    min_return=0.01,
    min_win_rate=0.5,
    min_sharpe=0.1,
    top_k=10
)
```

## How It Works

### Algorithm

1. **Calculate total stocks**: Count all unique stocks across all clusters
2. **Compute target**: `target_stocks = total_stocks × top_k_percent_coverage`
3. **Rank clusters**: Sort clusters by chosen metric (descending - best first)
4. **Accumulate coverage**: Iterate through ranked clusters, adding stock counts
5. **Stop at target**: Select clusters until cumulative stocks ≥ target

### Example

Suppose you have:
- 100 total clusters
- 1000 total unique stocks across all clusters
- `top_k_percent_coverage=0.3` (30%)
- `ranking_metric='mean_return'`

The algorithm will:
1. Calculate target: 1000 × 0.3 = 300 stocks
2. Rank all 100 clusters by mean return (best to worst)
3. Select clusters starting from the top:
   - Cluster 42: 50 stocks (cumulative: 50)
   - Cluster 17: 80 stocks (cumulative: 130)
   - Cluster 5: 60 stocks (cumulative: 190)
   - Cluster 91: 70 stocks (cumulative: 260)
   - Cluster 33: 55 stocks (cumulative: 315) ✓ Target reached!
4. Save clusters [42, 17, 5, 91, 33] as "best clusters"

## Benefits

1. **Intuitive Parameter**: "Top 30% of stocks" is easier to understand than "min return 0.01, min win rate 0.5, min Sharpe 0.1"
2. **Consistent Coverage**: Always get approximately the same proportion of dataset
3. **No Threshold Tuning**: No need to guess appropriate threshold values
4. **Data-Driven**: Adapts automatically to cluster performance distribution
5. **Flexible Ranking**: Can rank by any metric (return, Sharpe, win rate, etc.)

## Comparison: Threshold vs. Coverage

| Aspect | Threshold-Based | Coverage-Based |
|--------|----------------|----------------|
| Parameters | 3+ thresholds to tune | 1 intuitive percentage |
| Complexity | High (multiple hyperparameters) | Low (single parameter) |
| Consistency | Variable # of clusters | Consistent coverage |
| Interpretability | "Min 1% return, 50% win rate" | "Top 30% of stocks" |
| Adaptability | Fixed thresholds | Adapts to distribution |
| Use Case | Fine-grained control | General use (recommended) |

## Recommended Values

- **Conservative** (high quality): `0.1` - `0.2` (10-20% coverage)
- **Balanced**: `0.3` - `0.4` (30-40% coverage) ← **Recommended**
- **Aggressive** (more diversity): `0.5` - `0.7` (50-70% coverage)

## Output Files

The analysis saves several files to the output directory:

```
cluster_results/
├── best_clusters_1d.txt          # Best clusters for 1-day horizon
├── best_clusters_5d.txt          # Best clusters for 5-day horizon
├── best_clusters_10d.txt         # Best clusters for 10-day horizon
├── best_clusters_20d.txt         # Best clusters for 20-day horizon
├── cluster_ranking_1d.csv        # Full cluster rankings
├── cluster_performance_1d.png    # Visualizations
└── cluster_analysis.pkl          # Full analysis results
```

Each `best_clusters_*d.txt` file contains the cluster IDs to use for filtering:
```
# Best clusters for 5-day horizon
# Total: 5 clusters
42
17
5
91
33
```

## Integration with Backtesting

Once you've identified best clusters, use them with the cluster filter:

```bash
# 1. Analyze clusters and identify best ones
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path data/all_complete_dataset.h5 \
    --prices-path data/actual_prices.h5 \
    --top-k-percent-coverage 0.3

# 2. Run backtest with cluster filtering
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model.pt \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --cluster-top-k-percent 0.5 \  # Optional: use soft selection
    --output results.pt
```

## Example Scripts

See `examples/analyze_clusters_coverage.sh` for complete examples:
- Coverage-based selection with different percentages
- Ranking by different metrics (return, Sharpe, etc.)
- Comparison with threshold-based selection

## Performance Considerations

- **Speed**: Same as threshold-based (both iterate through all clusters)
- **Memory**: Minimal overhead (stores cluster metadata)
- **Scalability**: O(n log n) for sorting clusters, very fast even with 1000+ clusters

## Migration from Threshold-Based

If you're currently using threshold-based selection:

**Before:**
```bash
python -m cluster.analyze_clusters \
    --min-return 0.01 \
    --min-win-rate 0.5 \
    --min-sharpe 0.1 \
    --top-k 10
```

**After:**
```bash
python -m cluster.analyze_clusters \
    --top-k-percent-coverage 0.3 \
    --ranking-metric mean_return
```

**Benefits of switching:**
- One parameter instead of four
- More predictable results
- Easier to understand and tune
- More robust to cluster distribution changes

## Advanced: Custom Ranking Metrics

You can rank clusters by any available metric:

```python
# Available metrics:
# - mean_return: Average return (default)
# - median_return: Median return (more robust to outliers)
# - sharpe: Risk-adjusted return
# - win_rate: Probability of profit
# - skewness: Distribution asymmetry
# - volatility: Return standard deviation (lower is better - use negative for ranking)

# Example: Rank by Sharpe ratio
best_clusters = analyzer.identify_best_clusters(
    cluster_stats,
    horizon=5,
    top_k_percent_coverage=0.3,
    ranking_metric='sharpe'
)
```

## Summary

Coverage-based cluster selection provides a simpler, more intuitive way to identify profitable stock clusters:

1. **One parameter** instead of multiple thresholds
2. **Data-driven** approach that adapts to cluster distribution
3. **Consistent coverage** of your dataset
4. **Easy to understand**: "Select top 30% of stocks by mean return"
5. **Backward compatible**: Threshold mode still available if needed

For most use cases, we recommend using coverage-based selection with 30-40% coverage and mean return ranking.
