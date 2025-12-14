# Top-K Percent Cluster Filtering

## Overview

The cluster filtering system now supports **top-k percent selection** as an alternative to hard cluster assignment. Instead of only trading stocks that are assigned to specific "best" clusters, you can now select the top-k% of stocks closest to the best cluster centroids.

## Motivation

**Hard Cluster Assignment (Original Behavior):**
- Stocks are assigned to their nearest cluster centroid
- Only stocks in "best performing" clusters are considered for trading
- Binary decision: a stock is either IN or OUT
- Can be too restrictive, especially with small numbers of best clusters

**Top-K Percent Selection (New Behavior):**
- Compute distance from each stock to all best cluster centroids
- Rank stocks by their minimum distance to any best cluster centroid
- Select the top-k% closest stocks
- More flexible: includes stocks near cluster boundaries
- Better control over the number of candidate stocks

## Usage

### Command Line

Add the `--cluster-top-k-percent` flag to `backtest_simulation.py`:

```bash
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_1d.txt \
    --cluster-top-k-percent 0.3 \  # Select top 30% by distance
    --embeddings-cache data/embeddings_cache.h5 \
    --output results.pt
```

**Parameters:**
- `--cluster-top-k-percent`: Float between 0.0 and 1.0
  - `0.1` = top 10% of stocks closest to best centroids
  - `0.3` = top 30% of stocks closest to best centroids
  - `None` (omit flag) = hard cluster assignment (original behavior)

### Python API

```python
from cluster.dynamic_cluster_filter import DynamicClusterFilter

# Top-k percent selection
filter = DynamicClusterFilter(
    model_path='checkpoints/best_model.pt',
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_1d.txt',
    top_k_percent=0.3  # Select top 30%
)

# Hard assignment (original behavior)
filter = DynamicClusterFilter(
    model_path='checkpoints/best_model.pt',
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_1d.txt',
    top_k_percent=None  # Hard assignment
)

# Use the filter
features_dict = {...}  # {ticker: features}
allowed_stocks = filter.filter_stocks_for_date(features_dict)
```

## How It Works

### Distance Computation

1. **Encode stocks**: Convert stock features to embeddings using the transformer model
2. **Transform embeddings**: Apply the same scaler/PCA used during clustering
3. **Compute distances**: Calculate Euclidean distance from each stock to all cluster centroids
4. **Find minimum distance**: For each stock, find its distance to the nearest "best" cluster centroid
5. **Rank and select**: Sort stocks by distance (ascending) and select top-k%

### Example

Suppose you have:
- 100 candidate stocks on a trading day
- 3 best cluster centroids (clusters 5, 12, 31)
- `top_k_percent=0.3` (30%)

The filter will:
1. Compute distance from each of 100 stocks to centroids 5, 12, and 31
2. For each stock, take the minimum distance (closest best centroid)
3. Sort the 100 stocks by this minimum distance
4. Select the top 30 stocks (30% of 100)

## Benefits

1. **More Candidates**: Doesn't exclude stocks near cluster boundaries
2. **Finer Control**: Tune the percentage to balance quality vs. diversity
3. **Better Adaptation**: Works well even with small numbers of best clusters
4. **Backward Compatible**: Set `top_k_percent=None` for original behavior

## Comparison: Hard Assignment vs. Top-K Percent

| Aspect | Hard Assignment | Top-K Percent |
|--------|----------------|---------------|
| Selection | Binary (in/out) | Continuous ranking |
| Flexibility | Fixed by cluster boundaries | Adjustable via percentage |
| Candidates | Can vary widely per day | Consistent proportion |
| Sensitivity | High to cluster definition | Lower to cluster boundaries |
| Use Case | Well-separated clusters | Overlapping clusters |

## Recommended Values

- **Conservative** (high quality): `0.1` - `0.2` (10-20%)
- **Balanced**: `0.3` - `0.4` (30-40%)
- **Aggressive** (more diversity): `0.5` - `0.7` (50-70%)

## Performance Considerations

- **Speed**: Same as hard assignment (both compute embeddings)
- **With Cache**: Very fast (pre-computed embeddings from HDF5)
- **Without Cache**: Encoding dominates runtime (distance computation is negligible)

## Testing

Run the test script to verify the implementation:

```bash
python test_cluster_top_k.py
```

This will:
- Test top-k percent selection with various percentages
- Compare with hard cluster assignment
- Verify the correct proportion of stocks is selected

## Example Scripts

See `examples/backtest_with_cluster_top_k.sh` for complete examples comparing:
- Top 30% selection
- Top 50% selection
- Hard assignment (baseline)

## Implementation Details

**Files Modified:**
- `cluster/dynamic_cluster_filter.py`:
  - Added `top_k_percent` parameter to `__init__()`
  - Added `compute_distances_to_centroids()` method
  - Modified `filter_stocks_for_date()` to support both modes

- `inference/backtest_simulation.py`:
  - Added `--cluster-top-k-percent` command line argument
  - Passes parameter to `DynamicClusterFilter` initialization

**Key Methods:**
- `compute_distances_to_centroids(embeddings_dict)`: Returns `{ticker: {cluster_id: distance}}`
- `filter_stocks_for_date(features_dict, top_k_percent)`: Applies distance-based selection when `top_k_percent` is set
