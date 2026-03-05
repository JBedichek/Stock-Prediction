# Cluster-Based Stock Filtering

The idea here is to use the mean pooled transformer embeddings to build a set of k-means cluster, then analyze clusters of latent representation to find "favorable" clusters which fulfiill either one of two criteria:  

1. High return (mean, median return of stocks assigned to this cluster)
2. Predictability (model attains comparitively lower losses on predictions from within this cluster)

These two qualities are chosen because of their expected effect on trading.

This process can sucessfully identify clusters of high value, however it is dependent on the quality of the representation of the transformer.  The transfer of knowledge from past stock movements to future ones is difficult for classical models like the ones used here (At least within the scope of what I've tried), so this is not as magic a bullet as some of the cluster stats may indicate, because the clustering is "trained" on past data and doesn't generalize perfectly.  

## Overview


### How It Works

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CLUSTER FILTERING PIPELINE                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ENCODE                    2. CLUSTER                 3. ANALYZE          │
│  ┌─────────────┐              ┌─────────────┐            ┌─────────────┐     │
│  │ Transformer │──────────────│   K-Means   │────────────│  Backtest   │     │
│  │ Mean Pool   │  Embeddings  │  Clustering │  Clusters  │  Per Cluster│     │
│  └─────────────┘              └─────────────┘            └─────────────┘     │
│        ↑                                                        │            │
│    Features                                                Best Clusters     │
│                                                                 ↓            │
│                                                          ┌───────────────┐   │
│  4. FILTER                                               │Cache centroids│   │
│  During inference, assign stocks to nearest   ──────────>│ + best cluster│   │
│  centroid and filter to best clusters                    │     IDs       │   │
│                                                          └───────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Step 1: Create clusters from trained model
python -m cluster.create_clusters \
    --model-path checkpoints/fold_0_best.pt \
    --dataset-path all_complete_dataset.h5 \
    --n-clusters 50 \
    --output-dir cluster_results

# Step 2: Analyze which clusters are profitable
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path all_complete_dataset.h5 \
    --prices-path actual_prices_clean.h5

# Step 3: Use filter during inference
python -m inference.backtest_simulation \
    --cluster-filter-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt
```

## Scripts

| Script | Description |
|--------|-------------|
| `create_clusters.py` | Encode dataset and create clusters using K-means (all-in-one) |
| `analyze_clusters.py` | Compute returns, win rate, Sharpe for each cluster |
| `cluster_filter.py` | Static filter - fixed cluster assignments per stock |
| `dynamic_cluster_filter.py` | Dynamic filter - reassigns clusters daily based on current features |
| `cache_embeddings.py` | Pre-compute and cache embeddings for faster filtering |
| `compute_embeddings_lowmem.py` | Compute embeddings only (for large datasets - low memory workflow) |
| `cluster_embeddings_lowmem.py` | Cluster pre-computed embeddings (second step of low-memory workflow) |
| `gpu_kmeans.py` | GPU-accelerated K-means implementation |

## Detailed Documentation

| Document | Description |
|----------|-------------|
| [docs/STATIC_VS_DYNAMIC.md](docs/STATIC_VS_DYNAMIC.md) | Comparison of static vs dynamic cluster assignment |
| [docs/INFERENCE_INTEGRATION.md](docs/INFERENCE_INTEGRATION.md) | How to integrate filtering into inference pipelines |
| [docs/CACHING_GUIDE.md](docs/CACHING_GUIDE.md) | Pre-computing embeddings for production |

---

## Step 1: Create Clusters

Encode the dataset using transformer mean pooling and cluster the embeddings.

```bash
python -m cluster.create_clusters \
    --model-path checkpoints/fold_0_best.pt \
    --dataset-path all_complete_dataset.h5 \
    --n-clusters 50 \
    --method kmeans \
    --samples-per-stock 10 \
    --standardize \
    --output-dir cluster_results \
    --visualize
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | required | Path to trained model checkpoint |
| `--dataset-path` | required | Path to HDF5 dataset |
| `--n-clusters` | 50 | Number of clusters |
| `--method` | kmeans | Clustering method: kmeans, dbscan, agglomerative |
| `--samples-per-stock` | 10 | Random timesteps per stock for temporal diversity |
| `--standardize` | False | Standardize embeddings before clustering |
| `--use-pca` | False | Use PCA dimensionality reduction |
| `--pca-components` | 50 | Number of PCA components |

### Temporal Sampling

To build robust clusters, each stock is encoded at multiple random timesteps:

- Captures behavior across different market regimes
- A stock might appear in different clusters at different times
- Analysis aggregates by base ticker for performance metrics

### Output Files

```
cluster_results/
├── cluster_assignments.pkl    # {ticker: cluster_id}
├── embeddings.pkl             # {ticker: embedding_vector}
├── clustering_model.pkl       # Fitted K-means model
├── scaler.pkl                 # Fitted StandardScaler
└── cluster_visualization.png  # PCA projection plot
```

---

## Step 2: Analyze Clusters

Compute performance metrics for each cluster to identify profitable ones.

```bash
python -m cluster.analyze_clusters \
    --cluster-dir cluster_results \
    --dataset-path all_complete_dataset.h5 \
    --prices-path actual_prices_clean.h5 \
    --horizons 1 5 10 20 \
    --min-return 0.005 \
    --min-win-rate 0.52 \
    --min-sharpe 0.1 \
    --top-k 20
```

### Metrics Computed Per Cluster

- **Mean Return**: Average return at each horizon (1d, 5d, 10d, 20d)
- **Win Rate**: Probability of positive return
- **Sharpe Ratio**: Risk-adjusted return
- **Volatility**: Standard deviation of returns
- **Skewness**: Return distribution asymmetry

### Selection Criteria

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-return` | 0.005 | Minimum mean return (0.5%) |
| `--min-win-rate` | 0.52 | Minimum win rate (52%) |
| `--min-sharpe` | 0.1 | Minimum Sharpe ratio |
| `--top-k` | 20 | Select top K clusters by return |

### Output Files

```
cluster_results/
├── cluster_ranking_5d.csv      # Ranked clusters for 5-day horizon
├── best_clusters_5d.txt        # Best cluster IDs (one per line)
├── cluster_performance_5d.png  # Performance visualization
└── cluster_analysis.pkl        # Full analysis results
```

### Example Output

```
Top 10 Clusters (5-day returns):
cluster_id  num_stocks  mean_return  win_rate  sharpe  volatility
        23         127        0.0245     0.587   0.452      0.0321
         8         156        0.0218     0.564   0.411      0.0298
        41          98        0.0203     0.571   0.398      0.0287
```

---

## Step 3: Filter During Inference

### Static Filtering

Use fixed cluster assignments (stock always in same cluster):

```python
from cluster import ClusterFilter

# Initialize filter
cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Filter list of tickers
all_tickers = ['AAPL', 'GOOGL', 'MSFT', ...]
allowed_tickers = cluster_filter.filter_tickers(all_tickers)

# Check individual ticker
if cluster_filter.is_allowed('AAPL'):
    trade('AAPL')

# Get statistics
stats = cluster_filter.get_stats()
print(f"Filtering {stats['total_stocks']} → {stats['allowed_stocks']} stocks")
```

### Dynamic Filtering

Reassign clusters daily based on current features (captures regime changes):

```python
from cluster.dynamic_cluster_filter import DynamicClusterFilter

# Initialize filter
filter = DynamicClusterFilter(
    model_path='checkpoints/fold_0_best.pt',
    cluster_dir='cluster_results',
    best_cluster_ids=[1, 5, 8, 12]  # From analysis
)

# Every trading day
for date in trading_dates:
    features = get_features_for_date(date)

    # Filter to stocks in good clusters TODAY
    allowed_stocks = filter.filter_stocks_for_date(features)

    # Trade only allowed stocks
    predictions = model.predict(allowed_stocks)
    top_k = select_top_k(predictions)
```

See [docs/STATIC_VS_DYNAMIC.md](docs/STATIC_VS_DYNAMIC.md) for detailed comparison.

---

## Integration Examples

### With Backtesting

```python
from cluster import ClusterFilter
from inference.backtest_simulation import BacktestSimulator

cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Apply filter to backtest
cluster_filter.apply_cluster_filter_to_backtest(data_loader, inplace=True)

# Run backtest on filtered stocks only
simulator = BacktestSimulator(data_loader, predictor, ...)
results = simulator.run_backtest()
```

### With RL Training

```python
from cluster import ClusterFilter

cluster_filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Apply filter to RL environment
cluster_filter.apply_cluster_filter_to_rl_env(trainer.env, inplace=True)

# Train on filtered stocks only
trainer.train()
```

See [docs/INFERENCE_INTEGRATION.md](docs/INFERENCE_INTEGRATION.md) for more integration examples.

---

## Why Clustering Works

### Market Regimes

Different clusters represent different market regimes:

| Cluster Type | Characteristics | Trading Approach |
|--------------|----------------|------------------|
| High-momentum tech | High return, high volatility | Momentum trading |
| Stable dividends | Low return, low volatility | Hold positions |
| Mean-reverting cyclicals | Moderate return, predictable | Mean reversion |
| Noisy/unpredictable | Low Sharpe, erratic | **Avoid** |

### Learned Representations

The transformer encodes:
- **Price patterns**: Trends, reversals, volatility regimes
- **Fundamental relationships**: P/E ratios, growth metrics
- **Temporal dynamics**: Seasonality, cycles, momentum

Clustering groups stocks with similar learned representations.

### Profit Opportunity

Clusters with high returns + high win rates represent:
- Patterns the model has learned well
- Consistent edge across multiple stocks
- Lower variance in outcomes

---

## Troubleshooting

### Too few stocks after filtering

Relax criteria or select more clusters:
```bash
python -m cluster.analyze_clusters \
    --min-return 0.003 \
    --min-win-rate 0.50 \
    --top-k 30
```

### Unbalanced cluster sizes

Some patterns are more common. Solutions:
- Use more clusters (`--n-clusters 100`)
- Use DBSCAN for density-based clustering
- Post-process to filter by min/max cluster size

### No performance improvement

Possible causes:
1. Model hasn't learned meaningful patterns - train longer
2. Too many clusters - try `--n-clusters 20`
3. Wrong horizon - analyze multiple horizons
4. Overfitting - validate on separate time period

---

## File Structure

```
cluster/
├── README.md                     # This file
├── __init__.py
├── create_clusters.py               # Encode and cluster dataset (all-in-one)
├── analyze_clusters.py              # Analyze cluster performance
├── cluster_filter.py                # Static cluster filtering
├── dynamic_cluster_filter.py        # Dynamic (daily) cluster filtering
├── cache_embeddings.py              # Pre-compute embeddings for backtesting
├── compute_embeddings_lowmem.py     # Compute embeddings only (low-memory workflow)
├── cluster_embeddings_lowmem.py     # Cluster pre-computed embeddings (low-memory workflow)
├── gpu_kmeans.py                    # GPU-accelerated K-means
└── docs/
    ├── STATIC_VS_DYNAMIC.md         # Static vs dynamic comparison
    ├── INFERENCE_INTEGRATION.md     # Integration guide
    └── CACHING_GUIDE.md             # Embedding caching guide
```

---

## References

- Main documentation: [docs/cluster_top_k_filtering.md](../docs/cluster_top_k_filtering.md)
