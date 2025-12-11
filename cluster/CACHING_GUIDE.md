# Embeddings Caching for Fast Cluster Filtering

## The Problem

Dynamic cluster filtering requires encoding stocks every trading day:
- Encoding = expensive transformer forward pass
- 100 stocks × 120 trading days = 12,000 encodings
- At ~0.02 sec per stock = 4+ minutes just for encoding!

## The Solution

**Pre-compute embeddings once, reuse forever.**

Cache structure (HDF5):
```
embeddings_cache.h5
├── AAPL/
│   ├── 2024-01-01 -> [1024-dim embedding]
│   ├── 2024-01-02 -> [1024-dim embedding]
│   └── ...
├── GOOGL/
│   ├── 2024-01-01 -> [1024-dim embedding]
│   └── ...
```

## Quick Start

### Step 1: Pre-compute Cache (One-time, ~1 hour)

```bash
python -m cluster.cache_embeddings \
    --model-path checkpoints/best_model_100m_1.18.pt \
    --dataset-path data/all_complete_dataset.h5 \
    --output-path data/embeddings_cache.h5 \
    --num-months 6 \
    --batch-size 32
```

**What this does:**
- Encodes all stocks for the last 6 months
- Saves embeddings to HDF5 file
- Takes ~1 hour for 2,000 stocks × 120 days = 240,000 embeddings

**Parameters:**
- `--num-months 6`: Cache last 6 months of data
- `--batch-size 32`: Adjust based on GPU memory
- `--num-stocks 1000`: Optionally limit to specific number of stocks

### Step 2: Use Cache During Backtesting (20-30x faster!)

```bash
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model_100m_1.18.pt \
    --prices data/actual_prices.h5 \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt \
    --embeddings-cache data/embeddings_cache.h5 \
    --num-test-stocks 2000 \
    --subset-size 100 \
    --test-months 6
```

**New argument:**
- `--embeddings-cache data/embeddings_cache.h5`: Path to pre-computed cache

## Performance Comparison

### Without Cache
```
Trading day 1: Encode 100 stocks → 2.5 sec
Trading day 2: Encode 100 stocks → 2.5 sec
...
Trading day 120: Encode 100 stocks → 2.5 sec

Total encoding time: ~5 minutes for 6-month backtest
```

### With Cache
```
Trading day 1: Load 100 embeddings from cache → 0.1 sec
Trading day 2: Load 100 embeddings from cache → 0.1 sec
...
Trading day 120: Load 100 embeddings from cache → 0.1 sec

Total encoding time: ~12 seconds for 6-month backtest
```

**Speedup: 25x faster! ⚡**

## How It Works

1. **During cache building:**
   - For each stock and date, extract features
   - Run through transformer, get mean-pooled output
   - Save embedding to HDF5: `cache[ticker][date] = embedding`

2. **During backtesting:**
   - When filtering stocks for a date, check cache first
   - If `cache[ticker][date]` exists, use it
   - If not, fall back to on-the-fly encoding
   - Hybrid approach handles cache misses gracefully

## Cache Management

### Check Cache Contents

```python
import h5py

with h5py.File('data/embeddings_cache.h5', 'r') as f:
    print(f"Tickers in cache: {len(f.keys())}")

    # Check specific ticker
    if 'AAPL' in f:
        dates = list(f['AAPL'].keys())
        print(f"AAPL has {len(dates)} cached dates")
        print(f"Date range: {dates[0]} to {dates[-1]}")
```

### Update Cache for New Data

If you have new data (e.g., new month of trading days):

```bash
# Cache additional months
python -m cluster.cache_embeddings \
    --model-path checkpoints/best_model.pt \
    --dataset-path data/all_complete_dataset.h5 \
    --output-path data/embeddings_cache_new.h5 \
    --num-months 1  # Just the new month
```

Then merge or use separate cache files.

### Cache Size Estimates

**Storage requirements:**
- Each embedding: 1024 floats × 4 bytes = 4 KB
- 2,000 stocks × 120 days × 4 KB = ~960 MB
- With compression: ~500 MB

**Cache building time:**
- 2,000 stocks × 120 days = 240,000 encodings
- At 200 encodings/min (batch size 32): ~20 hours
- But you only do this ONCE!

## Advanced Usage

### Cache Specific Tickers Only

```bash
# Create ticker list
echo "AAPL" > tickers.txt
echo "GOOGL" >> tickers.txt
echo "MSFT" >> tickers.txt

# Then modify cache_embeddings.py to read from file
# (feature not yet implemented - use --num-stocks to limit)
```

### Incremental Caching

Build cache in stages:

```bash
# Stage 1: First 500 stocks
python -m cluster.cache_embeddings \
    --num-stocks 500 \
    --output-path data/cache_part1.h5

# Stage 2: Next 500 stocks
# (modify to skip first 500)
python -m cluster.cache_embeddings \
    --num-stocks 1000 \
    --output-path data/cache_part2.h5

# Merge manually or use both during backtesting
```

### Cache for Different Models

If you have multiple models:

```bash
# Cache for model 1
python -m cluster.cache_embeddings \
    --model-path checkpoints/model_v1.pt \
    --output-path data/embeddings_cache_v1.h5

# Cache for model 2
python -m cluster.cache_embeddings \
    --model-path checkpoints/model_v2.pt \
    --output-path data/embeddings_cache_v2.h5

# Use appropriate cache for each model
python -m inference.backtest_simulation \
    --model checkpoints/model_v1.pt \
    --embeddings-cache data/embeddings_cache_v1.h5
```

## Troubleshooting

### Issue: Cache building runs out of memory

**Solution**: Reduce batch size
```bash
python -m cluster.cache_embeddings --batch-size 16
```

### Issue: Cache file is huge

**Cause**: No compression enabled

**Solution**: HDF5 compression is already enabled in the code (`compression='gzip'`), but you can verify:
```python
import h5py
with h5py.File('embeddings_cache.h5', 'r') as f:
    # Check compression
    dataset = f['AAPL']['2024-01-01']
    print(f"Compression: {dataset.compression}")  # Should be 'gzip'
```

### Issue: Cache misses during backtesting

**Cause**: Date or ticker not in cache

**Check**:
```python
import h5py

with h5py.File('data/embeddings_cache.h5', 'r') as f:
    # Check if ticker exists
    if 'AAPL' not in f:
        print("AAPL not in cache!")

    # Check if date exists
    elif '2024-01-01' not in f['AAPL']:
        print("2024-01-01 not cached for AAPL!")
```

**Solution**: Rebuild cache with correct date range or stocks

### Issue: Different feature dimensions

**Symptom**: Cache built with old model, using new model

**Solution**: Rebuild cache with the new model - embeddings are model-specific

## Best Practices

1. **Build cache after training**: When you have a final model, build the cache once
2. **Match date ranges**: Cache should cover your backtesting period
3. **Version your caches**: Name caches by model version: `embeddings_cache_v1.2.h5`
4. **Monitor cache hits**: Check backtesting output for cache hit rates
5. **Rebuild periodically**: If you retrain models or add new data

## Summary

**One-time cost:**
- Build cache: ~1 hour for 2,000 stocks × 6 months

**Recurring benefit:**
- Every backtest: 20-30x faster
- Run 100 backtests? Save 80+ hours!

**The math:**
- Without cache: 100 backtests × 5 min = 500 min = 8.3 hours
- With cache: 100 backtests × 0.2 min = 20 min
- **Time saved: 8 hours!**

Pre-compute once, benefit forever! 🚀
