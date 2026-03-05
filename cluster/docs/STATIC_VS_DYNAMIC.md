# Static vs Dynamic Cluster Filtering

## Understanding the Two Approaches

There are two ways to use cluster filtering: **static** and **dynamic**. They serve different purposes.

## Static Filtering (cluster_filter.py)

**What it does:**
- Maps each stock to a FIXED cluster based on historical analysis
- "AAPL is always in cluster 5"
- Filters stock universe once at the beginning

**Use cases:**
- Quick stock screening
- Reducing dataset size for training
- Pre-filtering before backtesting starts

**Pros:**
- Very fast (no encoding needed during runtime)
- Simple to understand
- Good for initial stock selection

**Cons:**
- Doesn't adapt to changing market conditions
- Stock's cluster membership is fixed
- Misses regime changes

**Example:**
```python
from cluster.cluster_filter import ClusterFilter

# Filter stocks once
filter = ClusterFilter(
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# AAPL is in cluster 5 - it stays in cluster 5 forever
allowed_stocks = filter.filter_tickers(all_stocks)
```

## Dynamic Filtering (dynamic_cluster_filter.py) ✅ RECOMMENDED

**What it does:**
- Encodes stocks EVERY DAY using current features
- Assigns each stock to nearest cluster based on TODAY's state
- "Today, given current conditions, AAPL is in cluster 5"
- Filters stocks fresh each trading day

**Use cases:**
- Backtesting with regime awareness
- Live trading
- Situations where market conditions matter

**Pros:**
- Adapts to changing market conditions
- Stocks can change clusters as they change behavior
- Captures regime shifts
- More accurate filtering

**Cons:**
- Slower (requires encoding each day)
- More complex
- Uses more compute

**Example:**
```python
from cluster.dynamic_cluster_filter import DynamicClusterFilter

# Initialize filter once
filter = DynamicClusterFilter(
    model_path='checkpoints/best_model.pt',
    cluster_dir='cluster_results',
    best_clusters_file='cluster_results/best_clusters_5d.txt'
)

# Every day:
for date in trading_dates:
    # Get current features for all stocks
    features_dict = {ticker: get_features(ticker, date) for ticker in all_stocks}

    # Filter based on TODAY's cluster assignment
    # AAPL might be in cluster 5 today, cluster 8 tomorrow
    allowed_stocks = filter.filter_stocks_for_date(features_dict)

    # Trade only allowed stocks
    trade(allowed_stocks)
```

## When to Use Which?

### Use Static Filtering When:
- You want to quickly reduce the stock universe
- Compute efficiency is critical
- You're doing initial exploration
- You need to pre-filter a dataset for training

### Use Dynamic Filtering When: ✅
- You're backtesting a trading strategy
- You're running live trading
- You want to capture regime changes
- Accuracy > speed
- You want stocks to adapt to market conditions

## The Key Difference

**Static:**
```
Day 1: AAPL → Cluster 5 (good cluster) → TRADE ✓
Day 2: AAPL → Cluster 5 (good cluster) → TRADE ✓
Day 3: AAPL → Cluster 5 (good cluster) → TRADE ✓
```
AAPL is ALWAYS in cluster 5, regardless of current conditions.

**Dynamic:**
```
Day 1: AAPL → Cluster 5 (good cluster) → TRADE ✓
Day 2: AAPL → Cluster 8 (bad cluster)  → SKIP ✗
Day 3: AAPL → Cluster 5 (good cluster) → TRADE ✓
```
AAPL's cluster changes based on its current state!

## Integration with Backtesting

The `backtest_simulation.py` uses **dynamic filtering** because it's the right approach for backtesting:

```bash
python -m inference.backtest_simulation \
    --data data/all_complete_dataset.h5 \
    --model checkpoints/best_model.pt \
    --cluster-dir cluster_results \
    --best-clusters-file cluster_results/best_clusters_5d.txt
```

This will:
1. **Every trading day:** Encode all candidate stocks
2. **Assign to clusters:** Based on current features
3. **Filter:** Keep only stocks in good clusters TODAY
4. **Predict:** Get predictions for filtered stocks
5. **Trade:** Select top-k from predictions

## Performance Impact

### Static Filtering
```
Speed:     ⚡⚡⚡⚡⚡ (instant)
Accuracy:  ⭐⭐⭐ (moderate)
Adapts:    ❌ (fixed assignments)
```

### Dynamic Filtering
```
Speed:     ⚡⚡⚡ (slower, requires encoding)
Accuracy:  ⭐⭐⭐⭐⭐ (high)
Adapts:    ✅ (changes with market)
```

## Example: Why Dynamic Matters

Imagine AAPL during different market conditions:

**Bull Market (Day 1):**
- High momentum, strong growth
- Encoded features: [high momentum indicators]
- Assigned to: **Cluster 3 (high-momentum cluster)** ✓
- Filter result: ALLOWED (cluster 3 is a good cluster)

**Market Correction (Day 2):**
- Falling prices, high volatility
- Encoded features: [high volatility, negative momentum]
- Assigned to: **Cluster 9 (volatile/unpredictable cluster)** ✗
- Filter result: FILTERED OUT (cluster 9 is a bad cluster)

**Recovery (Day 3):**
- Stabilizing, moderate growth
- Encoded features: [moderate growth, lower volatility]
- Assigned to: **Cluster 5 (stable growth cluster)** ✓
- Filter result: ALLOWED (cluster 5 is a good cluster)

With **static filtering**, AAPL would be in the same cluster all three days, missing the regime changes!

## Recommendation

For **backtesting and live trading**, always use **dynamic filtering** (dynamic_cluster_filter.py).

For **quick dataset reduction** or **initial exploration**, static filtering (cluster_filter.py) is fine.

The extra compute cost of dynamic filtering is worth it for the improved accuracy and regime awareness!
