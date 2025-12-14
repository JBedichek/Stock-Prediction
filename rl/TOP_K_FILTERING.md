# Top-K Per Horizon Stock Filtering

## Overview

Instead of having the Q-network choose from all ~900 stocks each day, we pre-filter to the **top-K most promising stocks per time horizon** based on the predictor's expected returns. This dramatically reduces the action space while focusing on the best opportunities.

## How It Works

### 1. Extract Predictor Predictions

For each stock on each day, the predictor provides:
- **Expected returns** for 4 time horizons (1, 3, 5, 10 days ahead)
- Located at positions 1428:1432 in the 1444-dim feature vector

### 2. Rank and Filter

For each of the 4 time horizons:
1. Rank all stocks by expected return for that horizon
2. Select top-K stocks (default: K=10)
3. Take the **union** across all horizons

### 3. Final Action Space

- **Before filtering**: ~900 stocks
- **After filtering**: ~30-40 stocks (union of 4 × top-10, with overlap)
- **Reduction**: ~95% smaller action space!

## Benefits

### 1. Faster Training
- Fewer Q-network forward passes per step
- ~900 → ~40 stocks = **20x fewer computations**

### 2. Better Exploration
- Q-network focuses on the most promising stocks
- Avoids wasting time on low-potential stocks
- More efficient use of experience

### 3. Better Performance
- Predictor does heavy lifting (identifying opportunities)
- Q-network refines timing and position sizing
- Leverages both models' strengths

## Example

On a given day with 900 stocks available:

**Horizon 1 (1 day)**: Top 10 stocks by expected return
- AAPL, MSFT, GOOGL, TSLA, NVDA, ...

**Horizon 2 (3 days)**: Top 10 stocks
- AAPL, AMZN, META, NVDA, AMD, ...

**Horizon 3 (5 days)**: Top 10 stocks
- MSFT, NVDA, TSLA, AMD, NFLX, ...

**Horizon 4 (10 days)**: Top 10 stocks
- GOOGL, META, AAPL, NVDA, CRM, ...

**Union** (Final action space): 35 unique stocks
- The Q-network only sees these 35 stocks
- It decides which to buy, how much, and when to sell

## Implementation

### Environment Initialization

```python
env = TradingEnvironment(
    ...
    top_k_per_horizon=10  # Top-10 stocks per horizon
)
```

Set to `0` or `>= num_stocks` to disable filtering (use all stocks).

### Filtering Logic

Located in `rl_environment.py`:

```python
def _filter_top_k_stocks(self, cached_states):
    # Extract expected returns for all stocks
    for ticker, cached_state in cached_states.items():
        pred_features = cached_state[:1444]
        expected_returns = pred_features[1428:1432]  # 4 horizons

    # For each horizon, select top-k
    for horizon_idx in range(4):
        stocks_by_return.sort(key=lambda x: x[1], reverse=True)
        top_k = stocks_by_return[:self.top_k_per_horizon]
        selected_tickers.update(top_k)

    return list(selected_tickers)
```

Called in `_get_states()` before creating states for the Q-network.

## Configuration

In `train_rl_phase2.py`, you can adjust via config:

```python
config = {
    'top_k_per_horizon': 10,  # Default: 10
    ...
}
```

## Performance Impact

### Computational Savings

**Per step:**
- Q-network forward passes: 900 → 40 stocks
- Action selection: 900 → 40 options
- **Speedup: ~20x faster** per step!

### Memory Savings

- State tensors: Only create for 40 stocks instead of 900
- Replay buffer: Smaller transitions (fewer stocks per state)
- **Memory reduction: ~20x less**

### Training Quality

- **Expected improvement**: Faster convergence
- **Reason**: More focused exploration on promising stocks
- **Caveat**: May miss occasional dark horse stocks not in top-K

## Trade-offs

### Pros:
✅ Much faster training (20x fewer Q-network evaluations)
✅ More focused learning (best stocks only)
✅ Leverages predictor's strength (opportunity identification)
✅ Still diverse (union across 4 horizons)

### Cons:
⚠️ May miss stocks that predictor ranks low but Q-network would prefer
⚠️ Reduces diversity compared to full universe
⚠️ Relies on predictor quality

## Future Enhancements

1. **Dynamic K**: Adjust top_k based on market volatility
2. **Ensemble filtering**: Combine multiple predictors
3. **Negative selection**: Also include top-K worst for shorting
4. **Adaptive thresholds**: Use expected return thresholds instead of fixed K

## Summary

Top-K filtering is a **huge win** for RL training:
- **20x faster** training (fewer Q-network evaluations)
- **Better focus** (most promising stocks only)
- **Smart design** (predictor finds opportunities, Q-network times trades)

With default `top_k_per_horizon=10`, we get ~35-40 stocks per day instead of 900 - making training practical while maintaining quality!
