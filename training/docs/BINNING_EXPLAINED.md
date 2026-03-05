# Adaptive Binning for Stock Price Classification

## Overview

The classification approach uses **adaptive (quantile-based) binning** instead of uniform binning. This provides better class balance and more precise predictions around common price changes.

## How It Works

### 1. Bin Edge Computation

Bin edges are computed by:
1. Sampling ~50,000 price ratios from the dataset across all prediction horizons
2. Computing quantiles: `np.quantile(ratios, q=np.linspace(0, 1, num_bins+1))`
3. This creates bins where each bin contains approximately **equal number of samples**

### 2. Bin Distribution (100 bins, typical stock data)

```
Bin Range       | Price Change  | Bin Width    | Typical Use Case
----------------|---------------|--------------|------------------
Bins 0-10       | < -10%        | ~2-5% wide   | Extreme losses (rare)
Bins 10-30      | -10% to -2%   | ~0.5-1% wide | Large losses
Bins 30-45      | -2% to -0.5%  | ~0.1-0.3%    | Small losses
Bins 45-55      | -0.5% to +0.5%| ~0.02-0.05%  | Sideways (MOST GRANULAR)
Bins 55-70      | +0.5% to +2%  | ~0.1-0.3%    | Small gains
Bins 70-90      | +2% to +10%   | ~0.5-1% wide | Large gains
Bins 90-99      | > +10%        | ~2-5% wide   | Extreme gains (rare)
```

### 3. Class Balancing Effect

**Problem with uniform binning:**
- Most stocks move ±1% daily
- With uniform bins spanning -20% to +20%, most samples land in center bins
- Creates severe class imbalance (center bins have 100x more samples than edge bins)

**Solution with adaptive binning:**
- Each bin has ~equal number of samples (by design)
- Automatically creates finer granularity where needed (near 0%)
- Coarser bins for rare events (extreme moves)
- **Result: Balanced classes without resampling!**

### 4. Example Statistics

From a typical dataset:
```
📈 Distribution Statistics:
  Samples collected: 50,000
  Mean ratio: 1.0008
  Std ratio: 0.0234
  Median ratio: 1.0005

📏 Bin Width Statistics:
  Min width: 0.000021 (bins 48-52, near 0%)
  Max width: 0.083451 (bins 0-5, 95-99, extremes)
  Median width: 0.002314
  Center bins (45-55) avg width: 0.000156  <- Very fine
  Edge bins (0-10, 90-100) avg width: 0.029847  <- Very coarse
  Ratio (edge/center): 191.34x  <- Massive difference!
```

## Labels

**What are the labels?**
- Labels are **bin indices**: integers from 0 to 99 (for 100 bins)
- Each label represents a range of price ratios

**Example labels for typical distribution:**
- Label 0:   Price ratio 0.72-0.80 (losing 20-28%)
- Label 25:  Price ratio 0.95-0.97 (losing 3-5%)
- Label 48:  Price ratio 0.998-0.999 (losing 0.1-0.2%)
- Label 50:  Price ratio 1.000-1.001 (flat to +0.1%)
- Label 52:  Price ratio 1.001-1.002 (gaining 0.1-0.2%)
- Label 75:  Price ratio 1.03-1.05 (gaining 3-5%)
- Label 99:  Price ratio 1.25-1.35 (gaining 25-35%)

**NOTE:** Exact ranges depend on the actual data distribution and are computed automatically!

## Model Prediction

The model outputs a **probability distribution** over all bins:
```python
# Model output shape: (batch_size, num_bins, num_pred_days)
# For each prediction day:
#   - Each bin gets a probability
#   - Sum of probabilities = 1.0
#   - argmax gives most likely bin
#   - Can use full distribution for confidence estimation
```

## Training Loss

Uses **cross-entropy loss** between:
- Predicted distribution: (num_bins,) probabilities
- Target: one-hot vector with 1.0 at the true bin

This naturally handles:
- Uncertainty in predictions
- Similar bins (model can spread probability)
- Confidence estimation

## Benefits

1. **Automatic class balancing** - no need for resampling
2. **Fine-grained** predictions near common values (0%)
3. **Robust** to outliers (extreme moves get coarse bins)
4. **Interpretable** - bin edges show actual data distribution
5. **Efficient** - cached after first computation

## Usage

```python
from training.new_data_loader import compute_adaptive_bin_edges, convert_price_ratio_to_one_hot

# Compute bin edges (done once, cached)
bin_edges = compute_adaptive_bin_edges(
    dataset_path='data.pkl',
    num_bins=100,
    pred_days=[1, 5, 10, 20],
    max_samples=50000
)

# Convert price ratio to label
ratio = 1.05  # 5% gain
label = convert_price_ratio_to_one_hot(ratio, bin_edges=bin_edges)
# Returns one-hot tensor with 1.0 at the bin containing 1.05
```

## Comparison: Uniform vs Adaptive

| Aspect | Uniform Bins | Adaptive Bins |
|--------|--------------|---------------|
| Bin width | Constant (0.4% each) | Variable (0.02% to 5%) |
| Class balance | Poor (100:1 ratio) | Excellent (1:1 by design) |
| Precision at 0% | Low (0.4% resolution) | High (0.02% resolution) |
| Extreme moves | Over-represented | Properly handled |
| Training stability | Difficult | Easier |
| Performance | Lower | Higher |

## Conclusion

Adaptive binning is **strongly recommended** for stock price classification. It provides:
- Better class balance
- More precise predictions
- Easier training
- Better performance

The original implementation used this approach successfully!
