# Statistical Comparison: Model vs Random Selection

Rigorous statistical framework for comparing model's stock selection against random portfolios.

## Overview

This framework runs **30 independent trials** where each trial:
1. Samples a random subset of 200 stocks from the test set
2. Runs the model on this subset → selects top-5 stocks
3. Generates 100 random portfolios from the same subset
4. Simulates trading for all portfolios
5. Compares performance using multiple metrics

## Key Features

### **Bootstrap Design**
- **30 trials** with different random stock subsets
- **100 random portfolios per trial** (3,000 random portfolios total)
- Paired comparisons control for "universe luck"

### **Comprehensive Metrics**
- **Total Return %**: Raw performance
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return / max drawdown
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: % of profitable trades

### **Statistical Tests**
- **Paired t-test**: Tests if model outperforms on average
- **Effect size (Cohen's d)**: Magnitude of difference
- **95% Confidence intervals**: Range of likely true effect
- **Win rate**: % of trials where model beats random
- **Percentile ranking**: Model's rank vs all random portfolios

### **Visualizations**
- Distribution histograms for each metric
- Box plots comparing model vs random
- Clear visual indication of significance

## Usage

### Quick Start

```bash
python inference/run_statistical_comparison.py
```

### Custom Configuration

```bash
python inference/statistical_comparison.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --num-trials 30 \
    --num-random-per-trial 100 \
    --subset-size 200 \
    --top-k 5 \
    --horizon-idx 1 \
    --test-months 2 \
    --initial-capital 100000 \
    --output results.pt
```

### Arguments

**Data:**
- `--data`: Dataset path (HDF5 or pickle)
- `--prices`: Actual prices HDF5 (for normalized features)
- `--model`: Model checkpoint path
- `--bin-edges`: Bin edges cache path

**Test Set:**
- `--num-test-stocks`: Size of test set (default: 2000, last N alphabetically)

**Comparison:**
- `--num-trials`: Number of independent trials (default: 30)
- `--num-random-per-trial`: Random portfolios per trial (default: 100)
- `--subset-size`: Stocks per subset (default: 200)
- `--top-k`: Number of stocks to select (default: 5)
- `--horizon-idx`: Prediction horizon (0=1d, 1=5d, 2=10d, 3=20d)
- `--test-months`: Trading period length (default: 2)
- `--initial-capital`: Starting capital (default: $100,000)

**Other:**
- `--device`: cuda or cpu
- `--batch-size`: Inference batch size
- `--output`: Output file path
- `--seed`: Random seed for reproducibility

## Output

### Console Output

```
================================================================================
STATISTICAL ANALYSIS
================================================================================

────────────────────────────────────────────────────────────────────────────────
Metric: TOTAL_RETURN_PCT
────────────────────────────────────────────────────────────────────────────────

Summary Statistics:
  Model:  Mean =   15.234,  Std =    5.123
  Random: Mean =    8.456,  Std =    4.789
  Difference: +6.778

Paired t-test:
  t-statistic:    3.245
  p-value: 0.002891
  Significant: YES ✅
  Effect size (Cohen's d):    1.234
  95% CI: [+2.456, +11.100]
  Model wins: 23/30 trials (76.7%)
  Model percentile vs all random: 92.3th

────────────────────────────────────────────────────────────────────────────────
...

================================================================================
OVERALL CONCLUSION
================================================================================
Significant metrics: 4/5
✅ Model shows statistically significant improvement over random selection
```

### Saved Files

1. **`statistical_comparison_results.pt`** - Full results (load with `torch.load()`)
   ```python
   results = torch.load('statistical_comparison_results.pt')
   print(results['config'])  # Configuration
   print(results['trial_results'])  # All trial data
   ```

2. **`metric_distributions.png`** - Distribution histograms
   - Shows model vs random distributions for each metric
   - Visual assessment of separation

3. **`metric_boxplots.png`** - Box plot comparisons
   - Shows median, quartiles, outliers
   - Easy comparison of central tendency and spread

## Interpreting Results

### Statistical Significance

**p-value < 0.05**: Model significantly outperforms random at 95% confidence
- p < 0.05: Significant ✅
- p < 0.01: Highly significant ✅✅
- p < 0.001: Very highly significant ✅✅✅

**Effect size (Cohen's d)**:
- d < 0.2: Small effect
- d = 0.5: Medium effect
- d > 0.8: Large effect

### Practical Significance

Even if statistically significant, check:
- **Return difference**: Is it large enough to matter? (e.g., >5%)
- **Transaction costs**: Does profit exceed costs?
- **Consistency**: Does model win in >60% of trials?

### Red Flags

❌ **Model underperforms**: p-value > 0.05 and negative difference
❌ **High variance**: Model wins <50% of trials → unreliable
❌ **Only return significant**: Sharpe/Sortino not significant → just taking more risk
❌ **Small effect size**: d < 0.3 → difference may not be economically meaningful

### Green Flags

✅ **Multiple metrics significant**: Return, Sharpe, Sortino all p < 0.05
✅ **High win rate**: Model wins >70% of trials
✅ **Large effect size**: d > 0.8
✅ **95% CI excludes zero**: Confident the effect is real
✅ **High percentile**: Model in >90th percentile vs all random

## Example Interpretation

### Scenario 1: Strong Model
```
Total Return:  Model=15.2%, Random=8.4%, p=0.003, d=1.2, wins=76.7%
Sharpe Ratio:  Model=1.85, Random=1.12, p=0.008, d=1.1, wins=73.3%
```
**Conclusion**: Model significantly outperforms with large effect size.
Strong evidence of skill.

### Scenario 2: Weak Model
```
Total Return:  Model=9.8%, Random=8.4%, p=0.18, d=0.3, wins=56.7%
Sharpe Ratio:  Model=1.15, Random=1.12, p=0.45, d=0.1, wins=50.0%
```
**Conclusion**: No significant difference. Model may not add value.

### Scenario 3: Risky Model
```
Total Return:  Model=18.5%, Random=8.4%, p=0.001, d=1.5, wins=86.7%
Sharpe Ratio:  Model=0.95, Random=1.12, p=0.25, d=-0.4, wins=40.0%
Max Drawdown:  Model=25.3%, Random=12.1%, p=0.002
```
**Conclusion**: Higher returns but also higher risk. Not risk-adjusted better.
Model is just more volatile.

## Computational Time

**Approximate runtime** (for default config):
- Dataset: 2000 test stocks, 200 per subset
- 30 trials × 100 random portfolios = 3,030 simulations
- With feature preloading: ~15-30 minutes (GPU)
- Without preloading: ~2-4 hours

**To speed up**:
- Reduce `--num-trials` to 20
- Reduce `--num-random-per-trial` to 50
- Reduce `--subset-size` to 100
- Use GPU with large `--batch-size`

## Advanced Usage

### Compare Different Horizons

```bash
for horizon in 0 1 2 3; do
    python inference/statistical_comparison.py \
        --horizon-idx $horizon \
        --output results_horizon_${horizon}.pt
done
```

### Compare Different Top-K

```bash
for k in 3 5 10 20; do
    python inference/statistical_comparison.py \
        --top-k $k \
        --output results_topk_${k}.pt
done
```

### Load and Analyze Saved Results

```python
import torch
import numpy as np

# Load results
results = torch.load('statistical_comparison_results.pt')

# Extract model vs random for each trial
model_returns = [trial['model']['total_return_pct']
                 for trial in results['trial_results']]
random_returns = [trial['random_mean']['total_return_pct']
                  for trial in results['trial_results']]

# Custom analysis
print(f"Model median: {np.median(model_returns):.2f}%")
print(f"Random median: {np.median(random_returns):.2f}%")
print(f"Model 90th percentile: {np.percentile(model_returns, 90):.2f}%")
```

## Limitations

1. **Test set isolation**: Assumes last 2000 stocks were excluded from training
2. **Look-ahead bias**: Ensure model doesn't use future information
3. **Survivorship bias**: Dataset may exclude delisted stocks
4. **Transaction costs**: Not included in simulation
5. **Market impact**: Assumes trades don't move prices
6. **Data quality**: Results depend on accurate price data

## Future Extensions

Possible enhancements:
- [ ] Compare vs momentum/value strategies
- [ ] Add transaction costs
- [ ] Stratify by market regime (bull/bear)
- [ ] Add sector-neutral random baseline
- [ ] Bootstrap with replacement for CI estimation
- [ ] Permutation test implementation
- [ ] Time-series cross-validation

## Troubleshooting

**Issue**: Out of memory
- Solution: Reduce `--batch-size` or `--subset-size`

**Issue**: Too slow
- Solution: Reduce `--num-trials` or `--num-random-per-trial`

**Issue**: All trials show no difference
- Solution: Model may not work, or need more trials for power

**Issue**: High variance across trials
- Solution: Increase `--num-trials` for more stable estimates
