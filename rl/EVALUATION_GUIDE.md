# Statistical Significance Testing Guide

## Overview

This framework rigorously tests whether your RL agent is **statistically significantly better** than simple baselines over the past 2 years of data.

## Quick Start

```bash
# Run statistical evaluation (200 episodes)
python rl/evaluate_statistical_significance.py

# Results saved to: evaluation_results/
```

## Baseline Strategies

### 1. **TopK-1d/3d/5d/10d** (Primary Baselines)
- **Strategy**: Select stock with highest predicted return for matching time horizon
- **Why important**: This is what the predictor model was trained to do
- **Hypothesis**: RL should beat this by learning better timing and position sizing

### 2. **Random** (Sanity Check)
- **Strategy**: Random action selection
- **Why important**: RL should easily beat random
- **Hypothesis**: If RL doesn't beat random, something is wrong

### 3. **Buy-and-Hold** (Passive Baseline)
- **Strategy**: Buy top stock on day 1, hold until end
- **Why important**: Tests if active trading adds value
- **Hypothesis**: RL should beat this through active rebalancing

## Statistical Tests

### 1. Bootstrap Confidence Intervals

**What it is**: Resamples data 10,000 times to estimate uncertainty

**How to interpret**:
```
RL-Agent:
  Avg Return: 5.2% [3.1%, 7.3%]
TopK-1d:
  Avg Return: 2.8% [1.2%, 4.4%]
```

- **If CIs don't overlap**: Strong evidence RL is better
- **If CIs overlap slightly**: Weak evidence, need more tests
- **If CIs overlap heavily**: No significant difference

### 2. Paired t-Tests with Bonferroni Correction

**What it is**: Tests if RL returns are significantly higher than baseline on same episodes

**Bonferroni correction**: Adjusts p-value threshold for multiple comparisons
- Without correction: α = 0.05
- With 6 baselines: α = 0.05/6 = 0.0083

**How to interpret**:
```
RL vs TopK-1d:
  t-statistic: +3.42
  p-value:     0.0008
  Bonf. α:     0.0083
  Significant: ✅ YES
  Cohen's d:   +0.65
```

- **p-value < Bonf. α**: Statistically significant difference
- **t-statistic > 0**: RL is better
- **Cohen's d**: Effect size
  - 0.2 = small
  - 0.5 = medium
  - 0.8 = large

### 3. Jobson-Korkie Test (Sharpe Ratio)

**What it is**: Tests if Sharpe ratios are significantly different

**Why important**: Returns alone don't account for risk

**How to interpret**:
```
RL vs TopK-1d:
  RL Sharpe:       1.45
  TopK-1d Sharpe:  0.82
  Z-statistic:     +2.31
  p-value:         0.021
  Significant:     ✅ YES
```

- **p-value < 0.05**: RL's risk-adjusted performance is significantly better
- **Z-statistic > 1.96**: 95% confidence RL Sharpe is higher

## Evaluation Metrics

### Return-Based Metrics

1. **Average Return**: Mean % return per episode
2. **Sharpe Ratio**: Return / Volatility × √252
   - \> 1.0 = good
   - \> 2.0 = excellent
3. **Sortino Ratio**: Return / Downside Volatility × √252
   - Like Sharpe, but only penalizes downside volatility
4. **Calmar Ratio**: Return / Max Drawdown
   - Higher = better risk-adjusted performance

### Risk Metrics

1. **Max Drawdown**: Largest peak-to-trough decline
   - Lower is better
   - \> 20% = concerning for 30-day episodes
2. **Win Rate**: % of episodes with positive return
   - \> 50% = good
3. **Volatility**: Standard deviation of returns
   - Lower = more consistent

## Interpreting Results

### ✅ Strong Evidence RL Works

You have strong evidence if **ALL** of these are true:
1. Bootstrap CIs for return don't overlap with best baseline
2. Paired t-test: p < Bonferroni α for all baselines
3. Sharpe ratio significantly higher (Jobson-Korkie p < 0.05)
4. Cohen's d > 0.5 vs primary baselines

### ⚠️ Weak Evidence

Evidence is weak if:
1. RL beats baselines on average, but CIs overlap
2. Some t-tests significant, others not
3. Sharpe test not significant

**Action**: Need more episodes or longer evaluation period

### ❌ No Evidence / RL Not Better

No evidence if:
1. RL average return ≤ TopK baselines
2. None of the t-tests are significant
3. RL beats Random but loses to TopK strategies

**Action**:
- RL agent needs more training
- Hyperparameter tuning needed
- Architecture changes needed

## Expected Results (Well-Trained Agent)

```
Strategy          Avg Return    Sharpe    Sortino    Max DD    Win Rate
RL-Agent          +4.8%         1.42      1.65       12.3%     58.2%
TopK-1d           +2.1%         0.78      0.91       18.7%     51.3%
TopK-3d           +1.9%         0.71      0.84       19.2%     50.1%
TopK-5d           +1.7%         0.65      0.79       21.4%     49.8%
TopK-10d          +1.2%         0.52      0.61       23.1%     47.2%
Random            -0.8%         -0.15     -0.22      35.8%     45.1%
BuyAndHold        +1.4%         0.58      0.67       22.3%     N/A

Statistical Tests:
  RL vs TopK-1d:  t=3.42, p=0.0008 ✅ (Cohen's d=0.65)
  RL vs TopK-3d:  t=3.87, p=0.0002 ✅ (Cohen's d=0.71)
  ...all significant

Sharpe Tests:
  RL vs TopK-1d:  Z=2.31, p=0.021 ✅
  RL vs TopK-3d:  Z=2.54, p=0.011 ✅
  ...all significant
```

This would be **strong evidence** that RL adds value beyond predictor.

## Files Generated

1. **results_summary.csv**: Tabular results
2. **return_distributions.png**: Histogram of returns for each strategy
3. **performance_comparison.png**: Bar chart with error bars
4. **Console output**: Detailed statistical test results

## Advanced: Walk-Forward Validation

For even more rigor, modify the script to use walk-forward validation:

```python
# Split validation period into chunks
# Train on chunk 1, test on chunk 2
# Train on chunks 1-2, test on chunk 3
# etc.
```

This prevents any lookahead bias.

## Common Questions

**Q: How many episodes do I need?**
A: 200 is a good start. 500+ for high confidence. Use power analysis if critical.

**Q: What if results are mixed (some baselines beaten, others not)?**
A: Focus on beating TopK-1d (most relevant). Beating 3d/5d/10d is bonus.

**Q: My RL agent beats Random but not TopK. Is it learning?**
A: Yes, but not enough. Needs more training or better architecture.

**Q: All tests say "not significant". What now?**
A: Either (1) RL needs more training, (2) baselines are actually optimal, or (3) need more test episodes.

**Q: Can I test on training data?**
A: No! Use only held-out validation data (last 2 years). Testing on training = overfitting.

## Citation

If using this methodology in research:
```
Statistical methodology:
- Bootstrap CIs: Efron & Tibshirani (1993)
- Paired t-tests: Multiple comparison correction via Bonferroni
- Sharpe ratio tests: Jobson & Korkie (1981)
```
