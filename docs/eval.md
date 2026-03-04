# Principled Evaluation Guide

This document describes the `principled_evaluation.py` script, which provides statistically rigorous evaluation of stock prediction models.

## Table of Contents

1. [Overview](#overview)
2. [Key Metrics](#key-metrics)
3. [Evaluation Process](#evaluation-process)
4. [Baselines](#baselines)
5. [Statistical Tests](#statistical-tests)
6. [Sweep Mode](#sweep-mode)
7. [Output Files](#output-files)
8. [Interpreting Results](#interpreting-results)
9. [Usage Examples](#usage-examples)

---

## Overview

The principled evaluation framework assesses model quality using metrics from quantitative finance, rather than relying solely on trading simulation returns. This approach is more robust because:

1. **IC (Information Coefficient)** measures prediction quality directly, independent of position sizing or portfolio construction
2. **Statistical tests** distinguish skill from luck
3. **Multiple baselines** provide context for performance claims
4. **Bootstrap confidence intervals** quantify uncertainty

### Philosophy

Although trading simulations represent the use case of the model more directly, the returns are a **secondary metric**, not the primary one, because a model can have positive simulation returns by luck. Trading simulations are a high variance estimator.  Loss-like model metrics are more directly interpretable, and are not subject to high variance like trading simulations.  

---

## Key Metrics

### Information Coefficient (IC)

The **primary metric** for evaluating prediction quality.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Daily IC** | `pearsonr(predictions, actual_returns)` | Correlation between predictions and outcomes for one day |
| **Mean IC** | `mean(daily_ICs)` | Average predictive power across all days |
| **IC Std** | `std(daily_ICs)` | Consistency of predictions |
| **Information Ratio (IR)** | `mean_IC / std_IC` | Risk-adjusted IC (like Sharpe for predictions) |
| **Rank IC** | `spearmanr(predictions, actual_returns)` | Rank correlation (robust to outliers) |
| **% Positive IC** | `mean(daily_ICs > 0) × 100` | Percentage of days with positive correlation |

**Interpretation:**
- IC > 0.02: Weak but potentially useful signal
- IC > 0.05: Good predictive power
- IC > 0.10: Excellent (rare in practice)
- IR > 0.5: Consistent signal
- % Positive > 55%: Model is right more often than wrong

### Quantile Analysis

Tests whether top predictions actually outperform bottom predictions.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Top Decile Return** | Mean return of stocks in top 10% by prediction | What you'd earn going long the best predictions |
| **Bottom Decile Return** | Mean return of stocks in bottom 10% | What you'd earn (lose) going long the worst |
| **Long-Short Spread** | `Top Decile - Bottom Decile` | The "edge" from sorting stocks correctly |
| **Hit Rate** | % of top decile beating median | How often top picks actually outperform |

**Interpretation:**
- Spread > 0: Model ranks stocks correctly on average
- Spread > 0.1% per trade: Meaningful edge after costs
- Hit Rate > 55%: Top picks reliably outperform

### Baseline Comparisons

| Metric | Description |
|--------|-------------|
| **Model Return** | Mean per-trade return using model's top-k selection |
| **Momentum Return** | Mean return using 20-day momentum for selection |
| **Random Return** | Mean return from random stock selection |
| **Excess vs Random** | `Model - Random` (should be positive) |
| **Excess vs Momentum** | `Model - Momentum` (harder benchmark) |

### Simulation Metrics

Secondary metrics from simulated trading:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Total Return** | `(final_capital / initial - 1) × 100` | Cumulative profit/loss |
| **Sharpe Ratio** | `mean(returns) / std(returns) × √(trades_per_year)` | Risk-adjusted return |
| **Max Drawdown** | `max(peak - trough) / peak` | Worst peak-to-trough decline |

### Analytical Expected Return

Based on the **Fundamental Law of Active Management**:

```
E[return] ≈ IC × σ_cross × √(breadth) × TC
```

Where:
- `IC`: Information Coefficient
- `σ_cross`: Cross-sectional volatility of returns
- `breadth`: Number of independent bets per year (252 / horizon_days)
- `TC`: Transfer coefficient (≈ 0.8 for long-only top-k)

This provides a theoretical upper bound on expected returns given the measured IC.

### Bootstrap Confidence Intervals

Resampling-based uncertainty quantification:

| Metric | Description |
|--------|-------------|
| **Bootstrap Mean** | Mean return across 1000 bootstrap samples |
| **Bootstrap Std** | Standard deviation of bootstrap returns |
| **95% CI** | [5th percentile, 95th percentile] of bootstrap |
| **P(Positive)** | Probability of positive return |
| **VaR 5%** | 5th percentile return (Value at Risk) |
| **CVaR 5%** | Mean of worst 5% outcomes (Expected Shortfall) |

---

## Evaluation Process

### Standard Mode

For each fold checkpoint:

1. **Load model and test dates** from checkpoint
2. **For each test date** (non-overlapping by horizon):
   - Collect features for all stocks (up to `max_stocks`)
   - Get model predictions via batched inference
   - Compute actual future returns from price data
   - Calculate daily IC (correlation of predictions vs actuals)
   - Perform quantile analysis (top/bottom decile returns)
   - Compare against momentum and random baselines
   - Update trading simulation
3. **Aggregate metrics** across all test dates
4. **Compute statistical tests** (t-tests, bootstrap CIs)
5. **Generate plots and reports**

### Data Flow

```
Checkpoint → Model → Predictions
                         ↓
Features ─────────→ [Correlation] ← Actual Returns
                         ↓
                    Daily IC
                         ↓
              Aggregate Statistics
                         ↓
              T-tests, Bootstrap CIs
```

---

## Baselines

### Momentum Baseline

- **Definition**: Select top-k stocks by 20-day price momentum
- **Formula**: `momentum = (price_today / price_20d_ago) - 1`
- **Purpose**: Tests if model beats a simple technical strategy
- **Interpretation**: Beating momentum is harder than beating random

### Random Baseline

- **Definition**: Randomly select k stocks each day
- **Formula**: `np.random.choice(stocks, k, replace=False)`
- **Purpose**: Sanity check - any useful model must beat random
- **Interpretation**: Not beating random = no predictive value

### Why These Baselines?

1. **Random** establishes the floor - pure luck
2. **Momentum** is a well-known market anomaly that's hard to beat
3. Both are computed on the **same stocks/dates** as the model (fair comparison)
4. **Paired t-tests** account for day-to-day correlation

---

## Statistical Tests

### IC Significance (One-Sample T-Test)

Tests whether mean IC is significantly greater than zero.

```python
t_stat, p_value = stats.ttest_1samp(daily_ics, 0)
p_value = p_value / 2  # One-sided
```

**Interpretation:**
- p < 0.05: IC is significantly positive (95% confidence)
- p < 0.01: Strong evidence of predictive power
- p > 0.10: Cannot reject that IC = 0 (no evidence of skill)

### Baseline Comparison (Paired T-Test)

Tests whether model returns significantly exceed baseline returns.

```python
t_stat, p_value = stats.ttest_rel(model_returns, baseline_returns)
```

**Why paired?** Same dates, same market conditions - pairing removes market noise.

**Interpretation:**
- p < 0.05 vs Random: Model has predictive value
- p < 0.05 vs Momentum: Model beats a strong baseline
- Large t-stat (> 2): Economically meaningful difference

### Bootstrap Confidence Intervals

Non-parametric uncertainty estimation:

```python
for _ in range(1000):
    resampled = np.random.choice(daily_returns, size=n, replace=True)
    bootstrap_returns.append(compound(resampled))
ci_lower, ci_upper = np.percentile(bootstrap_returns, [5, 95])
```

**Interpretation:**
- CI excludes 0: Significantly positive/negative
- Narrow CI: Consistent performance
- Wide CI: High uncertainty, need more data

---

## Sweep Mode

Efficiently evaluates multiple configurations (horizons × top-k values) in a single run.

### Process

1. **Pre-load all data into RAM** (features and prices)
2. **For each fold**:
   - Run model inference **once** for all horizons
   - Cache predictions
3. **For each (horizon, top_k) configuration**:
   - Compute metrics using cached predictions (CPU-only)
   - No redundant GPU inference

### Efficiency

| Naive Approach | Sweep Mode |
|----------------|------------|
| 4 horizons × 5 top-k = 20 inference passes | 1 inference pass |
| ~20 minutes | ~2 minutes |

### Output

Generates a heatmap showing performance across configurations:

```
Horizon   Top-K   IC        IR        Spread    Excess
1d        5       +0.0234   +0.45     +0.082%   +0.041%
1d        10      +0.0198   +0.38     +0.071%   +0.035%
5d        5       +0.0312   +0.52     +0.156%   +0.078%
...
```

---

## Output Files

| File | Description |
|------|-------------|
| `principled_evaluation.json` | Full results in JSON format |
| `sweep_results.json` | Sweep mode results |
| `sweep_results.png` | Heatmaps of metrics across configurations |
| `ic_time_series.png` | IC over time (daily, cumulative, excess returns) |
| `fold_X_results.png` | Per-fold performance summary |

### IC Time Series Plot

Three panels showing:

1. **Daily IC**: Bar chart with 20-period rolling mean
   - Green bars: positive IC days
   - Red bars: negative IC days
   - Blue line: rolling average

2. **Cumulative IC**: Running sum of daily ICs
   - Rising line: consistent predictive power
   - Flat/declining: model losing edge

3. **Cumulative Excess Return**: Model vs random over time
   - Shows whether edge persists or is concentrated in specific periods

---

## Interpreting Results

### Strong Evidence of Skill

```
Mean IC:           +0.0450
Information Ratio: +0.82
IC P-value:        0.0012 ***
% Positive IC:     63.2%
Long-Short Spread: +0.21%
Excess vs Random:  +0.09% (p=0.008)
Excess vs Momentum:+0.04% (p=0.042)
Bootstrap P(>0):   94.2%
```

✓ IC significantly positive (p < 0.01)
✓ IR > 0.5 (consistent)
✓ Beats both baselines significantly
✓ High probability of positive return

### Weak/No Evidence

```
Mean IC:           +0.0082
Information Ratio: +0.15
IC P-value:        0.2341
% Positive IC:     51.8%
Long-Short Spread: +0.03%
Excess vs Random:  +0.01% (p=0.412)
Bootstrap P(>0):   52.1%
```

✗ IC not significantly different from zero
✗ Low IR (inconsistent)
✗ Cannot reject random performance
✗ Near coin-flip probability of profit

### Red Flags

- **Negative IC**: Model predictions are inversely correlated with returns
- **IC drops over time**: Model may be overfit to training period
- **Beats random but not momentum**: Capturing known effects, not novel alpha
- **High return but low IC**: Likely luck, won't persist

---

## Usage Examples

### Basic Evaluation

```bash
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5
```

### Sweep Mode (Recommended)

```bash
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --sweep \
    --sweep-horizons 0 1 2 3 \
    --sweep-top-ks 5 10 15 20
```

### Custom Configuration

```bash
python -m inference.principled_evaluation \
    --checkpoint-dir checkpoints/walk_forward \
    --data data/all_complete_dataset.h5 \
    --prices data/actual_prices_clean.h5 \
    --horizon-days 5 \
    --horizon-idx 1 \
    --top-k 10 \
    --max-stocks 1000 \
    --output results/eval_5d_top10
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | required | Directory with fold checkpoints |
| `--data` | required | Path to features HDF5 |
| `--prices` | None | Path to prices HDF5 |
| `--device` | cuda | Device for inference |
| `--max-stocks` | 500 | Max stocks to evaluate per day |
| `--sweep` | False | Run sweep across configurations |
| `--sweep-horizons` | 0 1 2 3 | Horizon indices (0=1d, 1=5d, 2=10d, 3=20d) |
| `--sweep-top-ks` | 5 10 15 20 25 | Top-k values to test |
| `--horizon-days` | from checkpoint | Override horizon days |
| `--top-k` | from checkpoint | Override top-k |
| `--output` | checkpoint-dir | Output directory |

---

## References

- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*
- Qian, E., Hua, R., & Sorensen, E. (2007). *Quantitative Equity Portfolio Management*
- Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
