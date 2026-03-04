# Statistical Background for Stock Prediction Analysis

This guide explains all statistical concepts used in the walk-forward training and evaluation system, assuming no prior knowledge. We build from basic concepts to advanced quantitative finance metrics.

## Table of Contents

1. [Foundations: Basic Statistics](#1-foundations-basic-statistics)
2. [Measuring Relationships: Correlation](#2-measuring-relationships-correlation)
3. [Hypothesis Testing: Is It Real or Luck?](#3-hypothesis-testing-is-it-real-or-luck)
4. [Effect Size: How Big Is the Difference?](#4-effect-size-how-big-is-the-difference)
5. [Bootstrap Methods: Uncertainty Without Assumptions](#5-bootstrap-methods-uncertainty-without-assumptions)
6. [Multiple Testing: When You Run Many Tests](#6-multiple-testing-when-you-run-many-tests)
7. [Financial Returns: Measuring Performance](#7-financial-returns-measuring-performance)
8. [Risk Metrics: Understanding Downside](#8-risk-metrics-understanding-downside)
9. [Information Coefficient: The Core Metric](#9-information-coefficient-the-core-metric)
10. [The Fundamental Law of Active Management](#10-the-fundamental-law-of-active-management)
11. [Walk-Forward Validation: Testing on Future Data](#11-walk-forward-validation-testing-on-future-data)
12. [Monte Carlo Methods: Testing Through Simulation](#12-monte-carlo-methods-testing-through-simulation)
13. [Survivorship Bias: The Hidden Trap](#13-survivorship-bias-the-hidden-trap)
14. [Putting It All Together](#14-putting-it-all-together)

---

## 1. Foundations: Basic Statistics

### Mean (Average)

The **mean** is the sum of all values divided by the count. It tells you the "center" of your data.

```
Mean = (x₁ + x₂ + ... + xₙ) / n
```

**Example**: If stock returns are [+2%, -1%, +3%, +1%, -2%], the mean is:
```
(2 + (-1) + 3 + 1 + (-2)) / 5 = 3/5 = 0.6%
```

### Median

The **median** is the middle value when data is sorted. Unlike the mean, it's not affected by extreme values (outliers).

**Example**: For returns [-10%, +1%, +2%, +3%, +100%]:
- Mean = 19.2% (pulled up by +100%)
- Median = +2% (the actual middle value)

The median often better represents "typical" performance when outliers exist.

### Standard Deviation (Std)

**Standard deviation** measures how spread out values are from the mean. Higher std = more variability.

```
Std = √[Σ(xᵢ - mean)² / (n-1)]
```

**Interpretation**:
- Low std (e.g., 1%): Returns are consistent, close to the mean
- High std (e.g., 10%): Returns vary wildly day to day

In finance, standard deviation of returns is often called **volatility**.

### Variance

**Variance** is simply the standard deviation squared:

```
Variance = Std²
```

Variance is used in many formulas, but std is easier to interpret because it's in the same units as your data.

### Percentiles

A **percentile** tells you what percentage of data falls below a value.

- **10th percentile**: 10% of values are below this
- **50th percentile**: Same as median
- **90th percentile**: 90% of values are below this

**Example**: If the 5th percentile of your returns is -8%, that means only 5% of the time did you lose more than 8%.

### Skewness

**Skewness** measures asymmetry in your data distribution.

```
Skewness = E[(x - mean)³] / std³
```

**Interpretation**:
- Skewness ≈ 0: Symmetric distribution (like a bell curve)
- Skewness > 0: Right-tailed (some very large positive values)
- Skewness < 0: Left-tailed (some very large negative values)

Stock returns often have negative skewness (crashes are more extreme than rallies).

---

## 2. Measuring Relationships: Correlation

### Pearson Correlation

**Pearson correlation** (r) measures the linear relationship between two variables. It ranges from -1 to +1.

```
r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / [(n-1) × std_x × std_y]
```

**Interpretation**:
- r = +1: Perfect positive relationship (when X goes up, Y goes up)
- r = 0: No linear relationship
- r = -1: Perfect negative relationship (when X goes up, Y goes down)

**Example**: If your model's predictions correlate with actual returns at r = 0.05, predictions explain about 0.25% of return variance (r² = 0.0025).

### Spearman Rank Correlation

**Spearman correlation** measures how well the *rankings* of two variables agree, not their actual values.

**Process**:
1. Rank both variables from lowest to highest
2. Compute Pearson correlation on the ranks

**Why use it?**
- Robust to outliers
- Captures non-linear monotonic relationships
- In stock prediction, we often care about ranking stocks correctly, not predicting exact returns

**Example**: If you predict Stock A > Stock B > Stock C, and actual returns show A > B > C, your rank correlation is perfect (+1), even if your predicted values were way off.

---

## 3. Hypothesis Testing: Is It Real or Luck?

When your model seems to work, how do you know it's skill and not luck? This is what hypothesis testing answers.

### The Null Hypothesis

The **null hypothesis (H₀)** assumes there's no real effect—any observed result is due to random chance.

**Example null hypotheses**:
- "The model has no predictive power (true IC = 0)"
- "The model performs the same as random selection"

### P-Values

The **p-value** is the probability of seeing your result (or more extreme) if the null hypothesis were true.

**Interpretation**:
- p = 0.05: 5% chance this result is pure luck
- p = 0.01: 1% chance this result is pure luck
- p = 0.001: 0.1% chance this result is pure luck

**Convention**:
- p < 0.05: "Statistically significant" (reject null hypothesis)
- p < 0.01: "Highly significant"
- p < 0.001: "Very highly significant"

**Warning**: P-values are often misunderstood. A p-value of 0.05 does NOT mean "95% chance the effect is real." It means "5% chance of seeing this if there's no effect."

### One-Tailed vs Two-Tailed Tests

- **Two-tailed**: Tests if there's any difference (positive or negative)
- **One-tailed**: Tests for difference in a specific direction

For stock prediction, we usually use one-tailed tests because we specifically want to know if the model is *better* than random, not just *different*.

```python
# Two-tailed p-value
p_two_tailed = stats.ttest_1samp(daily_ics, 0)[1]

# One-tailed p-value (testing if mean > 0)
p_one_tailed = p_two_tailed / 2
```

### T-Test

The **t-test** determines if a mean is significantly different from a hypothesized value (often 0).

#### One-Sample T-Test

Tests if the mean of your data differs from a specific value.

```python
t_stat, p_value = stats.ttest_1samp(daily_ics, 0)  # Test if mean IC ≠ 0
```

**Example**: If daily ICs average +0.03 with std of 0.10 over 100 days:
- t = 0.03 / (0.10 / √100) = 0.03 / 0.01 = 3.0
- p ≈ 0.003 (highly significant)

#### Paired T-Test

Compares two related measurements (same subjects, different conditions).

```python
t_stat, p_value = stats.ttest_rel(model_returns, random_returns)
```

**Why paired?** When comparing model vs random on the same days, market conditions affect both equally. Pairing removes this noise, making the test more powerful.

### Wilcoxon Signed-Rank Test

A **non-parametric** alternative to the paired t-test. It doesn't assume data is normally distributed.

```python
stat, p_value = stats.wilcoxon(model_returns - random_returns, alternative='greater')
```

**When to use**: When your data might have outliers or isn't bell-shaped.

### Permutation Test

A computer-intensive test that makes no distributional assumptions.

**Process**:
1. Calculate observed difference (e.g., model mean - random mean)
2. Randomly shuffle labels between groups many times (e.g., 10,000)
3. Calculate the difference each time
4. P-value = proportion of shuffled differences ≥ observed difference

```python
observed_diff = np.mean(model_returns) - np.mean(random_returns)
count_extreme = 0

for _ in range(10000):
    # Randomly swap labels
    shuffled = np.random.permutation(np.concatenate([model_returns, random_returns]))
    shuffled_diff = shuffled[:n].mean() - shuffled[n:].mean()
    if shuffled_diff >= observed_diff:
        count_extreme += 1

p_value = count_extreme / 10000
```

**Advantage**: Makes no assumptions about data distribution. Very robust.

---

## 4. Effect Size: How Big Is the Difference?

P-values tell you if an effect exists, but not how *big* it is. With enough data, tiny meaningless effects become "significant." Effect size measures magnitude.

### Cohen's d

**Cohen's d** measures the difference between means in standard deviation units.

```
d = (mean₁ - mean₂) / pooled_std
```

**Interpretation**:
| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

**Example**: If model returns average 5% with std 10%, and random averages 3% with std 10%:
```
d = (5 - 3) / 10 = 0.2 (small effect)
```

### Hedges' g

**Hedges' g** is Cohen's d with a correction for small samples:

```
g = d × (1 - 3/(4n - 9))
```

Use Hedges' g when sample size is small (< 50).

---

## 5. Bootstrap Methods: Uncertainty Without Assumptions

**Bootstrapping** estimates uncertainty by resampling your data with replacement.

### Basic Bootstrap

**Process**:
1. Take your data (n observations)
2. Randomly sample n observations *with replacement* (some appear multiple times, some not at all)
3. Calculate statistic of interest (mean, median, etc.)
4. Repeat many times (e.g., 10,000)
5. The distribution of these statistics shows your uncertainty

```python
def bootstrap_mean_ci(data, n_bootstrap=10000, ci=95):
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, 100 - (100-ci)/2)
    return lower, upper
```

### Confidence Intervals

A **95% confidence interval** means: if we repeated the experiment many times, 95% of the calculated intervals would contain the true value.

**Example**: Mean return = 5%, 95% CI = [2%, 8%]

This tells you the true mean is likely between 2% and 8%.

### Block Bootstrap

Standard bootstrap assumes observations are independent. Financial data is **autocorrelated** (today's return is related to yesterday's). Block bootstrap preserves this structure.

**Process**:
1. Divide data into blocks of consecutive observations
2. Randomly sample blocks (not individual points)
3. This preserves time-series patterns

```python
def block_bootstrap(data, block_size=5, n_bootstrap=1000):
    n = len(data)
    results = []

    for _ in range(n_bootstrap):
        # Sample block starting positions
        starts = np.random.randint(0, n - block_size + 1, size=n // block_size + 1)
        # Build resampled series from blocks
        resampled = np.concatenate([data[s:s+block_size] for s in starts])[:n]
        results.append(compute_statistic(resampled))

    return results
```

---

## 6. Multiple Testing: When You Run Many Tests

If you test 20 strategies and declare "significant" at p < 0.05, you expect 1 false positive by chance alone (20 × 0.05 = 1). This is the **multiple comparisons problem**.

### Bonferroni Correction

The simplest fix: divide your significance threshold by the number of tests.

```
α_adjusted = α / n_tests
```

**Example**: Testing 20 strategies at α = 0.05:
- Adjusted threshold: 0.05 / 20 = 0.0025
- Only declare significant if p < 0.0025

**Disadvantage**: Very conservative. May miss real effects.

### Holm-Bonferroni

A less conservative step-down procedure:

1. Sort p-values from smallest to largest
2. Compare smallest p-value to α/n
3. If significant, compare next to α/(n-1)
4. Continue until one fails

```python
def holm_bonferroni(p_values, alpha=0.05):
    n = len(p_values)
    sorted_idx = np.argsort(p_values)

    for i, idx in enumerate(sorted_idx):
        threshold = alpha / (n - i)
        if p_values[idx] > threshold:
            # This and all larger p-values are not significant
            break

    return significant_indices
```

### False Discovery Rate (FDR) - Benjamini-Hochberg

Controls the expected proportion of false positives among rejected hypotheses.

**Process**:
1. Sort p-values
2. Find largest i where p(i) ≤ (i/n) × α
3. Reject all hypotheses with smaller p-values

**Advantage**: More powerful than Bonferroni when many true effects exist.

---

## 7. Financial Returns: Measuring Performance

### Simple Returns

The **simple return** over a period:

```
R = (P_end - P_start) / P_start = P_end/P_start - 1
```

**Example**: Buy at $100, sell at $105:
```
R = (105 - 100) / 100 = 0.05 = 5%
```

### Compound Returns

Returns accumulate multiplicatively, not additively.

**Multi-period compound return**:
```
R_total = (1 + R₁) × (1 + R₂) × ... × (1 + Rₙ) - 1
```

**Example**: Three periods with returns +10%, -5%, +8%:
```
R_total = 1.10 × 0.95 × 1.08 - 1 = 1.1286 - 1 = 12.86%
```

Note: Simple sum would give 10 - 5 + 8 = 13%, which is wrong.

### Log Returns

**Log returns** (continuously compounded):

```
r = ln(P_end / P_start)
```

**Advantage**: Log returns are additive over time:
```
r_total = r₁ + r₂ + ... + rₙ
```

**Converting**: Simple return R and log return r are related by:
```
r = ln(1 + R)
R = e^r - 1
```

For small returns, they're approximately equal. For large returns, they differ significantly.

### Annualized Returns

To compare returns across different time periods, we annualize:

```
R_annual = (1 + R_total)^(252/n_trading_days) - 1
```

**Example**: 15% return over 6 months (126 trading days):
```
R_annual = (1.15)^(252/126) - 1 = 1.15² - 1 = 32.25%
```

---

## 8. Risk Metrics: Understanding Downside

### Sharpe Ratio

The **Sharpe ratio** measures risk-adjusted return: how much return you get per unit of risk.

```
Sharpe = (R_portfolio - R_risk_free) / σ_portfolio
```

Where:
- R_portfolio = portfolio return
- R_risk_free = risk-free rate (often set to 0 for simplicity)
- σ_portfolio = standard deviation of returns

**Annualized Sharpe** (assuming daily returns):
```
Sharpe_annual = Sharpe_daily × √252
```

**Interpretation**:
| Sharpe Ratio | Quality |
|--------------|---------|
| < 0 | Bad (losing money) |
| 0 - 1 | Acceptable |
| 1 - 2 | Good |
| 2 - 3 | Very good |
| > 3 | Excellent (or suspicious) |

### Maximum Drawdown

**Maximum drawdown (MDD)** is the largest peak-to-trough decline:

```
MDD = max[(Peak - Trough) / Peak]
```

**Example**: Portfolio value over time: $100 → $120 → $90 → $110
- Peak: $120
- Trough: $90
- Drawdown: (120 - 90) / 120 = 25%

MDD tells you the worst-case loss an investor would have experienced.

### Value at Risk (VaR)

**VaR** at confidence level α is the maximum loss expected with probability (1-α).

**Example**: "5% VaR is -$10,000" means there's a 5% chance of losing more than $10,000.

```python
var_5pct = np.percentile(returns, 5)  # 5th percentile of returns
```

**Interpretation**: The 5th percentile return is your 5% VaR.

### Conditional VaR (CVaR) / Expected Shortfall

**CVaR** is the average loss in the worst α% of cases. It answers: "When things go wrong, how bad is it on average?"

```python
var_5pct = np.percentile(returns, 5)
cvar_5pct = returns[returns <= var_5pct].mean()
```

**Example**: If 5% VaR is -8% but CVaR is -15%, on the worst 5% of days you lose 15% on average.

CVaR is considered better than VaR because it captures tail risk.

---

## 9. Information Coefficient: The Core Metric

The **Information Coefficient (IC)** is the primary metric for evaluating prediction quality in quantitative finance.

### Definition

IC is the correlation between predictions and actual returns:

```
IC = correlation(predictions, actual_returns)
```

Typically calculated daily (cross-sectionally across all stocks on each day) and then averaged.

### Daily IC Calculation

For each day:
1. Get predictions for all stocks
2. Get actual forward returns for all stocks
3. Calculate correlation

```python
daily_ic = pearsonr(predictions_day_t, actual_returns_day_t)[0]
```

### Mean IC

Average IC across all days:

```
Mean IC = (1/T) × Σ IC_t
```

**Interpretation**:
| Mean IC | Quality |
|---------|---------|
| < 0 | Model predicts wrong direction |
| 0 - 0.02 | Weak (possibly noise) |
| 0.02 - 0.05 | Moderate (potentially useful) |
| 0.05 - 0.10 | Good |
| > 0.10 | Excellent (rare in practice) |

### Information Ratio (IR)

The **Information Ratio** is the IC's Sharpe ratio:

```
IR = Mean(IC) / Std(IC)
```

It measures IC consistency. High IC with high variance is less valuable than moderate IC with low variance.

**Interpretation**:
| IR | Quality |
|----|---------|
| < 0.25 | Poor consistency |
| 0.25 - 0.50 | Moderate |
| 0.50 - 0.75 | Good |
| > 0.75 | Excellent |

### Rank IC (Spearman)

Uses rank correlation instead of Pearson:

```
Rank IC = spearman_correlation(predictions, actual_returns)
```

More robust to outliers. Often preferred in practice.

### Why IC Matters

Unlike backtest returns, IC:
1. **Directly measures prediction quality** (independent of portfolio construction)
2. **Is statistically stable** (computed daily, averaged)
3. **Enables analytical return estimation** (via Fundamental Law)
4. **Is harder to overfit** than trading returns

---

## 10. The Fundamental Law of Active Management

This law connects IC to expected portfolio returns.

### The Formula

```
E[Excess Return] ≈ IC × σ × √BR × TC
```

Where:
- **IC**: Information Coefficient
- **σ**: Cross-sectional volatility of returns (typically ~2% daily)
- **BR**: Breadth (number of independent bets per year)
- **TC**: Transfer Coefficient (implementation efficiency, typically 0.5-0.8)

### Understanding Each Component

#### Breadth (BR)

Number of independent investment decisions per year.

- Trading daily: BR ≈ 252
- Trading weekly: BR ≈ 52
- Holding 100 stocks, trading monthly: BR ≈ 100 × 12 = 1,200

More bets = more chances for edge to show.

#### Transfer Coefficient (TC)

How much of your signal gets into the portfolio.

- TC = 1: Perfect implementation (unrealistic)
- TC = 0.8: Typical for long-only portfolio
- TC = 0.5: Constrained portfolio

Constraints (sector limits, position limits) reduce TC.

### Example Calculation

With:
- IC = 0.05
- σ = 2% daily
- BR = 252 (daily trading)
- TC = 0.8

```
E[Return] = 0.05 × 0.02 × √252 × 0.8
         = 0.05 × 0.02 × 15.87 × 0.8
         = 0.0127 = 1.27% annual excess return
```

This shows that even a "good" IC of 0.05 only translates to modest excess returns.

### Implications

1. **Small ICs are valuable**: Even IC = 0.03 can generate meaningful returns with high breadth
2. **Frequency matters**: Trading more often (higher BR) amplifies small edges
3. **Implementation matters**: TC significantly impacts realized returns

---

## 11. Walk-Forward Validation: Testing on Future Data

### The Problem with Backtesting

If you optimize a model on historical data and test on the same data, you get overly optimistic results. The model "memorizes" the past.

### Walk-Forward Solution

**Walk-forward validation** trains on past data and tests on future data that the model has never seen.

```
Time ──────────────────────────────────────────────────►

Fold 1: [====TRAIN====][TEST]
Fold 2: [======TRAIN======][TEST]
Fold 3: [========TRAIN========][TEST]
Fold 4: [==========TRAIN==========][TEST]
```

Each fold:
1. Train on all data up to a point
2. Test on the next period (truly out-of-sample)
3. Move forward in time, repeat

### Expanding vs Rolling Window

**Expanding window** (shown above): Training data grows over time
- Pro: More training data in later folds
- Con: Old data may be less relevant

**Rolling window**: Fixed training window that slides forward
- Pro: Only uses recent, relevant data
- Con: Less training data

### Why It Matters

Walk-forward validation:
1. **Simulates real trading** (you only have past data when making decisions)
2. **Prevents look-ahead bias** (using future information)
3. **Tests regime changes** (model must adapt to new market conditions)
4. **Provides multiple test periods** (more robust than single test)

### Interpreting Results

When evaluating walk-forward results:
- **Consistent performance across folds**: Model is robust
- **Performance degrades in recent folds**: Market regime changed or model overfit to old data
- **High variance across folds**: Model is unstable, results may not persist

---

## 12. Monte Carlo Methods: Testing Through Simulation

### The Idea

Instead of deriving analytical formulas, simulate many random trials and observe what happens.

### Monte Carlo for Trading Strategies

**Process**:
1. Define a baseline (e.g., random stock selection)
2. Run many trials with random variations
3. Compare model performance to this distribution

```python
n_trials = 1000
random_returns = []

for _ in range(n_trials):
    # Randomly select stocks each day
    selected = np.random.choice(all_stocks, k, replace=False)
    trial_return = compute_return(selected)
    random_returns.append(trial_return)

# Compare model to random distribution
model_percentile = np.mean(random_returns < model_return) * 100
```

If model beats 95% of random trials, you have evidence of skill.

### Benefits

1. **No distributional assumptions**: Don't need to assume normal distribution
2. **Handles complexity**: Works when analytical solutions are impossible
3. **Intuitive**: "The model beat random 97% of the time"

### Our Implementation

The codebase runs Monte Carlo validation:
1. Randomly sample subsets of stocks
2. For each subset, compute model returns vs random selection
3. Test if model consistently beats random
4. Verify analytical IC predictions match simulation

---

## 13. Survivorship Bias: The Hidden Trap

### The Problem

**Survivorship bias** occurs when you only analyze data from entities that "survived" (still exist today), ignoring those that failed.

### In Stock Prediction

If you backtest on today's S&P 500 stocks:
- You exclude companies that went bankrupt (100% loss)
- You exclude companies that were delisted (often after large losses)
- You only include "survivors"

This makes *any* strategy look better than it would have in real-time.

### Example

Imagine in 2010 you could invest in 500 stocks:
- 490 are still trading in 2024
- 10 went bankrupt (lost 100%)

If you backtest only on the 490 survivors:
- Random selection looks great (+50% over period)
- But real random selection would have hit some bankruptcies
- Actual return might be +35%

### How to Detect

**Signs of survivorship bias**:
1. Random selection shows suspiciously high returns
2. Very few negative outliers in return distribution
3. Win rate is too high (e.g., 60%+ of trades profitable)

### How to Mitigate

1. **Use point-in-time data**: Stock universe as it existed at each date
2. **Include delisted stocks**: Track what happened when stocks were removed
3. **Compare to benchmarks**: Random selection should match market return
4. **Check return distribution**: Should see some extreme losses

### Our Diagnostics

The codebase includes survivorship bias warnings:
```
Market (EW) Return:        +28.5%
Avg Random Return:         +29.2%
Random ≈ Market:           ✓ Confirms no selection bias (+0.7% diff)
```

If random matches market, survivorship bias isn't causing the results.

---

## 14. Putting It All Together

### The Evaluation Pipeline

1. **Train Model** (walk-forward)
   - Train on historical data
   - Validate on held-out period
   - Repeat across multiple folds

2. **Compute Primary Metrics**
   - Daily IC: Correlation of predictions with returns
   - Mean IC: Average predictive power
   - IR: Consistency of IC

3. **Statistical Significance**
   - T-test: Is IC significantly > 0?
   - P-value: Probability this is luck
   - Bootstrap CI: Range of likely values

4. **Compare to Baselines**
   - Random: Does model beat chance?
   - Momentum: Does model beat simple strategy?
   - Paired tests: Account for market conditions

5. **Effect Size**
   - Cohen's d: Practical significance
   - Not just statistically significant, but meaningfully better

6. **Monte Carlo Validation**
   - Test many configurations
   - Verify analytical predictions match simulation
   - Check for survivorship bias

### Interpreting Results

**Strong evidence of skill**:
```
Mean IC:           +0.045
Information Ratio: +0.82
IC P-value:        0.0012 ***
Excess vs Random:  +0.09% (p=0.008)
Cohen's d:         +0.35
Bootstrap P(>0):   94.2%
Random ≈ Market:   ✓ (no bias)
```

**Weak/No evidence**:
```
Mean IC:           +0.008
Information Ratio: +0.15
IC P-value:        0.234
Excess vs Random:  +0.01% (p=0.412)
Cohen's d:         +0.05
Bootstrap P(>0):   52.1%
```

### Red Flags

- **Negative IC**: Model predicts the wrong direction
- **High return but low IC**: Probably luck, won't persist
- **Random beats model**: Something is wrong
- **Random >> Market**: Survivorship bias
- **p-value drops with more data**: Effect is noise, not signal

### Key Takeaways

1. **IC is king**: Focus on prediction quality, not backtest returns
2. **Statistical significance isn't enough**: Check effect size too
3. **Multiple testing matters**: Adjust for many comparisons
4. **Walk-forward is essential**: In-sample results are worthless
5. **Check for bias**: Survivorship bias inflates all results
6. **Monte Carlo validates**: Simulation should match theory
7. **Consistency matters**: IR tells you if IC is reliable

---

## Glossary

| Term | Definition |
|------|------------|
| **Autocorrelation** | Correlation of a series with its lagged values |
| **Breadth** | Number of independent bets per time period |
| **Confidence Interval** | Range likely to contain true parameter value |
| **Cross-sectional** | Across different stocks at the same time |
| **Effect Size** | Magnitude of a difference (e.g., Cohen's d) |
| **IC** | Information Coefficient - correlation of predictions with returns |
| **IR** | Information Ratio - risk-adjusted IC (mean IC / std IC) |
| **MDD** | Maximum Drawdown - largest peak-to-trough decline |
| **Null Hypothesis** | Assumption of no effect (to be disproven) |
| **OOS** | Out-of-sample (data not used for training) |
| **P-value** | Probability of result if null hypothesis true |
| **Sharpe Ratio** | Risk-adjusted return (return / volatility) |
| **Survivorship Bias** | Bias from only analyzing entities that survived |
| **TC** | Transfer Coefficient - implementation efficiency |
| **VaR** | Value at Risk - maximum expected loss at confidence level |
| **Walk-forward** | Training on past, testing on future, repeatedly |

---

## References

1. Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*
2. Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
3. Harvey, C. & Liu, Y. (2015). "Backtesting"
4. Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*
5. Qian, E., Hua, R., & Sorensen, E. (2007). *Quantitative Equity Portfolio Management*
