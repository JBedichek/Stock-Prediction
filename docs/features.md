# Feature Documentation

## Overview

The dataset contains **998 features** per stock per day, composed of:

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Base Features | 230 | Fundamentals, technicals, market-relative |
| News Embeddings | 768 | Nomic V1.5 text embeddings |
| **Total** | **998** | |

---

## Detailed Feature Breakdown

### 1. Fundamental/Quarterly Metrics (~100-120 features)

Sourced from FMP API (`fmp_comprehensive_scraper.py`):

#### Financial Statements (~40-50 features)
- **Income Statement**: revenue, netIncome, operatingIncome, ebitda, costOfGoodsSold, grossProfit, operatingExpenses, interestExpense, incomeBeforeTax, incomeTaxExpense
- **Balance Sheet**: totalAssets, totalLiabilities, stockholdersEquity, currentAssets, currentLiabilities, longTermDebt, shortTermDebt, totalDebt, cashAndEquivalents, inventory, receivables
- **Cash Flow**: operatingCashFlow, freeCashFlow, investingCashFlow, financingCashFlow, capitalExpenditures, dividendsPaid

#### Key Metrics (~25-30 features)
- **Valuation**: marketCap, enterpriseValue, peRatio, pbRatio, psRatio, pcfRatio, pegRatio, evToSales, evToEbitda
- **Profitability**: roe, roa, roic, grossMargin, operatingMargin, netMargin, profitMargin, ebitdaMargin
- **Efficiency**: assetTurnover, inventoryTurnover, receivablesTurnover, payablesTurnover
- **Growth**: revenueGrowth, earningsGrowth, bookValueGrowth, epsGrowth, dividendGrowth
- **Per-Share**: eps, bookValuePerShare, revenuePerShare, cashPerShare, freeCashFlowPerShare
- **Financial Health**: debtToEquity, currentRatio, quickRatio, debtRatio, interestCoverage

#### Enterprise Values & Financial Ratios (~30-40 features)
- Additional valuation metrics from FMP's enterprise_values endpoint
- Extended ratio metrics from financial_ratios endpoint
- Growth rate metrics from financial_growth endpoint

**Note**: Quarterly data is forward-filled to daily frequency.

---

### 2. Technical Indicators (~40-50 features)

From `derived_features_calculator.py`:

#### Volume Features (8-10 features)
| Feature | Description |
|---------|-------------|
| volume_ma_5d, 10d, 20d, 50d | Volume moving averages |
| volume_ratio_5d, 10d, 20d, 50d | Current volume / MA |
| dollar_volume | Price × Volume |
| dollar_volume_ma_20 | 20-day MA of dollar volume |
| volume_spike_2x, 3x | Binary: volume > 2x/3x MA |
| volume_trend | 5-day polyfit slope |
| volume_zscore | Normalized volume |

#### Price Features (18-20 features)
| Feature | Description |
|---------|-------------|
| intraday_volatility | (High - Low) / Open |
| volatility_ma_20 | 20-day MA of intraday volatility |
| gap_pct | (Open - Prev Close) / Prev Close |
| gap_up, gap_down | Binary gap indicators |
| high_low_ratio | High / Low |
| close_position | Where close falls in daily range [0,1] |
| dist_from_52w_high | % distance from 52-week high |
| dist_from_52w_low | % distance from 52-week low |
| dist_from_sma_20, 50, 200 | % distance from moving averages |
| sma20_above_sma50 | Binary trend indicator |
| sma50_above_sma200 | Binary trend indicator |
| return_1d, 5d, 10d, 20d | Percentage returns |
| realized_vol_5d, 20d | Rolling std of returns |

#### Bollinger Bands (7 features)
| Feature | Description |
|---------|-------------|
| bb_upper, bb_middle, bb_lower | Band values |
| bb_width | (Upper - Lower) / Middle |
| bb_position | Where price is between bands [0,1] |
| bb_squeeze | Width below threshold |
| touching_upper_bb, touching_lower_bb | Binary indicators |

#### Momentum Features (6 features)
| Feature | Description |
|---------|-------------|
| roc_5d, 10d, 20d | Rate of change |
| momentum_5d, 10d | Price differences |
| acceleration | Change in momentum |

---

### 3. Market-Relative Features (~15-20 features)

From `derived_features_calculator.py`:

#### S&P 500 Relative (~8 features)
| Feature | Description |
|---------|-------------|
| beta_spy | 60-day rolling beta |
| correlation_spy | 60-day rolling correlation |
| relative_return_1d, 5d, 20d | Stock return - SPY return |
| relative_performance | Cumulative relative return |

#### VIX Features (3 features)
| Feature | Description |
|---------|-------------|
| vix_level | Current VIX value |
| vix_above_20 | Binary: fear threshold |
| vix_above_30 | Binary: extreme fear |

#### Sector ETF Relative (~8 features)
Same structure as S&P 500 relative, computed vs sector ETF.

---

### 4. Cross-Sectional Percentile Rankings (~7-10 features)

From `cross_sectional_calculator.py`:

| Feature | Description |
|---------|-------------|
| return_1d_percentile | Percentile rank of 1-day return |
| return_5d_percentile | Percentile rank of 5-day return |
| return_20d_percentile | Percentile rank of 20-day return |
| volume_percentile | Percentile rank of volume |
| dollar_volume_percentile | Percentile rank of dollar volume |
| volatility_percentile | Percentile rank of volatility |
| volume_ratio_percentile | Percentile rank of volume ratio |
| return_1d_vs_sector | Return relative to sector median |
| return_5d_vs_sector | Return relative to sector median |
| volume_ratio_vs_sector | Volume ratio relative to sector |

---

### 5. News Embeddings (768 features)

From `news_embedder.py`:

- **Model**: Nomic Embed V1.5 (768-dimensional)
- **Content**: Averaged embedding of all news articles for that stock on that day
- **Fallback**: Zero vector if no news available
- **Normalization**: Pre-normalized by Nomic model (NOT cross-sectionally normalized)

---

## Normalization Strategy

### Cross-Sectional Normalization (across stocks, per date)

Applied to fundamental metrics to preserve relative ranking:
```
z_score = (value - mean_across_stocks) / std_across_stocks
```

**Features normalized this way:**
- All financial statement items
- Key metrics (PE, ROE, margins, etc.)
- Growth metrics
- Per-share metrics

### Temporal Normalization (per stock, over time)

Applied to price-based features:
```
z_score = (value - mean_over_time) / std_over_time
```

**Features normalized this way:**
- Returns, volatility
- Volume ratios
- Technical indicators (RSI, MACD, Bollinger)
- Beta, correlation

### No Normalization

**Features passed through as-is:**
- Binary indicators (gap_up, gap_down, etc.)
- Percentile features (already 0-100)
- Ratios already normalized by design
- News embeddings (pre-normalized)

---

## Known Issues & Precision Concerns

### 1. Scale Disparities Before Normalization

| Metric Type | Typical Range | After Z-Score |
|-------------|---------------|---------------|
| Market Cap | 1e6 - 3e12 | -2 to +3 |
| Ratios | 0 - 100 | -2 to +3 |
| Returns | -0.5 to +0.5 | -3 to +3 |
| News Embeddings | -1 to +1 | -1 to +1 |

**Problem**: After normalization, all features are z-scores (~same scale), but the original magnitude information is lost.

### 2. Quarterly Data Forward-Fill

Fundamental data is quarterly but forward-filled to daily:
- Same value repeated for ~63 trading days
- Creates artificial temporal correlation
- Model may learn spurious patterns from constant values

### 3. Missing Data Handling

- NaN values replaced with 0 after normalization
- Zero std features set to 0
- Padding missing columns with 0

**Risk**: Zero becomes ambiguous (missing vs. actual zero vs. market median).

### 4. News Embedding Scale Mismatch

- Base features (post-normalization): typically -3 to +3
- News embeddings: typically -1 to +1

News may be underweighted due to smaller magnitude.

### 5. Float32 Precision

All features stored as float32:
- Precision: ~7 significant digits
- Range: ±3.4e38

**Potential Issue**: When combining features with very different original scales, precision loss may occur in early layers.

---

---

## BUG FIX: Missing Daily Price Features (March 2026)

### Root Cause Identified

The `EnhancedFMPDataProcessor` class had a bug where yfinance price data was passed to the constructor but never used:

```python
# BUG: process_derived_features() looked for:
stock_prices = self.raw_data.get('daily_prices')  # Always None!

# FIX: Now uses yfinance price data passed to constructor:
stock_prices = self.price_data if self.price_data is not None else self.raw_data.get('daily_prices')
```

This affected both `process_derived_features()` and `process_market_relative_features()` in `data_scraping/fmp_enhanced_processor.py`.

### Before Fix (998 features)

| Feature Range | Autocorrelation | Unique Values | Behavior |
|---------------|-----------------|---------------|----------|
| 0-223 | > 0.99 | 1-7% | **Nearly constant** (quarterly, forward-filled) |
| 224-229 | 0.01-0.63 | 48-62% | **Daily variation** (cross-sectional percentiles) |
| 230-997 | N/A | 2% | **98% zeros** (missing news data) |

Only 6 features changed daily. The model saw essentially the same input every day.

### After Fix (Expected ~1048 features)

| Component | Features | Daily Variation |
|-----------|----------|-----------------|
| Fundamentals | ~230 | Low (quarterly) |
| **Derived (NEW)** | **~50** | **High (daily)** |
| Cross-sectional percentiles | ~6 | High (daily) |
| News embeddings | 768 | Variable |

**New daily-varying derived features include:**
- `return_1d`, `return_5d`, `return_10d`, `return_20d` - Daily returns
- `volume_ratio_5d`, `volume_ratio_10d`, `volume_ratio_20d` - Volume vs MA
- `intraday_volatility`, `gap_pct` - Daily price dynamics
- `dist_from_sma20`, `dist_from_sma50`, `dist_from_sma200` - Moving average distances
- `bb_position`, `bb_width`, `bb_squeeze` - Bollinger Bands
- `roc_5`, `roc_10`, `roc_20`, `momentum_5` - Momentum indicators
- `beta`, `correlation`, `relative_return_1d` - Market-relative metrics

---

## Recommendations

### For Precision Issues

1. **First layer fp64**: Use float64 for input projection to preserve precision
2. **LayerNorm**: Apply before combining different feature groups
3. **Separate pathways**: Process fundamentals, technicals, and news through separate initial layers

### For Missing Signal

1. **Verify temporal alignment**: Ensure features at time T don't contain info from T+1
2. **Check normalization leakage**: Cross-sectional stats should only use data up to time T
3. **Raw price inclusion**: Consider including un-normalized close prices as additional features

### For Feature Engineering

1. **Shorter-term features**: Add 1-5 day momentum/mean-reversion signals
2. **Interaction features**: Combine price momentum with volume confirmation
3. **Regime features**: Market state indicators (trending vs. mean-reverting)
