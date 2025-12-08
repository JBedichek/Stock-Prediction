# FMP Daily Features - Actually Available

## Summary

After testing your API key, here's what **daily data beyond prices** is actually accessible:

### ✅ **WORKING** (Available with your current API plan)

#### 1. **Market Indices** (Daily OHLCV)
- S&P 500 (^GSPC)
- Nasdaq (^IXIC)
- VIX Volatility Index (^VIX)
- Dow Jones (^DJI)
- Russell 2000 (^RUT)
- Sector ETFs (XLF, XLE, XLK, XLV, XLI, XLP, XLY, XLU, XLB)

**Features Added:** ~10-20 daily features per stock
- Stock beta vs S&P 500
- Correlation with market indices
- Relative performance vs sector ETF
- VIX level (market fear gauge)
- Market regime indicators (bull/bear)

#### 2. **Derived Volume Features** (From existing OHLCV)
**No additional API calls needed!**

- Volume ratio vs 20-day average
- Dollar volume (price × volume)
- Volume trend (5-day, 20-day MA)
- Volume spike detection (>2x average)
- Relative volume vs sector/market
- Price-volume correlation

**Features Added:** ~8-12 features per stock

#### 3. **Derived Price Features** (From existing OHLCV)
**No additional API calls needed!**

- Intraday volatility: (High - Low) / Close
- Gap analysis: Open / PreviousClose - 1
- Price ranges: 52-week high/low distance
- Support/resistance levels
- Bollinger Bands (from SMA + std dev)
- VWAP approximation
- Returns correlation with indices

**Features Added:** ~10-15 features per stock

#### 4. **Additional Technical Indicators**
**Can add more via existing endpoint:**

Currently have: SMA, EMA, RSI, ADX, Williams %R

Can add:
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- CCI (Commodity Channel Index)
- ATR (Average True Range)
- Parabolic SAR
- Ichimoku Cloud components

**Features Added:** ~10-20 features per stock
**Additional API calls:** ~5-10 per stock

#### 5. **Cross-Sectional Features** (Derived from all stocks)
**No additional API calls needed!**

For each stock, compute:
- Percentile rank by return (1-day, 5-day, 20-day)
- Percentile rank by volume
- Percentile rank by volatility
- Sector-relative metrics (vs sector median)
- Momentum rank (vs all stocks)

**Features Added:** ~8-12 features per stock

### ❌ **NOT WORKING** (Require premium/higher-tier plan)

These all returned 403 Forbidden:
- Treasury rates
- Economic calendar
- Sector/industry P/E ratios
- Sector performance
- Top gainers/losers/actives
- Earnings calendar
- Dividend calendar
- Daily DCF
- Stock news/sentiment
- Press releases

## Recommended Implementation

### **Phase 1: Market Indices (Easy Win)**
Add market-wide data collection:

```python
def get_market_indices_data(self, from_date, to_date):
    """Get market indices (collected once for all stocks)"""
    indices = {
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'VIX': '^VIX',
        'DJI': '^DJI',
        'RUT': '^RUT',
    }

    sector_etfs = {
        'XLF': 'Financial',
        'XLE': 'Energy',
        'XLK': 'Technology',
        'XLV': 'Healthcare',
        'XLI': 'Industrial',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLU': 'Utilities',
        'XLB': 'Materials',
    }

    # Collect OHLCV for each index/ETF
    # Return: {date: {index: ohlcv_tensor}}
```

**Benefits:**
- Only ~15 API calls total (not per stock!)
- Adds 15-25 daily features per stock
- Market context is crucial for predictions

### **Phase 2: Derived Features (Zero API Cost)**
Calculate from existing price/volume data:

```python
def calculate_derived_features(self, price_df):
    """Calculate features from OHLCV data"""

    # Volume features
    features['volume_ratio_20d'] = volume / volume.rolling(20).mean()
    features['dollar_volume'] = close * volume
    features['volume_spike'] = (volume > 2 * volume.rolling(20).mean()).astype(float)

    # Price features
    features['intraday_volatility'] = (high - low) / close
    features['gap_pct'] = open / close.shift(1) - 1
    features['dist_from_52w_high'] = (close / close.rolling(252).max()) - 1

    # Technical patterns
    features['bollinger_upper'] = sma_20 + 2 * std_20
    features['bollinger_lower'] = sma_20 - 2 * std_20
    features['bollinger_position'] = (close - bollinger_lower) / (bollinger_upper - bollinger_lower)

    return features
```

**Benefits:**
- Zero API calls
- Adds 20-30 features per stock
- Proven predictive value

### **Phase 3: Additional Technical Indicators**
Expand technical indicator collection:

```python
# Add to get_all_technical_indicators()
('macd_12_26', 'macd', 12),  # MACD
('macd_signal', 'macd', 26),
('stoch_14', 'stoch', 14),    # Stochastic
('cci_20', 'cci', 20),        # CCI
('atr_14', 'atr', 14),        # ATR (volatility)
```

**Benefits:**
- ~5-10 additional API calls per stock
- Adds 10-15 proven technical indicators
- Widely used by traders

### **Phase 4: Cross-Sectional Rankings**
Compute relative metrics across all stocks:

```python
def calculate_cross_sectional_features(self, all_stocks_data, date):
    """Calculate percentile ranks for each stock"""

    # Get all stocks' returns for this date
    returns_1d = {stock: data[date]['return_1d'] for stock, data in all_stocks_data.items()}

    # Percentile rank
    for stock in all_stocks_data:
        features['return_1d_percentile'] = percentileofscore(returns_1d.values(), returns_1d[stock])
        features['volume_percentile'] = percentileofscore(volumes.values(), volume[stock])
        features['volatility_percentile'] = percentileofscore(volatilities.values(), vol[stock])

    return features
```

**Benefits:**
- Zero API calls
- Adds 10-15 relative positioning features
- Captures cross-sectional alpha

## Total Feature Expansion

| Phase | Features Added | API Calls Added | Difficulty |
|-------|----------------|-----------------|------------|
| **Phase 1: Market Indices** | 15-25 | ~15 total (not per-stock!) | Easy |
| **Phase 2: Derived Features** | 20-30 | 0 | Easy |
| **Phase 3: More Technicals** | 10-15 | ~5-10 per stock | Medium |
| **Phase 4: Cross-Sectional** | 10-15 | 0 | Medium |
| **TOTAL** | **55-85 new daily features** | **Minimal** | **High ROI** |

## Implementation Plan

1. **Start with Phase 1** - Market indices give great ROI
   - Modify `fmp_comprehensive_scraper.py` to add `get_market_indices()`
   - Store in separate file: `market_indices_data.pkl`
   - Merge with stock data in `fmp_data_processor.py`

2. **Add Phase 2** - Derived features in processor
   - Modify `fmp_data_processor.py` to add `calculate_derived_features()`
   - No changes to scraper needed

3. **Expand Phase 3** - More technical indicators
   - Update `get_all_technical_indicators()` in scraper
   - Minimal API cost increase

4. **Optimize Phase 4** - Cross-sectional features
   - Add to dataset creation in `dataset_processing.py`
   - Compute on-the-fly during training data generation

## Next Steps

Want me to:
1. **Implement Phase 1** (Market indices scraper + processor)?
2. **Implement Phase 2** (Derived features calculator)?
3. **Test the expanded features** on a small dataset?
4. **Create integration guide** for your existing pipeline?

Let me know which phase you want to start with!
