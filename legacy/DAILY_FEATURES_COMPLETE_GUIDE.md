# Complete Guide: Daily Feature Expansion

## Overview

This guide covers the **complete implementation** of all 4 phases of daily feature expansion for the Stock Prediction model. We've added **50-100+ new daily features** beyond just prices, using FMP API data and derived calculations.

## Summary of All Phases

| Phase | Features Added | API Calls | Status |
|-------|----------------|-----------|--------|
| **Phase 1: Market Indices** | 15-25 | ~15 total | ✅ Complete |
| **Phase 2: Derived Features** | 40-50 | 0 (free!) | ✅ Complete |
| **Phase 3: Extended Technicals** | 10-20 | ~10 per stock | ✅ Complete |
| **Phase 4: Cross-Sectional** | 15-20 | 0 (free!) | ✅ Complete |
| **TOTAL** | **80-115 features** | **Minimal** | ✅ Complete |

---

## Phase 1: Market Indices

### What It Does
Scrapes daily OHLCV data for market indices and sector ETFs, then calculates market-relative features for each stock.

### Data Collected
- **Major Indices:** S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX
- **Sector ETFs:** XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLB, XLU, XLRE, XLC
- **Additional ETFs:** SPY, QQQ, IWM, TLT, GLD, USO

### Features Created
Per stock, adds:
- Stock beta vs S&P 500 (rolling 60-day)
- Correlation with major indices
- Relative return vs market (1d, 5d, 20d)
- VIX level and regime indicators
- Sector-relative performance
- Market regime features (bull/bear signals)

**Total: ~15-25 features per stock**

### Usage

```bash
# Step 1: Scrape market indices (one-time, covers all stocks)
python data_scraping/market_indices_scraper.py \
    --api_key YOUR_KEY \
    --from_date 2000-01-01 \
    --output market_indices_data.pkl

# Output: market_indices_data.pkl (~15 API calls total)
```

### Code Example

```python
from data_scraping.market_indices_scraper import MarketIndicesScraper

scraper = MarketIndicesScraper(api_key="YOUR_KEY")

# Scrape all indices
data = scraper.scrape_all_indices(
    from_date='2000-01-01',
    to_date='2024-12-31',
    include_sector_etfs=True,
    include_additional=True
)

# Save
scraper.save_data(data, 'market_indices_data.pkl')

# Result: {
#   'SP500': DataFrame with OHLCV,
#   'VIX': DataFrame with OHLCV,
#   'XLK': DataFrame with OHLCV,
#   ...
# }
```

---

## Phase 2: Derived Features

### What It Does
Calculates additional features from existing OHLCV data. **No additional API calls needed!**

### Features Created

#### Volume Features (10)
- `volume_ratio_5d/10d/20d/50d` - Volume vs moving average
- `dollar_volume` - Price × Volume
- `volume_spike_2x/3x` - Unusual volume detection
- `volume_trend_5d` - Volume momentum
- `volume_zscore` - Normalized volume

#### Price Features (18)
- `intraday_volatility` - (High - Low) / Close
- `gap_pct` - Overnight gap percentage
- `dist_from_52w_high/low` - Distance from 52-week extremes
- `dist_from_sma20/50/200` - Distance from moving averages
- `return_1d/5d/10d/20d` - Returns over various periods
- `volatility_5d/20d` - Realized volatility

#### Bollinger Bands (7)
- `bb_upper/middle/lower` - Band levels
- `bb_width` - Band width (volatility measure)
- `bb_position` - Where price sits within bands (0-1)
- `bb_squeeze` - Low volatility regime indicator
- `touching_upper_bb/lower_bb` - Price at band extremes

#### Momentum Features (6)
- `roc_5/10/20` - Rate of change
- `momentum_5/10` - Price momentum
- `acceleration_5` - Change in momentum

#### Market-Relative Features (8)
- `beta` - Rolling 60-day beta vs index
- `correlation` - Rolling correlation with market
- `relative_return_1d/5d/20d` - Excess return vs market
- `relative_performance` - Cumulative outperformance

**Total: ~40-50 features per stock**

### Usage

```python
from data_scraping.derived_features_calculator import DerivedFeaturesCalculator

calculator = DerivedFeaturesCalculator()

# Your stock OHLCV DataFrame
stock_df = pd.DataFrame({
    'date': dates,
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Calculate all derived features
derived_features = calculator.calculate_all_features(
    df=stock_df,
    market_df=sp500_df  # Optional, for market-relative features
)

# Result: DataFrame with 40-50 additional features
```

---

## Phase 3: Extended Technical Indicators

### What It Does
Expands the technical indicators collected from FMP API beyond the basic set.

### Core Indicators (Already Had)
- SMA (10, 20, 50, 200)
- EMA (10, 20, 50)
- RSI (14)
- ADX (14)
- Williams %R (14)

### Extended Indicators (Added)
- **EMA 200** - Long-term exponential MA
- **WMA** (10, 20, 50) - Weighted moving averages
- **TEMA** (10, 20) - Triple exponential MA
- **Standard Deviation** (10, 20) - Volatility measures
- **Additional RSI** (7, 21) - Different periods
- **Additional ADX** (7, 21) - Trend strength

**Total: ~20-25 technical indicators**

### Usage

```python
from data_scraping.fmp_comprehensive_scraper import FMPComprehensiveScraper

scraper = FMPComprehensiveScraper(api_key="YOUR_KEY")

# Get extended technical indicators
indicators = scraper.get_all_technical_indicators(
    ticker='AAPL',
    interval='daily',
    include_extended=True  # NEW: Enable extended indicators
)

# Result: {
#   'sma_10': DataFrame,
#   'sma_20': DataFrame,
#   ...
#   'wma_20': DataFrame,
#   'tema_10': DataFrame,
#   'stddev_20': DataFrame,
#   ...
# }
```

---

## Phase 4: Cross-Sectional Rankings

### What It Does
Calculates percentile ranks and relative metrics across all stocks for each date. Shows where each stock stands in the universe.

### Features Created

#### Percentile Rankings (7)
- `return_1d_percentile` - 1-day return rank (0-100)
- `return_5d_percentile` - 5-day return rank
- `return_20d_percentile` - 20-day return rank
- `volume_percentile` - Volume rank
- `dollar_volume_percentile` - Dollar volume rank
- `volatility_percentile` - Volatility rank
- `volume_ratio_percentile` - Relative volume rank

#### Momentum Rankings (4)
- `momentum_5d_rank` - 5-day momentum percentile
- `momentum_10d_rank` - 10-day momentum percentile
- `momentum_20d_rank` - 20-day momentum percentile
- `momentum_60d_rank` - 60-day momentum percentile

#### Sector-Relative (3) *if sector data provided*
- `return_1d_vs_sector` - Return vs sector median
- `return_5d_vs_sector` - 5-day return vs sector
- `volume_ratio_vs_sector` - Volume vs sector median

#### Decile Ranks (7)
- Same as percentiles but 0-9 bins

**Total: ~15-20 features per stock**

### Usage

```python
from data_scraping.cross_sectional_calculator import CrossSectionalCalculator

calculator = CrossSectionalCalculator(sector_dict=sector_mapping)

# Calculate for specific date
date_features = calculator.calculate_cross_sectional_features_for_date(
    all_stocks_data={
        'AAPL': aapl_df,
        'MSFT': msft_df,
        'GOOGL': googl_df,
        ...
    },
    date=datetime.date(2024, 12, 7)
)

# Result: {
#   'AAPL': {
#       'return_1d_percentile': 85.3,
#       'volume_percentile': 62.1,
#       'momentum_20d_rank': 91.2,
#       ...
#   },
#   'MSFT': {...},
#   ...
# }
```

---

## Complete Pipeline: Enhanced Data Processor

### What It Does
Integrates all 4 phases into a single processor that creates comprehensive feature tensors.

### Usage

```bash
# Full pipeline command-line
python data_scraping/fmp_enhanced_processor.py \
    --raw_data s_lot_fmp_comprehensive.pkl \
    --market_indices market_indices_data.pkl \
    --sector_dict s_lot_sector_dict.pkl \
    --output s_lot_enhanced_features.pkl \
    --start_date 2000-01-01

# Output: s_lot_enhanced_features.pkl
# Format: {ticker: {date: tensor([...])}}
```

### Code Example

```python
from data_scraping.fmp_enhanced_processor import process_enhanced_data

# Process everything
enhanced_data = process_enhanced_data(
    raw_data_file='s_lot_fmp_comprehensive.pkl',
    market_indices_file='market_indices_data.pkl',
    sector_dict_file='s_lot_sector_dict.pkl',
    output_file='s_lot_enhanced_features.pkl',
    start_date='2000-01-01',
    add_cross_sectional=True
)

# Result: {
#   'AAPL': {
#       datetime.date(2020, 1, 1): tensor([...]),  # 300-400 features
#       datetime.date(2020, 1, 2): tensor([...]),
#       ...
#   },
#   ...
# }
```

---

## Testing

### Run All Tests

```bash
# Test all phases
python tests/test_enhanced_features.py --api_key YOUR_KEY

# Test specific phase
python tests/test_enhanced_features.py --api_key YOUR_KEY --phase 1
python tests/test_enhanced_features.py --api_key YOUR_KEY --phase 2
python tests/test_enhanced_features.py --api_key YOUR_KEY --phase 3
python tests/test_enhanced_features.py --api_key YOUR_KEY --phase 4

# Test full pipeline
python tests/test_enhanced_features.py --api_key YOUR_KEY --phase full
```

### Expected Output

```
================================================================================
ENHANCED FMP FEATURES TEST SUITE
================================================================================

TEST PHASE 1: MARKET INDICES SCRAPER
...
✅ SUCCESS!
Collected 23 indices/ETFs

TEST PHASE 2: DERIVED FEATURES CALCULATOR
...
✅ ALL DERIVED FEATURES: 49 total features

TEST PHASE 3: EXTENDED TECHNICAL INDICATORS
...
✅ SUCCESS!
Collected 22 technical indicators

TEST PHASE 4: CROSS-SECTIONAL RANKINGS
...
✅ SUCCESS for date 2024-12-01!
Features calculated for 5 stocks

FULL PIPELINE TEST: ONE STOCK WITH ALL FEATURES
...
✅ ENHANCED FEATURES CREATED!
  Shape: (365, 287)
  Total features: 287

ALL TESTS COMPLETE!
```

---

## Complete Workflow: From Scratch

### Step 1: Scrape Market Indices (One-time)

```bash
python data_scraping/market_indices_scraper.py \
    --api_key YOUR_KEY \
    --from_date 2000-01-01 \
    --output market_indices_data.pkl
```

**Time:** ~2 minutes
**API Calls:** ~15
**Output:** market_indices_data.pkl

### Step 2: Scrape Stock Data with Extended Technicals

```bash
python data_scraping/fmp_comprehensive_scraper.py \
    --api_key YOUR_KEY \
    --dataset s_lot \
    --start_date 2000-01-01 \
    --output s_lot_comprehensive.pkl
```

**Time:** ~30-60 minutes (370 stocks)
**API Calls:** ~11,000
**Output:** s_lot_comprehensive.pkl

**Note:** The scraper now includes extended technical indicators by default.

### Step 3: Process with All Enhancements

```bash
python data_scraping/fmp_enhanced_processor.py \
    --raw_data s_lot_comprehensive.pkl \
    --market_indices market_indices_data.pkl \
    --sector_dict s_lot_sector_dict.pkl \
    --output s_lot_enhanced.pkl \
    --start_date 2000-01-01
```

**Time:** ~10-20 minutes
**API Calls:** 0
**Output:** s_lot_enhanced.pkl

### Step 4: Use in Training

```python
from utils.utils import pic_load

# Load enhanced features
enhanced_features = pic_load('s_lot_enhanced.pkl')

# Access data
aapl_data = enhanced_features['AAPL']
date_tensor = aapl_data[datetime.date(2024, 12, 7)]

print(f"Features for AAPL on 2024-12-07: {date_tensor.shape}")
# Output: Features for AAPL on 2024-12-07: torch.Size([287])

# Use in your dataset class
class EnhancedQTrainingData(Dataset):
    def __init__(self, enhanced_features_path, ...):
        self.enhanced_features = pic_load(enhanced_features_path)

    def __getitem__(self, idx):
        company, date = self.data_index[idx]

        # Get enhanced features for this date
        features = self.enhanced_features[company][date]

        return features, target
```

---

## Feature Count Summary

### Before Enhancement
- **Quarterly fundamentals:** ~200-300 (from FMP comprehensive)
- **Daily prices:** 5 (OHLCV)
- **Basic technicals:** 10 (SMA, EMA, RSI, ADX, Williams)
- **Total:** ~215-315 features

### After Enhancement
- **Quarterly fundamentals:** ~200-300 (unchanged)
- **Daily prices:** 5 (unchanged)
- **Extended technicals:** 20-25 (+10-15)
- **Derived features:** 40-50 (+40-50 NEW)
- **Market indices features:** 15-25 (+15-25 NEW)
- **Cross-sectional features:** 15-20 (+15-20 NEW)
- **Total:** ~295-420 features (**+80-105 daily features**)

---

## Files Created

### Scrapers
- `data_scraping/market_indices_scraper.py` - Market indices and ETF scraper
- `data_scraping/fmp_comprehensive_scraper.py` - Updated with extended technicals

### Processors
- `data_scraping/derived_features_calculator.py` - Derived features calculator
- `data_scraping/cross_sectional_calculator.py` - Cross-sectional rankings
- `data_scraping/fmp_enhanced_processor.py` - Integrated enhanced processor

### Tests
- `tests/test_enhanced_features.py` - Comprehensive test suite

### Documentation
- `FMP_DAILY_DATA_ANALYSIS.md` - Analysis of available endpoints
- `FMP_DAILY_AVAILABLE_FEATURES.md` - What's available with current API
- `DAILY_FEATURES_COMPLETE_GUIDE.md` - This guide

---

## Benefits

### Model Performance
- **More signal:** 80-100+ additional daily features
- **Market context:** VIX, sector performance, market regime
- **Relative positioning:** Cross-sectional rankings show stock strength
- **Richer patterns:** Derived features capture momentum, volatility dynamics

### API Efficiency
- **Market indices:** 15 calls total (not per-stock)
- **Derived features:** 0 calls (calculated from existing data)
- **Cross-sectional:** 0 calls (calculated from existing data)
- **Extended technicals:** ~10 additional calls per stock

### Implementation Quality
- **Modular:** Each phase is independent
- **Tested:** Comprehensive test suite
- **Documented:** Full usage examples
- **Integrated:** Single enhanced processor combines all

---

## Next Steps

1. **Run the test suite** to verify everything works
2. **Scrape market indices** (one-time, fast)
3. **Process your existing data** with enhanced processor
4. **Update your dataset class** to use enhanced features
5. **Retrain your model** with expanded features
6. **Compare performance** against baseline

---

## Questions?

Check:
- Code examples in `tests/test_enhanced_features.py`
- Individual module docstrings
- `FMP_COMPREHENSIVE_GUIDE.md` for base features
- `FMP_DAILY_AVAILABLE_FEATURES.md` for API limitations

**Ready to train with 300-400+ features per day!** 🚀
