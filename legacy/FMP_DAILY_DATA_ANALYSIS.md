# FMP Daily Data Expansion Analysis

## Currently Implemented Daily Features

### Already in `fmp_comprehensive_scraper.py`:
1. **Daily OHLCV Prices** - `get_daily_prices_full()`
2. **Technical Indicators** - `get_all_technical_indicators()`
   - SMA (10, 20, 50, 200)
   - EMA (10, 20, 50)
   - RSI (14)
   - ADX (14)
   - Williams %R (14)
3. **Market Cap History** - `get_market_cap_history()`

## Available Daily Data NOT Yet Implemented

### 1. **Economic & Macro Indicators** (Daily Updates)
**API Endpoints:**
- Treasury Rates: `/treasury?from=X&to=Y`
- Economic Calendar: `/economic_calendar` (GDP, unemployment, CPI, etc.)
- Market Risk Premium: `/market_risk_premium`
- Sector P/E Ratios: `/sector_price_earnings_ratio`
- Industry P/E Ratios: `/industry_price_earnings_ratio`

**Why Useful:**
- Macro conditions affect all stocks
- Interest rates impact discount rates for valuations
- Economic events drive market-wide movements

**Features Added:** ~10-20 daily macro features

### 2. **Relative Market Performance** (Daily)
**API Endpoints:**
- Sector Performance: `/sectors-performance`
- Historical Sector Performance: `/historical-sectors-performance`
- Industry Performance (implied from sector data)
- Top Gainers/Losers: `/stock_market/gainers`, `/stock_market/losers`
- Most Active: `/stock_market/actives`

**Why Useful:**
- Relative strength vs sector/market
- Sector rotation patterns
- Market regime identification (growth vs value, etc.)

**Features Added:** ~15-25 daily relative performance features

### 3. **Calendar Events** (Daily Flags)
**API Endpoints:**
- Earnings Calendar: `/earning_calendar?from=X&to=Y`
- Dividend Calendar: `/stock_dividend_calendar?from=X&to=Y`
- Stock Splits Calendar: `/stock_split_calendar?from=X&to=Y`
- Economic Calendar: `/economic_calendar?from=X&to=Y`
- IPO Calendar: `/ipo_calendar?from=X&to=Y`

**Why Useful:**
- Binary flags for "earnings this week", "dividend next week"
- Days until next earnings announcement
- Pre/post earnings announcement drift patterns

**Features Added:** ~10-15 daily event indicator features

### 4. **Enhanced Price Metrics** (Daily)
**API Endpoints:**
- Discounted Cash Flow (Daily): `/historical-daily-discounted-cash-flow`
- Price Changes: Can derive from existing price data
- Gap Analysis: Open vs previous close
- Intraday Volatility: High-Low range

**Why Useful:**
- DCF estimates provide fundamental valuation baseline
- Price gaps signal important overnight information
- Intraday volatility measures risk/uncertainty

**Features Added:** ~5-10 daily valuation features

### 5. **Volume Analysis** (Daily, from existing OHLCV)
**Derived Features (no new API needed):**
- Volume ratio vs 20-day average
- Volume trend (increasing/decreasing)
- Price-volume correlation
- Unusual volume detection
- Dollar volume (price × volume)

**Why Useful:**
- Volume confirms price movements
- Institutional activity detection
- Liquidity measures

**Features Added:** ~8-12 derived volume features

### 6. **Index & Benchmark Data** (Daily)
**API Endpoints:**
- S&P 500 Historical: `/historical-price-eod/full?symbol=^GSPC`
- Nasdaq Historical: `/historical-price-eod/full?symbol=^IXIC`
- Dow Jones: `/historical-price-eod/full?symbol=^DJI`
- VIX (Volatility Index): `/historical-price-eod/full?symbol=^VIX`
- Sector ETF Performance: SPY, QQQ, IWM, etc.

**Why Useful:**
- Market beta calculations
- Relative performance vs benchmarks
- Market regime identification (bull/bear)
- Fear gauge (VIX)

**Features Added:** ~10-15 daily benchmark features

### 7. **Options Data** (Daily, if available in premium)
**Potential API Endpoints:**
- Put/Call Ratio
- Options Volume
- Implied Volatility
- Open Interest Changes

**Note:** May require higher-tier FMP plan
**Features Added:** ~5-10 daily options features (if available)

### 8. **News & Sentiment** (Daily, if available)
**Potential API Endpoints:**
- Stock News: `/stock_news?tickers=X&limit=100`
- Press Releases: `/press-releases`
- News Sentiment (if available)

**Note:** Would need NLP processing to convert to daily features
**Features Added:** ~5-10 daily sentiment features

## Recommended Implementation Priority

### **TIER 1 (High Priority - Clear Value):**
1. **Treasury Rates & Economic Indicators** - Market-wide risk factors
2. **Sector/Industry Performance** - Relative positioning
3. **Earnings Calendar Events** - Known predictive signals
4. **Index Benchmarks (S&P 500, VIX)** - Market context
5. **Derived Volume Features** - No additional API calls needed

**Estimated Additional Features:** ~50-70 daily features
**Additional API Calls per Stock:** ~5-10 (most are market-wide, not per-stock)

### **TIER 2 (Medium Priority - Good Value):**
1. **Daily DCF Valuations** - Fundamental anchor
2. **Dividend Calendar** - Income signals
3. **Sector/Industry P/E Ratios** - Valuation context
4. **Additional Technical Indicators** - MACD, Bollinger Bands, Stochastic

**Estimated Additional Features:** ~20-30 daily features
**Additional API Calls per Stock:** ~3-5

### **TIER 3 (Low Priority - Nice to Have):**
1. **Stock Splits Calendar** - Rare events
2. **IPO Calendar** - Limited applicability
3. **News Sentiment** - Requires NLP processing
4. **Options Data** - May not be available in current plan

**Estimated Additional Features:** ~10-20 daily features
**Additional API Calls per Stock:** ~2-5

## Total Potential Expansion

| Category | Current | After Tier 1 | After All Tiers |
|----------|---------|--------------|-----------------|
| Daily Features | ~30 | ~80-100 | ~110-150 |
| Quarterly Features | ~200-300 | ~200-300 | ~200-300 |
| **Total Features** | **~230-330** | **~280-400** | **~310-450** |

## Implementation Strategy

### Phase 1: Market-Wide Data (Easy Wins)
Add these endpoints to a new `get_market_data()` function:
- Treasury rates
- S&P 500, Nasdaq, VIX prices
- Sector performance
- Economic indicators

**Benefits:**
- Only collected ONCE for all stocks (not per-stock)
- Very low API call overhead
- High signal value (market regime matters)

### Phase 2: Per-Stock Calendar Data
Add these to existing `scrape_all_data()`:
- Earnings calendar (days until earnings)
- Dividend calendar (days until dividend)
- Daily DCF values

**Benefits:**
- Clear predictive signals
- Low API overhead (~3 calls/stock)

### Phase 3: Derived Features
Calculate from existing data (no new API calls):
- Volume ratios and trends
- Price gaps
- Intraday volatility measures
- Beta calculations vs indices

**Benefits:**
- Zero API cost
- Can add many features quickly

### Phase 4: Advanced Features (Optional)
If data quality looks good after Phase 1-3:
- News sentiment
- Options data (if available)
- Additional sector/industry metrics

## Next Steps

1. **Test Market-Wide Endpoints** - Verify data quality
2. **Create `FMPDailyDataExpander` class** - Extend comprehensive scraper
3. **Add to `fmp_data_processor.py`** - Process new features into tensors
4. **Integrate with Dataset** - Update data loaders
5. **A/B Test Performance** - Compare model with/without new features

---

**Estimated Total New Daily Features: 50-100+**
**Additional API Calls: Minimal (mostly market-wide, not per-stock)**
**Implementation Time: 4-8 hours for Tier 1**
