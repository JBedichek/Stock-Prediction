# FMP Stable API - Available Endpoints

## Overview
The `/stable/` API endpoint has replaced `/api/v3/` as the primary FMP API. However, many premium features from v3 are not available in the free/basic tier of the stable API.

## Query Parameter Format
All endpoints use query parameters (NOT path parameters):
```
https://financialmodelingprep.com/stable/{endpoint}?symbol=AAPL&apikey=YOUR_KEY
```

## ✅ Available Endpoints (Free/Basic Tier)

### Financial Statements
- `income-statement?symbol=AAPL&period=quarter&limit=400`
- `balance-sheet-statement?symbol=AAPL&period=quarter&limit=400`
- `cash-flow-statement?symbol=AAPL&period=quarter&limit=400`

### Metrics & Ratios
- `key-metrics?symbol=AAPL&period=quarter&limit=400`
  - Returns: P/E, P/B, ROE, ROA, market cap, EV, debt ratios, etc.
- `ratios?symbol=AAPL&period=quarter&limit=400`
  - Returns: 60+ financial ratios
- `enterprise-values?symbol=AAPL&period=quarter&limit=400`
- `financial-growth?symbol=AAPL&period=quarter&limit=400`

### Company Information
- `profile?symbol=AAPL`
  - Returns: sector, industry, description, employees, etc.
- `historical-market-capitalization?symbol=AAPL&limit=1000`

### Analyst Data (Limited)
- `analyst-estimates?symbol=AAPL&period=quarter&limit=100`
  - Returns: revenue & EPS estimates

### News
- `stock_news?tickers=AAPL&limit=100`

## ❌ Not Available (404 Errors)

### Price Data
- `historical-price-full` ❌
- `historical-price-eod/full` ❌
- `historical-price-eod/dividend` ❌
- `historical-price-eod/split` ❌
- `quote` ❌
- `historical-chart/*` ❌

**Alternative:** Use yfinance for all price data

### Technical Indicators
- `technical_indicator/daily` ❌
- `technical_indicator/1min` ❌

**Alternative:** Calculate from price data using pandas-ta or local functions

### Earnings & Events
- `historical/earning_calendar` ❌
- `earnings-surprises` ❌

### Ownership & Trading
- `insider-trading` ❌
- `insider-trading-statistics` ❌
- `institutional-holder` ❌

### Analyst Coverage
- `rating` ❌ (analyst rating changes)
- `price-target` ❌

## Data Collection Strategy

Given the limited availability of the `/stable/` API, use this hybrid approach:

1. **FMP Stable API** → Fundamental data only
   - Financial statements
   - Key metrics & ratios
   - Company profile
   - Analyst estimates

2. **yfinance** → All price & market data
   - Historical prices (OHLCV)
   - Dividends & splits
   - Real-time quotes
   - Market indices

3. **Local Calculation** → Derived features
   - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Volume features
   - Price momentum
   - Volatility metrics

4. **Web Scraping** → News & alternative data
   - Google News
   - Yahoo Finance RSS
   - FMP news endpoint (if available)

## Example Usage

```python
from data_scraping.fmp_comprehensive_scraper import FMPComprehensiveScraper

scraper = FMPComprehensiveScraper(api_key="YOUR_KEY")

# This will now work without 404 errors
data = scraper.scrape_all_data("AAPL", start_date="2000-01-01")

# Available data:
# - data['financial_statements'] ✅
# - data['key_metrics'] ✅
# - data['financial_ratios'] ✅
# - data['enterprise_values'] ✅
# - data['financial_growth'] ✅
# - data['analyst_estimates'] ✅
# - data['company_profile'] ✅
# - data['market_cap_history'] ✅

# Disabled (returns None):
# - data['daily_prices'] → Use yfinance
# - data['technical_indicators'] → Calculate locally
# - data['earnings_history'] → Not available
# - data['insider_trading'] → Not available
# - data['institutional_holdings'] → Not available
# - data['analyst_ratings'] → Not available
# - data['dividend_history'] → Use yfinance
```

## Migration from v3 to stable

If you have old code using `/api/v3/`, update as follows:

### URL Structure
```python
# OLD (v3)
url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={key}"

# NEW (stable)
url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&apikey={key}"
```

### Key Changes
- Ticker moved from path to query parameter: `/{ticker}` → `?symbol={ticker}`
- Many endpoints removed or require premium tier
- Response format generally the same (JSON lists/dicts)

## API Call Limits

Free tier: 250 calls/day

For comprehensive scraping:
- Per stock: ~8-10 API calls (statements + metrics + ratios + growth)
- 370 stocks (s_lot): ~3,000 calls = ~12 days at free tier
- Consider upgrading if scraping large datasets frequently
