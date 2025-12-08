# Stock Prediction Dataset - Complete Pipeline Documentation

This document describes the entire data pipeline from raw data scraping to training-ready tensors.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Data Collection Pipeline](#data-collection-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Data Assembly & Processing](#data-assembly--processing)
6. [Final Dataset Format](#final-dataset-format)
7. [Usage for Training](#usage-for-training)
8. [File Reference](#file-reference)

---

## Overview

The dataset combines **~1100-1200 features per day** for each stock, spanning 25+ years of historical data:

- **Quarterly fundamentals**: 200-300 features (forward-filled to daily)
- **Technical indicators**: 20-25 features
- **Derived features**: 40-50 features
- **Market-relative**: 15-25 features
- **Cross-sectional rankings**: 15-20 features
- **News embeddings**: 768 features

**Final Output Format:**
```python
{
    'AAPL': {
        datetime.date(2000, 1, 1): tensor([...]),  # 1100-1200 features
        datetime.date(2000, 1, 2): tensor([...]),
        ...
    },
    'MSFT': {...},
    ...
}
```

---

## Dataset Structure

### Temporal Hierarchy

```
Quarterly Data → Forward-filled → Daily Frequency
    ↓
Daily Price Data → Aligned → Daily Frequency
    ↓
News Data (irregular) → Aggregated → Daily Frequency (forward-filled)
    ↓
ALL ALIGNED TO SAME DAILY TIMELINE
```

### Data Sources

| Source | Type | Frequency | Features | API/Tool |
|--------|------|-----------|----------|----------|
| Financial Statements | Fundamental | Quarterly | ~150 | FMP Stable API |
| Key Metrics | Fundamental | Quarterly | ~60 | FMP Stable API |
| Financial Ratios | Fundamental | Quarterly | ~60 | FMP Stable API |
| Enterprise Values | Fundamental | Quarterly | ~15 | FMP Stable API |
| Growth Metrics | Fundamental | Quarterly | ~20 | FMP Stable API |
| Company Profile | Static | One-time | ~10 | FMP Stable API |
| Daily Prices (OHLCV) | Market | Daily | 5 | yfinance |
| Market Indices | Market | Daily | 23 indices | yfinance |
| Sector ETFs | Market | Daily | 11 ETFs | yfinance |
| News Articles | Text | Irregular | → 768 (embedded) | Google News, Yahoo RSS |
| Technical Indicators | Derived | Daily | 20-25 | Calculated locally |
| Derived Features | Derived | Daily | 40-50 | Calculated locally |
| Cross-sectional Ranks | Derived | Daily | 15-20 | Calculated locally |

---

## Data Collection Pipeline

The complete pipeline is orchestrated by `generate_full_dataset.py` which runs 5 sequential steps:

### Step 1: Market Indices Scraping

**File:** `data_scraping/market_indices_scraper.py`

**What it does:**
- Scrapes broad market indices and sector ETFs
- **Shared across all stocks** (scraped once, reused for all tickers)

**Indices collected:**
```python
MARKET_INDICES = {
    'SP500': '^GSPC',       # S&P 500
    'NASDAQ': '^IXIC',      # Nasdaq Composite
    'VIX': '^VIX',          # Volatility Index
    'DJI': '^DJI',          # Dow Jones
    'RUSSELL2000': '^RUT',  # Russell 2000
}

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}
```

**Output:** `market_indices_data.pkl`
```python
{
    'SP500': DataFrame(date, open, high, low, close, volume),
    'NASDAQ': DataFrame(...),
    'VIX': DataFrame(...),
    'XLK': DataFrame(...),
    ...
}
```

**API calls:** ~15 total (not per stock)

**Time:** ~2-3 minutes

---

### Step 2: Stock Fundamentals Scraping

**File:** `data_scraping/fmp_comprehensive_scraper.py`

**What it does:**
- Scrapes comprehensive fundamental data for each stock
- Collects quarterly financial statements going back 30+ years

**Data collected per stock:**

1. **Financial Statements** (quarterly, limit=400 = 100 years)
   - Income statement (~50 fields)
   - Balance sheet (~60 fields)
   - Cash flow statement (~40 fields)

2. **Key Metrics** (quarterly, limit=400)
   - P/E ratio, P/B ratio, ROE, ROA
   - Market cap, enterprise value
   - Current ratio, debt ratios
   - EV/Sales, EV/EBITDA
   - ~60 pre-calculated metrics

3. **Financial Ratios** (quarterly, limit=400)
   - Profitability ratios
   - Liquidity ratios
   - Leverage ratios
   - Efficiency ratios
   - ~60 comprehensive ratios

4. **Enterprise Values** (quarterly, limit=400)
   - Enterprise value over time
   - Market cap history
   - Net debt calculations

5. **Financial Growth** (quarterly, limit=400)
   - Revenue growth rates
   - Earnings growth
   - Asset growth
   - ~20 growth metrics

6. **Company Profile** (static)
   - Sector, industry
   - Description, CEO
   - Employee count, founded date

7. **Analyst Estimates** (quarterly, limit=100)
   - Revenue estimates
   - EPS estimates

**FMP API Endpoints Used:**
```python
# All use query parameter format with /stable/ base
GET /stable/income-statement?symbol=AAPL&period=quarter&limit=400
GET /stable/balance-sheet-statement?symbol=AAPL&period=quarter&limit=400
GET /stable/cash-flow-statement?symbol=AAPL&period=quarter&limit=400
GET /stable/key-metrics?symbol=AAPL&period=quarter&limit=400
GET /stable/ratios?symbol=AAPL&period=quarter&limit=400
GET /stable/enterprise-values?symbol=AAPL&period=quarter&limit=400
GET /stable/financial-growth?symbol=AAPL&period=quarter&limit=400
GET /stable/profile?symbol=AAPL
GET /stable/historical-market-capitalization?symbol=AAPL&limit=1000
GET /stable/analyst-estimates?symbol=AAPL&period=quarter&limit=100
```

**Output:** `{dataset}_fmp_comprehensive.pkl`
```python
{
    'AAPL': {
        'ticker': 'AAPL',
        'collection_date': '2024-12-07T10:30:00',
        'financial_statements': {
            'income': DataFrame(date, revenue, netIncome, ...),
            'balance': DataFrame(date, totalAssets, totalLiabilities, ...),
            'cashflow': DataFrame(date, operatingCashFlow, freeCashFlow, ...)
        },
        'key_metrics': DataFrame(date, peRatio, pbRatio, roe, roa, ...),
        'financial_ratios': DataFrame(date, currentRatio, debtToEquity, ...),
        'enterprise_values': DataFrame(date, enterpriseValue, ...),
        'financial_growth': DataFrame(date, revenueGrowth, earningsGrowth, ...),
        'analyst_estimates': DataFrame(date, estimatedRevenue, estimatedEPS, ...),
        'company_profile': {...},
        'market_cap_history': DataFrame(date, marketCap),

        # These are disabled (not available in /stable/):
        'daily_prices': None,  # Use yfinance instead
        'technical_indicators': {},  # Calculate locally
        'earnings_history': None,
        'insider_trading': None,
        'institutional_holdings': None,
        'analyst_ratings': None,
        'dividend_history': None,
        'stock_splits': None,

        'api_calls': 10
    },
    'MSFT': {...},
    ...
}
```

**API calls:** ~10 per stock

**Time:** ~30 seconds per stock
- 370 stocks (s_lot): ~3 hours
- Rate limit: 250 calls/day free tier

**Important Notes:**
- `limit=400` means "get up to 400 quarterly periods" (100 years), NOT 400 API calls
- Each endpoint call is 1 API call, regardless of limit
- Quarterly data goes back to company founding (AAPL has ~50 years = ~200 quarters)
- Data will be forward-filled to daily frequency later

---

### Step 3: News Articles Scraping

**File:** `data_scraping/news_scraper.py`

**What it does:**
- Scrapes news articles from multiple sources
- Each article includes **timestamp** for temporal alignment

**News Sources:**

1. **Google News** (via `gnews` library)
   - Can specify date range: `start_date` to `end_date`
   - Full article text available
   - Timestamp format: `"Mon, 07 Dec 2024 14:30:00 GMT"`
   - Free, no API key required
   - Rate limited by web scraping

2. **Yahoo Finance RSS**
   - Recent articles only (~50 latest)
   - Timestamp format: `"Sun, 08 Dec 2024 01:23:45 +0000"`
   - Free, reliable

3. **FMP Stock News** (optional, may be restricted)
   - Endpoint: `/stable/stock_news?tickers=AAPL&limit=100`
   - Timestamp field: `publishedDate`
   - Format: `"2024-12-07T14:30:00Z"`
   - May return empty on free tier

**Article Data Structure:**
```python
{
    'ticker': 'AAPL',
    'title': 'Apple Announces New Product Line',
    'description': 'Summary of the article...',
    'url': 'https://...',
    'published_date': '2024-12-07 14:30:00',  # CRITICAL for temporal alignment
    'publisher': 'TechCrunch',
    'full_text': 'Full article content...',
    'source': 'google_news'  # or 'yahoo_rss', 'fmp'
}
```

**Deduplication:**
- Articles are deduplicated by URL across all sources
- If same article appears in multiple sources, keep only one

**Output:** `{dataset}_news_data.pkl`
```python
{
    'AAPL': [
        {article1}, {article2}, ...
    ],
    'MSFT': [
        {article1}, {article2}, ...
    ],
    ...
}
```

**Time:** Variable
- Depends on article availability
- Google News: ~5-10 articles per stock (fast)
- Can take 1-3 hours for 370 stocks

**Edge Cases Handled:**
1. **No news found** → Empty list `[]` (will get zero embeddings later)
2. **Failed scraping** → Exception caught, continue with other stocks
3. **Missing timestamps** → Use current date as fallback
4. **Duplicate URLs** → Removed during deduplication
5. **Long articles** → Truncated to 5000 chars during embedding

---

### Step 4: News Embedding

**File:** `data_scraping/news_embedder.py`

**What it does:**
- Embeds news articles using Nomic AI model
- Aggregates multiple articles per date
- Forward-fills to daily frequency

**Model:** `nomic-ai/nomic-embed-text-v1.5`
- 768-dimensional embeddings
- Supports long context (8192 tokens)
- Optimized for semantic search

**Process:**

1. **Article Embedding**
   ```python
   # Combine article components
   text = f"Title: {title} Summary: {description} Content: {full_text[:5000]}"

   # Embed with Nomic
   embedding = model.encode(text)  # Returns (768,) vector
   ```

2. **Date Parsing**
   - Extracts date from `published_date` field
   - Supports multiple formats:
     - `2024-12-07`
     - `2024-12-07 14:30:00`
     - `Mon, 07 Dec 2024 14:30:00 GMT`
     - `2024-12-07T14:30:00Z`
   - Falls back to pandas parsing
   - Converts to `datetime.date` object

3. **Temporal Aggregation**
   ```python
   # Group articles by date
   date_to_embeddings = {
       datetime.date(2024, 1, 1): [emb1, emb2],  # 2 articles on Jan 1
       datetime.date(2024, 1, 5): [emb3],        # 1 article on Jan 5
       ...
   }

   # Aggregate embeddings for same date (mean pooling)
   aggregated = {
       datetime.date(2024, 1, 1): mean([emb1, emb2]),  # Shape: (768,)
       datetime.date(2024, 1, 5): emb3,                # Shape: (768,)
   }
   ```

4. **Forward-Filling to Daily**
   ```python
   # Create daily embeddings from irregular article dates
   for date in range(start_date, end_date):
       if date in aggregated:
           last_embedding = aggregated[date]  # New article published
       elif last_embedding is not None:
           embedding = last_embedding  # Forward-fill: use last known embedding
       else:
           embedding = zeros(768)  # No news yet: use zeros

       daily_embeddings[date] = embedding
   ```

**Example Timeline:**
```
Date        Articles Published    Embedding Used
------------------------------------------------------
2024-01-01  [Article A]          → Embedding of A
2024-01-02  (none)               → Forward-fill A
2024-01-03  (none)               → Forward-fill A
2024-01-04  [Article B, C]       → Mean(B, C)
2024-01-05  (none)               → Forward-fill Mean(B, C)
2024-01-06  [Article D]          → Embedding of D
```

**Output:** `{dataset}_news_embeddings.pkl`
```python
{
    'AAPL': {
        datetime.date(2000, 1, 1): tensor([768 values]),  # All zeros if no news yet
        datetime.date(2000, 1, 2): tensor([768 values]),
        ...
        datetime.date(2024, 12, 7): tensor([768 values]),
    },
    'MSFT': {...},
    ...
}
```

**Device:**
- Uses GPU if available (`cuda`)
- Falls back to CPU
- Batch size: 32 (configurable)

**Time:**
- ~10-20 minutes for 370 stocks (with GPU)
- ~30-60 minutes (CPU only)

**Edge Cases Handled:**
1. **Stock with no articles** → All zeros (768,) for all dates
2. **Failed embedding** → Exception caught, return zeros as fallback
3. **Missing date in article** → Use today's date
4. **Empty text** → Return zero vector
5. **Article too long** → Truncate to 5000 chars

---

### Step 5: Enhanced Feature Processing

**File:** `data_scraping/fmp_enhanced_processor.py`

**What it does:**
- Combines all data sources
- Adds derived features
- Adds market-relative features
- Adds cross-sectional rankings
- Aligns everything to daily frequency
- Normalizes features

**Processing Phases:**

#### Phase 1: Base Processing (from fmp_data_processor.py)

Processes quarterly FMP data:

```python
# 1. Financial Statements
financial_features = process_financial_statements()
# Returns: ~150 features from income/balance/cash statements

# 2. Key Metrics
key_metrics = process_key_metrics()
# Returns: ~60 features (P/E, ROE, ROA, etc.)

# 3. Financial Ratios
ratios = process_financial_ratios()
# Returns: ~60 features (liquidity, leverage, profitability ratios)

# 4. Enterprise Values
enterprise = process_enterprise_values()
# Returns: ~15 features

# 5. Growth Metrics
growth = process_financial_growth()
# Returns: ~20 features (revenue growth, earnings growth, etc.)
```

**Total from Phase 1:** ~305 quarterly features

#### Phase 2: Derived Features

**File:** `data_scraping/derived_features_calculator.py`

Calculates features from OHLCV data (no API calls):

1. **Volume Features**
   - Volume ratios vs moving averages (5, 10, 20, 60-day)
   - Volume spikes (2x, 3x average)
   - Dollar volume (price × volume)
   - Volume momentum
   - ~10 features

2. **Price Features**
   - Intraday gap (open vs previous close)
   - High-low range
   - Close position in range
   - Price gaps
   - ~8 features

3. **Momentum Features**
   - Multiple timeframe returns (1d, 5d, 20d, 60d, 252d)
   - Price acceleration
   - Momentum oscillators
   - ~10 features

4. **Volatility Features**
   - Historical volatility (10, 20, 60-day)
   - Parkinson volatility (high-low range based)
   - Volatility ratios
   - ~8 features

5. **Bollinger Bands**
   - Upper/lower bands (20-day)
   - Distance from bands
   - Bandwidth
   - %B indicator
   - ~5 features

**Total from Phase 2:** ~40-50 features

#### Phase 3: Market-Relative Features

Compares stock to market indices:

1. **Relative Returns** (vs S&P 500, Nasdaq)
   - Outperformance (stock return - index return)
   - Multiple timeframes (1d, 5d, 20d, 60d)
   - ~8 features per index

2. **Beta Calculations**
   - 60-day rolling beta vs S&P 500
   - Beta vs Nasdaq
   - Beta vs sector ETF
   - ~3 features

3. **Correlation**
   - Rolling correlation vs indices
   - ~3 features

4. **VIX Features**
   - VIX level
   - VIX high/low indicators
   - Stock-VIX correlation
   - ~5 features

5. **Sector-Relative**
   - Performance vs sector ETF
   - Sector beta
   - ~5 features

**Total from Phase 3:** ~15-25 features

#### Phase 4: Cross-Sectional Rankings

**File:** `data_scraping/cross_sectional_calculator.py`

Ranks each stock against all others on the same date:

```python
# For each date, calculate percentile rank (0-100)
features_to_rank = {
    'return_1d': 'return_1d_percentile',
    'return_5d': 'return_5d_percentile',
    'return_20d': 'return_20d_percentile',
    'volume': 'volume_percentile',
    'volatility_20d': 'volatility_percentile',
    'market_cap': 'market_cap_percentile',
    'pe_ratio': 'pe_percentile',
    'roe': 'roe_percentile',
    ...
}

# Example for 2024-12-07:
# AAPL: return_1d = 2.5% → Percentile = 85 (better than 85% of stocks)
# MSFT: return_1d = 1.2% → Percentile = 62
# TSLA: return_1d = -1.5% → Percentile = 15
```

**Rankings calculated:**
- Short-term performance (1d, 5d, 20d returns)
- Volume activity
- Volatility
- Valuation metrics (P/E, P/B, EV/Sales)
- Profitability (ROE, ROA, margins)
- Growth rates
- Size (market cap)

**Total from Phase 4:** ~15-20 features

#### Phase 5: News Embeddings

Adds pre-computed news embeddings:

```python
# For each date, concatenate news embedding
if date in news_embeddings:
    news_vector = news_embeddings[date]  # Shape: (768,)
else:
    news_vector = zeros(768)  # Fallback if missing
```

**Total from Phase 5:** 768 features

---

### Daily Alignment Process

**Critical Step:** All features must align to the same daily timeline

```python
# 1. Create daily date range
date_range = pd.date_range(start='2000-01-01', end='2025-12-31', freq='D')

# 2. Align quarterly data (forward-fill)
quarterly_data = pd.DataFrame(...)  # Indexed by quarter-end dates
quarterly_daily = quarterly_data.reindex(date_range, method='ffill')
# Example:
#   2024-03-31: [Q1 2024 fundamentals]
#   2024-04-01: [Q1 2024 fundamentals]  ← forward-filled
#   2024-04-02: [Q1 2024 fundamentals]  ← forward-filled
#   ...
#   2024-06-30: [Q2 2024 fundamentals]  ← new quarter

# 3. Align daily data (as-is)
daily_data = pd.DataFrame(...)  # Indexed by trading days
daily_aligned = daily_data.reindex(date_range)
# Missing days (weekends, holidays) get NaN → filled with 0

# 4. Align news embeddings (forward-fill)
news_daily = pd.DataFrame(...)  # Indexed by article publication dates
news_aligned = news_daily.reindex(date_range, method='ffill')

# 5. Concatenate all features
combined = pd.concat([
    quarterly_daily,  # ~305 features
    daily_aligned,    # ~100 features
    news_aligned      # 768 features
], axis=1)

# 6. Handle missing values
combined = combined.fillna(0)  # NaN → 0
combined = combined.replace([np.inf, -np.inf], 0)  # Infinity → 0
```

**Result:** Every stock has a tensor for every date with the same shape

---

### Normalization

Each feature is normalized independently using **z-score normalization**:

```python
# For each feature column
mean = feature.mean()
std = feature.std()

if std == 0 or isnan(std):
    normalized = feature * 0  # Constant feature → all zeros
else:
    normalized = (feature - mean) / std  # Standard scaling

# Clip outliers (optional)
normalized = clip(normalized, -10, 10)
```

**Alternatives used in different contexts:**
- **Robust scaling** (median + IQR) for fundamentals with outliers
- **MinMax scaling** (0-1 range) for bounded features
- **Log scaling** for highly skewed distributions

**Output:** `{dataset}_fmp_comprehensive_processed.pkl`
```python
{
    'AAPL': {
        datetime.date(2000, 1, 1): tensor([305 fundamentals, 100 technical/derived, 768 news]),
        datetime.date(2000, 1, 2): tensor([...]),
        ...
    },
    'MSFT': {...},
    ...
}
```

---

## Final Dataset Format

**File:** `{dataset}_complete_dataset.pkl`

### Structure

```python
{
    'AAPL': {
        datetime.date(2000, 1, 1): torch.Tensor([f1, f2, ..., f1173]),  # Shape: (1173,)
        datetime.date(2000, 1, 2): torch.Tensor([...]),
        datetime.date(2000, 1, 3): torch.Tensor([...]),
        ...
        datetime.date(2024, 12, 7): torch.Tensor([...]),
    },
    'MSFT': {
        datetime.date(2000, 1, 1): torch.Tensor([...]),
        ...
    },
    ...  # 370 stocks for s_lot dataset
}
```

### Feature Composition

**Total: ~1100-1200 features per day**

| Category | Count | Source | Frequency |
|----------|-------|--------|-----------|
| Income Statement | ~50 | FMP | Quarterly (forward-filled) |
| Balance Sheet | ~60 | FMP | Quarterly (forward-filled) |
| Cash Flow | ~40 | FMP | Quarterly (forward-filled) |
| Key Metrics | ~60 | FMP | Quarterly (forward-filled) |
| Financial Ratios | ~60 | FMP | Quarterly (forward-filled) |
| Enterprise Values | ~15 | FMP | Quarterly (forward-filled) |
| Growth Metrics | ~20 | FMP | Quarterly (forward-filled) |
| **Subtotal: Fundamentals** | **~305** | | |
| | | | |
| Volume Features | ~10 | Derived | Daily |
| Price Features | ~8 | Derived | Daily |
| Momentum Features | ~10 | Derived | Daily |
| Volatility Features | ~8 | Derived | Daily |
| Bollinger Bands | ~5 | Derived | Daily |
| Market-Relative | ~20 | Derived | Daily |
| Cross-sectional Ranks | ~15 | Derived | Daily |
| **Subtotal: Technical/Derived** | **~76** | | |
| | | | |
| News Embeddings | 768 | Nomic AI | Daily (forward-filled) |
| **Subtotal: News** | **768** | | |
| | | | |
| **TOTAL** | **~1149** | | |

### Data Quality

**Handled Edge Cases:**
1. ✅ Missing quarterly data → Forward-filled from last known values
2. ✅ Missing daily prices → Filled with 0
3. ✅ Missing news → Zero embeddings (768 zeros)
4. ✅ NaN values → Replaced with 0
5. ✅ Infinity values → Replaced with 0
6. ✅ Non-trading days → Features still generated (using forward-filled data)
7. ✅ Stocks with sparse data → Included with partial features

**Consistency:**
- ✅ All stocks have same tensor shape on any given date
- ✅ All dates have data for all stocks (even if zeros)
- ✅ Temporal alignment is perfect (all features for same date)

---

## Usage for Training

### Loading the Dataset

```python
from utils.utils import pic_load
import datetime

# Load complete dataset
data = pic_load('s_lot_complete_dataset.pkl')

print(f"Stocks: {len(data)}")  # 370

# Access single stock
aapl_data = data['AAPL']
print(f"Days: {len(aapl_data)}")  # ~9000 days (25 years)

# Access single day
date = datetime.date(2024, 12, 7)
features = aapl_data[date]
print(features.shape)  # torch.Size([1149])
```

### Creating Training Sequences

The model uses **600-day rolling windows** as input:

```python
def create_sequence(ticker_data, end_date, sequence_length=600):
    """
    Create a sequence of features ending on end_date.

    Returns:
        tensor of shape (sequence_length, num_features)
    """
    dates = sorted([d for d in ticker_data.keys() if d <= end_date])

    if len(dates) < sequence_length:
        return None  # Not enough history

    # Get last 600 days
    sequence_dates = dates[-sequence_length:]

    # Stack features
    sequence = torch.stack([ticker_data[d] for d in sequence_dates])
    # Shape: (600, 1149)

    return sequence
```

### Creating Targets

The model predicts **probability distributions over next 4 days**:

```python
def create_targets(price_data, end_date, num_days=4):
    """
    Create binned target distributions for next 4 days.

    Returns:
        targets: tensor of shape (num_bins, num_days)
    """
    # Get prices for next 4 days after end_date
    future_dates = get_next_n_trading_days(end_date, num_days)
    current_price = price_data[end_date]

    # Calculate returns
    returns = []
    for future_date in future_dates:
        future_price = price_data[future_date]
        ret = (future_price - current_price) / current_price
        returns.append(ret)

    # Convert returns to bin indices
    # Bins defined in 'Bin_Edges_300' file (320 bins covering -20% to +20%)
    bin_edges = load_bin_edges()

    targets = torch.zeros(320, 4)
    for i, ret in enumerate(returns):
        bin_idx = get_bin_index(ret, bin_edges)
        targets[bin_idx, i] = 1.0  # One-hot encoding

    return targets
```

### PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StockPredictionDataset(Dataset):
    def __init__(self, data_dict, sequence_length=600, num_future_days=4):
        """
        Args:
            data_dict: Complete dataset from pickle file
            sequence_length: Number of historical days to use (600)
            num_future_days: Number of days to predict (4)
        """
        self.data = data_dict
        self.sequence_length = sequence_length
        self.num_future_days = num_future_days

        # Build index of valid (ticker, date) pairs
        self.samples = []
        for ticker, ticker_data in data_dict.items():
            dates = sorted(ticker_data.keys())

            # Need enough history + future
            for i in range(sequence_length, len(dates) - num_future_days):
                self.samples.append((ticker, dates[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, end_date = self.samples[idx]
        ticker_data = self.data[ticker]

        # Get 600-day sequence ending on end_date
        dates = sorted(ticker_data.keys())
        end_idx = dates.index(end_date)
        sequence_dates = dates[end_idx - self.sequence_length + 1:end_idx + 1]

        # Stack features
        features = torch.stack([ticker_data[d] for d in sequence_dates])
        # Shape: (600, 1149)

        # Get target (next 4 days)
        # (Target creation logic here - depends on your binning scheme)
        target = self._create_target(ticker, end_date)
        # Shape: (320, 4)

        return features, target

    def _create_target(self, ticker, end_date):
        # Implement target creation
        # See create_targets() function above
        pass

# Usage
dataset = StockPredictionDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_features, batch_targets in dataloader:
    # batch_features: (32, 600, 1149)
    # batch_targets: (32, 320, 4)

    predictions = model(batch_features)
    loss = criterion(predictions, batch_targets)
    # ...
```

### Model Input Format

The transformer model expects:

```python
# Input
batch_features.shape = (batch_size, sequence_length, num_features)
                     = (32, 600, 1149)

# Where:
# - batch_size: Number of samples per batch
# - sequence_length: 600 days of history
# - num_features: 1149 features per day

# Output
predictions.shape = (batch_size, num_bins, num_future_days)
                  = (32, 320, 4)

# Where:
# - num_bins: 320 price change bins
# - num_future_days: 4 days ahead predictions
```

### Company Embeddings (Legacy)

The original model also used RoBERTa embeddings of company descriptions:

```python
# Company text info (from Stock.py)
summary_embeddings = {
    'AAPL': tensor(768),  # RoBERTa embedding of "sector + industry + description"
    'MSFT': tensor(768),
    ...
}

# These were reshaped and concatenated:
# (768,) → reshape to (4, 218) → concat with price sequences
# Input becomes: (604, 218) where 604 = 600 price days + 4 summary chunks

# NOTE: With new dataset, you have 1149 features per day instead
# So you may need to adapt the model architecture
```

---

## File Reference

### Input Files (You Create)

- `data_scraping/Stock.py` - Contains ticker dictionaries:
  - `s_lot_of_stocks` (~370 tickers)
  - `a_lot_of_stocks` (~800 tickers)
  - `all_stocks` (~4000 tickers)
  - `test_stock_tickers` (3 tickers for testing)

### Intermediate Files (Generated by Pipeline)

| File | Size | Description |
|------|------|-------------|
| `market_indices_data.pkl` | ~50 MB | Market indices & sector ETFs (shared) |
| `s_lot_fmp_comprehensive.pkl` | ~500 MB | Raw fundamental data for 370 stocks |
| `s_lot_news_data.pkl` | ~200 MB | Raw news articles with timestamps |
| `s_lot_news_embeddings.pkl` | ~800 MB | 768-dim embeddings per day per stock |
| `s_lot_fmp_comprehensive_processed.pkl` | ~1 GB | Processed fundamentals + derived features |

### Final Output File

| File | Size | Description |
|------|------|-------------|
| `s_lot_complete_dataset.pkl` | ~2 GB | **FINAL TRAINING-READY DATASET** |

Structure:
```python
{ticker: {date: tensor(1149)}}
```

### Configuration Files

- `Bin_Edges_300` - Defines 320 bins for price change discretization
- `DATASET_GENERATION_GUIDE.txt` - Usage guide
- `QUICK_COMMANDS.txt` - Common commands
- `NEWS_EDGE_CASES_HANDLING.txt` - Edge case documentation
- `FMP_STABLE_API_ENDPOINTS.md` - API endpoint reference

### Code Files

**Data Collection:**
- `data_scraping/market_indices_scraper.py` - Step 1
- `data_scraping/fmp_comprehensive_scraper.py` - Step 2
- `data_scraping/news_scraper.py` - Step 3
- `data_scraping/news_embedder.py` - Step 4

**Feature Engineering:**
- `data_scraping/derived_features_calculator.py` - Derived features
- `data_scraping/cross_sectional_calculator.py` - Cross-sectional ranks

**Data Processing:**
- `data_scraping/fmp_data_processor.py` - Base processor
- `data_scraping/fmp_enhanced_processor.py` - Enhanced processor with all phases

**Orchestration:**
- `generate_full_dataset.py` - **MAIN SCRIPT** - Runs all 5 steps

**Legacy (Original Model):**
- `Stock.py` - Original scraper using yfinance + RoBERTa
- `training.py` - Dataset generation for original model
- `models.py` - Model definitions

---

## Pipeline Execution

### Quick Start

```bash
# Test with 3 stocks, 1 year
python generate_full_dataset.py --dataset test --years 1

# Production with 370 stocks, 25 years
python generate_full_dataset.py --dataset s_lot --years 25

# Full dataset, 4000+ stocks
python generate_full_dataset.py --dataset all --years 25
```

### Step-by-Step Execution

```bash
# Run only specific steps (if some are already done)
python generate_full_dataset.py --dataset s_lot --skip-steps 1 2  # Skip indices & fundamentals

# Resume interrupted fundamentals scraping
python generate_full_dataset.py --dataset s_lot --resume-from MSFT

# Force rescrape everything
python generate_full_dataset.py --dataset s_lot --force-rescrape
```

### Customization

```bash
# Custom output directory
python generate_full_dataset.py --dataset s_lot --output-dir /data/stocks

# Skip cross-sectional features (30% faster)
python generate_full_dataset.py --dataset s_lot --skip-cross-sectional

# Use CPU instead of GPU for embedding
python generate_full_dataset.py --dataset s_lot --device cpu

# Different news period than stock data
python generate_full_dataset.py --dataset s_lot --years 25 --news-years 5
```

### Background Execution

```bash
# Run overnight
nohup python generate_full_dataset.py --dataset all --years 25 > generation.log 2>&1 &

# Monitor progress
tail -f generation.log

# Check if still running
ps aux | grep generate_full_dataset
```

---

## Dataset Statistics

### s_lot Dataset (370 Stocks, 25 Years)

- **Trading days:** ~6,300
- **Total samples:** 370 stocks × 6,300 days = ~2.3M daily records
- **Training sequences:** ~2.3M - 604 (window size) = ~2.3M sequences
- **Features per day:** 1,149
- **Total feature values:** 2.3M × 1,149 = ~2.6 billion values
- **Storage size:** ~2-5 GB (compressed pickle)
- **Generation time:** 2-4 hours
- **API calls:** ~3,000 (FMP) + web scraping (news)

### all_stocks Dataset (4000+ Stocks, 25 Years)

- **Trading days:** ~6,300
- **Total samples:** 4,000 × 6,300 = ~25M daily records
- **Features per day:** 1,149
- **Total feature values:** ~29 billion values
- **Storage size:** ~20-50 GB
- **Generation time:** 24-48 hours
- **API calls:** ~40,000 (FMP) + web scraping

---

## Quality Assurance

### Validation Checks

After generation, verify:

```python
from utils.utils import pic_load

# Load dataset
data = pic_load('s_lot_complete_dataset.pkl')

# 1. Check stock count
assert len(data) == 370, f"Expected 370 stocks, got {len(data)}"

# 2. Check tensor shapes are consistent
shapes = set()
for ticker, ticker_data in data.items():
    for date, tensor in ticker_data.items():
        shapes.add(tensor.shape)

assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"
print(f"✅ All tensors have shape: {shapes.pop()}")

# 3. Check date coverage
for ticker in ['AAPL', 'MSFT', 'GOOGL']:  # Sample tickers
    dates = sorted(data[ticker].keys())
    print(f"{ticker}: {dates[0]} to {dates[-1]} ({len(dates)} days)")

# 4. Check for NaN/Inf
has_issues = False
for ticker, ticker_data in data.items():
    for date, tensor in ticker_data.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"⚠️  {ticker} on {date} has NaN/Inf")
            has_issues = True
            break
    if has_issues:
        break

if not has_issues:
    print("✅ No NaN or Inf values")

# 5. Check feature statistics
sample_ticker = 'AAPL'
sample_tensors = torch.stack(list(data[sample_ticker].values()))
print(f"\nFeature statistics for {sample_ticker}:")
print(f"  Mean: {sample_tensors.mean(dim=0).mean():.3f}")
print(f"  Std: {sample_tensors.std(dim=0).mean():.3f}")
print(f"  Min: {sample_tensors.min():.3f}")
print(f"  Max: {sample_tensors.max():.3f}")
```

### Expected Output

```
✅ All tensors have shape: torch.Size([1149])
AAPL: 2000-01-01 to 2024-12-07 (9131 days)
MSFT: 2000-01-01 to 2024-12-07 (9131 days)
GOOGL: 2004-08-19 to 2024-12-07 (7412 days)
✅ No NaN or Inf values

Feature statistics for AAPL:
  Mean: 0.002
  Std: 0.987
  Min: -10.000
  Max: 10.000
```

---

## Troubleshooting

### Common Issues

**1. API Rate Limits**
```
Error: 429 Too Many Requests
```
- **Solution:** Wait 24 hours for rate limit reset (250 calls/day free tier)
- **Prevention:** Use `--resume-from TICKER` to continue where you left off
- **Alternative:** Upgrade FMP API plan

**2. Out of Memory (GPU)**
```
RuntimeError: CUDA out of memory
```
- **Solution:** Use `--device cpu` instead
- **Alternative:** Reduce batch size in news_embedder.py

**3. Missing News for Stock**
```
⚠️ Stock XYZ has no news articles
```
- **Expected:** Small-cap stocks often have limited news coverage
- **Handled:** Zero embeddings (768 zeros) are used automatically

**4. Inconsistent Tensor Shapes**
```
RuntimeError: Expected tensor of size [1149] but got [1000]
```
- **Cause:** Pipeline was interrupted and resumed with different configuration
- **Solution:** Delete intermediate files and regenerate from scratch with `--force-rescrape`

**5. Disk Space**
```
OSError: No space left on device
```
- **Check:** `df -h .`
- **Required:** 5 GB for s_lot, 50 GB for all_stocks
- **Solution:** Use `--output-dir /path/with/more/space`

---

## Performance Optimization

### Speed Improvements

1. **Skip cross-sectional features** (30% faster):
   ```bash
   python generate_full_dataset.py --dataset s_lot --skip-cross-sectional
   ```

2. **Use GPU for embedding** (3-5x faster):
   ```bash
   python generate_full_dataset.py --dataset s_lot --device cuda
   ```

3. **Reduce news period** (faster scraping):
   ```bash
   python generate_full_dataset.py --dataset s_lot --years 25 --news-years 5
   ```

4. **Parallel processing** (for multiple datasets):
   ```bash
   # Terminal 1
   python generate_full_dataset.py --dataset s_lot &

   # Terminal 2
   python generate_full_dataset.py --dataset a_lot &
   ```

### Memory Optimization

If running on limited memory:

1. Process in smaller batches (modify code)
2. Use `--device cpu` to avoid GPU memory
3. Reduce `batch_size` in news_embedder.py
4. Process fewer stocks at once

---

## Next Steps

After generating the dataset:

1. **Verify dataset:**
   ```python
   from utils.utils import pic_load
   data = pic_load('s_lot_complete_dataset.pkl')
   # Run validation checks above
   ```

2. **Create PyTorch Dataset:**
   ```python
   dataset = StockPredictionDataset(data)
   print(f"Total sequences: {len(dataset)}")
   ```

3. **Create train/val/test splits:**
   ```python
   train_size = int(0.8 * len(dataset))
   val_size = int(0.1 * len(dataset))
   test_size = len(dataset) - train_size - val_size

   train, val, test = torch.utils.data.random_split(
       dataset, [train_size, val_size, test_size]
   )
   ```

4. **Train model:**
   ```python
   from models import t_Dist_Pred

   model = t_Dist_Pred(
       input_dim=1149,  # Updated for new feature count
       num_bins=320,
       num_days=4
   )

   # Training loop
   for batch in train_loader:
       # ...
   ```

---

## Appendix: Feature List

### Fundamental Features (from FMP) - ~305 total

**Income Statement (~50):**
- revenue, costOfRevenue, grossProfit
- operatingExpenses, operatingIncome
- netIncome, eps, epsDiluted
- ebitda, ebit, researchAndDevelopment
- sellingGeneralAndAdministrative
- (and ~40 more fields)

**Balance Sheet (~60):**
- totalAssets, totalLiabilities, stockholderEquity
- cash, shortTermInvestments
- inventory, accountsReceivable
- totalDebt, longTermDebt, shortTermDebt
- goodwill, intangibleAssets
- (and ~50 more fields)

**Cash Flow (~40):**
- operatingCashFlow, freeCashFlow
- capitalExpenditure
- dividendsPaid, stockRepurchased
- cashFromInvesting, cashFromFinancing
- (and ~30 more fields)

**Key Metrics (~60):**
- peRatio, pbRatio, priceToSales
- marketCap, enterpriseValue
- evToSales, evToEbitda
- roe, roa, roic
- currentRatio, quickRatio
- debtToEquity, debtToAssets
- (and ~45 more fields)

**Financial Ratios (~60):**
- grossProfitMargin, operatingProfitMargin, netProfitMargin
- returnOnEquity, returnOnAssets, returnOnCapitalEmployed
- assetTurnover, inventoryTurnover
- daysSalesOutstanding, daysInventoryOutstanding
- (and ~50 more fields)

**Enterprise Values (~15):**
- enterpriseValue, marketCapitalization
- totalDebt, totalCash, netDebt
- evToOperatingCashFlow, evToFreeCashFlow

**Growth Metrics (~20):**
- revenueGrowth, epsgrowth, netIncomeGrowth
- operatingIncomeGrowth, assetGrowth
- bookValuePerShareGrowth, debtGrowth
- (and ~15 more fields)

### Technical/Derived Features - ~76 total

**Volume Features (~10):**
- volume_ratio_5d, volume_ratio_10d, volume_ratio_20d
- volume_spike_2x, volume_spike_3x
- dollar_volume, volume_momentum

**Price Features (~8):**
- intraday_gap, high_low_range
- close_position_in_range, price_gap_from_previous

**Momentum Features (~10):**
- return_1d, return_5d, return_20d, return_60d, return_252d
- price_acceleration, momentum_oscillator

**Volatility Features (~8):**
- volatility_10d, volatility_20d, volatility_60d
- parkinson_volatility, volatility_ratio

**Bollinger Bands (~5):**
- bb_upper, bb_lower, bb_width
- bb_percent_b, distance_from_bands

**Market-Relative (~20):**
- beta_sp500, beta_nasdaq, beta_sector
- correlation_sp500, correlation_nasdaq
- outperformance_1d, outperformance_5d, outperformance_20d
- vix_level, vix_high, vix_extreme

**Cross-sectional Rankings (~15):**
- return_1d_percentile, return_5d_percentile
- volume_percentile, volatility_percentile
- market_cap_percentile, pe_percentile
- roe_percentile, revenue_growth_percentile

### News Features - 768 total

- 768-dimensional Nomic AI embeddings
- Aggregated daily from all news articles
- Forward-filled for continuity

---

**Total Features: ~1,149**

This comprehensive dataset enables the transformer model to learn patterns from:
- Fundamental business health (quarterly reports)
- Technical price action (daily movements)
- Market context (indices, sectors, volatility)
- Competitive positioning (cross-sectional ranks)
- Information flow (news sentiment embeddings)

The combination provides a rich, multi-dimensional view of each stock's state at any point in time.
