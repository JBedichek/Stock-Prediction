# FMP Comprehensive Data Scraper Guide

## Overview

This guide covers the **comprehensive FMP data scraper** that dramatically expands the time-series data fed to your transformer model. Instead of just 27 fundamental metrics, you now get **hundreds of features** from multiple data sources.

## What Data Do You Get?

### 1. **Financial Statements** (Quarterly, 30+ years)
- Income Statements: Revenue, EBITDA, Operating Income, Net Income, etc.
- Balance Sheets: Assets, Liabilities, Equity, Cash, Debt, etc.
- Cash Flow Statements: Operating CF, Investing CF, Financing CF, Free Cash Flow, etc.
- **~100+ metrics** per statement, changing quarterly

### 2. **Key Financial Metrics** (Quarterly)
- P/E Ratio, P/B Ratio, ROE, ROA, Gross Margin, Operating Margin
- Current Ratio, Quick Ratio, Debt-to-Equity
- Enterprise Value metrics
- **60+ derived metrics** calculated by FMP

### 3. **Financial Ratios** (Quarterly)
- Profitability ratios, Liquidity ratios, Leverage ratios
- Efficiency ratios, Valuation ratios
- **60+ comprehensive ratios**

### 4. **Daily Price Data** (30+ years, OHLCV)
- Open, High, Low, Close, Volume
- Price changes, Returns
- Intraday volatility measures

### 5. **Technical Indicators** (Daily)
- Simple/Exponential/Weighted Moving Averages (10, 20, 50, 200-day)
- RSI (Relative Strength Index)
- ADX (Average Directional Index)
- Williams %R, Standard Deviation
- **20+ technical indicators**

### 6. **Earnings Data** (Quarterly)
- Actual EPS vs Estimated EPS
- Earnings surprises (beat/miss)
- Earnings call dates

### 7. **Insider Trading** (Transaction-level)
- Buy/Sell transactions by insiders
- Shares traded, Prices paid
- Aggregated daily: Total insider buying/selling

### 8. **Institutional Holdings** (Quarterly via 13F)
- Top institutional holders
- Shares held, Value of holdings
- Changes in ownership

### 9. **Analyst Data** (Ongoing)
- Analyst ratings (Buy/Hold/Sell) over time
- Price targets from analysts
- Consensus estimates (Revenue, EPS)

### 10. **Market Data**
- Market capitalization history
- Sector/Industry performance
- Dividend history, Stock splits

## Total Features

After processing, you get **200-500+ features per day** depending on data availability:
- ~150 fundamental metrics (quarterly, forward-filled daily)
- ~50 price & volume features (daily)
- ~20 technical indicators (daily)
- ~30 analyst/insider/institutional features (irregular, forward-filled)

Compare this to the original 27 metrics!

## Premium Tier Benefits

Assuming premium FMP tier ($20-50/month):
- **Unlimited API calls** (or very high limits like 100K+/day)
- **30+ years of historical data** (vs 20 years free)
- **Intraday data** (1min, 5min, 15min, 30min, 1hour, 4hour)
- **Real-time data** (if needed for live trading)
- **Faster API response times**

Free tier: 250 calls/day → Can do ~80 stocks/day
Premium tier: Unlimited → Can do **all 4000 stocks in a few hours**

## Setup & Usage

### Step 1: Get Your API Key

1. Go to: https://site.financialmodelingprep.com/pricing-plans
2. Sign up for Premium tier ($20-50/month depending on plan)
3. Get your API key from the dashboard

### Step 2: Test Your API Connection

```bash
# Edit the test file and add your API key
nano tests/test_fmp_comprehensive.py

# Set this line:
API_KEY = "your_actual_api_key_here"

# Run the tests
python tests/test_fmp_comprehensive.py
```

You should see:
```
TEST 1: API CONNECTION
  ✅ Success! Company profile:
     Company: Apple Inc.
     Sector: Technology
     ...

TEST 2: FINANCIAL STATEMENTS
  ✅ Income Statement: 120 periods
  ✅ Balance Sheet: 120 periods
  ✅ Cash Flow: 120 periods

...

🎉 ALL TESTS PASSED!
```

### Step 3: Scrape Comprehensive Data

```bash
# For s_lot_of_stocks (~370 stocks)
# Premium tier: ~30 minutes
# Free tier: ~5 days (80 stocks/day)

python data_scraping/fmp_comprehensive_scraper.py \
    --api_key YOUR_API_KEY \
    --dataset s_lot \
    --start_date 2000-01-01

# Output: s_lot_fmp_comprehensive.pkl
```

Options:
- `--dataset`: Choose 's_lot' (~370 stocks), 'a_lot' (~800 stocks), or 'all' (~4000 stocks)
- `--start_date`: How far back to collect data (default: 2000-01-01)
- `--resume TICKER`: Resume from a specific ticker if interrupted
- `--output`: Custom output filename

### Step 4: Process Data into Tensors

```bash
# Convert raw FMP data into daily tensors
python data_scraping/fmp_data_processor.py \
    --input s_lot_fmp_comprehensive.pkl \
    --start_date 2000-01-01

# Output: s_lot_fmp_comprehensive_processed.pkl
```

This creates:
```python
{
    'AAPL': {
        datetime.date(2000, 1, 1): tensor([...]),  # 200-500 features
        datetime.date(2000, 1, 2): tensor([...]),
        ...
        datetime.date(2024, 12, 7): tensor([...]),
    },
    'MSFT': { ... },
    ...
}
```

### Step 5: Integrate with Training Pipeline

Update your dataset loader to use the comprehensive FMP data:

```python
# In dataset_creation/dataset_processing.py

class QTrainingData(Dataset):
    def __init__(self, ..., fmp_data_pth=None):
        # Load comprehensive FMP data
        if fmp_data_pth:
            self.fmp_data = pic_load(fmp_data_pth)
            print(f'✅ Loaded FMP data for {len(self.fmp_data)} stocks')

    def get_company_chunk_data(self, company, date):
        # ... existing code ...

        # Get FMP features for this specific date
        if self.fmp_data and company in self.fmp_data:
            if date in self.fmp_data[company]:
                fmp_features = self.fmp_data[company][date]
            else:
                # Date not available, use zeros
                fmp_features = torch.zeros(250)  # Adjust size as needed
        else:
            fmp_features = torch.zeros(250)

        return (comp_data, c, pca_stats, price, rel_data, fundamentals, fmp_features)
```

## API Call Estimation

For reference, here's how many API calls the comprehensive scraper makes per stock:

| Data Source | Calls per Stock |
|-------------|----------------|
| Financial Statements | 3 (income, balance, cash) |
| Key Metrics | 1 |
| Financial Ratios | 1 |
| Enterprise Values | 1 |
| Financial Growth | 1 |
| Daily Prices | 1 |
| Technical Indicators | 10 (various types/periods) |
| Earnings | 2 |
| Insider Trading | 2 |
| Institutional Holdings | 1 |
| Analyst Ratings | 3 |
| Company Profile | 1 |
| Market Cap History | 1 |
| Dividends & Splits | 2 |
| **Total** | **~30 calls/stock** |

For 370 stocks (s_lot): **~11,000 API calls**
For 4000 stocks (all): **~120,000 API calls**

**Free tier**: 250 calls/day → 44 days for all stocks
**Premium tier** (100K calls/day): < 2 hours for all stocks

## Cost-Benefit Analysis

| Option | Monthly Cost | Features | Time for 4000 Stocks | Data History |
|--------|-------------|----------|---------------------|--------------|
| Current (yfinance fundamentals) | Free | 27 | N/A | 4-5 years |
| FMP Free Tier | Free | 200-500+ | 44 days | 20 years |
| **FMP Premium** | **$20-50** | **200-500+** | **2 hours** | **30+ years** |

**Recommendation**: FMP Premium is worth it if you're serious about this project. The 10-20x feature expansion could dramatically improve model performance.

## Troubleshooting

### "Rate limit exceeded"
- **Free tier**: You've hit 250 calls/day. Use `--resume TICKER` to continue tomorrow.
- **Premium tier**: Very unlikely, but wait a few minutes if it happens.

### "API key invalid"
- Check you copied the full key
- Verify your subscription is active
- Try regenerating the key on FMP website

### "No data for ticker XYZ"
- Some small-cap or international stocks may not have full data
- The scraper automatically skips them
- Check FMP website to see if data exists for that ticker

### Script crashes mid-run
- Use `--resume TICKER` to continue from where it left off
- The script saves progress every 10 stocks

## Advanced Features

### Intraday Data (Future Enhancement)

The scraper supports intraday data but it's not enabled by default:

```python
# In fmp_comprehensive_scraper.py
scraper.get_intraday_prices('AAPL', '1min', '2024-01-01', '2024-12-31')
```

You could add:
- 1-minute OHLCV for last 30 days
- 5-minute data for volatility calculations
- Hourly data for longer-term patterns

### Custom Feature Engineering

Modify `fmp_data_processor.py` to add custom features:

```python
def process_custom_features(self) -> pd.DataFrame:
    """Add your own derived features."""
    df = self.raw_data.get('daily_prices')

    # Example: Momentum features
    features = {}
    features['momentum_5'] = df['close'].pct_change(5)
    features['momentum_20'] = df['close'].pct_change(20)

    # Example: Volatility features
    features['volatility_10'] = df['close'].rolling(10).std()

    return pd.DataFrame(features)
```

## Next Steps

After collecting comprehensive FMP data:

1. **Expand model architecture** to handle 200-500 features instead of 27
2. **Add feature selection** to identify most important features
3. **Experiment with feature engineering** (ratios, differences, rolling stats)
4. **Compare performance** against baseline (27 features only)
5. **Consider ensemble models** that specialize in different feature subsets

## Questions?

- FMP Documentation: https://site.financialmodelingprep.com/developer/docs
- FMP Support: support@financialmodelingprep.com
- Check `tests/test_fmp_comprehensive.py` for working examples

---

**Ready to dramatically expand your feature space? Get your API key and start scraping!** 🚀
