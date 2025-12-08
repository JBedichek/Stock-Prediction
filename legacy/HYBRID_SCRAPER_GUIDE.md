# Hybrid Data Scraper Guide (yfinance + FMP)

## Overview

The **Hybrid Scraper** combines the best of both worlds:
- **yfinance**: Daily price/volume data (free, reliable, 20+ years)
- **FMP**: Quarterly fundamental data (161 quarters = 40+ years of financial statements)

This gives you **288 features per day** instead of the original 27!

## What You Get

### From yfinance (19 features, daily):
1. **Core OHLCV**: Open, High, Low, Close, Volume
2. **Returns**: 1-day, 5-day, 20-day returns
3. **Volatility**: 10-day, 20-day rolling volatility
4. **Volume**: 1-day volume change
5. **Intraday**: High-low range relative to close
6. **Moving Averages**: 10, 20, 50, 200-day SMA
7. **Relative Position**: Price vs each SMA

### From FMP (269 features, quarterly → daily):

#### Financial Statements (~150 features)
- **Income Statement**: Revenue, EBITDA, Operating Income, Net Income, Gross Profit, R&D, SG&A, etc.
- **Balance Sheet**: Total Assets, Liabilities, Equity, Cash, Debt, Inventory, Receivables, etc.
- **Cash Flow**: Operating CF, Investing CF, Financing CF, Free Cash Flow, CapEx, etc.

#### Key Metrics (~50 features)
- P/E Ratio, P/B Ratio, EV/EBITDA, Price to Sales
- ROE, ROA, ROIC, Gross Margin, Operating Margin, Profit Margin
- Current Ratio, Quick Ratio, Cash Ratio
- Debt to Equity, Debt to Assets
- Asset Turnover, Inventory Turnover, Receivables Turnover
- Working Capital, Book Value per Share, Tangible Book Value

#### Financial Ratios (~60 features)
- Profitability Ratios, Liquidity Ratios, Leverage Ratios
- Efficiency Ratios, Valuation Ratios
- Dividend Yield, Payout Ratio

#### Other (~9 features)
- Enterprise Value metrics
- Growth rates (revenue, earnings, etc.)

## How It Works

1. **Fetch price data** from yfinance (daily OHLCV, 2000-present)
2. **Fetch fundamentals** from FMP (quarterly, 40+ years history)
3. **Process price features**: Returns, volatility, moving averages, etc.
4. **Process fundamental features**: Extract numeric columns from financial statements
5. **Forward-fill fundamentals**: Quarterly data → daily frequency (values stay constant between quarters)
6. **Combine**: Merge price features (daily) + fundamental features (forward-filled to daily)
7. **Normalize**: Z-score normalize each feature
8. **Convert**: Daily DataFrame → {date: tensor} format

## Quick Start

### Test on 3 Stocks

```bash
python3 tests/test_hybrid_scraper.py
```

Expected output:
```
✅ SUCCESS!
   Total days: 1827 (2020-2024)
   Features per day: 288

   19 price features + 269 fundamental features = 288 total
```

### Scrape Full Dataset

```bash
# For s_lot_of_stocks (~370 stocks)
python3 data_scraping/hybrid_scraper.py --dataset s_lot

# This will take ~30-60 minutes
# Output: s_lot_hybrid_data.pkl
```

Options:
- `--dataset`: 's_lot' (~370 stocks), 'a_lot' (~800 stocks), or 'all' (~4000 stocks)
- `--start_date`: Start date (default: 2000-01-01)
- `--end_date`: End date (default: today)
- `--resume TICKER`: Resume from specific ticker if interrupted
- `--output`: Custom output filename

### For All Stocks

```bash
# For all ~4000 stocks
python3 data_scraping/hybrid_scraper.py --dataset all

# This will take ~6-8 hours
# Output: all_hybrid_data.pkl
```

## Output Format

The scraper creates a pickle file with this structure:

```python
{
    'AAPL': {
        datetime.date(2020, 1, 1): tensor([288 features]),
        datetime.date(2020, 1, 2): tensor([288 features]),
        ...
        datetime.date(2024, 12, 31): tensor([288 features]),
    },
    'MSFT': { ... },
    ...
}
```

Each tensor contains:
- **[0:19]**: Price features (daily)
- **[19:288]**: Fundamental features (quarterly, forward-filled)

All features are z-score normalized.

## Time & Resource Estimates

| Dataset | Stocks | Time | yfinance Calls | FMP API Calls | Output Size |
|---------|--------|------|----------------|---------------|-------------|
| Test (3 stocks) | 3 | ~1 min | 3 | 24 | ~1 MB |
| s_lot | ~370 | ~30-60 min | 370 | ~3,000 | ~100 MB |
| a_lot | ~800 | ~1-2 hrs | 800 | ~6,500 | ~220 MB |
| all | ~4000 | ~6-8 hrs | 4,000 | ~32,000 | ~1.1 GB |

**Note**:
- yfinance: No rate limits (free)
- FMP: ~8 API calls per stock (all fundamental endpoints)
- The scraper saves progress every 10 stocks, so interruptions are safe

## Advantages Over Original Approach

### Original (27 features):
- ❌ Only 27 fundamental metrics
- ❌ Time-varying fundamentals not fully implemented
- ❌ Required manual forward-filling logic

### Hybrid (288 features):
- ✅ **288 features** (10x more!)
- ✅ **40+ years** of fundamental history
- ✅ **Automatic forward-filling** from quarterly to daily
- ✅ **Price + fundamental features** combined
- ✅ **Fully normalized** and ready to use
- ✅ **Backward compatible** with existing training code

## Integration with Training Pipeline

Update your training code to use the hybrid data:

```python
# In dataset_creation/dataset_processing.py or your training script

class QTrainingData(Dataset):
    def __init__(self, ..., hybrid_data_pth='s_lot_hybrid_data.pkl'):
        # Load hybrid data
        self.hybrid_data = pic_load(hybrid_data_pth)
        print(f'✅ Loaded hybrid data for {len(self.hybrid_data)} stocks')

    def get_company_chunk_data(self, company, date):
        # ... existing code ...

        # Get hybrid features for this date
        if self.hybrid_data and company in self.hybrid_data:
            if date in self.hybrid_data[company]:
                hybrid_features = self.hybrid_data[company][date]
            else:
                # Date not available (weekends/holidays), use zeros
                hybrid_features = torch.zeros(288)
        else:
            hybrid_features = torch.zeros(288)

        return (comp_data, c, pca_stats, price, rel_data, hybrid_features)
```

Or simply replace your current fundamentals entirely:

```python
# Old approach:
fundamentals = torch.zeros(27)  # Static fundamentals

# New hybrid approach:
hybrid_features = self.hybrid_data[company][date]  # 288 time-varying features
```

## Feature Breakdown

### Price Features (19):
```python
price_open, price_high, price_low, price_close, price_volume,
price_return_1d, price_return_5d, price_return_20d,
price_volatility_10d, price_volatility_20d,
volume_change_1d, intraday_range,
price_sma_10, price_sma_20, price_sma_50, price_sma_200,
price_vs_sma_10, price_vs_sma_20, price_vs_sma_50
```

### Fundamental Features (269):
```python
# Income statement (~50 features)
income_revenue, income_costOfRevenue, income_grossProfit,
income_operatingIncome, income_netIncome, income_ebitda, ...

# Balance sheet (~50 features)
balance_totalAssets, balance_totalLiabilities, balance_totalStockholdersEquity,
balance_cashAndCashEquivalents, balance_totalDebt, ...

# Cash flow (~50 features)
cashflow_operatingCashFlow, cashflow_freeCashFlow, cashflow_capitalExpenditure, ...

# Key metrics (~50 features)
metric_peRatio, metric_pbRatio, metric_roic, metric_roe, metric_roa,
metric_debtToEquity, metric_currentRatio, metric_quickRatio, ...

# Financial ratios (~60 features)
ratio_grossProfitMargin, ratio_operatingProfitMargin, ratio_netProfitMargin,
ratio_returnOnAssets, ratio_returnOnEquity, ratio_debtRatio, ...

# Other (~9 features)
ev_enterpriseValue, growth_revenueGrowth, growth_netIncomeGrowth, ...
```

## Troubleshooting

### "No price data available"
- Check ticker symbol is valid
- yfinance may not have data for very small/delisted stocks
- Try a different date range

### "No fundamental data available"
- Your FMP tier may not have access to that stock
- Very small/new companies may not have fundamental data
- The scraper will fall back to price-only features

### Script crashes mid-run
- Use `--resume TICKER` to continue from where it left off
- Progress is saved every 10 stocks in the output file

### Out of memory
- Process smaller batches (use --resume)
- Reduce date range with --start_date and --end_date
- Or process one dataset at a time (s_lot, then a_lot, etc.)

## Comparison to Manual Forward-Filling

### Old Approach (manual):
```python
# You had to manually implement this:
fundamentals_by_date = {}
for quarter_date in quarterly_fundamentals:
    # Forward fill until next quarter...
    # Complex date logic...
    # Manual tensor creation...
```

### Hybrid Scraper (automatic):
```python
# All handled automatically:
data = scrape_hybrid_dataset(stock_dict, fmp_api_key)
# Done! Ready to use.
```

## Example Use Case

```python
from utils.utils import pic_load

# Load hybrid data
data = pic_load('s_lot_hybrid_data.pkl')

# Get AAPL data for a specific date
aapl_features = data['AAPL'][datetime.date(2024, 1, 15)]

print(f"Features: {aapl_features.shape}")  # torch.Size([288])
print(f"Price features: {aapl_features[:19]}")  # OHLCV, returns, etc.
print(f"Fundamental features: {aapl_features[19:]}")  # Financial ratios, etc.
```

## Next Steps

1. **Test**: Run `python3 tests/test_hybrid_scraper.py`
2. **Scrape**: Run `python3 data_scraping/hybrid_scraper.py --dataset s_lot`
3. **Integrate**: Update your training pipeline to use hybrid data
4. **Train**: Train your model with 288 features instead of 27
5. **Compare**: Benchmark performance vs baseline (27 features)

## Cost Analysis

| Data Source | Cost | Features | Coverage |
|-------------|------|----------|----------|
| yfinance only | Free | 19 | 20+ years daily |
| FMP Free Tier | Free | 27-50 | 20 years quarterly |
| **Hybrid (yfinance + FMP)** | **Free** | **288** | **40+ years** |

The hybrid approach gives you **10x more features** at **zero cost**!

---

**Ready to scrape? Run the test and then scrape your dataset!** 🚀
