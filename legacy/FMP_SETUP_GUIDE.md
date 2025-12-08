# Financial Modeling Prep (FMP) Setup Guide

## What You Get

- **20+ years** of quarterly fundamental data
- **27 fundamental metrics** per stock, changing over time
- **Completely FREE** (up to 250 API calls/day)

## Step-by-Step Setup

### 1. Get Your Free API Key

1. Go to: https://site.financialmodelingprep.com/developer/docs
2. Click **"Get your Free API Key Today"**
3. Fill out the signup form (name, email)
4. Check your email and verify
5. Copy your API key (looks like: `abcdef123456...`)

### 2. Test the API

```bash
# Edit test_fmp_api.py and replace YOUR_API_KEY with your actual key
nano test_fmp_api.py  # Or use your preferred editor

# Run the test
python test_fmp_api.py
```

You should see output like:
```
✅ Success! Retrieved 120 periods
Date range: 2005-06-30 to 2025-09-30
Span: 20.2 years
```

### 3. Generate Fundamentals for Your Dataset

FMP free tier allows **250 API calls per day**.
- Each stock requires 3 calls (income + balance + cash flow)
- **~80 stocks per day** maximum

For **s_lot_of_stocks (370 stocks)**:
- Total calls needed: 370 × 3 = 1,110 calls
- **Estimated time: 5 days** (running once per day)

#### Day 1: First ~80 stocks
```bash
python generate_fmp_fundamentals.py \
    --dataset s_lot \
    --api_key YOUR_API_KEY
```

The script will automatically stop at 240 calls and tell you the last ticker processed.

#### Day 2-5: Resume from where you left off
```bash
python generate_fmp_fundamentals.py \
    --dataset s_lot \
    --api_key YOUR_API_KEY \
    --resume LAST_TICKER  # Replace with ticker from previous run
```

### 4. Alternative: Smaller Test First

Test with just a few stocks by creating a small dictionary:

```python
# test_small.py
from generate_fmp_fundamentals import *

test_stocks = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'TSLA': 'Tesla Inc.',
    'JPM': 'JPMorgan Chase'
}

fundamentals = generate_fmp_fundamentals(
    test_stocks,
    "test",
    "YOUR_API_KEY",
    start_date='2015-01-01',
    end_date='2025-12-31'
)

fundamentals = normalize_fundamentals(fundamentals)
save_pickle(fundamentals, 'test_fmp_fundamentals.pkl')
print("✅ Test complete!")
```

## Data Structure

The output pickle file contains:

```python
{
    'AAPL': {
        datetime.date(2015, 1, 1): tensor([0.31, 0.42, ...]),  # 27 metrics
        datetime.date(2015, 1, 2): tensor([0.31, 0.42, ...]),  # Same values (forward-filled)
        ...
        datetime.date(2015, 4, 1): tensor([0.33, 0.44, ...]),  # NEW quarter, different values!
        ...
    },
    'MSFT': { ... },
    ...
}
```

**Key point**: Metrics actually **change over time** (quarterly), solving the original problem!

## 27 Metrics Extracted

### Profitability (4 metrics)
- Gross Margin
- Operating Margin
- Profit Margin
- EBITDA Margin

### Returns (2 metrics)
- Return on Equity (ROE)
- Return on Assets (ROA)

### Financial Health (3 metrics)
- Current Ratio
- Quick Ratio
- Debt to Equity

### Cash & Debt (3 metrics)
- Total Cash (billions)
- Total Debt (billions)
- Operating Cash Flow (billions)

### Per-Share (3 metrics)
- Revenue per Share
- EPS (Earnings per Share)
- Book Value per Share

### Valuation (5 metrics)
- Price to Book
- Price to Sales
- Enterprise Value / Revenue
- Enterprise Value / EBITDA
- Payout Ratio

### Market (3 metrics)
- Market Cap (billions)
- Forward P/E
- Forward EPS

### Growth (1 metric)
- Revenue Growth (QoQ)

### Placeholder (3 metrics - not in statements)
- Beta
- 52-Week High
- 52-Week Low

## Next Steps After Data Generation

Once you have `s_lot_of_stocks_fmp_fundamentals.pkl`:

### 1. Update training.py to load FMP data

```python
# In QTrainingData.__init__()
fundamentals_pth = data_dict_pth.replace('DataDict', 'fmp_fundamentals').replace('.pt', '.pkl')
self.fundamentals = pic_load(fundamentals_pth)
```

### 2. Modify get_company_chunk_data to use dates

```python
def get_company_chunk_data(self, company, date):
    # ... existing code ...

    # Get fundamentals for this specific date
    if self.fundamentals is not None and company in self.fundamentals:
        if date in self.fundamentals[company]:
            fundamentals = self.fundamentals[company][date]
        else:
            # Date not available, use zeros
            fundamentals = torch.zeros(27)
    else:
        fundamentals = torch.zeros(27)

    return (comp_data, c, pca_stats, price, rel_data, fundamentals)
```

### 3. Update models.py (next phase)

Add fundamental embedding layer and integrate into transformer.

## Troubleshooting

### "Rate limit exceeded"
You've hit the 250 calls/day limit. Wait 24 hours and resume.

### "API key invalid"
- Check you copied the full key
- Verify email is confirmed
- Try regenerating key on FMP website

### "No data for ticker"
Some small-cap or international stocks may not have data on FMP. The script will skip them automatically.

## Cost Comparison

| Option | Cost | Data History | Ease |
|--------|------|--------------|------|
| yfinance | Free | 4-5 years | ✅ Easy |
| **FMP Free Tier** | Free | 20+ years | ✅ Easy |
| FMP Professional | $20/mo | 30+ years, 500 calls/min | ✅ Easy |

For your use case, **FMP Free Tier is perfect**.

---

**Ready to start?** Get your API key and run the test script!
