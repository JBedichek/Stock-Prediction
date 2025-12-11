# Daily Stock Predictions - Complete Guide

Get daily stock recommendations with one command.

## Two Approaches

### Approach 1: Quick Predictions (Uses Existing Dataset)

**Best for**: Daily predictions when your dataset is already up-to-date

```bash
./get_daily_stocks.sh 10
```

This uses your existing `all_complete_dataset.h5` which should contain at least 10 years of data up to recent dates.

### Approach 2: Fresh Data Pipeline (Creates New Dataset)

**Best for**: When you want the absolute latest data

```bash
python daily_prediction_pipeline.py \
    --raw-data-dir data/ \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10
```

This creates a brand new dataset with the most recent data before running predictions.

---

## Quick Start

### Option 1: Use Existing Dataset (Fastest)

```bash
# Get top 10 stocks
./get_daily_stocks.sh 10

# Get top 5 stocks
./get_daily_stocks.sh 5
```

**Requirements:**
- `all_complete_dataset.h5` - dataset with 10 years of historical data
- `actual_prices.h5` - actual price data
- `checkpoints/best_model.pt` - trained model
- `adaptive_bin_edges.pt` - bin edges file

### Option 2: Create Fresh Dataset (Most Accurate)

```bash
# Step 1: Update raw data (run your data collection scripts)
python data_scraping/yfinance_price_scraper.py
python data_scraping/fmp_comprehensive_scraper.py
python data_scraping/news_embedder.py

# Step 2: Run pipeline with fresh data
python daily_prediction_pipeline.py \
    --raw-data-dir data/ \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt
```

---

## Detailed Setup

### Dataset Requirements

Your dataset must contain **at least 10 years** of historical data (approximately 2,520 trading days) to fit the model's sequence length requirement of 2000.

#### Check Your Dataset

```python
import h5py

h5f = h5py.File('all_complete_dataset.h5', 'r')
ticker = list(h5f.keys())[0]
dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]

print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Total dates: {len(dates)}")
print(f"Years of data: {len(dates) / 252:.1f}")  # ~252 trading days/year

h5f.close()
```

Expected output:
```
Date range: 2014-12-10 to 2024-12-10
Total dates: 2520
Years of data: 10.0
```

#### If You Need to Create a Dataset

If your dataset is outdated or missing, you need to create one with 10 years of historical data:

1. **Collect Raw Data** (prices, fundamentals, news):
   ```bash
   # Update this with your actual data collection process
   python data_scraping/yfinance_price_scraper.py --years 10
   python data_scraping/fmp_comprehensive_scraper.py --years 10
   python data_scraping/news_embedder.py
   ```

2. **Create HDF5 Dataset**:
   ```bash
   # Use your dataset creation script
   python dataset_creation/create_hdf5_dataset.py \
       --bulk-prices data/bulk_prices.pickle \
       --fundamentals data/fundamentals.pickle \
       --news-embeddings data/news_embeddings.pickle \
       --output all_complete_dataset.h5 \
       --years 10
   ```

---

## Usage Examples

### Example 1: Quick Daily Prediction

```bash
./get_daily_stocks.sh 10
```

Output:
```
================================================================================
📈 TOP 10 STOCK RECOMMENDATIONS FOR 2024-12-10
================================================================================

Rank   Ticker   Expected Return    Confidence   Current Price
--------------------------------------------------------------------------------
1      AAPL      +3.45%            0.8234       $   175.23
2      MSFT      +2.87%            0.8156       $   370.45
...
```

### Example 2: Conservative Strategy (20-day hold)

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 20 \
    --horizon-idx 3 \
    --confidence-percentile 0.5
```

### Example 3: Aggressive Day Trading (1-day hold)

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 5 \
    --horizon-idx 0 \
    --confidence-percentile 0.7
```

### Example 4: Fresh Data Pipeline

```bash
# Only works if you have raw data sources set up
python daily_prediction_pipeline.py \
    --raw-data-dir data/ \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --lookback-years 10
```

---

## Automation

### Run Predictions Every Weekday at 9 AM

Add to crontab:

```bash
crontab -e
```

Add line:
```
0 9 * * 1-5 cd /home/james/Desktop/Stock-Prediction && ./get_daily_stocks.sh 10 >> predictions.log 2>&1
```

### Email Results

```bash
#!/bin/bash
# email_predictions.sh

./get_daily_stocks.sh 10 > results.txt

# Extract top stocks
TOP_STOCKS=$(grep "1" results.txt | head -10)

# Send email
echo "Daily Stock Recommendations

$TOP_STOCKS

See attached for full results." | mail -s "Daily Stocks $(date +%Y-%m-%d)" -A predictions_*.pt your@email.com

rm results.txt
```

---

## Understanding the Output

### Console Output

```
================================================================================
📈 TOP 10 STOCK RECOMMENDATIONS FOR 2024-12-10
================================================================================

Holding period: 5 trading days
Total candidates analyzed: 3245

Rank   Ticker   Expected Return    Confidence   Current Price
--------------------------------------------------------------------------------
1      AAPL      +3.45%            0.8234       $   175.23
2      MSFT      +2.87%            0.8156       $   370.45
...
```

- **Expected Return**: Model's prediction for price change over holding period
- **Confidence**: Model's confidence in the prediction (higher = more reliable)
- **Current Price**: Latest price from dataset

### Saved Results

```python
import torch

results = torch.load('predictions_20241210.pt')

# Access recommendations
for rec in results['recommendations']:
    ticker = rec['ticker']
    return_pct = rec['expected_return_pct']
    confidence = rec['confidence']
    price = rec['current_price']

    print(f"{ticker}: {return_pct:+.2f}% (confidence: {confidence:.4f}) @ ${price:.2f}")
```

---

## Strategy Parameters

All parameters match `backtest_simulation.py` for consistency:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 10 | Number of stocks to recommend |
| `--horizon-idx` | 1 (5 days) | Holding period (0=1d, 1=5d, 2=10d, 3=20d) |
| `--confidence-percentile` | 0.6 | Filter threshold (0.6 = keep top 40%) |
| `--batch-size` | 256 | Inference batch size |
| `--device` | cuda | Device (cuda or cpu) |

### Horizon Index Mapping

- `0`: 1 trading day (day trading)
- `1`: 5 trading days (swing trading) ← **default**
- `2`: 10 trading days (short-term)
- `3`: 20 trading days (medium-term)

### Confidence Percentile

- `0.5`: Keep top 50% (less selective, more stocks)
- `0.6`: Keep top 40% ← **default**
- `0.7`: Keep top 30% (more selective, fewer stocks)
- `0.8`: Keep top 20% (very selective, high confidence only)

---

## Data Freshness Workflow

### Daily Workflow (Recommended)

```bash
#!/bin/bash
# daily_workflow.sh - Run this every trading day

echo "=== Daily Stock Prediction Workflow ==="
echo "Date: $(date)"

# Option A: Use existing dataset (fast)
./get_daily_stocks.sh 10

# Option B: Update data and create fresh dataset (slower but more accurate)
# Uncomment if you want fresh data:
# python data_scraping/yfinance_price_scraper.py --incremental
# python data_scraping/fmp_comprehensive_scraper.py --incremental
# python daily_prediction_pipeline.py --raw-data-dir data/ --model checkpoints/best_model.pt --bin-edges adaptive_bin_edges.pt

echo "=== Complete ==="
```

### Weekly Data Update (Full Refresh)

```bash
#!/bin/bash
# weekly_data_update.sh - Run this every Sunday

echo "=== Weekly Data Refresh ==="

# Collect fresh data (10 years)
python data_scraping/yfinance_price_scraper.py --years 10
python data_scraping/fmp_comprehensive_scraper.py --years 10
python data_scraping/news_embedder.py

# Create new dataset
python dataset_creation/create_hdf5_dataset.py \
    --bulk-prices data/bulk_prices.pickle \
    --fundamentals data/fundamentals.pickle \
    --news-embeddings data/news_embeddings.pickle \
    --output all_complete_dataset.h5 \
    --years 10

# Backup old dataset
mv all_complete_dataset.h5 all_complete_dataset_$(date +%Y%m%d).h5.bak
mv new_dataset.h5 all_complete_dataset.h5

echo "=== Dataset Updated ==="
```

---

## Comparison: Fresh Data vs Existing Dataset

| Aspect | Existing Dataset | Fresh Data Pipeline |
|--------|------------------|---------------------|
| **Speed** | ⚡ Very fast (~1-2 min) | 🐌 Slower (~30-60 min) |
| **Accuracy** | ✅ Good if dataset recent | ✅✅ Best - latest data |
| **Requirements** | Dataset file only | Raw data sources |
| **When to Use** | Daily predictions | Weekly refresh |
| **Command** | `./get_daily_stocks.sh` | `daily_prediction_pipeline.py` |

**Recommendation**:
- Use **existing dataset** for daily predictions (fast)
- Run **fresh data pipeline** weekly to update your dataset
- This gives you the best balance of speed and accuracy

---

## Troubleshooting

### "Dataset is X years old, not 10 years"

Your dataset doesn't have enough historical data. You need at least 10 years (2,520 trading days) because the model uses seq_len=2000.

**Solution**: Recreate your dataset with 10 years of data:
```bash
python dataset_creation/create_hdf5_dataset.py --years 10 ...
```

### "No valid data for today's date"

Your dataset doesn't include today. It may be several days/weeks old.

**Solution**: Either:
1. The script will use the latest available date automatically
2. Update your dataset with recent data

### "Data is X hours old"

When using `daily_prediction_pipeline.py`, it checks if raw data is fresh (< 24 hours).

**Solution**: Update your raw data:
```bash
python data_scraping/yfinance_price_scraper.py --incremental
python data_scraping/fmp_comprehensive_scraper.py --incremental
```

Or use `--force` to skip the check:
```bash
python daily_prediction_pipeline.py --force ...
```

### "CUDA out of memory"

Reduce batch size:
```bash
python inference/predict_current_day.py --batch-size 64 ...
```

Or use CPU (slower):
```bash
python inference/predict_current_day.py --device cpu ...
```

---

## Integration with Trading

### Paper Trading Example

```python
import torch
from datetime import datetime, timedelta

# Load predictions
results = torch.load('predictions_20241210.pt')

# Simulate portfolio
capital = 100000  # $100k
num_stocks = len(results['recommendations'])
position_size = capital / num_stocks

print("📊 Paper Trading Portfolio")
print(f"Capital: ${capital:,.2f}")
print(f"Position size: ${position_size:,.2f} each")
print()

for rec in results['recommendations']:
    ticker = rec['ticker']
    price = rec['current_price']
    expected_return = rec['expected_return_pct']

    shares = int(position_size / price)
    investment = shares * price

    print(f"{ticker}:")
    print(f"  Buy {shares} shares @ ${price:.2f} = ${investment:,.2f}")
    print(f"  Expected return: +{expected_return:.2f}%")
    print(f"  Expected value: ${investment * (1 + expected_return/100):,.2f}")
    print()
```

### Backtesting Your Strategy

```bash
# After running daily predictions for a while, backtest them
python inference/backtest_simulation.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10 \
    --horizon-idx 1 \
    --test-months 3
```

---

## Summary

### For Daily Use (Fastest)

```bash
./get_daily_stocks.sh 10
```

### For Fresh Data (Most Accurate)

```bash
# 1. Update raw data
python data_scraping/your_data_collection_scripts.py

# 2. Run pipeline
python daily_prediction_pipeline.py \
    --raw-data-dir data/ \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt
```

### Files Created

- `predictions_YYYYMMDD.pt` - Full predictions (loadable with torch)
- `daily_predictions_YYYYMMDD_HHMMSS.txt` - Simple text summary
- `daily_predictions_YYYYMMDD_HHMMSS.pt` - Full results from pipeline

---

## Next Steps

1. **Verify your dataset has 10 years of data** ← Start here!
2. **Run your first prediction**: `./get_daily_stocks.sh 10`
3. **Review the results** and understand the metrics
4. **Set up weekly data refresh** to keep dataset current
5. **Automate daily predictions** with cron
6. **Track performance** by comparing predictions to actual returns

## Disclaimer

This tool is for educational and research purposes only. Always do your own research and consult with a financial advisor before making investment decisions.
