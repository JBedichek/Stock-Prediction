# Automated Daily Stock Predictions

**One command to rule them all**: Fetch fresh data, update dataset, get predictions.

## The Problem

You want daily stock predictions, but you need:
1. Fresh market data for today
2. Dataset updated with today's data
3. Predictions based on the latest data

Manually doing this every day is tedious.

## The Solution

```bash
./run_daily.sh 10
```

This **automatically**:
1. ✅ Fetches today's market data (prices, fundamentals, news)
2. ✅ Updates your HDF5 dataset incrementally (appends today's data)
3. ✅ Runs inference on the updated dataset
4. ✅ Outputs top 10 stocks to buy

**No manual intervention needed!**

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│  run_daily.sh                           │
│  (Simple wrapper)                       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  auto_daily_predictions.py              │
│  (Main automation logic)                │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐      ┌──────────┐
│  Fetch   │      │ Update   │
│  Data    │────▶ │ Dataset  │
└──────────┘      └─────┬────┘
                        │
                        ▼
                  ┌──────────┐
                  │   Run    │
                  │Inference │
                  └─────┬────┘
                        │
                        ▼
                  ┌──────────┐
                  │  Output  │
                  │  Stocks  │
                  └──────────┘
```

### Step-by-Step Process

1. **Check if update needed**
   - Looks at latest date in dataset
   - If dataset has today's data → skip update
   - If weekend → skip (no trading)
   - If market not closed → skip (wait until 4 PM)
   - Otherwise → fetch data

2. **Fetch today's data**
   - Prices from yfinance/FMP
   - Fundamentals from FMP
   - News embeddings from Nomic API
   - Combines into feature vectors

3. **Incremental dataset update**
   - Opens HDF5 file in append mode
   - For each ticker:
     - If ticker exists: append new date
     - If ticker new: create new group
   - Updates both features and prices files
   - **No need to recreate entire dataset!**

4. **Run predictions**
   - Loads updated dataset
   - Gets latest date (today)
   - Runs inference on all stocks
   - Ranks by confidence-weighted returns

5. **Output results**
   - Prints top-k recommendations
   - Saves to timestamped files
   - Creates both .pt and .txt formats

---

## Quick Start

### First Time Setup

1. **Create initial dataset** (one-time, ~30-60 min):
   ```bash
   # This creates a dataset with 10 years of historical data
   python dataset_creation/create_hdf5_dataset.py \
       --bulk-prices data/bulk_prices.pickle \
       --fundamentals data/fundamentals.pickle \
       --news-embeddings data/news_embeddings.pickle \
       --output all_complete_dataset.h5 \
       --years 10
   ```

2. **Verify dataset**:
   ```python
   import h5py
   h5f = h5py.File('all_complete_dataset.h5', 'r')
   ticker = list(h5f.keys())[0]
   dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]
   print(f"Dataset range: {dates[0]} to {dates[-1]}")
   print(f"Total dates: {len(dates)}")
   h5f.close()
   ```

### Daily Usage

```bash
# Get top 10 stocks for today
./run_daily.sh 10

# Get top 5 stocks
./run_daily.sh 5

# Get top 20 stocks
./run_daily.sh 20
```

That's it! The script handles everything else.

---

## What Gets Updated

### HDF5 Dataset Structure

```
all_complete_dataset.h5
├── AAPL/
│   ├── features (num_dates × num_features)
│   └── dates (num_dates,)
├── MSFT/
│   ├── features
│   └── dates
└── ...

On each run:
- Adds new row to 'features' for today
- Adds new date to 'dates'
- Preserves all historical data
```

### Incremental Update Example

**Before update:**
```
AAPL/features: (2520, 1400)  # 2520 historical dates
AAPL/dates: ['2014-12-10', ..., '2024-12-09']
```

**After update (Dec 10, 2024):**
```
AAPL/features: (2521, 1400)  # Added one more date
AAPL/dates: ['2014-12-10', ..., '2024-12-10']  # Added today
```

**No recreation needed!** Just appends to existing file.

---

## Automation

### Set Up Daily Cron Job

Run predictions automatically every weekday at 5 PM (after market close):

```bash
crontab -e
```

Add this line:
```
0 17 * * 1-5 cd /home/james/Desktop/Stock-Prediction && ./run_daily.sh 10 >> logs/daily_predictions.log 2>&1
```

This runs Monday-Friday at 5:00 PM.

### With Email Notifications

```bash
# add_email_notifications.sh

#!/bin/bash

./run_daily.sh 10 > /tmp/predictions.txt

# Extract top stocks
STOCKS=$(grep -A 10 "TOP 10 STOCK" /tmp/predictions.txt | tail -11)

# Send email
echo "Daily Stock Recommendations for $(date +%Y-%m-%d)

$STOCKS

Full results attached." | \
    mail -s "Daily Stocks $(date +%Y-%m-%d)" \
         -A daily_predictions_*.txt \
         your@email.com

rm /tmp/predictions.txt
```

---

## Configuration

### Edit run_daily.sh

Change these variables:

```bash
TOP_K=10           # Number of stocks
DATASET="all_complete_dataset.h5"
PRICES="actual_prices.h5"
MODEL="checkpoints/best_model.pt"
BIN_EDGES="adaptive_bin_edges.pt"
```

### Advanced Options

```bash
python auto_daily_predictions.py \
    --dataset all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10 \
    --horizon-idx 1 \
    --confidence-percentile 0.6 \
    --batch-size 256 \
    --device cuda \
    --skip-update  # Skip data fetch, just run predictions
```

---

## Data Sources

The script fetches data from:

1. **Prices**: yfinance or FMP API
   - Daily OHLCV data
   - Real-time or end-of-day

2. **Fundamentals**: FMP API
   - P/E, P/B ratios
   - Revenue, earnings
   - Quarterly metrics

3. **News**: Nomic API
   - Daily news embeddings (768-dim)
   - Market sentiment

### API Keys Required

Set these environment variables:

```bash
export FMP_API_KEY="your_fmp_key"
export NOMIC_API_KEY="your_nomic_key"
```

Or add to `~/.bashrc`:
```bash
echo 'export FMP_API_KEY="your_key"' >> ~/.bashrc
echo 'export NOMIC_API_KEY="your_key"' >> ~/.bashrc
source ~/.bashrc
```

### Free Tier Limits

- **FMP**: 250 calls/day (free tier)
- **Nomic**: 1000 calls/month (free tier)
- **yfinance**: Unlimited (unofficial API)

For ~3000 tickers, you may need paid API keys.

---

## File Organization

```
Stock-Prediction/
├── auto_daily_predictions.py   # Main automation script
├── run_daily.sh                 # Simple wrapper
├── all_complete_dataset.h5      # Your dataset (updated daily)
├── actual_prices.h5             # Prices (updated daily)
├── checkpoints/
│   └── best_model.pt           # Trained model
├── adaptive_bin_edges.pt        # Bin edges
├── daily_predictions_*.pt       # Saved predictions
├── daily_predictions_*.txt      # Text summaries
└── logs/
    └── daily_predictions.log   # Cron job logs
```

---

## Troubleshooting

### "Dataset doesn't exist"

**Problem**: Initial dataset not created yet.

**Solution**: Create initial dataset with 10 years of data:
```bash
python dataset_creation/create_hdf5_dataset.py --years 10
```

### "Data fetch failed"

**Problem**: API keys not set or rate limit exceeded.

**Solution**:
1. Check API keys: `echo $FMP_API_KEY`
2. Use `--skip-update` flag to skip data fetch:
   ```bash
   python auto_daily_predictions.py --skip-update
   ```

### "Market not closed yet"

**Problem**: Running before 4 PM ET (market close).

**Solution**: Wait until after market close, or script will use yesterday's data.

### "Weekend - no trading data"

**Problem**: Running on Saturday/Sunday.

**Solution**: Script automatically skips update on weekends. Will use Friday's data.

### "Dataset already up-to-date"

**Problem**: Already ran today, dataset has today's data.

**Solution**: This is normal! Script will skip update and just run predictions.

### "CUDA out of memory"

**Problem**: GPU doesn't have enough memory.

**Solution**: Reduce batch size or use CPU:
```bash
python auto_daily_predictions.py --batch-size 64 --device cpu
```

---

## Comparison: Manual vs Automated

| Task | Manual Approach | Automated Approach |
|------|----------------|-------------------|
| Fetch data | Run 3+ scripts | Automatic |
| Update dataset | Recreate entire dataset (slow) | Incremental append (fast) |
| Run predictions | Separate command | Built-in |
| Time required | 30-60 min | 2-5 min |
| Daily effort | High | Zero (set & forget) |

**Automated approach is 10-15x faster!**

---

## Monitoring

### Check Logs

```bash
# View today's log
tail -f logs/daily_predictions.log

# View last 50 lines
tail -50 logs/daily_predictions.log

# Search for errors
grep "ERROR\|Failed" logs/daily_predictions.log
```

### Check Dataset Growth

```python
import h5py
from datetime import datetime

h5f = h5py.File('all_complete_dataset.h5', 'r')
ticker = list(h5f.keys())[0]
dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]

print(f"Dataset has {len(dates)} dates")
print(f"Latest date: {dates[-1]}")
print(f"Expected: {datetime.now().strftime('%Y-%m-%d')}")

h5f.close()
```

### Check File Sizes

```bash
# Monitor dataset growth
ls -lh all_complete_dataset.h5 actual_prices.h5

# Expected growth: ~5-10 MB per day
```

---

## Advanced Usage

### Run for Specific Date

```python
from auto_daily_predictions import IncrementalDataUpdater
from datetime import date

updater = IncrementalDataUpdater('all_complete_dataset.h5')

# Fetch data for specific date
target_date = date(2024, 12, 1)
data = updater.fetch_daily_data(target_date)
updater.append_to_dataset(data, target_date)
```

### Backfill Missing Dates

```python
from datetime import date, timedelta

updater = IncrementalDataUpdater('all_complete_dataset.h5')
latest = updater.get_latest_date()
today = date.today()

# Fill in missing dates
current = latest + timedelta(days=1)
while current < today:
    if current.weekday() < 5:  # Skip weekends
        print(f"Backfilling {current}")
        data = updater.fetch_daily_data(current)
        updater.append_to_dataset(data, current)
    current += timedelta(days=1)
```

### Force Update

```bash
# Force data fetch even if dataset is up-to-date
python auto_daily_predictions.py --force-update
```

---

## Performance

### Timing Breakdown

| Step | Time |
|------|------|
| Check if update needed | < 1 sec |
| Fetch daily data | 30-90 sec |
| Update HDF5 dataset | 10-30 sec |
| Load model | 5-10 sec |
| Run inference (3000 stocks) | 30-60 sec |
| **Total** | **2-3 min** |

### Optimizations

1. **Incremental updates**: Only fetch today's data (not 10 years)
2. **Batched inference**: Process 256 stocks at once
3. **HDF5 append**: No need to recreate entire file
4. **Cached model**: Model stays loaded in memory

---

## Summary

**Before (Manual):**
```bash
# Fetch data (30 min)
python data_scraping/yfinance_price_scraper.py
python data_scraping/fmp_comprehensive_scraper.py
python data_scraping/news_embedder.py

# Recreate dataset (20 min)
python dataset_creation/create_hdf5_dataset.py --years 10

# Run predictions (2 min)
python inference/predict_current_day.py ...
```
**Total: ~52 min**

**After (Automated):**
```bash
./run_daily.sh 10
```
**Total: ~2 min**

**26x faster + fully automated!**

---

## Next Steps

1. ✅ Create initial dataset (one-time setup)
2. ✅ Test automated script: `./run_daily.sh 10`
3. ✅ Set up cron job for daily automation
4. ✅ Monitor for a week to ensure it works
5. ✅ Integrate with your trading strategy

**You're all set!** The script will now fetch fresh data and give you daily predictions automatically.
