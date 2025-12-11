# Quick Start - Get Stock Predictions in 3 Steps

## Step 1: Verify You Have the Required Files ✅

Run this check:

```bash
ls -lh all_complete_dataset.h5 actual_prices.h5 checkpoints/best_model.pt adaptive_bin_edges.pt
```

**What you need:**
- ✅ `all_complete_dataset.h5` - HDF5 dataset with 10 years of stock data
- ✅ `actual_prices.h5` - Actual price data for accurate returns
- ✅ `checkpoints/best_model.pt` - Your trained model
- ✅ `adaptive_bin_edges.pt` - Bin edges for classification

**If files are missing:** See [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md) for setup instructions.

---

## Step 2: Run Your First Prediction 🚀

```bash
./get_daily_stocks.sh 10
```

This will:
1. Load your dataset (finds the most recent date available)
2. Load your trained model
3. Run inference on all stocks
4. Output top 10 stocks to buy

**Expected runtime:** 1-2 minutes

---

## Step 3: Review Results 📊

You'll see output like:

```
================================================================================
📈 TOP 10 STOCK RECOMMENDATIONS FOR 2024-12-10
================================================================================

Rank   Ticker   Expected Return    Confidence   Current Price
--------------------------------------------------------------------------------
1      AAPL      +3.45%            0.8234       $   175.23
2      MSFT      +2.87%            0.8156       $   370.45
3      GOOGL     +2.65%            0.7923       $   140.12
...

================================================================================
💡 INVESTMENT STRATEGY
================================================================================
  • Buy these 10 stocks today
  • Hold for 5 trading days
  • Split capital equally among selected stocks
  • Rebalance after holding period
```

Results are also saved to `predictions_YYYYMMDD.pt`

---

## What's Next?

### Daily Usage

Run this every trading day:
```bash
./get_daily_stocks.sh 10
```

### Different Strategies

```bash
# Conservative (top 20 stocks, 20-day hold)
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 20 \
    --horizon-idx 3

# Aggressive (top 5 stocks, 1-day hold)
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 5 \
    --horizon-idx 0
```

### Automation

Set up daily predictions at 9 AM:
```bash
crontab -e
```

Add:
```
0 9 * * 1-5 cd /home/james/Desktop/Stock-Prediction && ./get_daily_stocks.sh 10 >> predictions.log 2>&1
```

---

## Understanding Your Dataset

Check how much historical data you have:

```python
import h5py

h5f = h5py.File('all_complete_dataset.h5', 'r')
ticker = list(h5f.keys())[0]
dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]

print(f"First date: {dates[0]}")
print(f"Last date: {dates[-1]}")
print(f"Total dates: {len(dates)}")
print(f"Years of data: {len(dates) / 252:.1f}")

h5f.close()
```

**Required:** At least 2,520 dates (10 years × 252 trading days)

Your model needs `seq_len=2000` historical data points to make predictions, so 10 years ensures you have enough context.

---

## Fresh Data Option

If you want to create a NEW dataset with the most recent data:

```bash
# 1. Update your raw data sources
python data_scraping/yfinance_price_scraper.py --years 10
python data_scraping/fmp_comprehensive_scraper.py --years 10
python data_scraping/news_embedder.py

# 2. Create new dataset
python dataset_creation/create_hdf5_dataset.py \
    --bulk-prices data/bulk_prices.pickle \
    --fundamentals data/fundamentals.pickle \
    --news-embeddings data/news_embeddings.pickle \
    --output all_complete_dataset_new.h5 \
    --years 10

# 3. Run predictions
./get_daily_stocks.sh 10
```

---

## Troubleshooting

### Error: "Dataset not found"

```bash
# Check if file exists
ls -lh all_complete_dataset.h5

# If not, you need to create it
# See DAILY_PREDICTIONS.md for instructions
```

### Error: "No valid data for today"

Your dataset doesn't include today's date. The script will automatically use the most recent available date.

To update:
```bash
# Option 1: Use existing dataset (it will use latest available date)
./get_daily_stocks.sh 10

# Option 2: Update dataset with fresh data
# Run your data collection and dataset creation scripts
```

### Error: "CUDA out of memory"

```bash
# Reduce batch size
python inference/predict_current_day.py --batch-size 64 --device cuda ...

# Or use CPU (slower but less memory)
python inference/predict_current_day.py --device cpu ...
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `get_daily_stocks.sh` | Simple script - get predictions fast |
| `predict_current_day.py` | Python script for predictions |
| `daily_prediction_pipeline.py` | Full pipeline with fresh data creation |
| `DAILY_PREDICTIONS.md` | Complete documentation |
| `QUICK_START.md` | This file |

---

## Key Parameters

| Parameter | What It Does | Default |
|-----------|-------------|---------|
| Top-K | Number of stocks to buy | 10 |
| Horizon | How long to hold | 5 days |
| Confidence | How selective | Keep top 40% |

Modify in `get_daily_stocks.sh` or pass to `predict_current_day.py`

---

## Summary

1. ✅ Check you have required files
2. 🚀 Run `./get_daily_stocks.sh 10`
3. 📊 Review recommendations
4. 💰 Invest (at your own risk!)

**That's it!** You now have a one-command solution to get daily stock predictions.

For more details, see [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md)

---

## Support

- 📖 Full docs: [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md)
- 🔙 Backtesting: [inference/BACKTEST_README.md](inference/BACKTEST_README.md)
- 🎯 Predictions: [inference/PREDICT_README.md](inference/PREDICT_README.md)
