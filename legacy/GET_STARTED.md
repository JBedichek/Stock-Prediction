# Get Started - Daily Stock Predictions

## What Was Created

You now have a complete automated stock prediction system with **3 approaches**:

### 🚀 Approach 1: Fully Automated (RECOMMENDED)
**File**: `./run_daily.sh`

Automatically fetches fresh data, updates dataset, runs predictions.

```bash
./run_daily.sh 10
```

**Best for**: Daily automated predictions with fresh data
**Time**: 2-3 minutes
**See**: [AUTOMATED_DAILY_README.md](AUTOMATED_DAILY_README.md)

---

### ⚡ Approach 2: Quick Predictions
**File**: `./get_daily_stocks.sh`

Uses existing dataset (no data fetching).

```bash
./get_daily_stocks.sh 10
```

**Best for**: When your dataset is already up-to-date
**Time**: 1-2 minutes
**See**: [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md)

---

### 🔧 Approach 3: Manual/Advanced
**File**: `inference/predict_current_day.py`

Full control over all parameters.

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10
```

**Best for**: Custom strategies, testing, debugging
**See**: [inference/PREDICT_README.md](inference/PREDICT_README.md)

---

## Quick Decision Guide

**What should I use?**

```
┌─────────────────────────────────────────┐
│ Do you want fully automated daily       │
│ predictions with fresh data?            │
└────────┬────────────────────────────────┘
         │
    YES  │  NO
         │
    ┌────▼────┐           ┌───────────────┐
    │         │           │ Is your       │
    │  Use    │           │ dataset       │
    │run_daily│           │ current?      │
    │  .sh    │           └───┬───────────┘
    │         │               │
    └─────────┘          YES  │  NO
                              │
                      ┌───────▼──────┐     ┌──────────┐
                      │              │     │          │
                      │ Use          │     │ Run data │
                      │get_daily     │     │ scripts  │
                      │_stocks.sh    │     │ first    │
                      │              │     │          │
                      └──────────────┘     └──────────┘
```

**My recommendation**: Start with `./run_daily.sh` - it handles everything!

---

## First Run Checklist

### ✅ Prerequisites

1. **Dataset exists** (10 years of historical data)
   ```bash
   ls -lh all_complete_dataset.h5
   # Should be 5-15 GB
   ```

2. **Model trained**
   ```bash
   ls -lh checkpoints/best_model.pt
   # Should exist
   ```

3. **Bin edges computed**
   ```bash
   ls -lh adaptive_bin_edges.pt
   # Should exist
   ```

4. **Prices file** (optional but recommended)
   ```bash
   ls -lh actual_prices.h5
   # Should exist
   ```

### ✅ If Missing Dataset

Create initial dataset (one-time, 30-60 min):

```bash
python dataset_creation/create_hdf5_dataset.py \
    --bulk-prices data/bulk_prices.pickle \
    --fundamentals data/fundamentals.pickle \
    --news-embeddings data/news_embeddings.pickle \
    --output all_complete_dataset.h5 \
    --years 10
```

### ✅ Verify Dataset

```python
import h5py

h5f = h5py.File('all_complete_dataset.h5', 'r')
ticker = list(h5f.keys())[0]
dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]

print(f"✓ Dataset range: {dates[0]} to {dates[-1]}")
print(f"✓ Total dates: {len(dates)} (need >= 2000)")
print(f"✓ Years: {len(dates)/252:.1f}")

if len(dates) >= 2000:
    print("\n✅ Dataset is ready!")
else:
    print("\n⚠️  Need more historical data")

h5f.close()
```

---

## Run Your First Prediction

### Option 1: Automated (Recommended)

```bash
./run_daily.sh 10
```

### Option 2: Quick

```bash
./get_daily_stocks.sh 10
```

### What You'll See

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

---

## Automation Setup

### Daily Cron Job

Run automatically every weekday at 5 PM:

```bash
crontab -e
```

Add:
```
0 17 * * 1-5 cd /home/james/Desktop/Stock-Prediction && ./run_daily.sh 10 >> logs/daily.log 2>&1
```

Create logs directory:
```bash
mkdir -p logs
```

### Check It's Working

```bash
# Next day, check the log
tail -100 logs/daily.log

# Should see output from yesterday's run
```

---

## Understanding the Output

### Metrics Explained

- **Expected Return**: Model's prediction for % gain/loss
  - Example: `+3.45%` means model predicts 3.45% gain in 5 days

- **Confidence**: How confident the model is (0-1)
  - Higher = more reliable
  - Example: `0.8234` is high confidence

- **Current Price**: Latest price from dataset
  - Used to calculate position sizes

### Files Created

Each run creates:

1. **predictions_YYYYMMDD_HHMMSS.pt**
   - Full predictions (torch format)
   - Loadable with `torch.load()`
   - Contains all stocks, not just top-k

2. **daily_predictions_YYYYMMDD_HHMMSS.txt**
   - Simple text summary
   - Top-k stocks listed
   - Easy to read/email

### Loading Results

```python
import torch

results = torch.load('predictions_20241210_143022.pt')

# Get recommendations
for rec in results['recommendations']:
    print(f"{rec['ticker']}: {rec['expected_return_pct']:+.2f}%")

# Check date and parameters
print(f"\nDate: {results['date']}")
print(f"Horizon: {results['horizon_days']} days")
```

---

## Common Questions

### Q: How often should I run this?

**A**: Once per trading day, after market close (4 PM ET)

The automated script checks if it's a trading day and if market is closed.

### Q: What if my dataset is old?

**A**: Use `./run_daily.sh` - it automatically fetches fresh data and updates the dataset.

### Q: Can I change the holding period?

**A**: Yes, edit the `--horizon-idx` parameter:
- `0` = 1 day
- `1` = 5 days (default)
- `2` = 10 days
- `3` = 20 days

### Q: How many stocks should I buy?

**A**: Start with `--top-k 10`. Adjust based on your capital and risk tolerance.

More stocks = more diversification, less risk

### Q: What if data fetching fails?

**A**: Script falls back to existing dataset. You can also use `--skip-update` flag.

### Q: Is this safe to use with real money?

**A**: **No!** This is for research and education only. Always:
- Paper trade first
- Do your own research
- Consult a financial advisor
- Never invest more than you can afford to lose

---

## Files Reference

### Main Scripts

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_daily.sh` | Automated pipeline | Daily automated predictions |
| `get_daily_stocks.sh` | Quick predictions | When dataset is current |
| `auto_daily_predictions.py` | Full automation | Advanced usage |
| `predict_current_day.py` | Manual predictions | Custom parameters |

### Documentation

| File | Content |
|------|---------|
| `GET_STARTED.md` | This file - overview |
| `QUICK_START.md` | Fastest way to get started |
| `AUTOMATED_DAILY_README.md` | Full automation guide |
| `DAILY_PREDICTIONS.md` | Complete prediction docs |
| `inference/PREDICT_README.md` | predict_current_day.py docs |
| `inference/BACKTEST_README.md` | Backtesting guide |

### Data Files

| File | Size | Description |
|------|------|-------------|
| `all_complete_dataset.h5` | 5-15 GB | Main dataset (10 years) |
| `actual_prices.h5` | 100-500 MB | Price data |
| `checkpoints/best_model.pt` | 500 MB | Trained model |
| `adaptive_bin_edges.pt` | < 1 MB | Classification bins |

---

## Workflow Comparison

### Manual Workflow (Old)

```bash
# Step 1: Fetch data (30 min)
python data_scraping/yfinance_price_scraper.py
python data_scraping/fmp_comprehensive_scraper.py
python data_scraping/news_embedder.py

# Step 2: Create dataset (30 min)
python dataset_creation/create_hdf5_dataset.py --years 10

# Step 3: Run predictions (2 min)
python inference/predict_current_day.py ...
```
**Total: ~62 minutes**

### Automated Workflow (New)

```bash
./run_daily.sh 10
```
**Total: ~2 minutes**

**31x faster!**

---

## Next Steps

1. ✅ **Verify prerequisites** (dataset, model, bin edges)
2. ✅ **Run first prediction**: `./run_daily.sh 10`
3. ✅ **Review results** - understand the metrics
4. ✅ **Set up automation** - cron job for daily runs
5. ✅ **Monitor for a week** - check logs, verify it works
6. ✅ **Backtest your strategy** - see historical performance
7. ✅ **Paper trade** - track predictions vs actual returns

### Recommended Reading Order

1. [QUICK_START.md](QUICK_START.md) - 3 steps to first prediction
2. [AUTOMATED_DAILY_README.md](AUTOMATED_DAILY_README.md) - How automation works
3. [inference/BACKTEST_README.md](inference/BACKTEST_README.md) - Test your strategy

---

## Support

### Documentation

- 📖 This file: Overview and quick start
- 📘 [AUTOMATED_DAILY_README.md](AUTOMATED_DAILY_README.md): Full automation guide
- 📗 [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md): Prediction details
- 📙 [QUICK_START.md](QUICK_START.md): Fastest start

### Troubleshooting

See the troubleshooting sections in:
- [AUTOMATED_DAILY_README.md](AUTOMATED_DAILY_README.md#troubleshooting)
- [DAILY_PREDICTIONS.md](DAILY_PREDICTIONS.md#troubleshooting)

### Common Issues

1. **Dataset not found** → Create initial dataset
2. **Data fetch failed** → Use `--skip-update` flag
3. **CUDA out of memory** → Reduce batch size or use CPU
4. **Weekend** → Script automatically skips (no trading)

---

## Summary

You now have **3 ways** to get daily stock predictions:

1. **🚀 Fully Automated**: `./run_daily.sh 10`
   - Fetches fresh data
   - Updates dataset incrementally
   - Runs predictions
   - **Recommended for daily use**

2. **⚡ Quick**: `./get_daily_stocks.sh 10`
   - Uses existing dataset
   - Fast predictions
   - Good when dataset is current

3. **🔧 Manual**: `python inference/predict_current_day.py ...`
   - Full control
   - Custom parameters
   - Advanced usage

**Start with**: `./run_daily.sh 10`

**Then**: Set up daily cron job and forget about it!

---

## Disclaimer

This tool is for **research and educational purposes only**.

- Not financial advice
- No guarantee of profits
- Past performance ≠ future results
- Always do your own research
- Consult a financial advisor
- Never risk more than you can afford to lose

**Use at your own risk.**

---

## You're All Set! 🎉

Everything is configured and ready to go.

**Your first command**:
```bash
./run_daily.sh 10
```

**Questions?** Check the docs in the links above.

**Happy trading!** 📈
