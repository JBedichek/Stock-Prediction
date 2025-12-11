# Current Day Stock Prediction

Get a list of stocks to buy based on model predictions for the current day.

## Quick Start

### Method 1: Simple Script (Recommended)

```bash
# Predict top 10 stocks to buy
./predict_stocks.sh

# Predict top 5 stocks to buy
./predict_stocks.sh 5

# Predict top 20 stocks to buy
./predict_stocks.sh 20
```

### Method 2: Direct Python Call

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10 \
    --horizon-idx 1
```

## How It Works

The script:

1. **Loads your existing dataset** (`all_complete_dataset.h5`)
   - This should contain at least 10 years of historical data
   - The model requires 2000 historical data points (seq_len=2000)
   - 10 years ≈ 2520 trading days, which is sufficient

2. **Finds the most recent date** in the dataset
   - Uses the latest date available in your dataset
   - Or you can specify a date with `--date YYYY-MM-DD`

3. **Runs inference** on all stocks for that date
   - Uses batched inference for speed (default batch_size=256)
   - Applies the same model and parameters as backtest_simulation.py

4. **Filters and ranks** stocks
   - Filters by confidence percentile (default: keep top 40%)
   - Ranks by confidence-weighted expected return
   - Returns top-k stocks

5. **Outputs recommendations**
   - Prints a ranked list to console
   - Optionally saves to file with `--output`

## Output Format

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
3      GOOGL     +2.65%            0.7923       $   140.12
4      NVDA      +2.34%            0.7845       $   495.67
5      TSLA      +2.12%            0.7734       $   242.89
6      META      +1.98%            0.7689       $   325.45
7      AMZN      +1.87%            0.7623       $   151.23
8      AMD       +1.76%            0.7567       $    98.45
9      NFLX      +1.65%            0.7512       $   456.78
10     CRM       +1.54%            0.7489       $   234.56

================================================================================
💡 INVESTMENT STRATEGY
================================================================================
  • Buy these 10 stocks today
  • Hold for 5 trading days
  • Split capital equally among selected stocks
  • Rebalance after holding period
```

### Saved File (--output predictions.pt)

```python
import torch

results = torch.load('predictions_20241210_143022.pt')

# Access recommendations
for rec in results['recommendations']:
    print(f"{rec['ticker']}: {rec['expected_return_pct']:.2f}% (confidence: {rec['confidence']:.4f})")

# Structure:
{
    'date': '2024-12-10',
    'horizon_days': 5,
    'top_k': 10,
    'recommendations': [
        {
            'rank': 1,
            'ticker': 'AAPL',
            'expected_return': 1.0345,
            'expected_return_pct': 3.45,
            'confidence': 0.8234,
            'current_price': 175.23
        },
        ...
    ],
    'all_predictions': [...]  # All stocks, not just top-k
}
```

## Arguments

### Required Arguments

- `--data`: Path to HDF5 dataset with features (e.g., `all_complete_dataset.h5`)
- `--model`: Path to model checkpoint (e.g., `checkpoints/best_model.pt`)
- `--bin-edges`: Path to bin edges file (e.g., `adaptive_bin_edges.pt`)

### Optional Arguments

#### Strategy Parameters (same as backtest_simulation.py)

- `--top-k`: Number of stocks to recommend (default: 10)
- `--horizon-idx`: Prediction horizon (default: 1)
  - 0 = 1 trading day
  - 1 = 5 trading days
  - 2 = 10 trading days
  - 3 = 20 trading days
- `--confidence-percentile`: Confidence filter (default: 0.6 = keep top 40%)
  - Lower value = more aggressive filtering
  - Higher value = less filtering

#### Other Parameters

- `--prices`: Path to actual prices HDF5 file (optional)
- `--date`: Specific date to predict for (default: latest available)
- `--batch-size`: Inference batch size (default: 256)
- `--device`: Device to use (default: cuda)
- `--compile`: Use torch.compile for faster inference
- `--output`: Save predictions to file

## Examples

### Conservative Long-Term Strategy

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

- Top 20 stocks
- 20-day holding period
- Keep top 50% by confidence (more selective)

### Aggressive Short-Term Strategy

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

- Top 5 stocks only
- 1-day holding period (day trading)
- Keep top 30% by confidence (less selective)

### Save Predictions for Later

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --top-k 10 \
    --output predictions_$(date +%Y%m%d).pt
```

### Predict for Specific Date

```bash
python inference/predict_current_day.py \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --date 2024-12-01
```

## Dataset Requirements

Your dataset (`all_complete_dataset.h5`) must contain:

1. **Sufficient historical data**: At least 2000 trading days (≈8 years)
   - The model uses seq_len=2000 for predictions
   - 10 years of data (≈2520 trading days) provides a good buffer

2. **Recent data**: Up to the date you want to predict for
   - The script uses the latest available date by default
   - Check your dataset's date range: `h5ls all_complete_dataset.h5/AAPL/dates`

3. **Feature format**: HDF5 format with structure:
   ```
   ticker/
     features: (num_dates, num_features)
     dates: (num_dates,)
   ```

4. **Optional prices file**: For accurate current prices
   - `actual_prices.h5` with same date range
   - If not provided, uses features[0] as price

## Checking Your Dataset

```python
import h5py

# Open dataset
h5f = h5py.File('all_complete_dataset.h5', 'r')

# Check a sample ticker
ticker = list(h5f.keys())[0]
dates = [d.decode('utf-8') for d in h5f[ticker]['dates'][:]]

print(f"Sample ticker: {ticker}")
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Total dates: {len(dates)}")
print(f"Feature shape: {h5f[ticker]['features'].shape}")

h5f.close()
```

Expected output:
```
Sample ticker: A
Date range: 2014-12-10 to 2024-12-10
Total dates: 2520
Feature shape: (2520, 1400)
```

## Creating a New Dataset (if needed)

If your dataset doesn't have recent data, you'll need to recreate it with updated data sources.

See:
- `dataset_creation/CREATE_INFERENCE_DATASET.md` for old format
- Contact the maintainer for HDF5 dataset creation scripts

## Troubleshooting

### "No valid data for date YYYY-MM-DD"

- Your dataset doesn't contain data for that date
- Check available dates: use `--date` with a date that exists in your dataset
- Or update your dataset with recent data

### "Insufficient historical data"

- Your dataset has fewer than 2000 trading days
- The model cannot make predictions without enough context
- You need to recreate your dataset with more historical data

### "Model checkpoint not found"

- Check that `checkpoints/best_model.pt` exists
- Or specify the correct path with `--model`

### "Bin edges file not found"

- For classification models, bin edges are required
- Check that `adaptive_bin_edges.pt` exists
- Or compute them: see `training/train_new_format.py`

### "CUDA out of memory"

- Reduce batch size: `--batch-size 128` or `--batch-size 64`
- Or use CPU: `--device cpu` (slower but uses less memory)

## Integration with Trading

### Automated Daily Predictions

Create a cron job to run predictions daily:

```bash
# Add to crontab (crontab -e)
# Run every weekday at 9:00 AM
0 9 * * 1-5 cd /home/james/Desktop/Stock-Prediction && ./predict_stocks.sh 10 >> predictions.log 2>&1
```

### Programmatic Access

```python
import torch
from datetime import datetime

# Load predictions
results = torch.load('predictions_20241210.pt')

# Get top stocks
top_stocks = [rec['ticker'] for rec in results['recommendations']]
print(f"Buy: {', '.join(top_stocks)}")

# Calculate position sizes (equal weight)
capital = 100000  # $100k
position_size = capital / len(top_stocks)

for rec in results['recommendations']:
    shares = position_size / rec['current_price']
    print(f"{rec['ticker']}: Buy {shares:.0f} shares @ ${rec['current_price']:.2f}")
```

### Paper Trading

Test predictions without real money:

```bash
# Day 1: Get predictions
./predict_stocks.sh 10 > predictions_day1.txt

# Day 6: Check actual returns
python inference/backtest_simulation.py \
    --data all_complete_dataset.h5 \
    --model checkpoints/best_model.pt \
    --test-months 1 \
    --quiet
```

## Comparison with Backtest

The prediction script uses **identical logic** to backtest_simulation.py:

| Component | Backtest | Prediction |
|-----------|----------|------------|
| Model loading | ✅ Same | ✅ Same |
| Feature loading | ✅ Same | ✅ Same |
| Inference | ✅ Same | ✅ Same |
| Confidence filtering | ✅ Same | ✅ Same |
| Ranking | ✅ Same | ✅ Same |
| Parameters | ✅ Same | ✅ Same |

The only difference:
- **Backtest**: Simulates trades over historical period
- **Prediction**: Makes predictions for a single day (today)

## Next Steps

After getting your stock recommendations:

1. **Review the list** - Check for any obvious issues
2. **Verify prices** - Ensure current prices are accurate
3. **Check confidence** - Higher confidence = more reliable predictions
4. **Consider diversification** - Don't put all eggs in one basket
5. **Set stop losses** - Protect against unexpected moves
6. **Monitor positions** - Track performance over holding period
7. **Rebalance** - Run predictions again after holding period ends

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consult with a financial advisor before making investment decisions.
