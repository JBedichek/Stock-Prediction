# Stock Prediction Backtesting Framework

A modular framework for backtesting stock prediction models on historical data.

## Features

- **Modular Design**: Clean separation of concerns with 4 main classes
- **Flexible Stock Selection**: Test on last N stocks alphabetically or random subsets
- **Multiple Strategies**: Configurable top-k selection and holding periods
- **Comprehensive Metrics**: Win rate, Sharpe ratio, max drawdown, and more
- **Easy Extension**: Add new strategies, risk management, or analysis features

## Architecture

### 1. `DatasetLoader`
Loads and manages the test dataset.

**Responsibilities:**
- Load pickle/HDF5 dataset
- Select test stocks (last N alphabetically)
- Optional random subsampling
- Provide features and prices for any ticker/date
- Calculate future prices for trade evaluation

**Key Methods:**
- `get_trading_period(num_months)`: Get last N months of trading days
- `get_features_and_price(ticker, date)`: Get model inputs
- `get_future_price(ticker, date, horizon_days)`: Get future price for evaluation

### 2. `ModelPredictor`
Runs inference and calculates expected returns.

**Responsibilities:**
- Load model checkpoint
- Run inference on feature tensors
- Calculate expected value from bin probabilities (classification mode)
- Handle both classification and regression modes

**Key Methods:**
- `predict_expected_return(features, horizon_idx)`: Get expected price ratio

### 3. `TradingSimulator`
Simulates trading strategy.

**Responsibilities:**
- Select top-k stocks based on predictions
- Simulate buying and selling
- Track capital over time
- Record trade history

**Key Methods:**
- `select_top_stocks(date)`: Rank stocks by predicted return
- `simulate_trade(date, capital)`: Execute one trade
- `run_simulation(trading_dates)`: Run full backtest

**Strategy:**
1. On each trading day, run inference on all test stocks
2. Select top-k stocks with highest predicted returns
3. Split capital equally among selected stocks
4. Hold for `horizon_days` trading days
5. Sell and calculate returns
6. Repeat

### 4. `PerformanceReporter`
Reports and visualizes results.

**Responsibilities:**
- Print summary statistics
- Show recent trades
- Save detailed results

**Metrics:**
- Total return %
- Win rate (% of profitable trades)
- Average return per trade
- Standard deviation of returns
- Sharpe ratio (risk-adjusted return)
- Maximum drawdown

## Usage

### Basic Usage

```bash
python inference/backtest_simulation.py \
    --data all_complete_dataset.pkl \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --num-test-stocks 1000 \
    --top-k 10 \
    --horizon-days 5 \
    --horizon-idx 1 \
    --test-months 2
```

### Arguments

**Data:**
- `--data`: Path to dataset (pickle or HDF5)
- `--model`: Path to model checkpoint
- `--bin-edges`: Path to cached bin edges (for classification mode)

**Test Set:**
- `--num-test-stocks`: Number of stocks from end of alphabet (default: 1000)
- `--subset-size`: Randomly sample N stocks from test set (optional)

**Strategy:**
- `--top-k`: Number of stocks to buy each period (default: 10)
- `--horizon-days`: Holding period in trading days (default: 5)
- `--horizon-idx`: Prediction horizon (0=1day, 1=5day, 2=10day, 3=20day)
- `--test-months`: Number of months to backtest (default: 2)
- `--initial-capital`: Starting capital (default: $100,000)

**Other:**
- `--device`: Device to run on (default: cuda)
- `--output`: Path to save results (default: backtest_results.pt)
- `--seed`: Random seed (default: 42)

### Example: Quick Test on 100 Stocks

```bash
python inference/backtest_simulation.py \
    --data all_complete_dataset.pkl \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --subset-size 100 \
    --top-k 5 \
    --horizon-days 5 \
    --horizon-idx 1
```

### Example: Day Trading Strategy

```bash
python inference/backtest_simulation.py \
    --data all_complete_dataset.pkl \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --subset-size 100 \
    --top-k 3 \
    --horizon-days 1 \
    --horizon-idx 0 \
    --initial-capital 50000
```

### Example: Long-term Strategy

```bash
python inference/backtest_simulation.py \
    --data all_complete_dataset.pkl \
    --model checkpoints/best_model.pt \
    --bin-edges adaptive_bin_edges.pt \
    --subset-size 100 \
    --top-k 5 \
    --horizon-days 20 \
    --horizon-idx 3
```

### Run All Examples

```bash
python inference/run_backtest_examples.py
```

## Output

### Console Output

```
================================================================================
LOADING DATASET
================================================================================
📦 Loading dataset from: all_complete_dataset.pkl
  ✅ Loaded 5000 tickers
  📊 Selected last 1000 tickers (alphabetically)
  📈 Test set: 1000 stocks
  Range: TSLA to ZZZZZ
  📅 Date range: 2020-01-01 to 2024-12-01
  Total dates: 1200

================================================================================
LOADING MODEL
================================================================================
📦 Loading checkpoint: checkpoints/best_model.pt
🏗️  Building model...
  ✅ Model loaded (epoch 50, val_loss: 0.234567)
📊 Loading bin edges: adaptive_bin_edges.pt
  ✅ Loaded 101 bin edges

================================================================================
RUNNING SIMULATION
================================================================================
Strategy: Buy top-10 stocks, hold for 5 days
Initial capital: $100,000.00

Simulating 40 trades...
Trading: 100%|████████████████| 40/40 [00:30<00:00,  1.33it/s]

================================================================================
SIMULATION RESULTS
================================================================================

💰 Capital Performance:
  Initial capital:  $  100,000.00
  Final capital:    $  115,234.56
  Total return:          15.23%

📊 Trading Statistics:
  Number of trades:           40
  Win rate:                62.50%
  Avg return:               0.38%
  Std deviation:            2.15%
  Sharpe ratio:             2.89
  Max drawdown:             8.45%
```

### Saved Results

Results are saved to a PyTorch file containing:

```python
{
    'initial_capital': 100000.0,
    'final_capital': 115234.56,
    'total_return_pct': 15.23,
    'num_trades': 40,
    'win_rate': 62.5,
    'avg_return_pct': 0.38,
    'std_return_pct': 2.15,
    'sharpe_ratio': 2.89,
    'max_drawdown_pct': 8.45,
    'capital_history': [100000.0, 101234.5, ...],  # Capital after each trade
    'trade_history': [                               # Detailed trade info
        {
            'date': '2024-10-01',
            'num_stocks': 10,
            'capital_invested': 100000.0,
            'capital_returned': 101234.5,
            'return_pct': 1.234,
            'stocks': [
                {
                    'ticker': 'AAPL',
                    'buy_price': 150.0,
                    'sell_price': 152.5,
                    'expected_return': 1.03,  # Model predicted 3% gain
                    'actual_return': 1.017,    # Actually gained 1.7%
                    'profit_pct': 1.67
                },
                ...
            ]
        },
        ...
    ],
    'daily_returns': [1.234, -0.567, ...]  # Return % for each trade
}
```

Load results:
```python
import torch
results = torch.load('backtest_results.pt')
print(f"Total return: {results['total_return_pct']:.2f}%")
```

## Extension Ideas

### 1. Risk Management
Add position sizing based on prediction confidence:

```python
def select_top_stocks_with_sizing(self, date: str):
    predictions = []
    for ticker in self.data_loader.test_tickers:
        features, price = self.data_loader.get_features_and_price(ticker, date)
        expected_return = self.predictor.predict_expected_return(features)
        confidence = self.predictor.get_confidence(features)  # New method
        predictions.append((ticker, expected_return, price, confidence))

    # Weight positions by confidence
    # ...
```

### 2. Stop Loss / Take Profit
Exit positions early based on price movements:

```python
def simulate_trade_with_stops(self, date: str, stop_loss: float = 0.05):
    # Monitor position daily and exit if stop loss hit
    # ...
```

### 3. Multiple Strategies
Compare different strategies:

```python
strategies = [
    {'top_k': 5, 'horizon': 1},   # Aggressive day trading
    {'top_k': 10, 'horizon': 5},  # Moderate swing trading
    {'top_k': 20, 'horizon': 20}  # Conservative long-term
]

for strategy in strategies:
    simulator = TradingSimulator(..., **strategy)
    results = simulator.run_simulation(dates)
```

### 4. Sector Diversification
Ensure stocks are from different sectors:

```python
def select_diversified_stocks(self, date: str):
    # Group predictions by sector
    # Select top from each sector
    # ...
```

### 5. Transaction Costs
Add realistic trading costs:

```python
def simulate_trade(self, date: str, capital: float, commission: float = 0.001):
    # Deduct commission on buy and sell
    buy_cost = capital_per_stock * (1 + commission)
    sell_proceeds = sell_value * (1 - commission)
```

### 6. Visualization
Plot equity curve and metrics:

```python
import matplotlib.pyplot as plt

def plot_equity_curve(results):
    plt.plot(results['capital_history'])
    plt.xlabel('Trade Number')
    plt.ylabel('Capital ($)')
    plt.title('Equity Curve')
    plt.show()
```

## Notes

- **Test Set Isolation**: Uses last 1000 stocks alphabetically (assumed to be excluded from training)
- **No Look-Ahead Bias**: Only uses data available at prediction time
- **Equal Weighting**: Splits capital equally among selected stocks (can be modified)
- **Holding Period**: Fixed holding period (not dynamic exit)
- **No Short Selling**: Only long positions (can be extended)

## Troubleshooting

**Issue**: "No predictions available"
- Check that test stocks have data for the selected dates
- Verify model checkpoint and bin edges are correct

**Issue**: "Future price not available"
- Dataset may not have enough future data
- Reduce `test_months` or `horizon_days`

**Issue**: Slow inference
- Reduce `subset_size` for faster testing
- Use GPU with `--device cuda`
- Enable torch.compile in model loading (TODO)

## Future Enhancements

- [ ] Support for HDF5 datasets
- [ ] Batch inference for speed
- [ ] Parallel backtesting of multiple strategies
- [ ] Visualization of results
- [ ] Comparison with buy-and-hold baseline
- [ ] Rolling window backtesting
- [ ] Walk-forward validation
