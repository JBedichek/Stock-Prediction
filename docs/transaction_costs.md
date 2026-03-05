# Transaction Cost Modeling in Walk-Forward Training

This document explains how transaction costs and portfolio turnover are modeled in `training/walk_forward_training.py`.

## Overview

Real-world trading incurs costs that erode returns. The walk-forward training system models these costs to provide realistic performance estimates:

- **Turnover tracking**: Monitors how much of the portfolio changes each period
- **Proportional costs**: Applies costs based on the fraction of positions changed
- **Gross vs Net metrics**: Reports both pre-cost and post-cost performance

## Key Concepts

### Basis Points (bps)

Transaction costs are specified in **basis points**:
- 1 basis point = 0.01% = 0.0001
- 10 bps = 0.1%
- 100 bps = 1%

### Turnover

**Turnover** measures what fraction of the portfolio changes between periods:

```
turnover = (positions_exited + positions_entered) / (2 × portfolio_size)
```

- **Turnover = 0**: No changes, all positions held
- **Turnover = 0.5**: Half the portfolio changed
- **Turnover = 1.0**: Complete portfolio replacement

### Round-Trip Cost

The cost model assumes **round-trip costs** - each changed position incurs both a sell (exit old) and buy (enter new) cost. The `transaction_cost_bps` parameter represents this total round-trip cost.

## Implementation Details

### Cost Calculation

From `run_principled_evaluation()` (line 547):

```python
# Convert bps to decimal
cost_rate = transaction_cost_bps / 10000.0  # e.g., 10 bps -> 0.001

# For each trading period:
transaction_cost = model_turnover * cost_rate
net_model_return = gross_model_return - transaction_cost
```

### Turnover Calculation

From line 717-727:

```python
# Calculate turnover (fraction of portfolio that changed)
if len(prev_model_tickers) > 0:
    # symmetric_difference gives positions that changed (exited OR entered)
    positions_changed = len(prev_model_tickers.symmetric_difference(current_model_tickers))
    model_turnover = positions_changed / (2 * top_k)
else:
    # First period: assume we're starting fresh, full buy cost
    model_turnover = 1.0
```

**Example**: With `top_k=25` stocks:
- 5 stocks exit, 5 new stocks enter
- `positions_changed = 10`
- `turnover = 10 / (2 × 25) = 0.2` (20% turnover)

### Metrics Reported

The evaluation returns both gross and net metrics:

| Metric | Description |
|--------|-------------|
| `model_mean_return` | Average daily return (gross) |
| `model_mean_return_net` | Average daily return (net of costs) |
| `total_return_pct` | Cumulative return (gross) |
| `total_return_pct_net` | Cumulative return (net) |
| `sharpe_ratio` | Sharpe ratio (gross) |
| `sharpe_ratio_net` | Sharpe ratio (net) |
| `mean_turnover` | Average daily turnover |
| `total_turnover` | Sum of all daily turnovers |
| `total_transaction_costs_pct` | Total costs paid as % of initial capital |

## Configuration

### Command Line

```bash
python training/walk_forward_training.py \
    --transaction-cost-bps 10 \   # 10 bps = 0.1% round-trip cost
    --top-k 25                     # Portfolio size (affects turnover)
```

### Typical Cost Values

| Market Segment | Typical Cost (bps) | Description |
|----------------|-------------------|-------------|
| Large-cap liquid | 5-20 | S&P 500 stocks, high volume |
| Mid-cap | 20-50 | Russell 2000, moderate liquidity |
| Small-cap | 50-100+ | Low volume, wider spreads |
| Emerging markets | 50-150 | Additional FX and settlement costs |

### Setting to Zero

To disable transaction cost modeling:

```bash
--transaction-cost-bps 0
```

When set to 0, the system still tracks turnover but reports gross = net metrics.

## Output Examples

### Console Output

```
📈 Returns (Net of 10 bps costs):
   Model:    +0.082% per trade
   vs Random:+0.045%
   Turnover: 35.2% avg per period

💰 Backtest Simulation (258 days):
   Total:    +22.45% (net)
   Costs:    3.21% total paid
   Sharpe:   1.24 (gross) / 1.08 (net)
```

### JSON Results

```json
{
  "model_mean_return": 0.085,
  "model_mean_return_net": 0.082,
  "total_return_pct": 25.66,
  "total_return_pct_net": 22.45,
  "sharpe_ratio": 1.24,
  "sharpe_ratio_net": 1.08,
  "mean_turnover": 0.352,
  "total_turnover": 90.82,
  "total_transaction_costs_pct": 3.21,
  "transaction_cost_bps": 10.0
}
```

## How Turnover Affects Strategy Viability

### Break-Even Analysis

A strategy is viable if:
```
gross_return > turnover × cost_rate
```

**Example**:
- Gross return: 0.05% per day
- Turnover: 40% per day
- Cost rate: 10 bps = 0.1%

Net return = 0.05% - (0.4 × 0.1%) = 0.05% - 0.04% = **0.01%**

The strategy barely survives costs!

### Turnover vs Alpha Trade-off

Higher turnover strategies need proportionally higher gross alpha:

| Turnover | 10 bps Cost | Required Gross Alpha |
|----------|-------------|---------------------|
| 20% | 0.02% | Minimal |
| 50% | 0.05% | Moderate |
| 100% | 0.10% | Significant |
| 200% | 0.20% | Very high |

## Comparison with Baselines

The system applies the same turnover-based cost model to baselines:

### Model Strategy
- Tracks actual position changes
- Applies costs based on measured turnover

### Momentum Baseline
- Tracks momentum portfolio changes separately
- Often has lower turnover (momentum is persistent)

### Random Baseline
- Assumes 100% turnover (complete random reselection)
- Represents worst-case turnover scenario

This ensures fair comparison: if the model has lower turnover than random, that's an additional source of alpha.

## Implementation Notes

### First Period Handling

The first trading period assumes 100% turnover (full portfolio purchase):

```python
if len(prev_model_tickers) > 0:
    # Normal turnover calculation
    ...
else:
    # First period: assume we're starting fresh
    model_turnover = 1.0
```

### Daily vs Per-Trade Costs

The current implementation applies costs **per evaluation period** (typically daily). For intraday strategies, costs would need adjustment.

### Slippage Not Modeled

The current model captures:
- Commission/fees (explicit costs)
- Bid-ask spread (implicit costs)

Not modeled:
- Market impact (price moves against large orders)
- Slippage (execution price vs. signal price)

For large portfolios, consider increasing `transaction_cost_bps` to account for market impact.

## Code References

- **Cost rate conversion**: `walk_forward_training.py:547`
- **Turnover calculation**: `walk_forward_training.py:717-727`
- **Net return calculation**: `walk_forward_training.py:733-737`
- **Results aggregation**: `walk_forward_training.py:919-930`
- **Command line argument**: `walk_forward_training.py:4011-4013`
- **WalkForwardTrainer init**: `walk_forward_training.py:1751, 1834`

## Best Practices

1. **Start conservative**: Use higher cost estimates (20-50 bps) for initial analysis
2. **Compare gross vs net**: Large gaps indicate turnover-sensitive strategies
3. **Monitor turnover trends**: Increasing turnover over time may indicate overfitting
4. **Consider holding periods**: Lower frequency rebalancing reduces turnover
5. **Validate with real costs**: Backtest against actual brokerage fee schedules
