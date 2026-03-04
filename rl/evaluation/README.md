# DQN Agent Evaluation Framework

Comprehensive evaluation system for testing trained DQN trading agents on out-of-sample data.

## Overview

This framework provides:

- **Out-of-sample testing**: Evaluate on held-out test data (e.g., 2024)
- **Multiple episodes**: Run 100+ episodes with random stock selections and start dates
- **Risk-adjusted metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown
- **Baseline comparisons**: Compare against random, hold, long-only, and other strategies
- **Statistical significance**: T-tests and effect sizes to validate performance
- **Comprehensive reports**: JSON output with all metrics and statistics

## Quick Start

### 1. Train Your DQN Agent

First, train a DQN agent and save a checkpoint:

```bash
python rl/train_dqn_simple.py
```

This will save checkpoints to `checkpoints/dqn_episode_*.pt`

### 2. Run Evaluation

Evaluate the trained agent on held-out 2024 data:

```bash
python -m rl.evaluation.run_evaluation \
    --checkpoint checkpoints/dqn_best.pt \
    --test-start 2024-01-01 \
    --test-end 2024-12-31 \
    --num-episodes 100 \
    --baselines random hold long
```

### 3. View Results

Results are saved to `rl/evaluation/results/evaluation_report_TIMESTAMP.json`

The console will print a summary like:

```
================================================================================
EVALUATION SUMMARY
================================================================================

DQN AGENT:
  Total Return:    0.0850 ± 0.1240
  Sharpe Ratio:    1.2400 ± 0.4500
  Max Drawdown:    0.0850 ± 0.0320
  Win Rate:        0.6200 ± 0.0850
  Num Trades:      12.5 ± 3.2

BASELINE COMPARISONS:

  vs RANDOM:
    Mean return difference: +0.0920
    p-value: 0.0012
    Significant: YES

  vs HOLD:
    Mean return difference: +0.0850
    p-value: 0.0001
    Significant: YES
```

## Command Line Options

### Required Arguments

- `--checkpoint`: Path to trained DQN checkpoint file

### Test Period

- `--test-start`: Start date for test period (default: `2024-01-01`)
- `--test-end`: End date for test period (default: `2024-12-31`)

### Evaluation Parameters

- `--num-episodes`: Number of evaluation episodes (default: `100`)
  - More episodes = more reliable statistics
  - Minimum recommended: 30 for statistical validity
  - 100+ recommended for publication-quality results

- `--episode-length`: Episode length in trading days (default: `30`)
  - Should match training episode length

### Baselines

- `--baselines`: List of baseline strategies to compare against
  - Available: `random`, `hold`, `long`, `short`, `momentum`, `long_short`
  - Default: `random hold long`

### File Paths

- `--dataset`: Path to dataset (default: `data/train_data_enhanced.h5`)
- `--prices`: Path to prices file (default: `data/prices.h5`)
- `--stock-selections`: Stock selection cache (default: `data/rl_stock_selections_4yr.h5`)
- `--state-cache`: State cache file (default: `data/state_cache.h5`)
- `--predictor-checkpoint`: Predictor checkpoint (default: `checkpoints/contrastive_best.pt`)

### Resources

- `--device`: Device to use (default: `cuda`)
- `--parallel`: Number of parallel environments (default: `1`)
  - Use 1 for evaluation (ensures reproducibility)

### Output

- `--output-dir`: Output directory for results (default: `rl/evaluation/results`)

## Metrics Explained

### Return Metrics

- **Total Return**: `(final_value - initial_value) / initial_value`
  - Simple percentage return over the episode
  - Example: 0.15 = 15% return

### Risk-Adjusted Returns

- **Sharpe Ratio**: `(mean_return - risk_free_rate) / std_return * sqrt(252)`
  - Measures return per unit of risk
  - Higher is better (>1.0 is good, >2.0 is excellent)
  - Industry standard for comparing strategies

- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
  - Better for strategies with asymmetric returns
  - Higher is better

- **Calmar Ratio**: `annualized_return / max_drawdown`
  - Return per unit of maximum drawdown
  - Higher is better

### Risk Metrics

- **Max Drawdown**: Worst peak-to-trough loss
  - Example: 0.15 = 15% maximum drawdown
  - Lower is better (less risk)

- **Volatility**: Standard deviation of returns
  - Lower is better (more stable)

### Trade Statistics

- **Win Rate**: Percentage of profitable trades
  - Example: 0.62 = 62% of trades profitable
  - Random should be ~50%, good strategy >55%

- **Profit Factor**: `sum(winning_trades) / sum(losing_trades)`
  - >1.0 means profitable overall
  - >1.5 is good, >2.0 is excellent

- **Average Trade Duration**: Mean holding period in days
  - Helps understand strategy behavior

## Baseline Strategies

The framework includes several baseline strategies for comparison:

### Random
- Selects actions uniformly at random (0-8)
- Any reasonable agent should beat this

### Hold
- Never trades, stays in cash
- Returns exactly 0% (baseline for doing nothing)

### Long
- Always goes long on the top predicted stock
- Tests if simply buying winners works

### Short
- Always shorts the bottom predicted stock
- Tests if simply shorting losers works

### Momentum
- Simple momentum-based strategy
- Buys stocks with positive momentum

### Long-Short
- Market-neutral strategy
- Alternates between long and short positions

## Statistical Significance

The framework performs t-tests comparing DQN returns to baseline returns:

- **p-value < 0.05**: DQN is significantly different from baseline
- **Cohen's d**: Effect size
  - 0.2 = small effect
  - 0.5 = medium effect
  - 0.8 = large effect

Example interpretation:
```
vs RANDOM:
  p-value: 0.0012
  Significant: YES
  Cohen's d: 0.85
```
This means: DQN is significantly better than random with a large effect size.

## Output Format

The evaluation generates a JSON report with the following structure:

```json
{
  "metadata": {
    "test_start_date": "2024-01-01",
    "test_end_date": "2024-12-31",
    "num_episodes": 100,
    "episode_length": 30,
    "initial_capital": 100000,
    "timestamp": "20250101_120000"
  },
  "dqn": {
    "episodes": [
      {
        "total_return": 0.0850,
        "sharpe_ratio": 1.24,
        "max_drawdown": 0.085,
        "win_rate": 0.62,
        ...
      }
    ],
    "aggregated": {
      "total_return": {
        "mean": 0.0850,
        "std": 0.1240,
        "min": -0.15,
        "max": 0.35,
        "median": 0.082
      },
      ...
    }
  },
  "baselines": {
    "random": { ... },
    "hold": { ... },
    "long": { ... }
  },
  "comparisons": {
    "random": {
      "t_statistic": 3.45,
      "p_value": 0.0012,
      "cohens_d": 0.85,
      "significant": true
    },
    ...
  }
}
```

## Best Practices

### 1. Use Held-Out Test Data

**NEVER** evaluate on data the agent saw during training!

```bash
# WRONG: Evaluating on training data
--test-start 2020-01-01 --test-end 2023-12-31  # Training period!

# RIGHT: Evaluating on held-out data
--test-start 2024-01-01 --test-end 2024-12-31  # Unseen test data
```

### 2. Run Enough Episodes

- Minimum: 30 episodes (for basic statistics)
- Recommended: 100 episodes (for reliable results)
- Publication-quality: 200+ episodes

### 3. Compare Multiple Baselines

Don't just beat a random agent:

```bash
--baselines random hold long momentum long_short
```

### 4. Check Statistical Significance

A strategy that's only slightly better than baselines might just be lucky.
Look for:
- p-value < 0.05
- Large effect size (Cohen's d > 0.5)

### 5. Analyze Risk Metrics

High returns mean nothing without considering risk:
- Check max drawdown (can you tolerate a 30% loss?)
- Check Sharpe ratio (are returns worth the volatility?)
- Check win rate (is it consistent or lucky?)

## Example Workflows

### Basic Evaluation

```bash
# Evaluate on 2024 data with default settings
python -m rl.evaluation.run_evaluation \
    --checkpoint checkpoints/dqn_best.pt \
    --num-episodes 100
```

### Comprehensive Evaluation

```bash
# Full evaluation with all baselines
python -m rl.evaluation.run_evaluation \
    --checkpoint checkpoints/dqn_best.pt \
    --test-start 2024-01-01 \
    --test-end 2024-12-31 \
    --num-episodes 200 \
    --baselines random hold long short momentum long_short \
    --output-dir rl/evaluation/results/comprehensive
```

### Walk-Forward Analysis

```bash
# Test on multiple time periods
for year in 2022 2023 2024; do
    python -m rl.evaluation.run_evaluation \
        --checkpoint checkpoints/dqn_best.pt \
        --test-start ${year}-01-01 \
        --test-end ${year}-12-31 \
        --num-episodes 100 \
        --output-dir rl/evaluation/results/year_${year}
done
```

## Troubleshooting

### "Not enough trading days in test period"

Your test period is too short for the episode length.

**Solution**: Use a longer test period or shorter episodes:
```bash
--test-start 2024-01-01 --test-end 2024-12-31 --episode-length 20
```

### "Checkpoint not found"

The checkpoint path is incorrect.

**Solution**: Check that the checkpoint exists:
```bash
ls checkpoints/
```

### "CUDA out of memory"

Not enough GPU memory.

**Solution**: Use CPU or reduce parallel environments:
```bash
--device cpu
# or
--parallel 1
```

### Results are not statistically significant

The DQN might not actually be better than baselines.

**Possible causes**:
1. Not enough training
2. Overfitting to training data
3. Bad hyperparameters
4. Strategy genuinely doesn't work

**Solution**: Train longer, try different hyperparameters, or rethink the approach.

## Module Structure

```
rl/evaluation/
├── README.md              # This file
├── __init__.py           # Package init
├── metrics.py            # Metric calculations
├── baselines.py          # Baseline strategies
├── evaluator.py          # Main evaluation class
├── run_evaluation.py     # CLI script
└── results/              # Output directory
    └── evaluation_report_*.json
```

## API Usage

You can also use the evaluation framework programmatically:

```python
from rl.evaluation.evaluator import DQNEvaluator
from rl.evaluation.metrics import compute_all_metrics
from rl.evaluation.baselines import get_baseline_strategy

# Create evaluator
evaluator = DQNEvaluator(
    agent=trained_agent,
    vec_env=vec_env,
    stock_selections_cache=stock_cache,
    test_start_date='2024-01-01',
    test_end_date='2024-12-31',
    num_episodes=100
)

# Run evaluation
report_path = evaluator.run_full_evaluation(
    baseline_names=['random', 'hold', 'long']
)

print(f"Report saved to: {report_path}")
```

## Contributing

To add new metrics, edit `metrics.py`.
To add new baseline strategies, edit `baselines.py`.

## License

Part of the Stock-Prediction project.
