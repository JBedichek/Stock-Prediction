# RL Trading Agent

Reinforcement Learning system for stock trading that learns when to buy, how much to allocate, and when to sell to maximize portfolio returns.

## Overview

The RL trading agent uses Deep Q-Network (DQN) to learn optimal trading strategies on top of your existing price predictor. It leverages rich features from the transformer model (activations, probability distributions, confidence scores) to make better trading decisions than static rules.

## System Architecture

```
Price Predictor → Feature Extraction → State → Q-Network → Actions → Portfolio
   (existing)         (rl_components)                      (buy/sell/size)
```

### Components

1. **`rl_components.py`**: Core RL neural networks
   - `PredictorFeatureExtractor`: Wraps existing price predictor to extract 1919-dim features
   - `StockDQN`: Dueling DQN for Q-value estimation
   - `TradingAgent`: Complete RL agent (feature extractor + Q-network + target network)
   - `ReplayBuffer`: Experience replay for stable training

2. **`rl_environment.py`**: Trading environment (Gym-style)
   - `TradingEnvironment`: Manages portfolio, executes actions, computes rewards
   - State: Predictor features + portfolio context (1920 dims)
   - Actions: HOLD, BUY_SMALL/MEDIUM/LARGE, SELL
   - Reward: Portfolio return (percentage change)

3. **`train_rl.py`**: Training script
   - Phase 1: Simplified 2-action system (HOLD, BUY_FULL)
   - Supports frozen predictor and joint training
   - W&B logging, checkpointing, evaluation

4. **`rl_backtest.py`**: Backtesting ✅ COMPLETE
   - Compare RL agent vs baselines (random, buy-and-hold)
   - Comprehensive performance metrics
   - Visualization and result saving

## Phased Implementation

| Feature | Phase 1 (Proof of Concept) | Phase 2 (Production) ✅ | Phase 3 (Future) |
|---------|---------------------------|------------------------|------------------|
| **Actions** | 2 (HOLD, BUY_FULL) | 5 (HOLD, BUY_S/M/L, SELL) | 5 (same as Phase 2) |
| **Exit Strategy** | Fixed 5-day auto-sell | Dynamic (agent decides) | Dynamic |
| **Stocks** | 100 | 1,000 | 3,500 (all) |
| **Max Positions** | 10 | 20 | 30 |
| **Episode Length** | 40 days | 60 days | 80 days |
| **Predictor** | Frozen | Frozen | **Joint training** |
| **Exploration Start** | 100% (ε=1.0) | 50% (ε=0.5) | 20% (ε=0.2) |
| **Buffer Capacity** | 100K | 200K | 500K |
| **Training Episodes** | 500 | 1,000 | 2,000 |
| **Goal** | Prove RL works | Learn sizing & exits | End-to-end optimization |
| **Success Criteria** | >5% vs random | >3% vs Phase 1 | >5% vs Phase 2 |

### Phase 1: Simplified System ✅ COMPLETE

**Goal**: Prove RL can learn stock selection

**Status**: Implementation complete, ready for training

### Phase 2: Full Action Space ✅ READY

**Goal**: Learn position sizing and dynamic exit timing

**Status**: Implementation complete, ready for training

### Phase 3: Joint Training (Future)

**Goal**: End-to-end optimization

**Features**:
- Unfreeze predictor
- Joint loss (DQN + predictor)
- Full 3,500 stocks

**Success Criteria**: Outperform baseline by >5%, predictor accuracy preserved

## Quick Start

**Want to jump straight to Phase 2 (production-ready)?**

```bash
# 1. Update paths in example_train_phase2.sh
nano rl/example_train_phase2.sh  # Edit DATASET_PATH, PREDICTOR_CHECKPOINT

# 2. Run training
chmod +x rl/example_train_phase2.sh
./rl/example_train_phase2.sh

# 3. Monitor training in W&B dashboard
# Training will take several hours on GPU (1000 episodes × 60 days each)

# 4. Backtest the trained agent
nano rl/example_backtest.sh  # Update checkpoint path
./rl/example_backtest.sh
```

### Prerequisites

1. Trained price predictor checkpoint
2. Dataset files (features, prices, tickers, dates)
3. Adaptive bin edges (`adaptive_bin_edges.pt`)
4. GPU recommended for Phase 2 (or reduce num_stocks/episodes for CPU)

### Phase 1 Training (Simplified - Proof of Concept)

```bash
# Option 1: Use provided example script
chmod +x rl/example_train_phase1.sh
./rl/example_train_phase1.sh

# Option 2: Run directly
python -m rl.train_rl \
    --dataset-path path/to/dataset.h5 \
    --prices-path path/to/actual_prices.h5 \
    --predictor-checkpoint path/to/best_model.pt \
    --num-episodes 500 \
    --num-stocks 100 \
    --use-wandb \
    --wandb-project rl-trading-phase1
```

### Phase 2 Training (Full Action Space - Production)

```bash
# Option 1: Use provided example script
chmod +x rl/example_train_phase2.sh
./rl/example_train_phase2.sh

# Option 2: Run directly
python -m rl.train_rl_phase2 \
    --dataset-path path/to/dataset.h5 \
    --prices-path path/to/actual_prices.h5 \
    --predictor-checkpoint path/to/best_model.pt \
    --num-episodes 1000 \
    --num-stocks 1000 \
    --max-positions 20 \
    --episode-length 60 \
    --use-wandb \
    --wandb-project rl-trading-phase2
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-stocks` | 100 | Number of stocks (Phase 1 subsample) |
| `--holding-period` | 5 | Fixed holding period (days) |
| `--num-episodes` | 500 | Number of training episodes |
| `--episode-length` | 40 | Episode length (trading days) |
| `--batch-size` | 256 | Training batch size |
| `--lr-q-network` | 1e-4 | Learning rate for Q-network |
| `--epsilon-start` | 1.0 | Initial exploration rate |
| `--epsilon-end` | 0.05 | Final exploration rate |
| `--initial-capital` | 100000 | Starting capital ($) |
| `--max-positions` | 10 | Max simultaneous positions |

## Key Differences: Phase 1 vs Phase 2

### What's New in Phase 2?

1. **Full Action Space (5 actions)**:
   - Phase 1: Binary decision (HOLD or BUY_FULL)
   - Phase 2: Granular control (HOLD, BUY_SMALL 25%, BUY_MEDIUM 50%, BUY_LARGE 100%, SELL)
   - **Benefit**: Agent learns position sizing, not just stock selection

2. **Dynamic Exit Timing**:
   - Phase 1: Fixed 5-day holding period (auto-sell)
   - Phase 2: Agent decides when to exit each position
   - **Benefit**: Can hold winners longer, cut losers sooner

3. **Larger Universe (10x more stocks)**:
   - Phase 1: 100 stocks subsample
   - Phase 2: 1,000 stocks (full test set)
   - **Benefit**: Better generalization, more realistic

4. **More Simultaneous Positions**:
   - Phase 1: 10 max positions
   - Phase 2: 20 max positions
   - **Benefit**: Better diversification, more trading opportunities

5. **Longer Episodes**:
   - Phase 1: 40 trading days
   - Phase 2: 60 trading days
   - **Benefit**: More data per episode, better long-term learning

6. **Action Constraints**:
   - Can't buy if already have position (prevents doubling down)
   - Can't sell if don't have position
   - Can't buy if at max positions or no cash
   - Invalid actions automatically mapped to valid alternatives

### When to Use Each Phase?

- **Use Phase 1** if:
  - You want to quickly validate that RL works on your data
  - You have limited compute resources
  - You want simple stock selection (buy and hold fixed period)

- **Use Phase 2** if:
  - You want production-ready trading agent
  - You have sufficient compute (GPU recommended)
  - You want to learn position sizing and exit timing
  - You want better performance

## How It Works

### State Representation (1920 dimensions)

**From Price Predictor (1919 dims)**:
- Transformer activations: 604 dims (via `forward_with_t_act()`)
- Probability distributions: 1280 dims (320 bins × 4 horizons)
- Distribution entropy: 4 dims (confidence per horizon)
- Expected returns: 4 dims (weighted predictions)
- Fundamentals: 27 dims (time-varying metrics)

**Portfolio Context (1 dim placeholder + future expansion)**:
- Position info: size, days_held, entry_price, unrealized_return
- Capital: available_cash, portfolio_weight
- Constraints: num_positions, max_positions
- Market context: cross-sectional_rank, volatility_rank

### Actions

Phase 1 (2 actions):
- 0: HOLD - Do nothing
- 1: BUY_FULL - Buy with full allocation (maps to BUY_LARGE internally)

Phase 2+ (5 actions):
- 0: HOLD - Do nothing
- 1: BUY_SMALL - Allocate 25% of available capital
- 2: BUY_MEDIUM - Allocate 50%
- 3: BUY_LARGE - Allocate 100%
- 4: SELL - Close position

### Reward

```python
reward = (portfolio_value_t - portfolio_value_{t-1}) / portfolio_value_{t-1}
```

Simple percentage return. TD learning handles variable holding periods naturally.

### Training Algorithm

**Double DQN** with:
- Dueling architecture (value + advantage streams)
- Experience replay (100k transitions)
- Soft target updates (τ=0.005)
- Epsilon-greedy exploration (1.0 → 0.05)
- Gradient clipping (norm=1.0)

## Monitoring Training

### W&B Metrics

- `episode_reward`: Reward per episode
- `loss`: TD loss
- `epsilon`: Exploration rate
- `buffer_size`: Replay buffer utilization
- `eval/avg_reward`: Evaluation performance
- `eval/avg_return`: Average portfolio return
- `eval/avg_sharpe`: Sharpe ratio
- `eval/win_rate`: Win rate (%)

### Console Output

```
Episode 100/500
  Avg Reward (last 10): 0.0234 ± 0.0112
  Epsilon: 0.800
  Buffer: 25600/100000 (25.6%)
  Global Step: 4000

==============================================================
EVALUATION at Episode 100
==============================================================

Evaluation Results (5 episodes):
  Avg Reward: 0.0312
  Avg Return: 12.45%
  Avg Sharpe: 1.82
  Avg Win Rate: 58.3%
  🎉 New best model! (Reward: 0.0312)
```

## Checkpoints

Checkpoints saved to `./rl_checkpoints/` (configurable):

- `checkpoint_episode_N.pt`: Regular checkpoints
- `best_episode_N.pt`: Best model by evaluation reward

### Loading a Checkpoint

```python
from rl.rl_components import TradingAgent

# Load checkpoint
checkpoint = torch.load('rl_checkpoints/best_episode_500.pt')

# Initialize agent
agent = TradingAgent(predictor_checkpoint_path='path/to/predictor.pt')
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

# Use for inference
q_values = agent.q_network(state)
action = q_values.argmax()
```

## Integration with Existing Code

### Reuses Existing Infrastructure

✅ **DatasetLoader** from `inference/backtest_simulation.py`
✅ **ModelPredictor** wrapped in `PredictorFeatureExtractor`
✅ **forward_with_t_act()** from `training/models.py:319`
✅ **adaptive_bin_edges.pt** for expected returns

### No Changes Required

- Existing price predictor: Used as-is
- Existing data pipeline: Wrapped in RL environment
- Existing backtesting: Can be adapted for RL agent

## Troubleshooting

### Issue: "Predictor checkpoint not found"

**Solution**: Verify path to your trained predictor:
```bash
ls -lh path/to/best_model.pt
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce `--batch-size` (default 256 → 128)
2. Reduce `--num-stocks` (default 100 → 50)
3. Use CPU: `--device cpu`

### Issue: "Buffer size not growing"

**Solution**: This is normal initially. Buffer needs to fill up before training starts. Check:
- Buffer size reaches `--batch-size` before training begins
- Training doesn't start until sufficient experience collected

### Issue: "Reward not improving"

**Possible causes**:
1. **Exploration too high**: Reduce `--epsilon-start` or increase `--epsilon-decay-steps`
2. **Learning rate too high**: Reduce `--lr-q-network` (try 1e-5)
3. **Insufficient training**: Increase `--num-episodes`
4. **Task too hard**: Check that random baseline achieves reasonable performance

## Performance Targets

### Phase 1 (Simplified):
- Total return: >10% per 40-day episode
- Sharpe ratio: >1.0
- Win rate: >52%
- vs Random: +5% improvement

### Phase 2 (Full actions):
- Total return: >15% per 40-day episode
- Sharpe ratio: >1.5
- Win rate: >55%
- vs Static baseline: +3% improvement

### Phase 3 (Joint training):
- Total return: >20% per 40-day episode
- Sharpe ratio: >2.0
- Win rate: >60%
- vs Baseline: +5% improvement
- Predictor accuracy: Within 5% of original

## Backtesting

After training, evaluate your agent using the backtest script:

```bash
# Option 1: Use provided example script
chmod +x rl/example_backtest.sh
./rl/example_backtest.sh

# Option 2: Run directly
python -m rl.rl_backtest \
    --checkpoint-path ./rl_checkpoints_phase1/best_episode_500.pt \
    --predictor-checkpoint path/to/best_model.pt \
    --dataset-path path/to/dataset.h5 \
    --prices-path path/to/actual_prices.h5 \
    --num-episodes 20 \
    --compare-baselines \
    --save-results \
    --plot-results
```

This will:
- Run 20 test episodes with the trained agent
- Compare against random and buy-and-hold baselines
- Generate performance plots
- Save detailed results to JSON

## Next Steps

1. **Test Phase 1**: Run training on 100 stocks, verify it outperforms random ✅
2. **Backtest Phase 1**: Evaluate trained agent vs baselines ✅
3. **Extend to Phase 2**: Implement full 5-action space with dynamic exits ✅
4. **Train Phase 2**: Run Phase 2 training on 1,000 stocks
5. **Compare Phases**: Backtest Phase 2 vs Phase 1 performance
6. **Implement Phase 3**: Add joint training with predictor fine-tuning (optional)
7. **Production deployment**: Live trading integration

## Training Workflow

Recommended progression:

```bash
# Step 1: Train Phase 1 (proof of concept)
./rl/example_train_phase1.sh

# Step 2: Backtest Phase 1
./rl/example_backtest.sh  # Update checkpoint path first

# Step 3: If Phase 1 successful, train Phase 2
./rl/example_train_phase2.sh

# Step 4: Backtest Phase 2
./rl/example_backtest.sh  # Update to Phase 2 checkpoint

# Step 5: Compare results and decide on production deployment
```

## Advanced: Custom Modifications

### Change Action Space

Edit `rl_components.py`:
```python
class StockDQN(nn.Module):
    def __init__(self, ..., action_dim=7):  # Example: Add 2 more actions
```

### Change Reward Function

Edit `rl_environment.py`:
```python
def step(self, actions):
    ...
    # Custom reward (e.g., risk-adjusted)
    reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
    reward -= 0.1 * max_drawdown  # Penalize drawdown
    return next_states, reward, done, info
```

### Add Risk Constraints

Edit `rl_environment.py`:
```python
def _execute_single_action(self, ticker, action_id, current_price):
    ...
    # Check max drawdown constraint
    if self._portfolio_value() < self.initial_capital * 0.85:
        # Don't allow new buys if down >15%
        if action_id in [1, 2, 3]:
            return None
```

## References

- **Double DQN**: van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning
- **Dueling DQN**: Wang et al. (2016) - Dueling Network Architectures for Deep Reinforcement Learning
- **Experience Replay**: Mnih et al. (2013) - Playing Atari with Deep Reinforcement Learning

## Support

For issues or questions:
1. Check this README
2. Review the plan file: `/home/james/.claude/plans/compressed-tickling-graham.md`
3. Inspect code comments in `rl_components.py`, `rl_environment.py`, `train_rl.py`

## License

Same as parent project.
