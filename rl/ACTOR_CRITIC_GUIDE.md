# Actor-Critic Trading System

Complete guide to the new actor-critic reinforcement learning framework for stock trading.

## Overview

This system replaces the previous Q-learning approach with a more sophisticated **actor-critic** architecture that separates policy learning (actor) from value estimation (critic).

### Why Actor-Critic?

**Benefits over Q-learning:**
- 🎯 **Better exploration**: Actor learns stochastic policy, naturally explores
- 📈 **Faster convergence**: Separate networks for policy and value
- 🧠 **More expressive**: Can learn complex, continuous-like policies
- ⚡ **Jumpstart learning**: Pre-train critic on synthetic data first

## Architecture

### Actor Network
- **Input**: State features (1469 dims)
- **Output**: Probability distribution over 3 actions [HOLD, BUY, SELL]
- **Training**: Policy gradient (maximize Q-values)
- **Role**: Decides which actions to take

### Critic Network
- **Input**: State features (1469 dims)
- **Output**: Q-values for all 3 actions
- **Training**: Temporal difference learning (Double DQN)
- **Role**: Estimates value of actions

### Feature Extractor
- **Frozen predictor model**: Extracts 1444-dim features
- **Portfolio context**: Adds 25-dim position/cash info
- **Total**: 1469-dim state representation

## Training Pipeline

The system uses a **3-phase training approach**:

### Phase 1: Generate Synthetic Data

**Script**: `rl/generate_critic_data.py`

**Purpose**: Create large dataset of reasonable trading experiences

**Programmatic Strategy**:
1. **Single stock at a time** (simplified action space)
2. **Buy logic**:
   - Randomly sample time horizon (1d, 3d, 5d, 10d)
   - Buy top-ranked stock for that horizon (based on predictor's expected return)
3. **Sell logic** (pseudo-deterministic):
   - Calculate percentile rank (where stock ranks among all stocks for its horizon)
   - Calculate sell probability:
     ```python
     time_factor = days_held / target_horizon
     rank_factor = max(0, (0.5 - percentile_rank) / 0.5)  # 0 if above median
     sell_prob = min(1.0, α * time_factor + β * rank_factor)
     ```
   - Sell with probability `sell_prob`
4. **Exploration**: 10% random actions for robustness

**Key Features**:
- ✅ Uses **relative ranking** (not absolute predictions)
- ✅ Sells more aggressively if stock falls below median
- ✅ Sells more as approaching target horizon
- ✅ **Randomized hyperparameters per episode** for wide distribution coverage
  - α (time weight): Uniform(0.2, 0.8) - creates patient vs impatient traders
  - β (rank weight): Uniform(0.2, 0.8) - creates conservative vs aggressive sellers
  - ε (exploration): Uniform(0.05, 0.25) - varying levels of randomness
- ✅ Generates diverse, reasonable experiences across different trading styles

**Output**: `data/critic_training_data.pkl` (~600K transitions for 20K episodes)

**Runtime**: ~2-5 minutes for 20K episodes (PARALLELIZED across all CPU cores!)
- Sequential: Would take ~3-4 hours
- Parallel: Uses all CPU cores → **40x faster!**

### Phase 2: Pre-train Critic

**Script**: `rl/pretrain_critic.py`

**Purpose**: Train critic on synthetic data BEFORE going online

**Training**:
- Loads synthetic data from Phase 1
- Trains critic using Double DQN loss
- Freezes actor and predictor (only critic trains)
- Uses standard Q-learning updates:
  ```
  Q_target = reward + γ * max_a Q_target(s', a)
  loss = MSE(Q_predicted(s,a), Q_target)
  ```

**Benefits**:
- ✅ Critic has good value estimates before online training
- ✅ Actor can use these estimates immediately (faster learning)
- ✅ Reduces early training instability

**Output**: `./checkpoints/pretrained_critic.pt`

**Runtime**: ~1-2 hours for 50 epochs

### Phase 3: Online Actor-Critic Training

**Script**: `rl/train_actor_critic.py`

**Purpose**: Train both actor and critic online with real environment

**Training Loop**:
1. **Collect experience**:
   - Actor selects actions (stochastic sampling)
   - Environment executes trades
   - Store transitions in replay buffer

2. **Train critic**:
   - Sample batch from replay buffer
   - Compute Double DQN loss
   - Update critic weights
   - Soft-update target critic

3. **Train actor** (every N steps):
   - Sample batch from replay buffer
   - Compute policy gradient loss:
     ```
     L_actor = -E[log π(a|s) * Q(s,a)] - β*H(π)
     ```
   - Where H(π) is entropy (encourages exploration)
   - Update actor weights

**Key Details**:
- Actor updates less frequently (every 2 steps)
- Uses entropy regularization for exploration
- Loads pre-trained critic from Phase 2
- Single-stock mode (max_positions=1)

**Output**:
- `./checkpoints/actor_critic_final.pt` (final model)
- `./checkpoints/actor_critic_training_history.csv` (metrics)

**Runtime**: ~2-4 hours for 1000 episodes

## Running the Pipeline

### Quick Start (All Phases)

Run the complete pipeline:
```bash
./rl/run_actor_critic_pipeline.sh
```

This executes all three phases sequentially.

### Individual Phases

**Phase 1 only** (generate data):
```bash
python rl/generate_critic_data.py
```

**Phase 2 only** (pre-train critic):
```bash
python rl/pretrain_critic.py
```

**Phase 3 only** (online training):
```bash
python rl/train_actor_critic.py
```

## Configuration

### Phase 1: Data Generation

Edit `rl/generate_critic_data.py` (bottom of file):
```python
buffer = generate_synthetic_data(
    num_episodes=20000,     # Number of episodes (parallelized, so can use many!)
    episode_length=30,      # Steps per episode
    num_test_stocks=100,    # Number of stocks
    save_path='data/critic_training_data.pkl',
    num_workers=None        # Use all CPU cores (default)
)
```

Strategy configuration (in `generate_synthetic_data`):
```python
strategy = ProgrammaticTradingStrategy(
    randomize_per_episode=True  # Randomize α, β, ε each episode
)
# Hyperparameters are sampled per episode:
# alpha ~ Uniform(0.2, 0.8)
# beta ~ Uniform(0.2, 0.8)
# epsilon ~ Uniform(0.05, 0.25)
```

To use fixed hyperparameters instead (not recommended):
```python
strategy = ProgrammaticTradingStrategy(
    alpha=0.5,
    beta=0.5,
    epsilon=0.1,
    randomize_per_episode=False
)
```

### Phase 2: Critic Pre-training

Edit `rl/pretrain_critic.py` (bottom of file):
```python
pretrain_critic(
    num_epochs=50,        # Training epochs
    batch_size=256,       # Batch size
    learning_rate=3e-4,   # Learning rate
    gamma=0.99,           # Discount factor
    tau=0.005             # Target network update rate
)
```

### Phase 3: Online Training

Edit `rl/train_actor_critic.py` (config dict at bottom):
```python
config = {
    # Training
    'num_episodes': 1000,      # Number of episodes
    'batch_size': 256,         # Batch size
    'actor_lr': 1e-4,          # Actor learning rate
    'critic_lr': 3e-4,         # Critic learning rate
    'gamma': 0.99,             # Discount factor
    'tau': 0.005,              # Target network update
    'entropy_coef': 0.01,      # Entropy regularization
    'actor_update_freq': 2,    # Update actor every N steps

    # Environment
    'episode_length': 20,      # Steps per episode (shorter for opportunity switching)
    'max_positions': 1,        # Single stock mode
    'top_k_per_horizon': 10,   # Stock filtering
}
```

## Key Design Decisions

### 1. Single Stock Mode

**Decision**: Only hold 1 stock at a time (max_positions=1)

**Rationale**:
- Simpler state/action space for network to learn
- Easier to attribute rewards to specific actions
- Still realistic (many traders focus on 1-2 positions)
- Can scale up later once working

### 2. Relative Ranking for Selling

**Decision**: Sell based on percentile rank, not absolute predictions

**Rationale**:
- More robust to market regime changes
- Focuses on comparative advantage
- Predictor is better at ranking than absolute values
- Naturally adapts to market conditions

### 3. Pseudo-Deterministic Selling

**Decision**: Use deterministic formula → probability → sample

**Rationale**:
- Deterministic rules are interpretable
- Stochasticity adds exploration/diversity
- Best of both worlds: principled + diverse

### 4. Pre-training Critic First

**Decision**: Train critic offline before online actor-critic training

**Rationale**:
- Critic provides stable value estimates immediately
- Actor can learn faster with good critic
- Avoids early instability of both learning simultaneously
- Synthetic data is "free" (no opportunity cost)

### 5. Top-K Filtering

**Decision**: Filter to top-10 stocks per time horizon

**Rationale**:
- Reduces action space ~20x (900 → ~40 stocks)
- Focuses on most promising opportunities
- Leverages predictor's strength
- Still diverse (union across 4 horizons)

### 6. Short Episodes (20 Days)

**Decision**: Use 20-day episodes instead of longer periods

**Rationale**:
- Allows learning opportunity switching (sell A to buy better B)
- Agent can discover: "Small profit now + next big winner > holding mediocre stock"
- Shorter feedback loops for faster learning
- Still long enough to evaluate multi-day holding strategies
- Encourages active trading and rebalancing decisions

### 7. Randomized Strategy Hyperparameters

**Decision**: Randomize α, β, ε per episode during synthetic data generation

**Rationale**:
- **Wide distribution coverage**: Critic sees diverse trading behaviors
- **Better generalization**: Can evaluate strategies it hasn't seen before
- **Prevents overfitting**: Not tied to one specific strategy
- **Actor flexibility**: Actor can discover novel strategies critic can still evaluate
- **Trading style diversity**:
  - High α, low β = patient, holds to horizon regardless of rank
  - Low α, high β = aggressive, sells immediately if rank drops
  - Mid α, mid β = balanced approach
  - Each episode explores different point in strategy space

### 8. Parallelized Data Generation

**Decision**: Run episode generation in parallel across all CPU cores

**Rationale**:
- **Massive speedup**: ~40x faster (3-4 hours → 2-5 minutes for 20K episodes)
- **CPU-bound task**: Strategy logic and environment stepping are pure Python
- **No GPU needed**: Workers use CPU, freeing GPU for other tasks
- **Scales linearly**: 16 cores = ~16x faster
- **Enables more data**: Can generate 20K episodes instead of 2K in same time
- **Better distribution coverage**: More episodes = better sampling of strategy space

## Performance Optimizations

### Caching System

**Two-level caching**:
1. **Feature cache** (`rl_feature_cache_4yr.h5`): Raw HDF5 features
2. **State cache** (`rl_state_cache_4yr.h5`): Precomputed transformer features

**First run**: ~20 minutes to precompute
**Subsequent runs**: <1 minute to load from cache

### 4-Year Window

**Uses only last 4 years** of market data instead of full 25 years:
- 6281 days → 1000 days (6x reduction)
- More relevant recent data
- 10x faster initialization
- Smaller cache files (10 GB → 2 GB)

### Batched Inference

- All stock features extracted in batches
- GPU-accelerated transformer forward passes
- State lookup from pre-computed cache (no on-the-fly computation)

## Monitoring Training

### Phase 2: Critic Pre-training

**Key metrics**:
- `loss`: TD-error (should decrease over time)
- `avg_epoch_loss`: Epoch average (track convergence)
- `best_loss`: Best achieved loss

**Expected behavior**:
- Loss decreases steadily
- Converges after 30-50 epochs
- Final loss: ~0.001-0.01 (depends on reward scale)

### Phase 3: Online Training

**Key metrics**:
- `return`: Episode return (should increase)
- `portfolio_value`: Portfolio value (should grow)
- `avg_critic_loss`: Critic TD-error (should stabilize)
- `avg_actor_loss`: Policy gradient loss (can fluctuate)
- `avg_entropy`: Policy entropy (should decrease slowly)

**Expected behavior**:
- Returns improve over first 200-500 episodes
- Portfolio value grows (ideally >$100K initial capital)
- Critic loss stabilizes
- Entropy decreases as policy becomes more confident

**Good signs**:
- ✅ Average return trending upward
- ✅ Portfolio value > initial capital
- ✅ Entropy decreasing gradually (not too fast!)
- ✅ Actor loss relatively stable

**Bad signs**:
- ⚠️ Returns oscillating wildly
- ⚠️ Portfolio value < initial capital consistently
- ⚠️ Entropy drops to near-zero (policy collapsed)
- ⚠️ Critic loss increasing

## Troubleshooting

### Issue: Critic loss not decreasing (Phase 2)

**Possible causes**:
- Learning rate too high
- Synthetic data too noisy
- Not enough training epochs

**Solutions**:
- Lower learning rate (3e-4 → 1e-4)
- Increase num_epochs (50 → 100)
- Regenerate data with lower epsilon (0.1 → 0.05)

### Issue: Actor loss exploding (Phase 3)

**Possible causes**:
- Learning rate too high
- Entropy coefficient too low
- Critic estimates too poor

**Solutions**:
- Lower actor_lr (1e-4 → 3e-5)
- Increase entropy_coef (0.01 → 0.05)
- Re-train critic (Phase 2)

### Issue: Policy entropy collapses to zero

**Possible causes**:
- Entropy coefficient too low
- Actor learning too fast

**Solutions**:
- Increase entropy_coef (0.01 → 0.1)
- Lower actor_lr
- Update actor less frequently (actor_update_freq: 2 → 4)

### Issue: Returns not improving

**Possible causes**:
- Episode length too short
- Top-K filtering too aggressive
- Reward signal too sparse

**Solutions**:
- Increase episode_length (50 → 100)
- Increase top_k_per_horizon (10 → 20)
- Check programmatic strategy is reasonable

## Next Steps

### After Training

1. **Evaluate performance**:
   - Run backtest on held-out data
   - Compare to buy-and-hold baseline
   - Analyze trade patterns

2. **Tune hyperparameters**:
   - Learning rates
   - Entropy coefficient
   - Update frequencies

3. **Scale up**:
   - Increase max_positions (1 → 3)
   - Use more stocks (100 → 500)
   - Longer episodes (50 → 100)

### Potential Improvements

1. **Multi-stock mode**: Train with max_positions > 1
2. **Adaptive position sizing**: Learn allocation amounts
3. **Risk management**: Add position limits, stop losses
4. **Curriculum learning**: Start simple, gradually increase difficulty
5. **Ensemble predictors**: Use multiple predictors for filtering

## Summary

The actor-critic system provides a robust framework for learning trading policies:

**Key advantages**:
- ✅ Pre-trained critic jumpstarts learning
- ✅ Programmatic strategy provides reasonable baseline
- ✅ Relative ranking adapts to market conditions
- ✅ Single-stock mode simplifies learning
- ✅ Fast training with caching optimizations

**Total pipeline time** (first run): ~4-7 hours
**Total pipeline time** (with caching): ~3-6 hours

**Expected outcome**: Trading policy that outperforms random baseline and potentially matches or exceeds programmatic strategy.

Good luck training! 🚀
