# Simplified DQN Implementation

## What Changed

Replaced the complex actor-critic approach with a much simpler **DQN (Deep Q-Network)** implementation.

## Actor-Critic vs DQN

### Actor-Critic (Old - Complex)

```
┌──────────────────────────────────────────────────────────┐
│  TWO NETWORKS                                             │
├──────────────────────────────────────────────────────────┤
│  Actor Network:  State → Action Probabilities            │
│  Critic Network: State → Q-values for all actions        │
│                                                           │
│  TWO OPTIMIZERS                                           │
│  - Actor optimizer (policy gradient)                     │
│  - Critic optimizer (TD learning)                        │
│                                                           │
│  TWO LOSS FUNCTIONS                                       │
│  - Actor loss: -E[log π(a|s) * Advantage]               │
│  - Critic loss: E[(Q(s,a) - target)²]                   │
│                                                           │
│  PROBLEMS:                                                │
│  1. Off-policy learning issues                           │
│  2. Critic learns Q-values for OLD policies              │
│  3. Actor optimizes based on OUTDATED Q-values           │
│  4. Complex training dynamics                            │
│  5. Portfolio values not improving despite critic        │
│     learning (decoupled learning)                        │
└──────────────────────────────────────────────────────────┘
```

### DQN (New - Simple)

```
┌──────────────────────────────────────────────────────────┐
│  ONE NETWORK                                              │
├──────────────────────────────────────────────────────────┤
│  Q-Network: State → Q-values for all 9 actions           │
│  Target Network: Stabilized copy of Q-network            │
│                                                           │
│  ONE OPTIMIZER                                            │
│  - Q-network optimizer (TD learning)                     │
│                                                           │
│  ONE LOSS FUNCTION                                        │
│  - TD loss: E[(r + γ*max Q_target(s',a') - Q(s,a))²]   │
│                                                           │
│  ACTION SELECTION                                         │
│  - Training: ε-greedy (explore vs exploit)               │
│  - Inference: argmax Q(s, a)                             │
│                                                           │
│  ADVANTAGES:                                              │
│  1. Direct Q-value optimization                          │
│  2. No actor-critic coupling issues                      │
│  3. Simpler training dynamics                            │
│  4. Well-understood exploration (ε-greedy)               │
│  5. Proven effective for discrete actions                │
└──────────────────────────────────────────────────────────┘
```

## Action Space (9 Actions)

```
Action 0: HOLD        → Stay in cash
Action 1: LONG_1      → Buy top_4_stocks[0]
Action 2: LONG_2      → Buy top_4_stocks[1]
Action 3: LONG_3      → Buy top_4_stocks[2]
Action 4: LONG_4      → Buy top_4_stocks[3]
Action 5: SHORT_1     → Short bottom_4_stocks[0]
Action 6: SHORT_2     → Short bottom_4_stocks[1]
Action 7: SHORT_3     → Short bottom_4_stocks[2]
Action 8: SHORT_4     → Short bottom_4_stocks[3]
```

## How DQN Training Works

### 1. Q-Value Prediction

```python
# Network outputs Q-value for each action
state = [stock features + portfolio context]  # (1469,)
q_values = q_network(state)  # → [Q_HOLD, Q_L1, Q_L2, Q_L3, Q_L4, Q_S1, Q_S2, Q_S3, Q_S4]
```

### 2. Action Selection (ε-greedy)

```python
if random() < epsilon:
    action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])  # Explore
else:
    action = argmax(q_values)  # Exploit (choose best action)
```

### 3. Training (TD Learning)

```python
# Sample batch from replay buffer
batch = buffer.sample(1024)

# Current Q-values
q_current = Q(s, a)

# Target Q-values
q_target = r + γ * max_a' Q_target(s', a')

# Loss: minimize TD error
loss = MSE(q_current, q_target)

# Backprop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 4. Target Network Update

```python
# Periodically copy Q-network → target network
if step % 1000 == 0:
    target_network.load_state_dict(q_network.state_dict())
```

## Why This Should Work Better

### Problem with Actor-Critic

Your observation was spot-on:
- **Critic loss decreasing** ✅ → Critic learning to fit Q-values
- **Portfolio values NOT improving** ❌ → Agent not getting better
- **Actor loss increasing** ❌ → Policy getting worse

**Root cause**: The critic was learning Q-values for a **mixture of old policies** from the replay buffer. The actor was then trying to optimize based on these Q-values, but they didn't reflect the current policy. This created a feedback loop where:

1. Actor generates experiences with policy π_old
2. Experiences stored in buffer (mixed with ancient policies)
3. Critic learns Q-values for the mixed policies
4. Actor updates based on these Q-values
5. Policy changes → Q-values no longer accurate
6. Repeat → divergence!

### How DQN Fixes This

1. **Direct Q-value optimization**: No separate actor to get out of sync
2. **Exploration is explicit**: ε-greedy guarantees exploration
3. **Target network stabilizes learning**: Prevents moving target problem
4. **Well-understood dynamics**: DQN is battle-tested (Atari, etc.)
5. **Simpler debugging**: One network, one loss, one optimizer

## Training Process

### Exploration Schedule

```
Episodes 0-1000:    ε = 1.0 → 0.8   (80-100% random, learn basics)
Episodes 1000-3000: ε = 0.8 → 0.3   (30-80% random, explore strategies)
Episodes 3000-5000: ε = 0.3 → 0.01  (1-30% random, exploit learned Q-values)
Episodes 5000+:     ε = 0.01        (1% random, mostly exploit)
```

### What to Expect

**Early training (ε > 0.5):**
- Portfolio values will be random/poor (lots of exploration)
- Q-values will be noisy
- Loss will be high

**Mid training (0.1 < ε < 0.5):**
- Portfolio values should start improving
- Q-values should stabilize
- Loss should decrease
- Agent learns which stocks tend to be profitable

**Late training (ε < 0.1):**
- Portfolio values should be consistently good
- Q-values should be accurate predictions
- Loss should be low and stable
- Agent exploits learned strategy with 1% random exploration

## Key Hyperparameters

```python
learning_rate: 1e-4        # How fast to update Q-network
gamma: 0.99                # Discount factor (value future rewards)
epsilon_start: 1.0         # Start with 100% exploration
epsilon_end: 0.01          # End with 1% exploration
epsilon_decay: 5000        # Decay over 5000 episodes
batch_size: 1024           # Large batches for stable gradients
buffer_capacity: 100000    # Store 100k transitions
target_update_freq: 1000   # Update target network every 1000 steps
updates_per_step: 1        # 1 gradient update per environment step
```

## Reused Infrastructure

You keep all your optimizations:
- ✅ **Vectorized environment** (32 parallel episodes)
- ✅ **GPU stock selection cache** (instant lookups)
- ✅ **Vectorized transition storage** (batched operations)
- ✅ **State creation optimized** (efficient state building)
- ✅ **Replay buffer** (experience replay)
- ✅ **Profiler** (track performance)

## Files Created

1. **`rl/train_dqn_simple.py`** - Simplified DQN trainer
   - SimpleDQNTrainer class
   - Uses existing StockDQN network
   - One optimizer, one loss function
   - ε-greedy exploration

2. **`rl/example_train_dqn.sh`** - Training script
   - Shows how to run DQN training
   - Uses all existing infrastructure
   - 32 parallel environments
   - GPU stock cache

## How to Run

```bash
# Train DQN agent
./rl/example_train_dqn.sh
```

Or in Python:

```python
from rl.train_dqn_simple import SimpleDQNTrainer
from rl.gpu_vectorized_env import GPUVectorizedEnvironment
from rl.gpu_stock_cache import GPUStockSelectionCache

config = {
    'device': 'cuda',
    'num_episodes': 10000,
    'num_parallel_envs': 32,
    'learning_rate': 1e-4,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_episodes': 5000,
    # ... (see example_train_dqn.sh for full config)
}

trainer = SimpleDQNTrainer(config)
trainer.train(vec_env, stock_cache)
```

## Monitoring Training

Watch these metrics:

1. **`ema_return`** - Should increase over time
2. **`portfolio_value`** - Should increase above initial capital
3. **`ema_q_value`** - Should stabilize and become positive
4. **`ema_loss`** - Should decrease
5. **`epsilon`** - Should decay from 1.0 to 0.01
6. **`buffer_size`** - Should fill up to capacity

## Next Steps

1. Run training and monitor portfolio values
2. If values improve → DQN is working!
3. If values still don't improve → check rewards (might need reward shaping)
4. Fine-tune hyperparameters based on results

---

**Created**: 2025-12-16
**Replaces**: Actor-critic training (too complex, off-policy issues)
**Status**: Ready for testing
