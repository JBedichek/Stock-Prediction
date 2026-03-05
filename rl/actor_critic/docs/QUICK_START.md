# Actor-Critic Quick Start

Quick reference for running the new actor-critic trading system.

## One-Command Pipeline

Run everything (3 phases):
```bash
./rl/run_actor_critic_pipeline.sh
```

**Time**: 4-7 hours (first run), 3-6 hours (with caching)

## Individual Phases

### Phase 1: Generate Synthetic Data

```bash
python rl/generate_critic_data.py
```

**What it does**: Creates ~600K trading experiences using programmatic strategy (PARALLELIZED!)
**Output**: `data/critic_training_data.pkl`
**Time**: 2-5 minutes (uses all CPU cores - ~40x faster than sequential!)

### Phase 2: Pre-train Critic

```bash
python rl/pretrain_critic.py
```

**What it does**: Trains critic on synthetic data offline
**Output**: `./checkpoints/pretrained_critic.pt`
**Time**: 1-2 hours

### Phase 3: Train Actor-Critic Online

```bash
python rl/train_actor_critic.py
```

**What it does**: Trains both actor and critic with live environment
**Output**:
- `./checkpoints/actor_critic_final.pt`
- `./checkpoints/actor_critic_training_history.csv`
**Time**: 2-4 hours

## Required Files

Before running, ensure you have:
- ✅ `data/all_complete_dataset.h5` (dataset)
- ✅ `data/actual_prices.h5` (price data)
- ✅ `./checkpoints/best_model_100m_1.18.pt` (trained predictor)
- ✅ `data/adaptive_bin_edges.pt` (bin edges for predictor)

## Caches (Generated Automatically)

First run creates:
- `data/rl_feature_cache_4yr.h5` (~400 MB)
- `data/rl_state_cache_4yr.h5` (~1.5 GB)

Subsequent runs load from cache (much faster!)

## Quick Config Changes

### More/Less Data (Phase 1)

Edit `rl/generate_critic_data.py`, line ~520:
```python
num_episodes=2000,  # Change this: more = more data, slower
```

### Faster Critic Training (Phase 2)

Edit `rl/pretrain_critic.py`, line ~248:
```python
num_epochs=50,  # Reduce for faster (e.g., 20), may hurt quality
```

### Longer Online Training (Phase 3)

Edit `rl/train_actor_critic.py`, line ~386:
```python
'num_episodes': 1000,  # Increase for longer training
'episode_length': 20,  # Shorter episodes allow opportunity switching
```

## Monitoring Progress

### Phase 1 (Data Generation)

Watch for:
- Progress bar showing episode completion
- Average episode return (should be reasonable, e.g., -0.1 to 0.1)

### Phase 2 (Critic Pre-training)

Watch for:
- Loss decreasing over epochs
- Best loss checkpoint saves

Good: Loss goes from ~0.1 → ~0.001-0.01
Bad: Loss stays flat or increases

### Phase 3 (Online Training)

Watch for (printed every 10 episodes):
- **Return**: Should trend upward over time
- **Portfolio value**: Should grow above $100K
- **Critic/Actor loss**: Should stabilize

Good signs:
- Returns improving
- Portfolio > $100K after 500+ episodes

Bad signs:
- Returns oscillating wildly
- Portfolio declining steadily
- Entropy = 0 (policy collapsed)

## Checkpoints

The system saves checkpoints automatically:

**Phase 2**:
- `./checkpoints/pretrained_critic.pt` (best critic)

**Phase 3**:
- `./checkpoints/actor_critic_ep100.pt` (every 100 episodes)
- `./checkpoints/actor_critic_ep200.pt`
- ...
- `./checkpoints/actor_critic_final.pt` (final)

## Common Issues

### CUDA Out of Memory

**Solution**: Reduce batch_size in config
```python
'batch_size': 128,  # Was 256
```

### Data Generation Too Slow

**Solution**: Reduce num_episodes or num_test_stocks
```python
num_episodes=1000,      # Was 2000
num_test_stocks=50,     # Was 100
```

### Critic Not Loading (Phase 3)

**Solution**: Make sure Phase 2 completed successfully
```bash
ls -lh ./checkpoints/pretrained_critic.pt
```

If missing, re-run Phase 2.

## Resume Training

To resume online training from checkpoint:

1. Edit `rl/train_actor_critic.py`
2. Add to config:
```python
'resume_checkpoint': './checkpoints/actor_critic_ep500.pt'
```
3. Add load logic in `__init__`:
```python
if 'resume_checkpoint' in config:
    checkpoint = torch.load(config['resume_checkpoint'])
    self.agent.load_state_dict(checkpoint['agent_state_dict'])
    # ... load optimizers, etc.
```

(Not implemented by default - manual modification needed)

## Next Steps

After training completes:

1. **Check training history**:
```bash
# View CSV with metrics
cat ./checkpoints/actor_critic_training_history.csv | column -t -s,
```

2. **Run evaluation/backtest**:
   - Use trained model for inference
   - Compare to baselines
   - Analyze trading patterns

3. **Tune hyperparameters**:
   - Adjust learning rates if unstable
   - Modify entropy coefficient if policy too deterministic/random
   - Change episode length if not learning long-term patterns

## Help

For detailed information:
- **Full guide**: `rl/ACTOR_CRITIC_GUIDE.md`
- **Architecture details**: `rl/rl_components.py` (docstrings)
- **Strategy details**: `rl/generate_critic_data.py` (ProgrammaticTradingStrategy class)

Questions? Check the documentation or review the code!
