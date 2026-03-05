# Training Loop Profiling Guide

## How to Enable Profiling

Set `'enable_profiling': True` in the config to enable detailed timing profiling.

## Usage

The profiler is already integrated into the training loop. To add profiling to a new section:

```python
with self.profiler.profile('section_name'):
    # Code to profile
    do_work()
```

## Recommended Profiling Sections

Here are the key sections that should be profiled in `train_episode_vectorized()`:

### 1. Stock Selection
```python
# Around lines 736-800
with self.profiler.profile('stock_selection'):
    # Pre-computed selection loading OR on-the-fly computation
    if self.use_precomputed_selections:
        # Load from cache
        ...
    else:
        # Compute top-K stocks
        ...
```

### 2. Current State Creation
```python
# Around lines 813-819
with self.profiler.profile('state_creation_current'):
    states_list_current = create_states_batch_optimized(...)
```

### 3. Agent Action Selection
```python
# Around lines 821-831
with self.profiler.profile('action_selection'):
    results = self.agent.select_actions_reduced_batch(...)
```

### 4. Environment Step
```python
# Around lines 849-852
with self.profiler.profile('environment_step'):
    next_states_list, rewards_list, dones_list, infos_list, next_positions_list = self.vec_env.step(...)
```

### 5. Next State Creation
```python
# Around lines 924-931
with self.profiler.profile('state_creation_next'):
    next_states_dict = create_next_states_batch_optimized(...)
```

### 6. Global State Construction
```python
# Around lines 860-867 and 937-944
with self.profiler.profile('global_state_construction'):
    global_state = create_global_state(...)
    next_global_state = create_global_state(...)
```

### 7. Transition Storage
```python
# Around lines 959-971
with self.profiler.profile('transition_storage'):
    transition = {...}
    self.buffer.push(**transition)
```

### 8. Critic Training
```python
# Around lines 1020-1029
with self.profiler.profile('critic_training'):
    critic_loss, critic_info = self._train_critic()
```

### 9. Actor Training
```python
# Around lines 1049-1059
with self.profiler.profile('actor_training'):
    actor_loss, actor_info = self._train_actor()
```

### 10. Target Network Updates
```python
# Around line 1062
with self.profiler.profile('target_network_update'):
    self.agent.update_target_critics(tau=self.config.get('tau', 0.005))
```

## Quick Integration Example

For a quick start, just wrap the entire episode training in one profile block:

```python
def train(self):
    pbar = tqdm(range(num_iterations), desc="Training")
    for iteration in pbar:
        # Profile entire episode
        with self.profiler.profile('full_episode'):
            stats_list = self.train_episode_vectorized()

        # Record episode completion
        self.profiler.record_episode()
```

## Output Format

The profiler generates reports like:

```
================================================================================
TRAINING PROFILER SUMMARY
================================================================================
Total episodes: 100
Total steps: 3000

Section                               Total (s)    Mean (ms)      Calls   % Total
--------------------------------------------------------------------------------
state_creation_current                    45.234      150.78      300      35.2%
action_selection                          28.561       95.20      300      22.3%
critic_training                           22.145       73.82      300      17.3%
environment_step                          15.892       52.97      300      12.4%
actor_training                            10.234       34.11      300       8.0%
... (other sections)

Time per episode: 1.282s
Time per step: 42.7ms
================================================================================
```

## CSV Output

Profiling data is also saved to `./profiling_results/profile_ep{episode}.csv` with columns:
- section: Section name
- total_time: Total time spent (seconds)
- mean_time: Mean time per call (ms)
- std_time: Std dev of time per call (ms)
- min_time: Min time per call (ms)
- max_time: Max time per call (ms)
- num_calls: Number of times called
- percent_total: Percentage of total profiled time

## Troubleshooting

**Q: Profiler shows 0% for all sections**
A: Make sure you're calling `self.profiler.record_step()` or `self.profiler.record_episode()` to track progress.

**Q: Some sections show very low percentages**
A: This is normal if those sections are fast. Focus on the top sections that take the most time.

**Q: Want to profile inside a function**
A: Import the profiler instance or use the `timed()` convenience function:
```python
from rl.profiling_utils import timed

with timed('my_section'):
    do_work()
```

---
*Guide created: 2025-12-16*
