# Walk-Forward Training Guide

This document explains how `training/walk_forward_training.py` works conceptually, covering the training loop, epoch behavior, and distributed training.

## Overview

Walk-forward training is a time-series cross-validation method that respects the temporal nature of financial data. Unlike standard k-fold cross-validation (which randomly splits data), walk-forward validation ensures:

1. **No look-ahead bias**: Training data always precedes test data in time
2. **Realistic evaluation**: Simulates how the model would be used in production
3. **Multiple evaluation periods**: Tests model performance across different market conditions

## Window Modes

### Expanding Window (Recommended)

```
Fold 1: [====TRAIN====][GAP][TEST]
Fold 2: [========TRAIN========][GAP][TEST]
Fold 3: [============TRAIN============][GAP][TEST]
```

- Training window grows with each fold
- Uses all available historical data
- More stable training but may include stale patterns

### Sliding Window

```
Fold 1: [====TRAIN====][GAP][TEST]
Fold 2:      [====TRAIN====][GAP][TEST]
Fold 3:           [====TRAIN====][GAP][TEST]
```

- Fixed-size training window that slides forward
- Uses only recent data
- Adapts better to regime changes but has less training data

## The Training Pipeline

### 1. Fold Generation

The pipeline first generates temporal folds based on configuration:

```python
folds = self._get_folds()  # Returns list of (train_dates, test_dates)
```

Key parameters:
- `--min-train-months`: Minimum training period (default: 12 months)
- `--test-months`: Test period per fold (default: 3 months)
- `--gap-days`: Gap between train and test to avoid data leakage (default: 5 days)

### 2. Per-Fold Training

For each fold, the trainer:

1. **Creates a fresh model** (no weight sharing between folds)
2. **Splits training data** into 90% train / 10% validation
3. **Trains for N epochs** (configurable via `--epochs-per-fold`)
4. **Evaluates on the test period** using backtesting simulation

## What Happens During an Epoch

```
┌─────────────────────────────────────────────────────────────────┐
│                        EPOCH STRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Set DistributedSampler epoch (for DDP shuffling)           │
│     └── train_sampler.set_epoch(epoch)                         │
│                                                                 │
│  2. TRAINING PHASE                                              │
│     ├── model.train()                                          │
│     ├── For each batch:                                        │
│     │   ├── Move data to GPU                                   │
│     │   ├── Forward pass: pred, confidence = model(features)   │
│     │   ├── Compute loss (cross-entropy or MSE)                │
│     │   ├── Backward pass: loss.backward()                     │
│     │   └── Optimizer step: optimizer.step()                   │
│     └── Collect training losses                                │
│                                                                 │
│  3. VALIDATION PHASE                                            │
│     ├── model.eval()                                           │
│     ├── torch.no_grad() context                                │
│     ├── For each batch:                                        │
│     │   ├── Forward pass only                                  │
│     │   └── Compute validation loss                            │
│     └── Collect validation losses                              │
│                                                                 │
│  4. SYNCHRONIZATION (DDP only)                                  │
│     ├── All-reduce training loss across GPUs                   │
│     └── All-reduce validation loss across GPUs                 │
│                                                                 │
│  5. LOGGING                                                     │
│     └── Print epoch summary (rank 0 only)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Computation

For classification mode (default):
```python
# Convert continuous price ratios to bin indices
bin_indices = convert_price_ratios_to_bins_vectorized(targets, bin_edges)

# Cross-entropy loss for each prediction horizon
loss = F.cross_entropy(pred[:, :, 0], bin_indices[:, 0])  # 1-day
loss += F.cross_entropy(pred[:, :, 1], bin_indices[:, 1])  # 5-day
loss += F.cross_entropy(pred[:, :, 2], bin_indices[:, 2])  # 10-day
loss += F.cross_entropy(pred[:, :, 3], bin_indices[:, 3])  # 20-day
loss = loss / 4.0  # Average across horizons
```

For regression mode:
```python
loss = F.mse_loss(pred, targets)
```

## What Happens After Each Fold

```
┌─────────────────────────────────────────────────────────────────┐
│                    POST-FOLD OPERATIONS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SYNCHRONIZATION (DDP only)                                  │
│     └── dist.barrier() - Wait for all GPUs to finish training  │
│                                                                 │
│  2. BACKTESTING EVALUATION (rank 0 only)                        │
│     ├── Save model to temporary checkpoint                     │
│     ├── Load test data for test_dates period                   │
│     ├── Run TradingSimulator:                                  │
│     │   ├── For each test day:                                 │
│     │   │   ├── Get model predictions for all stocks           │
│     │   │   ├── Filter by confidence threshold                 │
│     │   │   ├── Select top-k stocks                            │
│     │   │   └── Simulate trades and track P&L                  │
│     │   └── Compute metrics: return, Sharpe, drawdown, etc.    │
│     └── Clean up temporary files                               │
│                                                                 │
│  3. SAVE CHECKPOINT (rank 0 only)                               │
│     └── Save fold_{idx}_best.pt with:                          │
│         ├── model_state_dict                                   │
│         ├── config (hidden_dim, num_layers, etc.)              │
│         ├── fold_idx                                           │
│         └── train/test date ranges                             │
│                                                                 │
│  4. RECORD RESULTS                                              │
│     └── Create FoldTrainingResult with:                        │
│         ├── Training metrics (train_loss, val_loss)            │
│         └── Evaluation metrics (return, Sharpe, drawdown)      │
│                                                                 │
│  5. SYNCHRONIZATION (DDP only)                                  │
│     └── dist.barrier() - Ensure checkpoint saved before        │
│         proceeding to next fold                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Distributed Data Parallel (DDP)

DDP enables training across multiple GPUs for faster training.

### How DDP Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     DDP ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 0 (Rank 0)      GPU 1 (Rank 1)      GPU 2 (Rank 2)        │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐          │
│  │  Model   │        │  Model   │        │  Model   │          │
│  │  Copy    │        │  Copy    │        │  Copy    │          │
│  └────┬─────┘        └────┬─────┘        └────┬─────┘          │
│       │                   │                   │                 │
│  ┌────┴─────┐        ┌────┴─────┐        ┌────┴─────┐          │
│  │ Batch 0  │        │ Batch 1  │        │ Batch 2  │          │
│  │ (1/3 of  │        │ (1/3 of  │        │ (1/3 of  │          │
│  │  data)   │        │  data)   │        │  data)   │          │
│  └────┬─────┘        └────┬─────┘        └────┬─────┘          │
│       │                   │                   │                 │
│       └───────────────────┼───────────────────┘                 │
│                           │                                     │
│                     GRADIENT SYNC                               │
│                    (All-Reduce)                                 │
│                           │                                     │
│                     ┌─────┴─────┐                               │
│                     │ Averaged  │                               │
│                     │ Gradients │                               │
│                     └───────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key DDP Concepts

1. **DistributedSampler**: Ensures each GPU sees different data
   ```python
   train_sampler = DistributedSampler(
       train_dataset,
       num_replicas=world_size,  # Total GPUs
       rank=rank,                 # This GPU's ID
       shuffle=True
   )
   ```

2. **set_epoch()**: Must be called each epoch for proper shuffling
   ```python
   train_sampler.set_epoch(epoch)
   ```

3. **All-Reduce**: Synchronizes gradients across GPUs (automatic in DDP)

4. **Barriers**: Explicit synchronization points
   ```python
   dist.barrier()  # Wait for all GPUs
   ```

### Running with DDP

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --num-folds 5 \
    --ddp

# Multi-node (2 nodes, 4 GPUs each)
# On node 0:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=12355 \
    -m training.walk_forward_training --data data.h5 --ddp

# On node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=<MASTER_IP> --master_port=12355 \
    -m training.walk_forward_training --data data.h5 --ddp
```

## Memory Management

### Preloading vs Streaming

```bash
# Preload all data (faster, requires more RAM)
python -m training.walk_forward_training --data data.h5

# Stream from HDF5 (slower, lower memory)
python -m training.walk_forward_training --data data.h5 --no-preload
```

Use `--no-preload` when:
- Using large sequence lengths (e.g., `--seq-len 500`)
- Running on memory-constrained machines
- Dataset is very large

### torch.compile

```bash
# Enable torch.compile with max-autotune (default)
python -m training.walk_forward_training --data data.h5

# Disable for debugging
python -m training.walk_forward_training --data data.h5 --no-compile
```

## Output Structure

### Checkpoints

```
checkpoints/walk_forward/
├── fold_0_best.pt    # Model checkpoint for fold 0
├── fold_1_best.pt    # Model checkpoint for fold 1
├── fold_2_best.pt    # ...
├── fold_3_best.pt
└── fold_4_best.pt
```

### Results Files

```
walk_forward_training_results.pt   # PyTorch serialized results
walk_forward_training_results.json # Human-readable JSON
```

Results contain:
- Per-fold metrics (return, Sharpe, drawdown, win rate)
- Aggregate statistics (mean, std)
- Statistical significance tests (t-test vs. zero)

## Example Usage

### Basic Training

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --num-folds 5 \
    --mode expanding \
    --epochs-per-fold 10 \
    --batch-size 64 \
    --seq-len 60
```

### Production Configuration

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices.h5 \
    --num-folds 8 \
    --mode expanding \
    --min-train-months 24 \
    --test-months 3 \
    --epochs-per-fold 20 \
    --hidden-dim 768 \
    --num-layers 8 \
    --num-heads 12 \
    --batch-size 128 \
    --seq-len 120 \
    --top-k 10 \
    --output production_results.pt
```

### Multi-GPU Training

```bash
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --num-folds 5 \
    --epochs-per-fold 20 \
    --batch-size 256 \
    --ddp
```

## Interpreting Results

### Statistical Significance

The summary includes a t-test against zero returns:
- **p < 0.05**: Returns are statistically significant
- **95% CI not crossing zero**: Consistent positive/negative returns

### Key Metrics

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| Mean Return | > 5% per quarter | Positive alpha |
| Sharpe Ratio | > 1.0 | Good risk-adjusted return |
| Max Drawdown | < 15% | Limited downside risk |
| Win Rate | > 50% | More winning than losing trades |

### Warning Signs

- High variance across folds: Model may be overfitting
- Negative later folds: Model not adapting to regime changes
- Sharpe < 0.5 with high returns: Too much risk
