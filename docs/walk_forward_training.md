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

## Complete Argument Reference

This section documents every command-line argument available in `walk_forward_training.py`.

### Data Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | **required** | Path to HDF5 dataset containing features. Must have structure `{ticker}/features` and `{ticker}/dates`. |
| `--prices` | str | None | Path to actual prices HDF5 for backtesting. If not provided, uses prices from the main dataset. Should have `{ticker}/prices` and `{ticker}/dates`. |

### Walk-Forward Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-folds` | int | 10 | Number of temporal folds for cross-validation. More folds = more evaluation periods but less test data per fold. |
| `--mode` | str | expanding | Window mode: `expanding` (growing training window) or `sliding` (fixed-size window that moves forward). |
| `--min-train-months` | int | 160 | Minimum training period in months. For expanding mode, this is the initial training size. For sliding mode, this is the fixed window size. |
| `--test-months` | int | 6 | Test period per fold in months. Each fold evaluates on this many months of out-of-sample data. |
| `--gap-days` | int | 5 | Gap between train and test periods (in trading days). Prevents data leakage from features that use future information. |
| `--no-auto-span` | flag | False | Disable automatic date span calculation. When enabled, uses fixed `--min-train-months` and `--test-months` instead of auto-spanning the full dataset. |
| `--initial-train-fraction` | float | 0.5 | Fraction of data for initial training when auto-span is enabled. The remaining data is divided into test folds. |

### Model Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hidden-dim` | int | 1024 | Hidden dimension of the transformer. Controls model capacity. Larger = more expressive but slower and more memory. |
| `--num-layers` | int | 4 | Number of transformer encoder layers. Depth of the model. |
| `--num-heads` | int | 8 | Number of attention heads. Must evenly divide `hidden-dim`. More heads = more parallel attention patterns. |
| `--dropout` | float | 0.1 | Dropout rate for regularization. Applied after attention and feedforward layers. |
| `--pred-mode` | str | regression | Prediction mode: `classification` (predict distribution over return bins) or `regression` (predict raw returns). Classification is generally more stable. |

### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs-per-fold` | int | 1 | Number of training epochs per fold. More epochs = more training but risk of overfitting. |
| `--batch-size` | int | 128 | Batch size per GPU. Actual batch size with gradient accumulation = `batch_size × gradient_accumulation_steps`. |
| `--lr` | float | 1e-4 | Learning rate for AdamW optimizer. Lower values = more stable but slower convergence. |
| `--seq-len` | int | 1536 | Sequence length (number of historical days). Longer = more context but more memory and slower training. |
| `--gradient-accumulation-steps` | int | 4 | Accumulate gradients over N steps before optimizer update. Effective batch size = `batch_size × N`. Useful for large effective batches on limited GPU memory. |
| `--early-stopping-patience` | int | 3 | Stop training if validation loss doesn't improve for N epochs. Set to 0 to disable early stopping. |
| `--data-fraction` | float | 1.0 | Fraction of training data to use (0.0-1.0). Useful for fast experiments with subset of data. |

### Adaptive Batch Size

These arguments control automatic gradient accumulation increase when training plateaus.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--adaptive-batch-size` | flag | False | Enable adaptive batch size: automatically increase gradient accumulation when loss plateaus. |
| `--plateau-patience` | int | 15 | Number of optimizer steps without improvement before increasing batch size. |
| `--plateau-threshold` | float | 0.001 | Minimum relative improvement (0.1%) to count as "not plateaued". |
| `--max-grad-accum` | int | 32 | Maximum gradient accumulation steps. Adaptive batch size won't exceed this. |

### Ranking Loss

These arguments control auxiliary ranking losses that help the model learn cross-sectional stock ranking.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ranking-loss-weight` | float | 0.0 | Weight for ranking loss (0.0 = disabled). When > 0, adds ranking loss to primary CE/MSE loss. |
| `--ranking-loss-type` | str | pairwise | Type of ranking loss: `pairwise` (margin-based), `listnet` (distribution-based), or `correlation` (directly optimizes IC). See [ListNet documentation](listnet_loss.md). |
| `--ranking-margin` | float | 0.01 | Margin for pairwise ranking loss. Minimum score difference required for correctly ordered pairs. |
| `--ranking-only` | flag | False | Train with ONLY ranking loss (no cross-entropy/MSE). Directly optimizes for stock ranking rather than return prediction. |

### Monte Carlo Validation

Monte Carlo validation runs after each fold to estimate strategy robustness across random stock subsets.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-monte-carlo` | flag | False | Disable Monte Carlo validation (enabled by default). |
| `--mc-trials` | int | 50 | Number of Monte Carlo trials per fold. More trials = more reliable estimates but slower. |
| `--mc-stocks` | int | 120 | Number of stocks to randomly sample per trial. |
| `--mc-top-ks` | int[] | [1,2,3,4,5,10,15,20,30,50] | Top-k values to test in Monte Carlo. Tests strategy performance for different portfolio sizes. |

### Progressive Evaluation

These arguments control evaluation and logging during training.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval-every-n-steps` | int | 100 | Evaluate on validation set every N optimizer steps. Set to 0 to disable progressive evaluation. |
| `--loss-smoothing-window` | int | 50 | EMA window size for smoothed loss curves in plots. |
| `--save-loss-curves-every` | int | 100 | Save loss curve PNG every N steps. Set to 0 to only save at end of training. |

### Incremental Training

Incremental training reuses the previous fold's model weights instead of training from scratch.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--incremental` | flag | False | Enable incremental training. Loads previous fold checkpoint and fine-tunes on new data. |
| `--incremental-epochs` | int | 2 | Number of epochs for incremental training. Typically fewer than full training since starting from pretrained weights. |
| `--incremental-data-fraction` | float | 1.0 | Fraction of new data to use in incremental training (0.0-1.0). |

### Checkpoint Saving

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save-intermediate-checkpoints` | flag | False | Save checkpoints at regular intervals throughout training. Useful for analyzing training progression. |
| `--checkpoint-every-n-epochs` | int | 1 | Save intermediate checkpoint every N epochs (when `--save-intermediate-checkpoints` is enabled). |
| `--checkpoint-dir` | str | checkpoints/walk_forward | Directory to save model checkpoints. Automatically appended with loss type and seed. |
| `--output` | str | walk_forward_training_results.pt | Output file for training results (PyTorch format). JSON version saved alongside. |

### Evaluation Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--top-k` | int | 25 | Number of top stocks to select for portfolio. Affects both backtesting and turnover calculations. |
| `--horizon-idx` | int | 0 | Prediction horizon index to use for ranking (0=1-day, 1=5-day, 2=10-day, 3=20-day). |
| `--confidence-percentile` | float | 0.6 | Minimum confidence percentile for stock selection. Stocks below this confidence are filtered out. |
| `--subset-size` | int | 512 | Number of stocks to evaluate per day in backtesting. Reduces computation for large universes. |
| `--num-test-stocks` | int | 1000 | Maximum number of stocks to include in test set per fold. |
| `--max-eval-dates` | int | 60 | Maximum evaluation dates for principled evaluation. Randomly sampled if test period has more dates. |

### Transaction Costs

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--transaction-cost-bps` | float | 10.0 | Transaction cost in basis points per round-trip trade. 10 bps = 0.1%. See [Transaction Costs documentation](transaction_costs.md). Typical values: 5-20 bps (large-cap), 20-50 bps (mid-cap), 50-100+ bps (small-cap). |

### System Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | str | cuda | Device for training: `cuda`, `cpu`, or specific GPU like `cuda:0`. |
| `--seed` | int | 42 | Master random seed for full reproducibility. Controls all randomness (PyTorch, NumPy, Python random). |
| `--no-preload` | flag | False | Disable data preloading. Streams from HDF5 instead. Saves memory but slower. Use for large `--seq-len`. |
| `--num-workers` | int | 4 | Number of DataLoader workers for parallel data loading. Set to 0 for debugging hangs. |
| `--no-compile` | flag | False | Disable `torch.compile()`. Useful for debugging or when compilation fails. |
| `--compile-mode` | str | default | `torch.compile` mode: `default`, `reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs`. |
| `--ddp` | flag | False | Enable Distributed Data Parallel for multi-GPU training. Must be launched with `torchrun`. |

### Baseline Model Comparison

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--compare-models` | flag | False | Train and evaluate baseline models (Ridge, LightGBM, MLP) for comparison against transformer. |
| `--baseline-max-samples` | int | 50000 | Maximum training samples for baseline models. Limits memory usage for tree-based models. |

## Argument Interactions and Tips

### Effective Batch Size

The effective batch size is:
```
effective_batch = batch_size × gradient_accumulation_steps × num_gpus
```

For stable training, aim for effective batch sizes of 256-1024.

### Memory vs Speed Trade-offs

| Goal | Adjustments |
|------|-------------|
| Reduce memory | Lower `--batch-size`, increase `--gradient-accumulation-steps`, enable `--no-preload`, reduce `--seq-len` |
| Faster training | Increase `--batch-size`, use `--ddp` with multiple GPUs, enable `torch.compile` (default) |
| Faster experiments | Use `--data-fraction 0.1`, reduce `--epochs-per-fold`, reduce `--num-folds` |

### Checkpoint Directory Naming

The checkpoint directory is automatically extended based on configuration:
```
{checkpoint_dir}_{loss_type}_seed{seed}
```

Examples:
- `checkpoints/walk_forward_ce_seed42` - Classification with CE loss
- `checkpoints/walk_forward_mse_seed42` - Regression with MSE loss
- `checkpoints/walk_forward_ce+listnet_seed42` - CE + ListNet ranking loss
- `checkpoints/walk_forward_listnet_seed42` - Ranking-only with ListNet

### Recommended Configurations

**Quick Test Run:**
```bash
python -m training.walk_forward_training \
    --data data.h5 --num-folds 2 --epochs-per-fold 1 \
    --data-fraction 0.1 --no-monte-carlo
```

**Standard Training:**
```bash
python -m training.walk_forward_training \
    --data data.h5 --prices prices.h5 \
    --num-folds 10 --epochs-per-fold 5 \
    --hidden-dim 512 --num-layers 4 \
    --batch-size 128 --gradient-accumulation-steps 4
```

**Production Training:**
```bash
torchrun --nproc_per_node=4 -m training.walk_forward_training \
    --data data.h5 --prices prices.h5 \
    --num-folds 10 --epochs-per-fold 10 \
    --hidden-dim 768 --num-layers 6 \
    --batch-size 64 --gradient-accumulation-steps 8 \
    --ranking-loss-type listnet --ranking-loss-weight 0.1 \
    --transaction-cost-bps 15 \
    --ddp --compare-models
```

**Ablation Study:**
```bash
./sanity_checks/ablate.sh hidden-dim "256 512 768 1024" "0 1 2"
```
