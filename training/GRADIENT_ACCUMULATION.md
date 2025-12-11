# Gradient Accumulation Guide

## What is Gradient Accumulation?

Gradient accumulation is a technique to **simulate larger batch sizes** without actually loading more data into GPU memory. It works by:

1. Running multiple forward passes with small batches
2. Accumulating gradients across these batches
3. Only updating weights after N batches (accumulation steps)

This allows you to train with **effective batch sizes larger than your GPU memory allows**.

## When to Use It

Use gradient accumulation when:
- ✅ You want larger batch sizes but have limited GPU memory
- ✅ You're using large models (like the 2B parameter transformer)
- ✅ You have long sequences (seq_len=2000)
- ✅ You want to improve training stability with larger batches

## How to Use

### Command Line

```bash
# Example: Batch size 2, accumulate 8 steps = effective batch size 16
python -m training.train_new_format \
    --data dataset.pkl \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --use-amp

# Effective batch size = 2 × 8 = 16
```

### Common Configurations

| GPU Memory | Batch Size | Accum Steps | Effective Batch Size |
|------------|------------|-------------|---------------------|
| 24 GB      | 1          | 32          | 32                  |
| 24 GB      | 2          | 16          | 32                  |
| 40 GB      | 2          | 16          | 32                  |
| 40 GB      | 4          | 8           | 32                  |
| 80 GB      | 8          | 4           | 32                  |

**Rule of thumb:** Keep effective batch size around 32-64 for stable training.

## Memory vs Speed Tradeoff

### More Accumulation Steps (e.g., batch_size=1, accum_steps=32)
- ✅ **Less GPU memory** - can fit larger models
- ✅ **Longer sequences** - can use seq_len=2000+
- ❌ **Slower** - more forward/backward passes per update

### Fewer Accumulation Steps (e.g., batch_size=8, accum_steps=4)
- ✅ **Faster** - fewer passes per update
- ❌ **More GPU memory** - needs larger GPU
- ❌ **Shorter sequences** - may need to reduce seq_len

## Training Output

When you start training, you'll see:

```
🚀 Starting training...
  Epochs: 10
  Optimizer: lion
  Learning rate: 1e-05
  Weight decay: 0.01
  Device: cuda
  Batch size: 2
  Gradient accumulation steps: 16
  Effective batch size: 32 (2 × 16)  ← This is what matters!
  Mixed precision (FP16): ✅ Enabled
```

## How It Works Internally

```python
# Without gradient accumulation (batch_size=32)
for batch in dataloader:  # batch_size=32
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Update every batch

# With gradient accumulation (batch_size=2, accum_steps=16)
for i, batch in enumerate(dataloader):  # batch_size=2
    loss = model(batch) / 16  # Scale loss
    loss.backward()  # Accumulate gradients

    if (i + 1) % 16 == 0:  # Every 16 batches
        optimizer.step()  # Update with accumulated gradients
        optimizer.zero_grad()

# Result: Same effective batch size (32), less GPU memory!
```

## Benefits

1. **Larger effective batch sizes** without OOM errors
2. **More stable training** - larger batches = less noisy gradients
3. **Better convergence** - especially for large models
4. **Flexibility** - trade memory for speed as needed

## Tips

1. **Start with accum_steps=1** and increase if you get OOM errors
2. **Monitor GPU memory** with `nvidia-smi` to find optimal batch size
3. **Keep effective_batch_size consistent** across runs for fair comparison
4. **Adjust learning rate** if you change effective batch size significantly
   - Rule of thumb: `lr ∝ sqrt(batch_size)`
   - If you double batch size, multiply lr by √2 ≈ 1.4

## Example Workflow

```bash
# Step 1: Try without accumulation
python -m training.train_new_format --batch-size 8 --grad-accum-steps 1

# If OOM: Reduce batch size, increase accumulation
python -m training.train_new_format --batch-size 4 --grad-accum-steps 2

# Still OOM: Reduce more
python -m training.train_new_format --batch-size 2 --grad-accum-steps 4

# Still OOM: Go minimal
python -m training.train_new_format --batch-size 1 --grad-accum-steps 8

# Note: All have effective_batch_size = 8
```

## Advanced: Combined with Mixed Precision

For **maximum efficiency**, combine with `--use-amp`:

```bash
python -m training.train_new_format \
    --batch-size 2 \
    --grad-accum-steps 16 \
    --use-amp \
    --compile

# Benefits:
# - Mixed precision: 2x speedup + 50% less memory
# - Gradient accumulation: Larger effective batch (32)
# - torch.compile: Additional 20-40% speedup
# = Train large models fast!
```

## Monitoring

The training logs will show:
- Individual batch loss (per mini-batch)
- EMA smoothed loss (for stable monitoring)
- Global step counter (increments every mini-batch, not every weight update)

Note: With `grad_accum_steps=16`, weight updates happen every 16 global steps.

## Conclusion

Gradient accumulation is a **essential technique** for training large models on consumer GPUs. It allows you to:
- Train models that wouldn't fit otherwise
- Use optimal batch sizes for convergence
- Trade speed for memory efficiency

Start with the defaults and adjust based on your GPU memory!
