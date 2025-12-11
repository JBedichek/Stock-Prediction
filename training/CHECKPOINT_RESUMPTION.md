# Checkpoint Resumption Guide

## Overview

The training script automatically saves checkpoints during training and supports **resuming from any saved checkpoint**. This is useful for:

- ✅ **Resuming interrupted training** - Continue where you left off after crashes/interruptions
- ✅ **Fine-tuning models** - Start from a pretrained checkpoint with different hyperparameters
- ✅ **Continuing training** - Add more epochs to an already trained model
- ✅ **Experimenting** - Try different learning rates or other parameters on top of a checkpoint

## Quick Start

### Resume Training

```bash
# Continue training from a saved checkpoint
python -m training.train_new_format \
    --data dataset.pkl \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --epochs 20  # Train for 20 total epochs (will continue from checkpoint epoch)
```

### Fine-Tune with Different Learning Rate

```bash
# Resume but with a lower learning rate for fine-tuning
python -m training.train_new_format \
    --data dataset.pkl \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --lr 1e-6 \
    --epochs 15
```

## How Checkpoints Work

### What Gets Saved

Every checkpoint saves:
- ✅ **Model weights** (`model_state_dict`)
- ✅ **Optimizer state** (`optimizer_state_dict`) - Momentum, learning rate schedule, etc.
- ✅ **Gradient scaler state** (`scaler_state_dict`) - For mixed precision training
- ✅ **Training state**:
  - Current epoch number
  - Global step counter
  - Best validation loss so far
  - Training EMA loss
- ✅ **Configuration** (`config`) - All hyperparameters used

### When Checkpoints Are Saved

Checkpoints are saved in **two ways**:

1. **Best model checkpoint** - Saved whenever validation loss improves
   - Location: `./checkpoints/best_model.pt`
   - This is the checkpoint you usually want to resume from

2. **Step-based checkpoints** - Saved every `eval_every` steps during validation
   - Also saves when validation improves during step-based eval
   - Allows resumption from mid-epoch

## Usage Examples

### Example 1: Training Interrupted

```bash
# Start training
python -m training.train_new_format \
    --data dataset.pkl \
    --epochs 10 \
    --lr 1e-4

# Training crashes at epoch 5!
# Resume from where it stopped:
python -m training.train_new_format \
    --data dataset.pkl \
    --epochs 10 \
    --resume-from-checkpoint ./checkpoints/best_model.pt
```

**Output:**
```
📂 Loading checkpoint from: ./checkpoints/best_model.pt
  ✅ Loaded model weights
  ✅ Loaded optimizer state
  ✅ Loaded gradient scaler state
  ✅ Resuming from epoch 5
  ✅ Resuming from global step 2500
  ✅ Best val loss: 2.345678
  ✅ Train EMA loss: 2.401234

  🎯 Checkpoint loaded successfully!

🚀 Starting training...
  Epochs: 10 (starting from epoch 6)
  ...
```

### Example 2: Add More Epochs

```bash
# Model trained for 10 epochs, want to train for 5 more
python -m training.train_new_format \
    --data dataset.pkl \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --epochs 15  # Will train epochs 11-15
```

### Example 3: Fine-Tuning

```bash
# Fine-tune with lower learning rate and different dropout
python -m training.train_new_format \
    --data dataset.pkl \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --lr 5e-6 \
    --dropout 0.05 \
    --epochs 20
```

**Note:** Model architecture must match! You can change:
- Learning rate
- Weight decay
- Dropout (if not using torch.compile)
- Batch size / gradient accumulation
- Data augmentation settings

You cannot change:
- Model size (hidden_dim, num_layers, etc.)
- Input/output dimensions

## Advanced: Loading from Different Experiment

```bash
# Load checkpoint from a different experiment directory
python -m training.train_new_format \
    --data new_dataset.pkl \
    --resume-from-checkpoint /path/to/other/experiment/best_model.pt \
    --save-dir ./new_experiment \
    --epochs 10
```

## Checkpoint File Structure

A checkpoint file (`.pt`) contains:

```python
{
    'epoch': 5,                        # Epoch number when saved
    'global_step': 2500,               # Total training steps
    'model_state_dict': {...},         # Model weights
    'optimizer_state_dict': {...},     # Optimizer state (momentum, etc.)
    'scaler_state_dict': {...},        # Mixed precision scaler (if using AMP)
    'val_loss': 2.345678,              # Best validation loss
    'train_ema_loss': 2.401234,        # Training EMA loss
    'config': {                        # All hyperparameters
        'batch_size': 2,
        'lr': 0.0001,
        'hidden_dim': 2048,
        ...
    }
}
```

## Tips & Best Practices

### 1. **Always resume from `best_model.pt`**
This is the checkpoint with the lowest validation loss:
```bash
--resume-from-checkpoint ./checkpoints/best_model.pt
```

### 2. **Check what epoch you're resuming from**
The script will print:
```
✅ Resuming from epoch 5
```

### 3. **Adjust `--epochs` accordingly**
If you resume from epoch 5 and want to train 5 more epochs, set `--epochs 10`.

### 4. **Optimizer state is preserved**
- Momentum/Adam statistics are restored
- Learning rate schedule continues from where it left off
- This gives better training stability

### 5. **Compatible with all training features**
Resume works with:
- ✅ Mixed precision (`--use-amp`)
- ✅ Gradient accumulation (`--grad-accum-steps`)
- ✅ torch.compile (`--compile`)
- ✅ Wandb logging (`--use-wandb`)

### 6. **Checkpoint not found warning**
If checkpoint file doesn't exist:
```
⚠️  WARNING: Checkpoint file not found: ./checkpoints/best_model.pt
  Starting training from scratch...
```

### 7. **Model architecture must match**
If you get an error like:
```
RuntimeError: Error(s) in loading state_dict...
```

This means the checkpoint model architecture doesn't match your current model. Make sure:
- `--hidden-dim` matches
- `--num-layers` matches
- `--num-heads` matches
- Input feature dimensions match

## Wandb Integration

When resuming from checkpoint:
- Wandb will create a new run (not resume the old run)
- You can manually link runs in wandb UI
- Or use `wandb.init(id=run_id, resume="must")` to resume the same run (requires code modification)

## Common Scenarios

### Scenario 1: Out of Memory During Training

```bash
# Training fails with OOM at epoch 3
# Solution: Resume with smaller batch size

python -m training.train_new_format \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --batch-size 1 \
    --grad-accum-steps 64 \
    --epochs 10
```

### Scenario 2: Learning Rate Too High

```bash
# Loss exploded, reduce learning rate and resume

python -m training.train_new_format \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --lr 1e-6 \
    --epochs 15
```

### Scenario 3: Add More Training

```bash
# Model plateaued at epoch 10, train 5 more with lower LR

python -m training.train_new_format \
    --resume-from-checkpoint ./checkpoints/best_model.pt \
    --lr 5e-6 \
    --epochs 15  # Will run epochs 11-15
```

## Troubleshooting

### Issue: "Checkpoint file not found"
**Solution:** Check that the path is correct:
```bash
ls -lh ./checkpoints/best_model.pt
```

### Issue: "Error loading state_dict"
**Solution:** Model architecture mismatch. Use the same model hyperparameters:
```bash
# Must match the original training command
--hidden-dim 2048 \
--num-layers 24 \
--num-heads 16
```

### Issue: Training starts from epoch 0
**Solution:** Make sure checkpoint has 'epoch' field. Older checkpoints might not have it. The script will start from epoch 0 but with trained weights.

### Issue: Different dataset
**Solution:** You can resume with a different dataset, but:
- Feature dimensions must match
- Model will continue training on new data
- This is essentially transfer learning

## Conclusion

Checkpoint resumption is a **critical feature** for:
- Long training runs that might get interrupted
- Iterative experimentation with hyperparameters
- Fine-tuning pretrained models

Always use `--resume-from-checkpoint` when continuing training!
