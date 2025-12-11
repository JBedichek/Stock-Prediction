# Selection-Aware Training

## Overview

**Selection-aware training** focuses the model on what actually matters in practice: correctly ranking the stocks it predicts will perform best.

Since the model only trades its top predictions, traditional training on all examples equally wastes compute on stocks that would never be selected. This approach iteratively focuses on the model's current top picks.

## Algorithm

```
For each iteration:
  1. Sample 10,000 random sequences from training set
  2. Run inference to get model predictions
  3. Calculate expected return for each sequence
  4. Select top 100 (1%) with highest expected return
  5. Train on those 100 for 3 epochs
  6. Repeat
```

## Why This Works

### Problem with Standard Training
- Standard training treats all examples equally
- But in practice, you only trade the top 0.1-1% of predictions
- Model wastes capacity learning to rank stocks it would never select

### Solution: Hard Example Mining on Top Predictions
- **Focuses on discrimination at the high end**: Where it matters most
- **Self-paced learning**: Model identifies its own high-confidence predictions
- **Curriculum learning**: As model improves, top predictions change dynamically
- **Efficient**: Only trains on 1% of data per iteration (100x less than full batch)

### Comparison to Other Approaches

| Approach | What it learns | Efficiency | Task alignment |
|----------|---------------|-----------|----------------|
| **Standard training** | Rank all stocks equally | Low (trains on everything) | Poor (wastes compute on never-traded stocks) |
| **Random sampling** | Random subset | Medium | Poor |
| **Hard example mining (worst predictions)** | Fix errors everywhere | Medium | Medium |
| **Selection-aware (this)** | Discriminate among top performers | High | **Perfect** (trains on exactly what matters) |

## Usage

### Quick Start

```bash
python training/run_selection_aware_training.py
```

### Custom Configuration

```bash
python training/selection_aware_training.py \
    --data all_complete_dataset.h5 \
    --sample-size 10000 \
    --top-k 100 \
    --train-epochs-per-iteration 3 \
    --num-iterations 100 \
    --horizon-idx 1 \
    --lr 4e-5 \
    --batch-size 32
```

### Arguments

**Data:**
- `--data`: Dataset path (HDF5)
- `--seq-len`: Sequence length (default: 2000)

**Selection:**
- `--sample-size`: Sequences to sample each iteration (default: 10000)
- `--top-k`: Top sequences to keep (default: 100, i.e., 1%)
- `--horizon-idx`: Prediction horizon for selection (0=1d, 1=5d, 2=10d, 3=20d)

**Training:**
- `--train-epochs-per-iteration`: Epochs per subset (default: 3)
- `--num-iterations`: Total iterations (default: 100)
- `--lr`: Learning rate (default: 4e-5)
- `--batch-size`: Training batch size (default: 32)
- `--inference-batch-size`: Inference batch size for selection (default: 64)

**Model:**
- `--hidden-dim`: Hidden dimension (default: 1024)
- `--num-layers`: Transformer layers (default: 16)
- `--num-heads`: Attention heads (default: 16)
- `--dropout`: Dropout rate (default: 0.15)

**Checkpointing:**
- `--save-dir`: Checkpoint directory (default: ./checkpoints)
- `--resume-from`: Resume from checkpoint
- `--val-every`: Validate every N iterations (default: 5)

**Other:**
- `--use-amp`: Enable mixed precision (default: True)
- `--use-wandb`: Enable W&B logging (default: True)
- `--device`: cuda or cpu (default: cuda)

## Output

### Console Output

```
================================================================================
SELECTION-AWARE TRAINING
================================================================================

Strategy:
  Sample 10000 sequences
  Select top 100 (1.0%) by expected return
  Train on them for 3 epochs
  Repeat 100 times

📦 Loading data...
  Train: 1,234,567 sequences
  Val:   1,000 sequences

================================================================================
Iteration 1/100
================================================================================

  🎲 Sampling 10000 sequences and selecting top 100...
    📊 Expected return stats:
      Mean: 1.0023
      Median: 1.0015
      Top 100 mean: 1.0456
      Top 100 min: 1.0312

  🏋️  Training on top 100 for 3 epochs...
  ✅ Training loss: 2.3456

  📊 Validating...
  ✅ Validation loss: 2.4123
  ✅ Saved best model (val_loss: 2.4123)
```

### Saved Files

1. **`checkpoints/best_model_selection_aware.pt`** - Best model by validation loss
2. **`checkpoints/checkpoint_iter_10.pt`** - Periodic checkpoints every 10 iterations
3. **Weights & Biases logs** - Training curves, losses, metrics

## Expected Results

### Training Dynamics

**Early iterations (1-20):**
- Model learns to identify obvious winners
- Expected return of top-100 improves rapidly
- Validation loss decreases quickly

**Mid iterations (20-60):**
- Model refines discrimination among good stocks
- Top-100 expected return stabilizes at higher level
- Validation loss continues gradual improvement

**Late iterations (60-100):**
- Fine-tuning on subtle differences
- Top-100 expected return plateaus
- Risk of overfitting on top performers

### Performance Comparison

Compared to standard training, selection-aware training should show:

✅ **Better top-k accuracy**: More accurate predictions for top 1-5%
✅ **Higher backtest Sharpe**: Better performance in actual trading simulation
✅ **Faster convergence**: Reaches good performance in fewer iterations
❌ **Worse overall accuracy**: May perform worse on bottom 90% (but we don't care!)

## Implementation Details

### Expected Return Calculation

```python
# Convert model's probability distribution to expected value
probs = softmax(logits)  # (batch, num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
expected_return = sum(probs * bin_centers)
```

### Selection Strategy

1. **Sample randomly**: Ensures diversity, prevents overfitting to fixed subset
2. **Rank by expected return**: Model's belief about future performance
3. **Keep top-k**: Only train on highest predictions
4. **Repeat**: New sample each iteration keeps training dynamic

### Training on Subset

- **Small batch size**: 100 examples can use batch_size=32
- **3 epochs**: Enough to update without overfitting
- **Standard loss**: CrossEntropyLoss on all horizons

## Hyperparameter Tuning

### Sample Size vs Top-K Ratio

| Sample Size | Top-K | Ratio | Trade-off |
|-------------|-------|-------|-----------|
| 5,000 | 50 | 1% | Faster, less diverse |
| 10,000 | 100 | 1% | **Balanced (default)** |
| 20,000 | 200 | 1% | Slower, more diverse |
| 10,000 | 50 | 0.5% | More extreme, may overfit |
| 10,000 | 500 | 5% | Less extreme, more stable |

**Recommendation**: Start with default (10k/100), increase if overfitting.

### Epochs Per Iteration

| Epochs | Trade-off |
|--------|-----------|
| 1 | Fast, underfits subset |
| 3 | **Balanced (default)** |
| 5 | Slower, risk of overfitting subset |
| 10 | High risk of overfitting |

**Recommendation**: Use 3 epochs unless validation shows underfitting.

## Monitoring Training

### Key Metrics

1. **Top-100 mean expected return**: Should increase over iterations
2. **Top-100 min expected return**: Lower bound should also increase
3. **Validation loss**: Should decrease (but may not match standard training)
4. **Training loss on subset**: Should be low (<1.5 for classification)

### Warning Signs

❌ **Top-100 expected return not increasing**: Model not improving discrimination
❌ **Training loss not decreasing**: Learning rate too high or subset too hard
❌ **Validation loss increasing**: Overfitting to top performers
❌ **Top-100 min too high**: Possible numerical issues or data leakage

## Advanced: Combining with Standard Training

For best results, consider a **two-stage approach**:

```bash
# Stage 1: Standard training (50 epochs)
python training/train_new_format.py --epochs 50

# Stage 2: Selection-aware fine-tuning (50 iterations)
python training/selection_aware_training.py \
    --resume-from checkpoints/best_model.pt \
    --num-iterations 50
```

This gives:
- **Broad coverage** from standard training
- **Sharp discrimination** from selection-aware training

## Limitations

1. **May sacrifice overall accuracy**: Focuses on top predictions, may neglect bottom 99%
2. **Requires good initialization**: If random init predicts poorly, selection is meaningless
3. **Computationally intensive**: 2x overhead (inference for selection + training)
4. **Risk of overfitting**: Training repeatedly on similar high-performers

## Troubleshooting

**Issue**: Top-100 expected return not improving
- **Solution**: Increase sample size for more diversity
- **Solution**: Check if model is learning at all (training loss decreasing?)

**Issue**: Validation loss increasing while training loss decreases
- **Solution**: Reduce epochs per iteration (3→1)
- **Solution**: Add validation-based early stopping
- **Solution**: Increase regularization (dropout, weight decay)

**Issue**: Out of memory during inference
- **Solution**: Reduce `--inference-batch-size`
- **Solution**: Reduce `--sample-size`

**Issue**: Training too slow
- **Solution**: Reduce `--sample-size` (10000→5000)
- **Solution**: Use larger `--inference-batch-size` if memory allows
- **Solution**: Enable `--use-amp` for 2x speedup

## Future Extensions

Possible enhancements:
- [ ] **Stratified sampling**: Ensure diversity across sectors/market caps
- [ ] **Dynamic top-k**: Increase k as model improves
- [ ] **Negative mining**: Also train on worst predictions to avoid them
- [ ] **Multi-objective**: Balance top-performer accuracy with overall accuracy
- [ ] **Confidence weighting**: Weight loss by prediction confidence
- [ ] **Temporal consistency**: Prefer stocks that stay in top-k over multiple iterations
