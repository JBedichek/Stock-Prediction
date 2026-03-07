# Contrastive Pretraining for Multi-Modal Stock Prediction

## Overview

This document describes the contrastive learning implementation for unsupervised representation learning in the multi-modal stock prediction model. The approach uses a two-phase training strategy:

1. **Contrastive Pretraining**: Learn robust feature representations by maximizing agreement between augmented views of the same sample
2. **Supervised Finetuning**: Freeze the encoder and train only the prediction heads on the downstream task

This approach is motivated by the inherent noise in financial return prediction. By first learning general-purpose representations of market data without relying on noisy return labels, the model can capture meaningful structure in the data that transfers to the prediction task.

## Theoretical Background

### Why Contrastive Learning for Finance?

Stock return prediction suffers from several challenges that make contrastive pretraining attractive:

1. **Low Signal-to-Noise Ratio**: Daily returns have notoriously low predictability (R² often < 1%). Learning representations from returns directly may overfit to noise.

2. **Cross-Sectional Structure**: Stocks within the same sector, market cap, or factor exposure share similar characteristics. Contrastive learning can capture this structure.

3. **Temporal Dynamics**: Market regimes change over time. Representations learned from the data structure (rather than labels) may generalize better across regimes.

4. **Multi-Modal Fusion**: The model combines fundamentals, technicals, and news. Contrastive learning provides a principled way to learn joint representations across modalities.

### InfoNCE Loss

The implementation uses the InfoNCE (Noise-Contrastive Estimation) loss from the SimCLR framework:

```
L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```

Where:
- `z_i, z_j` are embeddings from two augmented views of the same sample (positive pair)
- `z_k` are embeddings from all other samples in the batch (negative pairs)
- `sim(·,·)` is cosine similarity
- `τ` is the temperature parameter

**Temperature Interpretation**:
- Lower temperature (e.g., 0.05): Harder negatives, more discriminative but may collapse
- Higher temperature (e.g., 0.5): Softer negatives, more stable but less discriminative
- Default (0.1): Good balance for most cases

## Architecture

### ContrastiveMultiModalModel

The contrastive wrapper extends the base `MultiModalStockPredictor` with additional components:

```
Input (batch, seq_len, features)
         │
         ▼
┌─────────────────────┐
│  TimeSeriesAugmenter │ ──► Two augmented views
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ View1 │ │ View2 │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌─────────────────────┐
│ MultiModalEncoder   │
│  ├─ FundamentalsEnc │
│  ├─ TechnicalEnc    │
│  ├─ NewsEncoder     │
│  └─ CrossModalFusion│
└─────────────────────┘
    │         │
    ▼         ▼
┌─────────────────────┐
│ ContrastiveProjection│ ──► L2-normalized embeddings
└─────────────────────┘
    │         │
    ▼         ▼
┌─────────────────────┐
│    InfoNCE Loss     │
└─────────────────────┘
```

### Component Details

#### 1. TimeSeriesAugmenter

Applies domain-appropriate augmentations for financial time series:

```python
class TimeSeriesAugmenter(nn.Module):
    def __init__(
        self,
        temporal_jitter: int = 5,      # Max timesteps to shift
        feature_mask_prob: float = 0.15,  # Probability of masking features
        noise_std: float = 0.05,       # Gaussian noise std
        crop_ratio: float = 0.8,       # Subsequence crop ratio (not used)
    )
```

**Augmentation Strategies**:

| Augmentation | Description | Rationale |
|--------------|-------------|-----------|
| Temporal Jitter | Roll sequence by ±N timesteps | Stock patterns are approximately time-translation invariant over short periods |
| Feature Masking | Randomly zero 15% of features | Forces model to not rely on any single feature; improves robustness |
| Gaussian Noise | Add N(0, 0.05) noise | Handles measurement noise in financial data |

**Why These Augmentations?**

Unlike image augmentation (rotation, color jitter), financial data requires domain-specific augmentations:

- **Temporal Jitter**: A bullish pattern starting on Monday vs Tuesday should produce similar representations
- **Feature Masking**: If RSI is unavailable, the model should still work from other indicators
- **Noise Injection**: Financial data contains measurement errors and bid-ask bounce

#### 2. ContrastiveProjectionHead

Projects encoder representations to a lower-dimensional space for contrastive learning:

```python
class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,    # Combined encoder output dimension
        hidden_dim: int = 256,
        output_dim: int = 128,  # Contrastive embedding dimension
    )
```

Architecture:
```
Linear(input_dim, hidden_dim) → BatchNorm → ReLU
Linear(hidden_dim, hidden_dim) → BatchNorm → ReLU
Linear(hidden_dim, output_dim) → L2Normalize
```

**Design Choices**:

1. **Three-layer MLP**: Follows SimCLR finding that deeper projection heads improve representation quality
2. **BatchNorm**: Stabilizes training and prevents mode collapse
3. **L2 Normalization**: Projects to unit hypersphere for cosine similarity
4. **Lower output dim (128)**: Contrastive loss works well in lower dimensions; encoder representations are higher-dimensional

#### 3. Encoder Architecture (Inherited)

The encoder is the full `MultiModalStockPredictor` without prediction heads:

| Component | Input Dim | Hidden Dim | Output Dim |
|-----------|-----------|------------|------------|
| FundamentalsEncoder | 224 | hidden_dim | hidden_dim/4 |
| TechnicalEncoder | 126 | hidden_dim | hidden_dim/2 |
| NewsEncoder | 770 | hidden_dim | hidden_dim/4 |
| CrossModalFusion | varies | hidden_dim | hidden_dim |

Combined output dimension: `hidden_dim + hidden_dim/4 + hidden_dim/2 + hidden_dim/4 = 2 * hidden_dim`

For `hidden_dim=256`: combined_dim = 512
For `hidden_dim=512`: combined_dim = 1024

## Training Procedure

### Two-Phase Training

#### Phase 1: Contrastive Pretraining

```python
for epoch in range(contrastive_epochs):
    for batch in train_loader:
        # Create two augmented views
        x1 = augmenter(batch)
        x2 = augmenter(batch)

        # Get representations
        h1 = encoder(x1)
        h2 = encoder(x2)

        # Project to contrastive space
        z1 = projection_head(h1)
        z2 = projection_head(h2)

        # Compute InfoNCE loss
        loss = info_nce_loss(z1, z2, temperature=0.1)

        # Update all parameters
        loss.backward()
        optimizer.step()
```

**Optimizer Configuration**:
- Optimizer: AdamW with weight_decay=0.01
- Learning Rate: 1e-4 (configurable via `--contrastive-lr`)
- Scheduler: CosineAnnealingLR over contrastive epochs
- Gradient Clipping: max_norm=1.0

#### Phase 2: Supervised Finetuning

After pretraining, the encoder is optionally frozen:

```python
def freeze_encoder(self):
    """Freeze encoder weights for finetuning."""
    for name, param in self.encoder.named_parameters():
        if 'pred_head' not in name and 'confidence_head' not in name:
            param.requires_grad = False

    # Also freeze projection head (not needed for prediction)
    for param in self.projection_head.parameters():
        param.requires_grad = False
```

This leaves only the prediction and confidence heads trainable:
- `pred_head`: Classification/regression output
- `confidence_head`: Prediction confidence scores

**Parameter Count Example (hidden_dim=256)**:
- Total parameters: ~4.8M
- Frozen after pretraining: ~4.7M (encoder + projection)
- Trainable for finetuning: ~0.1M (prediction heads only)

### Loss Function Implementation

```python
def info_nce_loss(
    z1: torch.Tensor,  # (batch, dim) normalized embeddings view 1
    z2: torch.Tensor,  # (batch, dim) normalized embeddings view 2
    temperature: float = 0.1,
) -> torch.Tensor:
    batch_size = z1.shape[0]

    # Concatenate both views: (2*batch, dim)
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix: (2*batch, 2*batch)
    sim = torch.mm(z, z.t()) / temperature

    # Mask self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))

    # Labels: positive pairs are (i, i+batch) and (i+batch, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ])

    # Cross-entropy over similarity scores
    loss = F.cross_entropy(sim, labels)
    return loss
```

**Complexity Analysis**:
- Similarity matrix: O(batch² × dim)
- Memory: O(batch²) for similarity matrix
- Recommendation: batch_size ≤ 512 for reasonable memory usage

## Integration with Walk-Forward Training

The contrastive pretraining is integrated into the `_train_fold` method:

```
For each fold:
    1. Create train/val dataloaders

    2. If contrastive_pretrain enabled:
        a. Run contrastive pretraining for contrastive_epochs
        b. Optionally freeze encoder

    3. Run supervised training for epochs_per_fold

    4. Evaluate on test set
```

This ensures that:
- Each fold gets its own pretrained representations (no data leakage)
- Pretraining uses only training data from that fold
- Representations are tuned to the specific time period

## Usage

### Command Line

```bash
python -m training.walk_forward_training \
    --data all_complete_dataset.h5 \
    --prices actual_prices_clean.h5 \
    --model-type multimodal \
    --hidden-dim 512 \
    --pred-mode classification \
    --contrastive-pretrain \
    --contrastive-epochs 5 \
    --contrastive-lr 1e-4 \
    --contrastive-temperature 0.1 \
    --epochs-per-fold 3 \
    --batch-size 128 \
    --seq-len 256 \
    --device cuda:0
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contrastive-pretrain` | flag | False | Enable contrastive pretraining |
| `--contrastive-epochs` | int | 5 | Number of pretraining epochs |
| `--contrastive-lr` | float | 1e-4 | Learning rate for pretraining |
| `--contrastive-temperature` | float | 0.1 | InfoNCE temperature |
| `--no-freeze-encoder` | flag | False | Keep encoder trainable after pretraining |

### Programmatic Usage

```python
from training.multimodal_model import (
    ContrastiveMultiModalModel,
    FeatureConfig,
    pretrain_contrastive,
)

# Create model
model = ContrastiveMultiModalModel(
    feature_config=FeatureConfig(),
    hidden_dim=512,
    projection_dim=128,
    num_technical_layers=4,
    num_technical_heads=4,
    pred_mode='classification',
    temperature=0.1,
)

# Pretrain (uses helper function)
model = pretrain_contrastive(
    model=model,
    dataloader=train_loader,
    num_epochs=5,
    lr=1e-4,
    device=torch.device('cuda'),
)

# Freeze encoder for finetuning
model.freeze_encoder()

# Now train prediction heads
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

## Hyperparameter Guidelines

### Temperature Selection

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.05 | Very hard negatives, risk of collapse | Large batches (512+), well-tuned augmentations |
| 0.1 | Balanced (default) | Most cases |
| 0.2 | Softer negatives | Small batches, high noise |
| 0.5 | Very soft | Debugging, unstable training |

### Batch Size Considerations

Contrastive learning benefits from larger batch sizes (more negatives):

| Batch Size | Effective Negatives | Recommendation |
|------------|---------------------|----------------|
| 32 | 62 | Too small, may not work well |
| 128 | 254 | Minimum recommended |
| 256 | 510 | Good default |
| 512 | 1022 | Best quality if memory allows |

### Pretraining Epochs

| Epochs | Observation |
|--------|-------------|
| 1-2 | Minimal benefit, representations not converged |
| 3-5 | Good balance of quality vs compute |
| 10+ | Diminishing returns, risk of overfitting augmentations |

### Encoder Freezing

| Strategy | Pros | Cons |
|----------|------|------|
| Freeze (default) | Prevents forgetting learned representations, faster finetuning | Less flexibility for task adaptation |
| No freeze | Full model adaptation | May overwrite good representations |

**Recommendation**: Start with frozen encoder. If performance is poor, try unfreezing with very low learning rate (1e-5).

## Monitoring Training

### Contrastive Loss Interpretation

| Loss Value | Interpretation |
|------------|----------------|
| ~log(batch_size) | Random initialization (e.g., ~5.5 for batch=256) |
| 3.0-4.0 | Learning, representations separating |
| 1.5-2.5 | Good representations |
| < 1.0 | Potential collapse, check augmentations |

### Signs of Mode Collapse

1. Loss drops very quickly then plateaus
2. All embeddings have high cosine similarity (> 0.9)
3. Downstream performance is random

**Solutions**:
- Increase temperature
- Reduce augmentation strength
- Check for bugs in augmentation (all views identical?)

## Theoretical Considerations

### Relationship to Other Methods

| Method | Positive Pairs | Negatives | Key Difference |
|--------|----------------|-----------|----------------|
| SimCLR (this impl.) | Augmented views | All other samples | Simple, effective |
| MoCo | Augmented views | Memory bank | Larger effective batch |
| BYOL | Augmented views | None (momentum encoder) | No negatives needed |
| Barlow Twins | Augmented views | None (redundancy reduction) | Cross-correlation objective |

### Why Not BYOL/Barlow Twins?

These methods don't require negatives but:
1. More complex architecture (momentum encoder, asymmetric design)
2. Require careful tuning to prevent collapse
3. SimCLR is simpler and works well with sufficient batch size

### Financial-Specific Considerations

1. **Cross-Sectional Negatives**: Stocks on the same day provide natural hard negatives (similar market conditions, different outcomes)

2. **Temporal Negatives**: Same stock on different days captures regime changes

3. **Sector Structure**: The model should learn that tech stocks are more similar to each other than to utilities

## Future Extensions

### Potential Improvements

1. **Momentum Encoder (MoCo)**: Maintain a queue of embeddings for more negatives without memory overhead

2. **Multi-Crop Strategy**: Use multiple augmentation strengths (global + local views)

3. **Cross-Modal Contrastive Loss**: Explicitly align modalities (e.g., technicals should predict fundamentals)

4. **Temporal Contrastive Loss**: Same stock at nearby times should be similar (but not identical due to information decay)

5. **Supervised Contrastive Loss**: Use return buckets as labels to create "soft" positives

### Research Directions

1. **Optimal Augmentations**: Which augmentations preserve financial meaning while creating useful variation?

2. **Negative Sampling**: Should negatives be random or strategically selected (e.g., same sector)?

3. **Pretraining Scope**: Pretrain once on all data or per-fold? Trade-off between compute and freshness.

## File Locations

| Component | Location |
|-----------|----------|
| Contrastive model | `training/multimodal_model.py` |
| TimeSeriesAugmenter | `training/multimodal_model.py` |
| Integration | `training/walk_forward_training.py:2788-2860` |
| CLI arguments | `training/walk_forward_training.py:4108-4120` |

## References

1. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), 2020
2. He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo), 2020
3. Grill et al., "Bootstrap Your Own Latent" (BYOL), 2020
4. Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", 2021
