# ListNet Loss in Walk-Forward Training

This document explains how the ListNet ranking loss works in `training/walk_forward_training.py` and how it integrates with the stock prediction training pipeline.

## Overview

ListNet is a **listwise learning-to-rank** algorithm that treats ranking as a probability distribution matching problem. Unlike pointwise methods (predicting individual scores) or pairwise methods (comparing pairs of items), ListNet considers the entire ranked list simultaneously.

In stock prediction, this is valuable because we care about **relative ranking** of stocks (which stocks will outperform others) rather than absolute return predictions.

## Mathematical Foundation

### Core Idea

Given a set of stocks on a particular day, ListNet:

1. Converts predicted scores into a **probability distribution** over stocks using softmax
2. Converts actual returns into a **target probability distribution** using softmax
3. Minimizes the **cross-entropy** between these two distributions

### Formulation

For a batch of `n` stocks with predicted scores `s = [s_1, s_2, ..., s_n]` and actual returns `y = [y_1, y_2, ..., y_n]`:

**Predicted distribution:**
```
P(i) = exp(s_i / τ) / Σ_j exp(s_j / τ)
```

**Target distribution:**
```
Q(i) = exp(y_i / τ) / Σ_j exp(y_j / τ)
```

**ListNet loss:**
```
L = -Σ_i Q(i) * log(P(i))
```

Where `τ` (temperature) controls the "softness" of the distribution:
- Lower τ → sharper distribution, more weight on top stocks
- Higher τ → softer distribution, more uniform weighting

## Implementation in walk_forward_training.py

### The `compute_listnet_loss` Function

Located at line 345:

```python
def compute_listnet_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_idx: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
```

**Key steps:**

1. **Extract prediction scores** from classification logits:
   ```python
   if predictions.dim() == 3:
       probs = F.softmax(predictions[:, :, horizon_idx], dim=1)
       bin_indices = torch.arange(probs.shape[1], device=probs.device)
       pred_scores = (probs * bin_indices).sum(dim=1)
   ```
   This converts bin probabilities to expected scores using bin indices as weights.

2. **Compute probability distributions:**
   ```python
   pred_dist = F.softmax(pred_scores / temperature, dim=0)
   target_dist = F.softmax(target_scores / temperature, dim=0)
   ```

3. **Calculate cross-entropy loss:**
   ```python
   loss = -torch.mean(target_dist * torch.log(pred_dist + 1e-8))
   ```

### Integration with Training Loop

The ranking loss (including ListNet) is computed and combined with the primary loss in the training loop (around line 2810):

```python
# Compute ranking loss
ranking_loss = compute_ranking_loss(
    pred, targets, bin_edges,
    horizon_idx=self.horizon_idx,
    loss_type=self.ranking_loss_type,  # 'listnet', 'pairwise', or 'correlation'
    margin=self.ranking_margin,
)

if self.ranking_only:
    # Use ONLY ranking loss
    loss = ranking_loss
else:
    # Standard CE loss for classification
    ce_loss = F.cross_entropy(pred[:, :, 0], bin_indices[:, 0])
    # ... sum across horizons ...

    # Add ranking loss if enabled
    if self.ranking_loss_weight > 0:
        loss = loss + self.ranking_loss_weight * ranking_loss
```

## Configuration Options

### Command Line Arguments

```bash
python training/walk_forward_training.py \
    --ranking-loss-type listnet \      # Use ListNet (vs 'pairwise' or 'correlation')
    --ranking-loss-weight 0.1 \        # Weight for ranking loss (0 = disabled)
    --ranking-only                      # Use ONLY ranking loss, no CE/MSE
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ranking-loss-type` | `pairwise` | Type of ranking loss: `pairwise`, `listnet`, or `correlation` |
| `--ranking-loss-weight` | `0.0` | Weight for ranking loss when combined with CE/MSE |
| `--ranking-only` | `False` | If set, use only ranking loss (ignore CE/MSE) |

## Comparison of Ranking Loss Types

### 1. ListNet (`listnet`)

**Pros:**
- Considers entire ranking simultaneously
- Differentiable everywhere
- Natural probabilistic interpretation
- Works well with small batches

**Cons:**
- Computationally more expensive than pairwise
- Sensitive to temperature hyperparameter

### 2. Pairwise (`pairwise`)

```python
# For each pair (i, j) where y_i > y_j:
loss = max(0, margin - (s_i - s_j))
```

**Pros:**
- Simple and intuitive
- Directly optimizes pairwise ordering

**Cons:**
- O(n²) pairs can be expensive
- Doesn't consider list-level properties

### 3. Correlation (`correlation`)

```python
# Negative Pearson correlation
loss = -corr(predictions, targets)
```

**Pros:**
- Directly optimizes Information Coefficient (IC)
- Scale-invariant

**Cons:**
- Can be unstable with small batches
- Assumes linear relationship

## How Model Predictions Become Ranking Scores

The `SimpleTransformerPredictor` in `training/model.py` outputs **classification logits** over 100 return bins for each prediction horizon:

```
Output shape: (batch_size, 100, 4)
              └─stocks─┘ └bins┘ └days┘
```

To convert these to ranking scores:

1. **Softmax over bins** → probability distribution over return bins
2. **Expected value** → weighted sum using bin midpoints or indices

```python
probs = F.softmax(predictions[:, :, horizon_idx], dim=1)  # (batch, 100)
bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2      # (100,)
expected_return = (probs * bin_midpoints).sum(dim=1)       # (batch,)
```

This expected return is the **ranking score** used by ListNet.

## Practical Recommendations

### When to Use ListNet

1. **Cross-sectional ranking** - When you care about ranking stocks relative to each other on each day
2. **Portfolio selection** - When selecting top-K stocks for a portfolio
3. **Small batches** - ListNet works well even with modest batch sizes (50-200 stocks)

### Temperature Selection

- **τ = 1.0** (default): Balanced weighting
- **τ < 1.0**: Focus more on correctly ranking top stocks
- **τ > 1.0**: More uniform weighting across all stocks

### Combining with Classification Loss

A common effective setup:

```bash
--ranking-loss-type listnet \
--ranking-loss-weight 0.1
```

This uses cross-entropy as the primary loss (for calibrated probability estimates) while adding 10% ListNet loss to encourage better ranking.

### Pure Ranking Mode

For maximum focus on ranking:

```bash
--ranking-loss-type listnet \
--ranking-only
```

This completely ignores bin classification accuracy and optimizes purely for ranking.

## Code References

- **ListNet loss function**: `walk_forward_training.py:345-387`
- **Unified ranking loss interface**: `walk_forward_training.py:390-469`
- **Training loop integration**: `walk_forward_training.py:2810-2844`
- **Model architecture**: `model.py:154-253`
- **Command line arguments**: `walk_forward_training.py:3955-3956`

## References

- Cao, Z., et al. (2007). "Learning to Rank: From Pairwise Approach to Listwise Approach." ICML.
- Original ListNet paper introduced the top-one probability model and cross-entropy loss for learning to rank.
