# Differentiable Portfolio Selection with Gumbel-Softmax

This document explains how the differentiable top-k stock selection works in `train_portfolio_differentiable.py`, and discusses potential improvements.

## The Problem

Traditional stock selection is **non-differentiable**:

```python
# Hard top-k selection - no gradient!
top_k_indices = torch.topk(scores, k).indices
selected_stocks = stocks[top_k_indices]
```

The `argmax`/`topk` operation has zero gradient almost everywhere, so we can't backpropagate through the selection to train the scoring model end-to-end.

## The Solution: Gumbel-Softmax

The Gumbel-Softmax trick (Jang et al. 2016, Maddison et al. 2016) provides a differentiable approximation to discrete sampling.

### Step 1: Add Gumbel Noise

For stochastic exploration during training:

```python
# Gumbel(0,1) noise via inverse CDF
gumbels = -log(-log(U))  # where U ~ Uniform(0,1)
perturbed_scores = (scores + gumbels) / temperature
```

This is equivalent to sampling from a categorical distribution - the argmax of `scores + gumbels` follows the same distribution as sampling proportional to `softmax(scores)`.

### Step 2: Iterative Soft Top-K

Since we need k selections (not just 1), we use an iterative approach:

```python
soft_mask = zeros_like(scores)
remaining = perturbed_scores.clone()

for _ in range(k):
    # Soft selection via softmax
    soft_selection = softmax(remaining, dim=-1)
    soft_mask += soft_selection

    # "Remove" selected item
    remaining = remaining - soft_selection * 1e9
```

Each iteration:
1. Softmax gives a probability distribution over remaining items
2. Add this distribution to our cumulative soft mask
3. Subtract a large value from selected items to exclude them

### Step 3: Straight-Through Estimator

For the actual forward pass, we want **hard** binary selections (0 or 1), but we need gradients to flow through the **soft** approximation:

```python
# Forward: hard selection
hard_mask = topk(scores, k)  # binary {0, 1}

# Backward: gradient flows through soft_mask
output = hard_mask - soft_mask.detach() + soft_mask
```

This clever trick:
- **Forward pass**: `output = hard_mask` (the detached terms cancel)
- **Backward pass**: `d_output/d_scores = d_soft_mask/d_scores` (hard_mask has no gradient)

## Temperature Annealing

The temperature controls the "sharpness" of selections:

| Temperature | Behavior |
|-------------|----------|
| High (1.0+) | Soft, exploratory - gradients spread across many stocks |
| Low (0.1)   | Sharp, deterministic - approaches hard top-k |

We anneal from high to low during training:

```python
temperature = min_temp + (initial_temp - min_temp) * (1 - progress)
```

**Early training**: High temperature allows exploration and gradient flow to many stocks.

**Late training**: Low temperature makes selections crisp, matching inference behavior.

## Full Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Features  │───▶│  Encoder    │───▶│   Scores    │───▶│  Gumbel     │
│ (N stocks)  │    │ (Transformer)│    │ (N values)  │    │  Top-K      │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Loss     │◀───│  Portfolio  │◀───│   Weights   │◀───│   Soft      │
│  (Return,   │    │   Return    │    │ (K selected)│    │   Mask      │
│   Sharpe)   │    └─────────────┘    └─────────────┘    └─────────────┘
```

The loss function combines:
1. **Return loss**: `-mean(portfolio_returns)` - maximize returns
2. **Sharpe loss**: `-mean/std` - risk-adjusted returns
3. **Concentration loss**: `entropy(weights)` - encourage decisive selection
4. **Confidence loss**: `-mean(selected_confidence)` - trust high-confidence picks

## Alternative Methods

The codebase also implements:

### NeuralSort (Grover et al. 2019)

Creates a differentiable soft permutation matrix:

```python
# Pairwise score differences
pairwise_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)

# Soft permutation matrix
P = softmax(pairwise_diff / temperature, dim=-1)

# Top-k probability = sum of probabilities for ranks 0 to k-1
top_k_probs = P[:, :k, :].sum(dim=1)
```

**Pros**: Theoretically cleaner, considers global ranking
**Cons**: O(n²) memory for n stocks

### Straight-Through Estimator (STE)

Simplest approach - just pass gradients through:

```python
# Forward: hard top-k
mask = topk(scores, k)

# Backward: gradient * softmax(scores)
grad = upstream_grad * softmax(scores)
```

**Pros**: Simple, fast
**Cons**: Biased gradient estimate

---

## Potential Improvements

### 1. Subset Sampling Instead of Sequential Selection

**Current approach**: Iteratively select k items one at a time.

**Problem**: Sequential selection can accumulate errors and the soft mask doesn't sum exactly to k.

**Improvement**: Use subset sampling methods like:
- **Plackett-Luce** models for direct k-subset sampling
- **Gumbel-Sinkhorn** for learning permutations

```python
# Gumbel-Sinkhorn approach
def gumbel_sinkhorn_topk(scores, k, temperature, n_iters=20):
    gumbels = sample_gumbel(scores.shape)
    log_alpha = (scores + gumbels) / temperature

    # Sinkhorn iterations for doubly-stochastic matrix
    for _ in range(n_iters):
        log_alpha = log_alpha - logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - logsumexp(log_alpha, dim=-2, keepdim=True)

    return log_alpha.exp()[:, :k].sum(dim=1)
```

### 2. Learn the Temperature

**Current**: Temperature is annealed on a fixed schedule.

**Improvement**: Make temperature a learnable parameter or use a learned schedule:

```python
class LearnedTemperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return F.softplus(self.log_temp) + 0.1  # min 0.1
```

### 3. Multi-Horizon Selection

**Current**: Selects based on single horizon (e.g., 1-day return).

**Improvement**: Select stocks that are good across multiple horizons:

```python
# Combine predictions across horizons
scores_1d = predictions[:, :, 0]
scores_5d = predictions[:, :, 1]
scores_20d = predictions[:, :, 2]

# Weighted combination
combined_scores = 0.5 * scores_1d + 0.3 * scores_5d + 0.2 * scores_20d
```

### 4. Sector/Diversity Constraints

**Current**: Purely score-based selection (may over-concentrate in one sector).

**Improvement**: Add differentiable diversity penalty:

```python
def sector_diversity_loss(weights, sector_ids, num_sectors):
    """Penalize concentration in single sector."""
    # One-hot encode sectors
    sector_onehot = F.one_hot(sector_ids, num_sectors).float()

    # Weighted sector exposure
    sector_exposure = (weights.unsqueeze(-1) * sector_onehot).sum(dim=1)

    # Entropy of sector allocation (higher = more diverse)
    sector_entropy = -(sector_exposure * torch.log(sector_exposure + 1e-8)).sum(dim=-1)

    return -sector_entropy.mean()  # Maximize entropy
```

### 5. Transaction Cost Awareness

**Current**: No consideration of turnover between days.

**Improvement**: Penalize changing the portfolio too much:

```python
def turnover_loss(current_weights, previous_weights, transaction_cost=0.001):
    """Penalize portfolio turnover."""
    turnover = (current_weights - previous_weights).abs().sum(dim=-1)
    return transaction_cost * turnover.mean()
```

This requires maintaining state across batches or using sequential training.

### 6. Conditional Top-K

**Current**: Fixed k regardless of market conditions.

**Improvement**: Learn when to hold more or fewer positions:

```python
class ConditionalTopK(nn.Module):
    def __init__(self, max_k):
        super().__init__()
        self.k_predictor = nn.Linear(hidden_dim, max_k)

    def forward(self, market_features, stock_scores):
        # Predict soft k values
        k_logits = self.k_predictor(market_features)
        k_weights = F.softmax(k_logits, dim=-1)  # (batch, max_k)

        # Weighted sum of top-1, top-2, ..., top-max_k selections
        weighted_selection = 0
        for k in range(1, max_k + 1):
            selection_k = gumbel_top_k(stock_scores, k)
            weighted_selection += k_weights[:, k-1:k] * selection_k

        return weighted_selection
```

### 7. Curriculum Learning on Temperature

**Current**: Linear temperature annealing.

**Improvement**: Use validation performance to guide annealing:

```python
class AdaptiveTemperature:
    def __init__(self, initial=1.0, min_temp=0.1, patience=3):
        self.temperature = initial
        self.min_temp = min_temp
        self.patience = patience
        self.best_sharpe = -float('inf')
        self.wait = 0

    def step(self, val_sharpe):
        if val_sharpe > self.best_sharpe:
            self.best_sharpe = val_sharpe
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce temperature when plateauing
                self.temperature = max(self.min_temp, self.temperature * 0.8)
                self.wait = 0
```

### 8. Better Gradient Estimation

**Current**: Straight-through estimator can have high variance.

**Improvement**: Use REINFORCE with baseline or Rao-Blackwellization:

```python
def rao_blackwell_gradient(scores, k, returns):
    """Lower variance gradient via Rao-Blackwellization."""
    # Sample multiple Gumbel noises
    n_samples = 10
    total_grad = 0

    for _ in range(n_samples):
        gumbels = sample_gumbel(scores.shape)
        perturbed = scores + gumbels

        # Hard selection
        _, indices = torch.topk(perturbed, k, dim=-1)
        selected_returns = returns.gather(-1, indices).sum(-1)

        # REINFORCE gradient
        log_prob = compute_log_prob(scores, indices)
        total_grad += selected_returns * log_prob.grad()

    return total_grad / n_samples
```

---

## References

1. Jang, E., Gu, S., & Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax. arXiv:1611.01144

2. Maddison, C. J., Mnih, A., & Teh, Y. W. (2016). The Concrete Distribution. arXiv:1611.00712

3. Grover, A., et al. (2019). Stochastic Optimization of Sorting Networks via Continuous Relaxations. ICLR.

4. Xie, S. M., & Ermon, S. (2019). Reparameterizable Subset Sampling via Continuous Relaxations. IJCAI.
