#!/usr/bin/env python3
"""
Test script to verify confidence model architecture works correctly.

Tests:
1. Model instantiation with confidence head
2. Forward pass returns correct shapes
3. Confidence values in [0, 1]
4. Loss computation with confidence
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from training.train_new_format import (
    SimpleTransformerPredictor,
    compute_confidence_targets,
    convert_price_ratios_to_bins_vectorized
)

def test_model_architecture():
    """Test 1: Model instantiation"""
    print("="*80)
    print("TEST 1: Model Instantiation")
    print("="*80)

    model = SimpleTransformerPredictor(
        input_dim=998,  # Based on your dataset
        hidden_dim=256,  # Smaller for quick test
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        num_pred_days=4,
        pred_mode='classification'
    )

    print("\n✅ Model instantiated successfully!")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def test_forward_pass(model):
    """Test 2: Forward pass and output shapes"""
    print("\n" + "="*80)
    print("TEST 2: Forward Pass and Output Shapes")
    print("="*80)

    # Create dummy input
    batch_size = 4
    seq_len = 100
    input_dim = 998

    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    pred, confidence = model(x)

    print(f"\n✅ Forward pass successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Confidence shape: {confidence.shape}")

    # Check shapes
    assert pred.shape == (batch_size, 100, 4), f"Expected pred shape (4, 100, 4), got {pred.shape}"
    assert confidence.shape == (batch_size, 4), f"Expected confidence shape (4, 4), got {confidence.shape}"

    print("\n✅ Output shapes correct!")
    return pred, confidence


def test_confidence_range(confidence):
    """Test 3: Confidence values in [0, 1]"""
    print("\n" + "="*80)
    print("TEST 3: Confidence Value Range")
    print("="*80)

    min_conf = confidence.min().item()
    max_conf = confidence.max().item()
    mean_conf = confidence.mean().item()

    print(f"\n📊 Confidence statistics:")
    print(f"   Min: {min_conf:.4f}")
    print(f"   Max: {max_conf:.4f}")
    print(f"   Mean: {mean_conf:.4f}")

    assert min_conf >= 0.0, f"Confidence min {min_conf} < 0!"
    assert max_conf <= 1.0, f"Confidence max {max_conf} > 1!"

    print("\n✅ Confidence values in valid range [0, 1]!")


def test_loss_computation(pred, confidence):
    """Test 4: Loss computation with confidence"""
    print("\n" + "="*80)
    print("TEST 4: Loss Computation")
    print("="*80)

    batch_size = pred.shape[0]
    num_pred_days = pred.shape[2]

    # Create dummy target prices
    prices = torch.rand(batch_size, num_pred_days) * 0.1 + 0.95  # Price ratios around 1.0

    # Create dummy bin edges (100 bins)
    bin_edges = torch.linspace(0.5, 1.5, 101)

    # Convert prices to bin indices
    bin_indices = convert_price_ratios_to_bins_vectorized(prices, bin_edges)

    # Compute main loss
    main_loss = 0
    for day_idx in range(num_pred_days):
        targets = bin_indices[:, day_idx]
        main_loss += F.cross_entropy(pred[:, :, day_idx], targets)

    # Compute confidence targets from CE loss
    confidence_targets = compute_confidence_targets(pred, bin_indices)

    # Compute confidence loss
    confidence_loss = F.mse_loss(confidence, confidence_targets)

    # Combined loss
    confidence_weight = 0.5
    total_loss = main_loss + confidence_weight * confidence_loss

    print(f"\n✅ Loss computation successful!")
    print(f"   Main loss: {main_loss.item():.4f}")
    print(f"   Confidence loss: {confidence_loss.item():.4f}")
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Mean confidence target: {confidence_targets.mean().item():.4f}")
    print(f"   Mean confidence output: {confidence.mean().item():.4f}")
    print(f"\n   📊 Confidence target range:")
    print(f"      Min: {confidence_targets.min().item():.4f}")
    print(f"      Max: {confidence_targets.max().item():.4f}")
    print(f"      (exp(-CE) mapping provides dense signal)")

    # Test backward pass
    total_loss.backward()

    print("\n✅ Backward pass successful!")

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients computed!"

    print("✅ Gradients computed successfully!")


def main():
    print("\n" + "="*80)
    print("CONFIDENCE MODEL TEST SUITE")
    print("="*80)

    # Run tests
    model = test_model_architecture()
    pred, confidence = test_forward_pass(model)
    test_confidence_range(confidence)
    test_loss_computation(pred, confidence)

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe confidence model implementation is working correctly!")
    print("You can now:")
    print("  1. Train with: python training/train_new_format.py")
    print("  2. Use selection-aware training: python training/selection_aware_training.py")
    print()


if __name__ == '__main__':
    main()
