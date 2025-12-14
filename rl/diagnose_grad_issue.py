#!/usr/bin/env python3
"""
Diagnostic script to identify the gradient issue in RL training.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent

def check_gradients():
    """Check if gradients are properly enabled."""

    print("\n" + "="*80)
    print("GRADIENT DIAGNOSTIC")
    print("="*80)

    # Initialize agent
    print("\n1. Initializing TradingAgent...")
    agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=1024,
        action_dim=5
    ).to('cuda')

    # Check Q-network parameters
    print("\n2. Checking Q-network parameters...")
    q_params_with_grad = sum(p.requires_grad for p in agent.q_network.parameters())
    q_params_total = sum(1 for _ in agent.q_network.parameters())

    print(f"   Q-network parameters: {q_params_total}")
    print(f"   Parameters with requires_grad=True: {q_params_with_grad}")
    print(f"   Parameters with requires_grad=False: {q_params_total - q_params_with_grad}")

    if q_params_with_grad == 0:
        print("   ❌ PROBLEM FOUND: All Q-network parameters have requires_grad=False!")
    else:
        print("   ✅ Q-network parameters have gradients enabled")

    # Check target network parameters (should be frozen)
    print("\n3. Checking target network parameters...")
    target_params_with_grad = sum(p.requires_grad for p in agent.target_network.parameters())
    target_params_total = sum(1 for _ in agent.target_network.parameters())

    print(f"   Target network parameters: {target_params_total}")
    print(f"   Parameters with requires_grad=True: {target_params_with_grad}")
    print(f"   Parameters with requires_grad=False: {target_params_total - target_params_with_grad}")

    if target_params_with_grad == 0:
        print("   ✅ Target network properly frozen")
    else:
        print("   ⚠️  Target network has unfrozen parameters (should all be frozen)")

    # Test forward pass and gradient computation
    print("\n4. Testing forward pass and gradient computation...")

    # Create dummy state
    dummy_state = torch.randn(32, 1469, device='cuda')

    # Forward pass
    q_values = agent.q_network(dummy_state)
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Q-values requires_grad: {q_values.requires_grad}")
    print(f"   Q-values has grad_fn: {q_values.grad_fn is not None}")

    if not q_values.requires_grad:
        print("   ❌ PROBLEM: Q-values don't require gradients!")
        print("   This means the Q-network parameters don't have requires_grad=True")
    else:
        print("   ✅ Q-values require gradients")

    # Try computing loss and backward
    print("\n5. Testing loss computation and backward pass...")
    try:
        target = torch.randn_like(q_values)
        loss = torch.nn.functional.mse_loss(q_values, target)

        print(f"   Loss: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss has grad_fn: {loss.grad_fn is not None}")

        if not loss.requires_grad or loss.grad_fn is None:
            print("   ❌ PROBLEM: Loss doesn't have gradient information!")
            return False

        loss.backward()
        print("   ✅ Backward pass successful!")
        return True

    except RuntimeError as e:
        print(f"   ❌ Backward pass failed: {e}")
        return False

    print("\n" + "="*80)

if __name__ == '__main__':
    success = check_gradients()

    if not success:
        print("\nDIAGNOSTIC SUMMARY:")
        print("The Q-network parameters don't have requires_grad=True.")
        print("\nPossible causes:")
        print("1. Parameters were explicitly set to requires_grad=False")
        print("2. Model was loaded from checkpoint with frozen parameters")
        print("3. torch.no_grad() context is active")
        print("\nSuggested fix:")
        print("Add this after creating the agent:")
        print("  for param in agent.q_network.parameters():")
        print("      param.requires_grad = True")
    else:
        print("\n✅ All gradient checks passed!")
