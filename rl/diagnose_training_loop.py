#!/usr/bin/env python3
"""
Diagnostic script to test the exact training loop scenario.
"""

import torch
import torch.optim as optim
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, ReplayBuffer, compute_dqn_loss

def test_training_scenario():
    """Test the exact scenario from the training loop."""

    print("\n" + "="*80)
    print("TRAINING LOOP DIAGNOSTIC")
    print("="*80)

    # Initialize agent
    print("\n1. Initializing TradingAgent...")
    agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=1024,
        action_dim=5
    ).to('cuda')

    # Initialize optimizer
    print("\n2. Initializing optimizer...")
    optimizer = optim.Adam(agent.q_network.parameters(), lr=5e-5)
    print(f"   ✅ Optimizer created")

    # Initialize replay buffer
    print("\n3. Initializing replay buffer...")
    buffer = ReplayBuffer(capacity=10000)

    # Simulate storing transitions
    print("\n4. Simulating transition storage...")
    for i in range(100):
        # Create dummy transitions (simulating environment)
        state = torch.randn(1469, device='cuda')  # Note: stored WITHOUT requires_grad
        action = torch.randint(0, 5, (1,)).item()
        reward = torch.randn(1).item()
        next_state = torch.randn(1469, device='cuda')
        done = False

        buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            ticker=f'STOCK{i}'
        )

    print(f"   ✅ Stored {len(buffer)} transitions")

    # Sample a batch
    print("\n5. Sampling batch from buffer...")
    batch = buffer.sample(32)
    print(f"   Batch size: {len(batch)}")

    # Check state properties
    print("\n6. Checking sampled states...")
    sample_state = batch[0]['state']
    print(f"   State shape: {sample_state.shape}")
    print(f"   State device: {sample_state.device}")
    print(f"   State requires_grad: {sample_state.requires_grad}")
    print(f"   State dtype: {sample_state.dtype}")

    # Compute loss
    print("\n7. Computing DQN loss...")
    try:
        loss = compute_dqn_loss(
            agent=agent,
            batch=batch,
            gamma=0.99,
            device='cuda'
        )

        print(f"   ✅ Loss computed: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")
        print(f"   Loss dtype: {loss.dtype}")
        print(f"   Loss device: {loss.device}")

        if not loss.requires_grad:
            print("   ❌ PROBLEM: Loss doesn't require gradients!")
            return False

        if loss.grad_fn is None:
            print("   ❌ PROBLEM: Loss doesn't have grad_fn!")
            return False

    except Exception as e:
        print(f"   ❌ Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try backward pass
    print("\n8. Testing backward pass...")
    try:
        optimizer.zero_grad()
        loss.backward()
        print(f"   ✅ Backward pass successful!")

        # Check if gradients were computed
        has_grads = False
        for name, param in agent.q_network.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                print(f"   ✅ Parameter '{name}' has gradients (sum: {param.grad.abs().sum():.4f})")
                break

        if not has_grads:
            print("   ⚠️  Warning: No gradients computed!")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_training_scenario()

    print("\n" + "="*80)
    if success:
        print("✅ ALL CHECKS PASSED - Training loop should work!")
        print("\nIf you're still getting errors, the issue might be:")
        print("1. Agent is in eval() mode - ensure agent.train() is called")
        print("2. States are being created with wrong device")
        print("3. There's a torch.no_grad() context active somewhere")
    else:
        print("❌ DIAGNOSTIC FAILED")
        print("\nThe issue has been identified above.")
    print("="*80)
