#!/usr/bin/env python3
"""
Full pipeline diagnostic - simulates exact training scenario.
"""

import torch
import torch.optim as optim
import sys
import os
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, ReplayBuffer, compute_dqn_loss
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader

# CRITICAL FIX: Re-enable gradients (backtest_simulation disables them globally)
torch.set_grad_enabled(True)

def test_full_pipeline():
    """Test the complete training pipeline."""

    print("\n" + "="*80)
    print("FULL PIPELINE DIAGNOSTIC")
    print("="*80)

    # Initialize data loader
    print("\n1. Loading data...")
    data_loader = DatasetLoader(
        dataset_path='data/all_complete_dataset.h5',
        prices_path='data/actual_prices.h5',
        num_test_stocks=100
    )
    print(f"   ✅ Loaded {len(data_loader.test_tickers)} stocks")

    # Initialize agent
    print("\n2. Initializing agent...")
    agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=1024,
        action_dim=5
    ).to('cuda')

    # Freeze predictor
    agent.feature_extractor.freeze_predictor()

    # Set Q-network to training mode
    agent.q_network.train()
    print("   ✅ Q-network in training mode")

    # Check parameters
    q_params_with_grad = sum(p.requires_grad for p in agent.q_network.parameters())
    print(f"   Q-network parameters with gradients: {q_params_with_grad}")

    # Initialize environment
    print("\n3. Initializing environment...")
    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=10,
        episode_length=20,
        device='cuda'
    )

    # Initialize replay buffer and optimizer
    buffer = ReplayBuffer(capacity=10000)
    optimizer = optim.Adam(agent.q_network.parameters(), lr=5e-5)

    # Reset environment and get initial states
    print("\n4. Resetting environment...")
    states = env.reset()
    print(f"   ✅ Got {len(states)} initial states")

    if len(states) > 0:
        sample_ticker = list(states.keys())[0]
        sample_state = states[sample_ticker]
        print(f"\n   Sample state (ticker: {sample_ticker}):")
        print(f"     Shape: {sample_state.shape}")
        print(f"     Device: {sample_state.device}")
        print(f"     Requires grad: {sample_state.requires_grad}")

    # Simulate a few steps to collect transitions
    print("\n5. Collecting transitions...")
    for step in range(5):
        # Select random actions
        actions = {ticker: random.randint(0, 4) for ticker in states.keys()}

        # Step environment
        next_states, reward, done, info = env.step(actions)

        # Store transitions
        for ticker in actions.keys():
            if ticker in states and ticker in next_states:
                buffer.push(
                    state=states[ticker],
                    action=actions[ticker],
                    reward=reward,
                    next_state=next_states[ticker],
                    done=done,
                    ticker=ticker
                )

        states = next_states
        if done:
            states = env.reset()

        print(f"   Step {step+1}: {len(buffer)} transitions in buffer")

    # Sample a batch and try training
    print("\n6. Sampling batch and computing loss...")
    batch = buffer.sample(32)

    # DEBUG: Check Q-network mode
    print(f"\n   DEBUG: Q-network training mode: {agent.q_network.training}")
    print(f"   DEBUG: Q-network first param requires_grad: {next(agent.q_network.parameters()).requires_grad}")

    # DEBUG: Check if we're in a no_grad context
    print(f"\n   DEBUG: torch.is_grad_enabled(): {torch.is_grad_enabled()}")

    # DEBUG: Manual forward pass
    print("\n   DEBUG: Testing manual Q-network forward pass...")
    test_states = torch.stack([b['state'] for b in batch[:4]]).to('cuda')
    print(f"     Input states shape: {test_states.shape}")
    print(f"     Input states requires_grad: {test_states.requires_grad}")

    # Explicitly enable gradients
    with torch.enable_grad():
        print(f"     Inside torch.enable_grad(), is_grad_enabled: {torch.is_grad_enabled()}")
        test_q_values = agent.q_network(test_states)
        print(f"     Q-values shape: {test_q_values.shape}")
        print(f"     Q-values requires_grad: {test_q_values.requires_grad}")
        print(f"     Q-values grad_fn: {test_q_values.grad_fn}")

    try:
        loss = compute_dqn_loss(
            agent=agent,
            batch=batch,
            gamma=0.99,
            device='cuda'
        )

        print(f"\n   Loss: {loss.item():.4f}")
        print(f"   Loss requires_grad: {loss.requires_grad}")
        print(f"   Loss grad_fn: {loss.grad_fn}")

        if not loss.requires_grad or loss.grad_fn is None:
            print("   ❌ PROBLEM: Loss doesn't have gradient information!")
            return False

    except Exception as e:
        print(f"   ❌ Error computing loss: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try backward pass
    print("\n7. Testing backward pass...")
    try:
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        has_grads = False
        for name, param in agent.q_network.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                print(f"   ✅ Gradients computed successfully")
                break

        if not has_grads:
            print("   ❌ No gradients computed!")
            return False

        # Try optimizer step
        optimizer.step()
        print("   ✅ Optimizer step successful!")
        return True

    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_full_pipeline()

    print("\n" + "="*80)
    if success:
        print("✅ FULL PIPELINE WORKS!")
        print("\nThe training loop should work now.")
        print("If you still get errors, check:")
        print("1. Is agent.q_network.train() being called?")
        print("2. Are Q-network parameters frozen somewhere?")
        print("3. Is there a torch.no_grad() context active during training?")
    else:
        print("❌ PIPELINE FAILED")
        print("\nThe issue has been identified above.")
    print("="*80)
