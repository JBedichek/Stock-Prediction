#!/usr/bin/env python3
"""
Profile training step to identify bottlenecks.
"""

import torch
import sys
import os
import time
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, ReplayBuffer, compute_dqn_loss
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients
torch.set_grad_enabled(True)

def profile_step():
    """Profile each part of a training step."""

    print("\n" + "="*80)
    print("PROFILING TRAINING STEP")
    print("="*80)

    # Initialize
    print("\nInitializing...")
    data_loader = DatasetLoader(
        dataset_path='data/all_complete_dataset.h5',
        prices_path='data/actual_prices.h5',
        num_test_stocks=1000  # Full 1000 stocks
    )

    agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=1024,
        action_dim=5
    ).to('cuda')

    agent.feature_extractor.freeze_predictor()
    agent.q_network.train()

    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=20,
        episode_length=60,
        device='cuda'
    )

    buffer = ReplayBuffer(capacity=200000)

    # Reset environment (includes precomputation)
    print("\n" + "="*80)
    print("1. EPISODE RESET (includes state precomputation)")
    print("="*80)
    t0 = time.time()
    states = env.reset()
    t1 = time.time()
    print(f"Total reset time: {t1-t0:.2f}s")
    print(f"Stocks available: {len(states)}")

    # Now profile a single step in detail
    print("\n" + "="*80)
    print("2. PROFILING SINGLE STEP")
    print("="*80)

    # 2a. Action selection
    print("\n2a. Action selection")
    t0 = time.time()

    # Stack states
    t_stack_start = time.time()
    tickers = list(states.keys())
    state_tensors = torch.stack([states[ticker] for ticker in tickers])
    t_stack_end = time.time()
    print(f"  - Stack states: {t_stack_end - t_stack_start:.3f}s")

    # Q-network forward pass
    t_qnet_start = time.time()
    with torch.no_grad():
        q_values_batch = agent.q_network(state_tensors)
    t_qnet_end = time.time()
    print(f"  - Q-network forward: {t_qnet_end - t_qnet_start:.3f}s")

    # Epsilon-greedy selection
    t_select_start = time.time()
    actions = {}
    for i, ticker in enumerate(tickers):
        action = random.randint(0, 4)  # Random for profiling
        actions[ticker] = action
    t_select_end = time.time()
    print(f"  - Action selection: {t_select_end - t_select_start:.3f}s")

    t1 = time.time()
    print(f"Total action selection: {t1-t0:.3f}s")

    # 2b. Environment step
    print("\n2b. Environment step")
    t0 = time.time()

    next_states, reward, done, info = env.step(actions)

    t1 = time.time()
    print(f"Total env.step(): {t1-t0:.3f}s")
    print(f"  Breakdown will require instrumenting env.step()")

    # 2c. Store transitions
    print("\n2c. Store transitions")
    t0 = time.time()
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
    t1 = time.time()
    print(f"Store transitions: {t1-t0:.3f}s")

    # Fill buffer for training test
    print("\nFilling buffer to test training...")
    for i in range(10):
        actions_random = {ticker: random.randint(0, 4) for ticker in next_states.keys()}
        next_next_states, reward, done, info = env.step(actions_random)
        for ticker in actions_random.keys():
            if ticker in next_states and ticker in next_next_states:
                buffer.push(
                    state=next_states[ticker],
                    action=actions_random[ticker],
                    reward=reward,
                    next_state=next_next_states[ticker],
                    done=done,
                    ticker=ticker
                )
        next_states = next_next_states
        if done:
            next_states = env.reset()

    # 2d. Training step
    if len(buffer) >= 512:
        print(f"\n2d. Training step (buffer size: {len(buffer)})")
        t0 = time.time()

        batch = buffer.sample(512)
        t_sample = time.time()
        print(f"  - Sample batch: {t_sample - t0:.3f}s")

        loss = compute_dqn_loss(agent, batch, gamma=0.99, device='cuda')
        t_loss = time.time()
        print(f"  - Compute loss: {t_loss - t_sample:.3f}s")

        loss.backward()
        t_backward = time.time()
        print(f"  - Backward pass: {t_backward - t_loss:.3f}s")

        t1 = time.time()
        print(f"Total training: {t1-t0:.3f}s")

    # Summary
    print("\n" + "="*80)
    print("BOTTLENECK SUMMARY")
    print("="*80)
    print("\nExpected time per step: <1s")
    print("Actual time: ~120s (2 minutes!)")
    print("\nMost likely bottlenecks:")
    print("1. env.step() - need to instrument internally")
    print("2. _get_states() - even with caching might be slow")
    print("3. Portfolio calculations")
    print("4. HDF5 reads (if not properly cached)")

if __name__ == '__main__':
    profile_step()
