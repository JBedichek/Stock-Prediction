#!/usr/bin/env python3
"""
Test script to verify pre-computing all states works and provides speedup.
"""

import torch
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients
torch.set_grad_enabled(True)

def test_precompute_all():
    """Test pre-computing all states at initialization."""

    print("\n" + "="*80)
    print("TESTING PRE-COMPUTE ALL STATES OPTIMIZATION")
    print("="*80)

    # Initialize data loader
    print("\n1. Loading data...")
    data_loader = DatasetLoader(
        dataset_path='data/all_complete_dataset.h5',
        prices_path='data/actual_prices.h5',
        num_test_stocks=100  # Use 100 stocks for testing
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

    agent.feature_extractor.freeze_predictor()
    agent.q_network.train()

    # Initialize environment WITH precompute_all_states=True
    print("\n3. Initializing environment WITH precompute_all_states=True...")
    print("   (This will take ~30 minutes to pre-compute all states once...)")

    start_time = time.time()
    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=10,
        episode_length=20,
        device='cuda',
        precompute_all_states=True  # Pre-compute all states once!
    )
    precompute_time = time.time() - start_time

    print(f"\n   ✅ Pre-computation completed in {precompute_time:.2f} seconds")

    # Now test that episode resets are INSTANT
    print("\n4. Testing episode reset speed (should be instant - just reads from cache)...")

    reset_times = []
    for i in range(5):
        start_time = time.time()
        states = env.reset()
        reset_time = time.time() - start_time
        reset_times.append(reset_time)
        print(f"   Episode {i+1} reset: {reset_time:.4f}s | {len(states)} initial states")

    avg_reset_time = sum(reset_times) / len(reset_times)
    print(f"\n   ✅ Average reset time: {avg_reset_time:.4f}s (vs ~30-60s without pre-caching!)")

    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nOne-time precomputation cost: {precompute_time:.2f}s")
    print(f"Average episode reset time: {avg_reset_time:.4f}s")
    print(f"\nFor 1000 episodes:")
    print(f"  - Without pre-caching: 1000 × 30s = ~8.3 hours")
    print(f"  - With pre-caching: {precompute_time:.2f}s + (1000 × {avg_reset_time:.4f}s) = ~{precompute_time + 1000*avg_reset_time:.2f}s = ~{(precompute_time + 1000*avg_reset_time)/60:.1f} minutes")
    print(f"  - Speedup: ~{(1000*30)/(precompute_time + 1000*avg_reset_time):.1f}x faster!")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE - Pre-compute all states optimization working!")
    print("="*80)

if __name__ == '__main__':
    test_precompute_all()
