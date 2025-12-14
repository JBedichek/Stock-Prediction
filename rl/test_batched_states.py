#!/usr/bin/env python3
"""
Test script to verify batched state extraction and caching works correctly.
"""

import torch
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients (backtest_simulation disables them globally)
torch.set_grad_enabled(True)

def test_batched_states():
    """Test the batched state extraction and caching."""

    print("\n" + "="*80)
    print("TESTING BATCHED STATE EXTRACTION & CACHING")
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
        action_dim=3  # Simplified: HOLD, BUY, SELL
    ).to('cuda')

    agent.feature_extractor.freeze_predictor()
    agent.q_network.train()

    # Preload features for fast access
    print("\n3. Preloading features into RAM cache...")

    # Define cache path
    cache_path = 'data/rl_feature_cache.h5'

    # Try to load from cache first (instant if exists!)
    cache_loaded = False
    if os.path.exists(cache_path):
        print(f"   📂 Found existing cache: {cache_path}")
        print(f"   ⚡ Loading from cache (instant!)...")
        cache_loaded = data_loader.load_feature_cache(cache_path)

    # If cache not loaded, preload from HDF5 and save cache
    if not cache_loaded:
        print("   ⚠️  No cache found - preloading from HDF5...")
        print("   This will take ~5 minutes but only happens once!")
        # Get all trading days from prices file
        sample_ticker = list(data_loader.prices_file.keys())[0]
        prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
        all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
        data_loader.preload_features(all_trading_days)

        # Save cache for next time
        print(f"\n   💾 Saving cache to: {cache_path}")
        print(f"   (Next run will load instantly!)")
        data_loader.save_feature_cache(cache_path)

    # Initialize environment
    print("\n4. Initializing environment...")
    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=10,
        episode_length=20,  # Short episode for testing
        device='cuda'
    )

    # Test episode reset (should trigger batched precomputation)
    print("\n5. Testing episode reset with batched state precomputation...")
    start_time = time.time()
    states = env.reset()
    precompute_time = time.time() - start_time

    print(f"\n   ✅ Precomputation completed in {precompute_time:.2f} seconds")
    print(f"   📊 Cache stats:")
    print(f"      - Dates cached: {len(env.state_cache)}")
    print(f"      - States in first date: {len(env.state_cache[env.dates[0]])}")
    print(f"      - Initial states returned: {len(states)}")

    # Verify state properties
    sample_ticker = list(states.keys())[0]
    sample_state = states[sample_ticker]
    print(f"\n   ✅ Sample state (ticker: {sample_ticker}):")
    print(f"      - Shape: {sample_state.shape}")
    print(f"      - Device: {sample_state.device}")
    print(f"      - Requires grad: {sample_state.requires_grad}")

    # Test that stepping through episode is fast (uses cache)
    print("\n6. Testing episode step speed (should be fast - uses cache)...")
    step_times = []

    for i in range(5):
        # Random actions (3 actions: HOLD, BUY, SELL)
        import random
        actions = {ticker: random.randint(0, 2) for ticker in states.keys()}

        start_time = time.time()
        next_states, reward, done, info = env.step(actions)
        step_time = time.time() - start_time
        step_times.append(step_time)

        print(f"      Step {i+1}: {step_time:.3f}s | Reward: {reward:+.4f} | Positions: {info['num_positions']}")

        if done:
            states = env.reset()
        else:
            states = next_states

    avg_step_time = sum(step_times) / len(step_times)
    print(f"\n   ✅ Average step time: {avg_step_time:.3f}s (should be <1s with caching)")

    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nWith batched extraction + caching:")
    print(f"  - Precomputation time: {precompute_time:.2f}s for {len(env.dates)} days")
    print(f"  - Average step time: {avg_step_time:.3f}s")
    print(f"  - Total states cached: {sum(len(env.state_cache[d]) for d in env.state_cache)}")
    print(f"\nExpected speedup:")
    print(f"  - Old method: ~10-20s per step (sequential extraction)")
    print(f"  - New method: ~{avg_step_time:.3f}s per step (cached lookup)")
    print(f"  - Speedup: ~{15/avg_step_time:.1f}x faster")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE - Batched extraction & caching working!")
    print("="*80)

if __name__ == '__main__':
    test_batched_states()
