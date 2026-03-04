"""Quick test to see if fixed baselines have reasonable variance."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import h5py
import numpy as np
from rl.evaluation.baselines import get_baseline_strategy, evaluate_baseline
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader

device = torch.device('cuda:1')

print("Loading...")
data_loader = DatasetLoader(
    dataset_path='data/all_complete_dataset.h5',
    prices_path='data/actual_prices.h5',
    num_test_stocks=100
)

dummy_agent = TradingAgent('./checkpoints/best_model_100m_1.18.pt', 1469, 512, 5).to(device)
dummy_agent.train()

with h5py.File('data/actual_prices.h5', 'r') as f:
    prices_file_keys = list(f.keys())

stock_cache = GPUStockSelectionCache('data/rl_stock_selections_4yr.h5', prices_file_keys, device)

temp_env = TradingEnvironment(
    data_loader=data_loader,
    agent=dummy_agent,
    initial_capital=100000,
    max_positions=1,
    episode_length=30,
    device=device,
    transaction_cost=0.0
)

if os.path.exists('data/rl_state_cache_4yr.h5'):
    temp_env.load_state_cache('data/rl_state_cache_4yr.h5')

test_dates = [d for d in sorted(stock_cache.selections.keys()) if '2023-07-01' <= d <= '2024-12-31']

vec_env = GPUVectorizedTradingEnv(
    num_envs=1,
    data_loader=data_loader,
    agent=dummy_agent,
    initial_capital=100000,
    max_positions=1,
    episode_length=30,
    transaction_cost=0.0,
    device=device,
    trading_days_filter=test_dates,
    top_k_per_horizon=50
)

vec_env.ref_env.state_cache = temp_env.state_cache
vec_env.ref_env.price_cache = temp_env.price_cache

print("\n" + "="*80)
print("TESTING BASELINE STRATEGIES (10 episodes each)")
print("="*80 + "\n")

for strategy_name in ['random', 'hold', 'long']:
    print(f"Testing {strategy_name}...")
    strategy = get_baseline_strategy(strategy_name, 100000)

    results = evaluate_baseline(
        strategy,
        vec_env,
        stock_cache,
        num_episodes=10,
        episode_length=30
    )

    returns = [r['total_return'] for r in results]
    final_values = [r['final_portfolio_value'] for r in results]

    print(f"  Returns: {[f'{r*100:.2f}%' for r in returns]}")
    print(f"  Mean return: {np.mean(returns)*100:.2f}% ± {np.std(returns)*100:.2f}%")
    print(f"  Mean final value: ${np.mean(final_values):,.2f} ± ${np.std(final_values):,.2f}")
    print(f"  Min/Max: {np.min(returns)*100:.2f}% / {np.max(returns)*100:.2f}%")
    print()
