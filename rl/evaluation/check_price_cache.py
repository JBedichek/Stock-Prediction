"""Check if price cache is populated correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import h5py
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader

device = torch.device('cuda:1')

# Load dataset
data_loader = DatasetLoader(
    dataset_path='data/all_complete_dataset.h5',
    prices_path='data/actual_prices.h5',
    num_test_stocks=100
)

# Create dummy agent
dummy_agent = TradingAgent(
    predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
    state_dim=1469,
    hidden_dim=512,
    action_dim=5
).to(device)
dummy_agent.train()

# Load stock cache
with h5py.File('data/actual_prices.h5', 'r') as f:
    prices_file_keys = list(f.keys())

stock_cache = GPUStockSelectionCache(
    h5_path='data/rl_stock_selections_4yr.h5',
    prices_file_keys=prices_file_keys,
    device=device
)

# Create temp environment
temp_env = TradingEnvironment(
    data_loader=data_loader,
    agent=dummy_agent,
    initial_capital=100000,
    max_positions=1,
    episode_length=30,
    device=device,
    transaction_cost=0.0
)

# Load state cache
if os.path.exists('data/rl_state_cache_4yr.h5'):
    temp_env.load_state_cache('data/rl_state_cache_4yr.h5')
    print(f"✅ State cache loaded")
    print(f"Price cache has {len(temp_env.price_cache)} dates")
    print(f"Sample dates: {list(temp_env.price_cache.keys())[:5]}")

    # Check a specific date
    test_date = '2023-09-29'
    if test_date in temp_env.price_cache:
        prices_on_date = temp_env.price_cache[test_date]
        print(f"\n{test_date}: {len(prices_on_date)} tickers with prices")
        print(f"Sample tickers: {list(prices_on_date.keys())[:10]}")

        # Check if HHS is there
        if 'HHS' in prices_on_date:
            print(f"HHS price on {test_date}: ${prices_on_date['HHS']:.2f}")
        else:
            print(f"HHS NOT in price cache for {test_date}!")
    else:
        print(f"{test_date} NOT in price cache!")

# Create vectorized env
test_start = '2023-07-01'
test_end = '2024-12-31'
all_dates = sorted(stock_cache.selections.keys())
test_dates = [d for d in all_dates if test_start <= d <= test_end]

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

# Share caches
vec_env.ref_env.state_cache = temp_env.state_cache
vec_env.ref_env.price_cache = temp_env.price_cache

print(f"\n✅ Vectorized env created")
print(f"vec_env.ref_env.price_cache has {len(vec_env.ref_env.price_cache)} dates")

# Check if same object
print(f"\nAre they the same object? {vec_env.ref_env.price_cache is temp_env.price_cache}")

# Reset and check what happens
states_list, positions_list = vec_env.reset()

if len(vec_env.episode_dates) > 0:
    first_date = vec_env.episode_dates[0][0]
    print(f"\nFirst episode date: {first_date}")

    if first_date in vec_env.ref_env.price_cache:
        prices = vec_env.ref_env.price_cache[first_date]
        print(f"Prices available for {first_date}: {len(prices)} tickers")
        print(f"Sample: {list(prices.items())[:5]}")
    else:
        print(f"ERROR: {first_date} NOT in vec_env.ref_env.price_cache!")
        print(f"Available dates: {list(vec_env.ref_env.price_cache.keys())[:10]}")
