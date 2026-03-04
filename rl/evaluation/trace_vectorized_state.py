"""
Trace the ACTUAL vectorized environment state, not the ref_env.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import h5py
import random
from rl.evaluation.baselines import get_baseline_strategy
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader


device = torch.device('cuda:1')

# Load everything
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

print("\n" + "="*100)
print("TRACING VECTORIZED ENVIRONMENT STATE (1 episode, random strategy)")
print("="*100)

# Get random strategy
strategy = get_baseline_strategy('random', 100000)
if hasattr(strategy, 'reset'):
    strategy.reset()

# Reset
states_list, positions_list = vec_env.reset()

# Get stock selections
current_date = vec_env.episode_dates[0][0]
num_samples = stock_cache.get_num_samples(current_date)
sample_idx = random.randint(0, num_samples - 1)
top_4_stocks, bottom_4_stocks = stock_cache.get_sample(current_date, sample_idx)

print(f"\nEpisode start: {current_date}")
print(f"Top 4: {[t[0] for t in top_4_stocks[:4]]}")
print(f"Bottom 4: {[t[0] for t in bottom_4_stocks[:4]]}\n")

# Run 10 steps
for step in range(10):
    print(f"{'─'*100}")
    print(f"STEP {step + 1}/10")

    # Get current date
    current_date = vec_env.episode_dates[0][vec_env.step_indices[0].item()]
    print(f"Date: {current_date}")

    # BEFORE action - show vectorized state
    print(f"\nBEFORE ACTION (Vectorized State):")
    print(f"  cash[0] = ${vec_env.cash[0].item():,.2f}")
    print(f"  position_tickers[0] = {vec_env.position_tickers[0]}")
    print(f"  position_sizes[0] = {vec_env.position_sizes[0].item():.2f}")
    print(f"  position_entry_prices[0] = ${vec_env.position_entry_prices[0].item():.2f}")

    # Calculate portfolio value manually
    if vec_env.position_tickers[0]:
        cached_prices = vec_env.ref_env.price_cache.get(current_date, {})
        current_price = cached_prices.get(vec_env.position_tickers[0], 0.0)
        equity = vec_env.position_sizes[0].item() * current_price
        manual_portfolio_value = vec_env.cash[0].item() + equity
        print(f"  current_price for {vec_env.position_tickers[0]} = ${current_price:.2f}")
        print(f"  equity = {vec_env.position_sizes[0].item():.2f} * ${current_price:.2f} = ${equity:,.2f}")
        print(f"  MANUAL portfolio_value = ${vec_env.cash[0].item():,.2f} + ${equity:,.2f} = ${manual_portfolio_value:,.2f}")
    else:
        print(f"  No position")
        print(f"  MANUAL portfolio_value = ${vec_env.cash[0].item():,.2f}")

    # Strategy selects action
    action = strategy.select_action({}, top_4_stocks, bottom_4_stocks)

    # Convert to trade
    trade = None
    if action == 0:
        trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
        print(f"\nAction: {action} (HOLD)")
    elif 1 <= action <= 4:
        stock_idx = action - 1
        if stock_idx < len(top_4_stocks):
            ticker = top_4_stocks[stock_idx][0]
            trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
            print(f"\nAction: {action} (BUY {ticker})")
        else:
            trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            print(f"\nAction: {action} (HOLD - no ticker)")
    elif 5 <= action <= 8:
        stock_idx = action - 5
        if stock_idx < len(bottom_4_stocks):
            ticker = bottom_4_stocks[stock_idx][0]
            trade = {'action': 'SHORT', 'ticker': ticker, 'position_type': 'SHORT'}
            print(f"\nAction: {action} (SHORT {ticker} - will be rejected if shorting disabled)")
        else:
            trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            print(f"\nAction: {action} (HOLD - no ticker)")
    else:
        trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
        print(f"\nAction: {action} (HOLD)")

    # Check price availability
    if trade['ticker']:
        cached_prices = vec_env.ref_env.price_cache.get(current_date, {})
        price = cached_prices.get(trade['ticker'], None)
        if price:
            print(f"  Price for {trade['ticker']}: ${price:.2f}")
        else:
            print(f"  ❌ NO PRICE for {trade['ticker']} on {current_date}")

    # Step
    next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
        [action], [[trade]], positions_list
    )

    # AFTER action - show vectorized state
    print(f"\nAFTER ACTION (Vectorized State):")
    print(f"  cash[0] = ${vec_env.cash[0].item():,.2f}")
    print(f"  position_tickers[0] = {vec_env.position_tickers[0]}")
    print(f"  position_sizes[0] = {vec_env.position_sizes[0].item():.2f}")
    print(f"  position_entry_prices[0] = ${vec_env.position_entry_prices[0].item():.2f}")

    # Show info dict
    if infos_list:
        info = infos_list[0]
        print(f"\nINFO DICT:")
        print(f"  portfolio_value = ${info['portfolio_value']:,.2f}")
        print(f"  cash = ${info['cash']:,.2f}")
        print(f"  trades = {info['trades']}")

    # Update
    states_list = next_states_list
    positions_list = next_positions_list

    if dones_list[0]:
        print("\nEpisode done")
        break

    # Get next stock selections
    if vec_env.step_indices[0] < len(vec_env.episode_dates[0]):
        current_date = vec_env.episode_dates[0][vec_env.step_indices[0].item()]
        num_samples = stock_cache.get_num_samples(current_date)
        sample_idx = random.randint(0, num_samples - 1)
        top_4_stocks, bottom_4_stocks = stock_cache.get_sample(current_date, sample_idx)

    print()

print(f"\n{'='*100}\n")
