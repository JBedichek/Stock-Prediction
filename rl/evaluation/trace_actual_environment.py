"""
Trace actual environment execution to find bugs.

Uses the real vec_env.step() to see what's happening.
"""

import os
import sys
import torch
import h5py
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.evaluation.baselines import get_baseline_strategy
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader


def trace_episode(strategy_name, vec_env, stock_cache, test_dates, episode_length=30):
    """Trace a single episode with detailed logging."""

    print(f"\n{'='*100}")
    print(f"TRACING {strategy_name.upper()} STRATEGY")
    print(f"{'='*100}\n")

    # Get baseline strategy
    if strategy_name == 'random':
        strategy = get_baseline_strategy('random', 100000)
    elif strategy_name == 'hold':
        strategy = get_baseline_strategy('hold', 100000)
    elif strategy_name == 'long':
        strategy = get_baseline_strategy('long', 100000)
    else:
        print(f"Unknown strategy: {strategy_name}")
        return

    # Reset strategy
    if hasattr(strategy, 'reset'):
        strategy.reset()

    # Reset environment
    states_list, positions_list = vec_env.reset()

    # Get initial stock selections
    if len(vec_env.episode_dates) > 0 and len(vec_env.episode_dates[0]) > 0:
        current_date = vec_env.episode_dates[0][0]
        num_samples = stock_cache.get_num_samples(current_date)
        sample_idx = random.randint(0, num_samples - 1)
        top_4_stocks, bottom_4_stocks = stock_cache.get_sample(current_date, sample_idx)

        print(f"Episode start date: {current_date}")
        print(f"Initial capital: ${vec_env.initial_capital:,.2f}")
        print(f"Episode length: {episode_length} days")
        print(f"\nInitial top 4 stocks: {[t[0] for t in top_4_stocks[:4]]}")
        print(f"Initial bottom 4 stocks: {[t[0] for t in bottom_4_stocks[:4]]}\n")
    else:
        print("ERROR: No episode dates available")
        return

    # Track portfolio
    portfolio_values = [vec_env.initial_capital]
    all_trades = []

    # Run episode
    for step in range(episode_length):
        print(f"{'─'*100}")
        print(f"STEP {step + 1}/{episode_length}")

        # Get current date
        if vec_env.step_indices[0] < len(vec_env.episode_dates[0]):
            current_date = vec_env.episode_dates[0][vec_env.step_indices[0].item()]
            print(f"Date: {current_date}")

        # Strategy selects action
        action = strategy.select_action(states_list[0] if states_list else {}, top_4_stocks, bottom_4_stocks)
        print(f"Action selected: {action}", end="")

        # Convert action to trade
        trade = None
        if action == 0:
            print(" (HOLD)")
            trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
        elif 1 <= action <= 4:
            stock_idx = action - 1
            if stock_idx < len(top_4_stocks):
                ticker = top_4_stocks[stock_idx][0]
                print(f" (BUY {ticker} - top stock #{stock_idx + 1})")
                trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
            else:
                print(" (BUY failed - no stock)")
                trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
        elif 5 <= action <= 8:
            stock_idx = action - 5
            if stock_idx < len(bottom_4_stocks):
                ticker = bottom_4_stocks[stock_idx][0]
                print(f" (SHORT {ticker} - DISABLED, will be rejected)")
                trade = {'action': 'SHORT', 'ticker': ticker, 'position_type': 'SHORT'}
            else:
                print(" (SHORT failed - no stock)")
                trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
        else:
            print(f" (UNKNOWN - defaulting to HOLD)")
            trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}

        # Step environment
        next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
            [action],
            [[trade]],
            positions_list
        )

        # Get info
        if len(infos_list) > 0:
            info = infos_list[0]
            portfolio_value = info.get('portfolio_value', vec_env.initial_capital)
            portfolio_values.append(portfolio_value)

            # Get position info
            cash = vec_env.ref_env.cash
            portfolio_dict = vec_env.ref_env.portfolio

            print(f"Cash: ${cash:,.2f}")
            print(f"Portfolio: {list(portfolio_dict.keys()) if portfolio_dict else 'Empty'}")

            # Show positions
            if portfolio_dict:
                for ticker, pos in portfolio_dict.items():
                    shares = pos['size']
                    entry_price = pos['entry_price']
                    current_price = vec_env.ref_env.price_cache.get(current_date, {}).get(ticker)
                    if current_price:
                        value = shares * current_price
                        pnl_pct = (current_price / entry_price - 1) * 100
                        print(f"  {ticker}: {shares:.2f} shares @ ${entry_price:.2f} → ${current_price:.2f} = ${value:,.2f} ({pnl_pct:+.2f}%)")
                    else:
                        print(f"  {ticker}: {shares:.2f} shares @ ${entry_price:.2f} → NO PRICE")

            print(f"Portfolio value: ${portfolio_value:,.2f}")
            print(f"Return: {(portfolio_value / vec_env.initial_capital - 1) * 100:+.2f}%")

            # Show trades executed
            episode_trades = info.get('trades', [])
            if episode_trades:
                for t in episode_trades:
                    if isinstance(t, list) and len(t) > 0:
                        t = t[0]
                    if isinstance(t, dict):
                        print(f"  ✅ TRADE: {t.get('action')} {t.get('ticker')} @ ${t.get('price', 0):.2f}")
                        all_trades.append(t)

        # Update state
        states_list = next_states_list
        positions_list = next_positions_list

        # Check if done
        if dones_list[0]:
            print("Episode ended early")
            break

        # Update stock selections for next step
        if vec_env.step_indices[0] < len(vec_env.episode_dates[0]):
            current_date = vec_env.episode_dates[0][vec_env.step_indices[0].item()]
            num_samples = stock_cache.get_num_samples(current_date)
            sample_idx = random.randint(0, num_samples - 1)
            top_4_stocks, bottom_4_stocks = stock_cache.get_sample(current_date, sample_idx)

        print()

    # Final summary
    print(f"\n{'='*100}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*100}")
    print(f"Initial capital: ${vec_env.initial_capital:,.2f}")
    print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1] / vec_env.initial_capital - 1) * 100:+.2f}%")
    print(f"Number of trades: {len(all_trades)}")

    # Show all trades
    if all_trades:
        print(f"\nAll trades:")
        for i, trade in enumerate(all_trades):
            action = trade.get('action')
            ticker = trade.get('ticker')
            price = trade.get('price', trade.get('entry_price', trade.get('exit_price', 0)))
            pnl = trade.get('pnl')
            print(f"  {i+1}. {action} {ticker} @ ${price:.2f}" + (f" | P&L: ${pnl:,.2f}" if pnl is not None else ""))

    print(f"\nPortfolio value history (first 10 and last 5):")
    for i, pv in enumerate(portfolio_values[:10]):
        print(f"  Step {i}: ${pv:,.2f} ({(pv/vec_env.initial_capital - 1)*100:+.2f}%)")
    if len(portfolio_values) > 15:
        print("  ...")
        for i, pv in enumerate(portfolio_values[-5:], len(portfolio_values) - 5):
            print(f"  Step {i}: ${pv:,.2f} ({(pv/vec_env.initial_capital - 1)*100:+.2f}%)")

    print(f"\n{'='*100}\n")


def main():
    """Run trace for different strategies."""

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load dataset
    print("Loading dataset...")
    data_loader = DatasetLoader(
        dataset_path='data/all_complete_dataset.h5',
        prices_path='data/actual_prices.h5',
        num_test_stocks=100
    )

    # Create dummy agent
    print("\nCreating dummy agent...")
    dummy_agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(device)
    dummy_agent.train()

    # Load stock cache
    print("\nLoading stock selection cache...")
    with h5py.File('data/actual_prices.h5', 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path='data/rl_stock_selections_4yr.h5',
        prices_file_keys=prices_file_keys,
        device=device
    )

    # Create temp environment for state cache
    print("\nCreating environment...")
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

    # Get test dates
    test_start = '2023-07-01'
    test_end = '2024-12-31'
    all_dates = sorted(stock_cache.selections.keys())
    test_dates = [d for d in all_dates if test_start <= d <= test_end]

    print(f"\nTest period: {len(test_dates)} trading days\n")

    # Create vectorized environment
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

    # Trace each strategy
    for strategy_name in ['random', 'hold', 'long']:
        trace_episode(strategy_name, vec_env, stock_cache, test_dates, episode_length=30)

        user_input = input(f"\nPress Enter to continue to next strategy (or 'q' to quit)...")
        if user_input.lower() == 'q':
            break


if __name__ == '__main__':
    main()
