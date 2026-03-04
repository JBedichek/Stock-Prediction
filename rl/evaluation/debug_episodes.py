"""
Debug script to trace through sample episodes for each strategy.

Shows prices, actions, trades, and portfolio values step-by-step.
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
from rl.rl_environment import TradingEnvironment
from rl.rl_components import TradingAgent
from inference.backtest_simulation import DatasetLoader
from rl.train_dqn_simple import SimpleDQNTrainer


def debug_episode(
    strategy_name: str,
    strategy,
    data_loader,
    stock_cache,
    dummy_agent,
    temp_env,
    test_dates,
    initial_capital: float = 100000,
    episode_length: int = 30,
    device='cuda'
):
    """
    Run a single episode with detailed logging.

    Args:
        strategy_name: Name of the strategy
        strategy: Strategy instance
        data_loader: DatasetLoader
        stock_cache: GPUStockSelectionCache
        dummy_agent: Agent for feature extraction
        temp_env: Reference environment
        test_dates: List of test dates
        initial_capital: Starting capital
        episode_length: Episode length in days
        device: Device
    """
    print(f"\n{'='*100}")
    print(f"DEBUGGING {strategy_name.upper()} STRATEGY")
    print(f"{'='*100}\n")

    # Reset strategy
    if hasattr(strategy, 'reset'):
        strategy.reset()

    # Pick random start date
    max_start_idx = len(test_dates) - episode_length - 1
    if max_start_idx < 0:
        print("ERROR: Not enough test dates")
        return

    start_idx = random.randint(0, max_start_idx)
    episode_dates = test_dates[start_idx:start_idx + episode_length]

    print(f"Episode start date: {episode_dates[0]}")
    print(f"Episode end date: {episode_dates[-1]}")
    print(f"Episode length: {len(episode_dates)} days")
    print(f"Initial capital: ${initial_capital:,.2f}\n")

    # Initialize tracking
    portfolio_values = [initial_capital]
    trades = []
    current_position = None
    position_entry_price = None
    position_shares = 0
    cash = initial_capital
    is_short = False

    # Run episode step by step
    for step_idx, date in enumerate(episode_dates):
        print(f"\n{'─'*100}")
        print(f"STEP {step_idx + 1}/{len(episode_dates)} | Date: {date}")
        print(f"{'─'*100}")

        # Get stock selections
        num_samples = stock_cache.get_num_samples(date)
        sample_idx = random.randint(0, num_samples - 1)
        top_4_stocks, bottom_4_stocks = stock_cache.get_sample(date, sample_idx)

        print(f"\nTop 4 stocks: {[f'{t[0]}' for t in top_4_stocks[:4]]}")
        print(f"Bottom 4 stocks: {[f'{t[0]}' for t in bottom_4_stocks[:4]]}")

        # Get prices for these stocks
        cached_prices = temp_env.price_cache.get(date, {})
        print(f"\nPrices:")
        for ticker, horizon in top_4_stocks[:4]:
            price = cached_prices.get(ticker)
            print(f"  {ticker}: ${price:.2f}" if price else f"  {ticker}: NO PRICE")

        # Get current state (simplified - just use cash and position info)
        state = {
            'cash': cash,
            'portfolio_value': cash + (position_shares * cached_prices.get(current_position, 0) if current_position else 0),
            'position': current_position,
            'is_short': is_short
        }

        # Strategy selects action
        action = strategy.select_action(state, top_4_stocks, bottom_4_stocks)

        print(f"\nCurrent Position: {current_position if current_position else 'None'}")
        print(f"Cash: ${cash:,.2f}")
        print(f"Action selected: {action}")

        # Convert action to trade
        trade = None
        if action == -1:  # RandomUniverseStrategy special case
            if hasattr(strategy, 'selected_ticker'):
                ticker = strategy.selected_ticker
                print(f"Trade: BUY {ticker} (random from universe)")
                trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
        elif action == -2:  # RandomUniverseStrategy close position
            print(f"Trade: CLOSE position")
            trade = {'action': 'SELL', 'ticker': None, 'position_type': None}
        elif action == 0:
            print(f"Trade: HOLD")
            trade = None
        elif 1 <= action <= 4:
            stock_idx = action - 1
            if stock_idx < len(top_4_stocks):
                ticker = top_4_stocks[stock_idx][0]
                print(f"Trade: BUY {ticker} (top stock #{stock_idx + 1})")
                trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
        elif 5 <= action <= 8:
            stock_idx = action - 5
            if stock_idx < len(bottom_4_stocks):
                ticker = bottom_4_stocks[stock_idx][0]
                print(f"Trade: SHORT {ticker} (bottom stock #{stock_idx + 1})")
                trade = {'action': 'SHORT', 'ticker': ticker, 'position_type': 'SHORT'}

        # Execute trade manually
        if trade:
            ticker = trade['ticker']
            price = cached_prices.get(ticker) if ticker else None

            if trade['action'] == 'BUY' and ticker and price:
                if current_position is None:
                    # Open long position
                    position_shares = cash / price
                    position_entry_price = price
                    current_position = ticker
                    is_short = False
                    cash = 0
                    print(f"  ✅ EXECUTED: Bought {position_shares:.2f} shares of {ticker} at ${price:.2f}")
                    print(f"  Position value: ${position_shares * price:,.2f}")
                else:
                    print(f"  ❌ REJECTED: Already have position in {current_position}")

            elif trade['action'] == 'SHORT' and ticker and price:
                if current_position is None:
                    # Open short position
                    position_shares = cash / price
                    position_entry_price = price
                    current_position = ticker
                    is_short = True
                    cash = cash * 2  # Get proceeds from short + original cash
                    print(f"  ✅ EXECUTED: Shorted {position_shares:.2f} shares of {ticker} at ${price:.2f}")
                    print(f"  Cash after short: ${cash:,.2f}")
                else:
                    print(f"  ❌ REJECTED: Already have position in {current_position}")

            elif trade['action'] == 'SELL':
                if current_position:
                    # Close position
                    exit_price = cached_prices.get(current_position)
                    if exit_price:
                        if is_short:
                            # Close short: return borrowed shares and calculate P&L
                            pnl = position_shares * (position_entry_price - exit_price)
                            cash = cash - (position_shares * exit_price) + pnl
                            print(f"  ✅ CLOSED SHORT: {current_position} at ${exit_price:.2f}")
                            print(f"  Entry: ${position_entry_price:.2f}, Exit: ${exit_price:.2f}")
                            print(f"  P&L: ${pnl:,.2f}")
                        else:
                            # Close long
                            cash = position_shares * exit_price
                            pnl = cash - initial_capital
                            print(f"  ✅ CLOSED LONG: {current_position} at ${exit_price:.2f}")
                            print(f"  Entry: ${position_entry_price:.2f}, Exit: ${exit_price:.2f}")
                            print(f"  P&L: ${pnl:,.2f}")

                        trades.append({
                            'ticker': current_position,
                            'entry_price': position_entry_price,
                            'exit_price': exit_price,
                            'shares': position_shares,
                            'pnl': pnl,
                            'is_short': is_short
                        })

                        current_position = None
                        position_entry_price = None
                        position_shares = 0
                        is_short = False
                    else:
                        print(f"  ❌ ERROR: No price for {current_position}")
                else:
                    print(f"  ⚠️  No position to close")

        # Calculate portfolio value
        if current_position:
            current_price = cached_prices.get(current_position, 0)
            if is_short:
                # Short position value: cash - (shares * current_price)
                unrealized_pnl = position_shares * (position_entry_price - current_price)
                portfolio_value = cash - (position_shares * current_price) + unrealized_pnl
            else:
                # Long position value: shares * current_price
                portfolio_value = position_shares * current_price
        else:
            portfolio_value = cash

        portfolio_values.append(portfolio_value)

        print(f"\nEnd of step portfolio value: ${portfolio_value:,.2f}")
        print(f"Return so far: {(portfolio_value / initial_capital - 1) * 100:+.2f}%")

    # Final summary
    print(f"\n{'='*100}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*100}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
    print(f"Total return: {(portfolio_values[-1] / initial_capital - 1) * 100:+.2f}%")
    print(f"Number of completed trades: {len(trades)}")
    print(f"Final position: {current_position if current_position else 'None (all cash)'}")

    if len(trades) > 0:
        print(f"\nCompleted trades:")
        for i, trade in enumerate(trades):
            print(f"  {i+1}. {'SHORT' if trade['is_short'] else 'LONG'} {trade['ticker']}: "
                  f"${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f}, "
                  f"P&L: ${trade['pnl']:,.2f}")

    print(f"\n{'='*100}\n")


def main():
    """Debug episodes for each strategy."""

    # Configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}\n")

    # Load dataset
    print("Loading dataset...")
    data_loader = DatasetLoader(
        dataset_path='data/all_complete_dataset.h5',
        prices_path='data/actual_prices.h5',
        num_test_stocks=100
    )

    all_tickers = list(data_loader.h5_file.keys()) if data_loader.is_hdf5 else list(data_loader.data.keys())
    all_tickers_with_prices = [ticker for ticker in all_tickers if ticker in data_loader.prices_file]
    print(f"✅ Loaded {len(all_tickers_with_prices)} tickers with price data\n")

    # Create dummy agent
    print("Creating dummy agent...")
    dummy_agent = TradingAgent(
        predictor_checkpoint_path='./checkpoints/best_model_100m_1.18.pt',
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(device)
    dummy_agent.train()
    print("✅ Dummy agent created\n")

    # Load stock cache
    print("Loading stock selection cache...")
    with h5py.File('data/actual_prices.h5', 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path='data/rl_stock_selections_4yr.h5',
        prices_file_keys=prices_file_keys,
        device=device
    )
    print(f"✅ Stock cache loaded ({len(stock_cache.selections)} dates)\n")

    # Create temp environment
    print("Creating environment...")
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=100000,
        max_positions=1,
        episode_length=30,
        device=device,
        transaction_cost=0.0
    )

    # Load caches
    if os.path.exists('data/rl_state_cache_4yr.h5'):
        temp_env.load_state_cache('data/rl_state_cache_4yr.h5')
        print("✅ State cache loaded\n")

    # Get test dates
    test_start = '2023-07-01'
    test_end = '2024-12-31'
    all_dates = sorted(stock_cache.selections.keys())
    test_dates = [d for d in all_dates if test_start <= d <= test_end]

    print(f"Test period: {len(test_dates)} trading days from {test_start} to {test_end}\n")

    # Debug each strategy
    strategies = {
        'DQN': None,  # Skip DQN for now, it's more complex
        'random': get_baseline_strategy('random', 100000),
        'hold': get_baseline_strategy('hold', 100000),
        'long': get_baseline_strategy('long', 100000),
        'random_universe': get_baseline_strategy('random_universe', 100000, all_tickers=all_tickers_with_prices)
    }

    for strategy_name, strategy in strategies.items():
        if strategy is None:
            continue

        debug_episode(
            strategy_name=strategy_name,
            strategy=strategy,
            data_loader=data_loader,
            stock_cache=stock_cache,
            dummy_agent=dummy_agent,
            temp_env=temp_env,
            test_dates=test_dates,
            initial_capital=100000,
            episode_length=30,
            device=device
        )

        input("Press Enter to continue to next strategy...")


if __name__ == '__main__':
    main()
