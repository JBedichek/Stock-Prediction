#!/usr/bin/env python3
"""
Inference/Backtesting script for trained Actor-Critic trading agent.

Runs the trained agent on recent market data (past 4 months) and reports:
- All buy/sell trades with timestamps
- Price changes for each position
- Portfolio value over time
- Performance metrics (returns, Sharpe ratio, etc.)
"""

import torch
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import ActorCriticAgent
from rl.rl_environment import TradingEnvironment
from rl.reduced_action_space import (
    get_top_k_stocks_per_horizon, sample_top_4_from_top_k,
    create_global_state, decode_action_to_trades
)
from inference.backtest_simulation import DatasetLoader


class ActorCriticInference:
    """
    Inference engine for trained actor-critic trading agent.
    """

    def __init__(self, checkpoint_path: str, config: Dict):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        print("="*80)
        print("ACTOR-CRITIC TRADING AGENT - INFERENCE MODE")
        print("="*80)

        # Load data
        print("\n1. Loading data...")
        self.data_loader = DatasetLoader(
            dataset_path=config.get('dataset_path', 'data/all_complete_dataset.h5'),
            prices_path=config.get('prices_path', 'data/actual_prices.h5'),
            num_test_stocks=config.get('num_test_stocks', 100)
        )
        # Count stocks with price data (used for inference)
        all_tickers = list(self.data_loader.h5_file.keys()) if self.data_loader.is_hdf5 else list(self.data_loader.data.keys())
        stocks_with_prices = sum(1 for ticker in all_tickers if ticker in self.data_loader.prices_file)
        print(f"   ✅ Loaded {len(all_tickers)} total stocks")
        print(f"   ✅ {stocks_with_prices} stocks have price data")
        print(f"   🌍 Inference will use ALL {stocks_with_prices} stocks (production-realistic)")

        # Initialize agent
        print("\n2. Loading trained agent...")
        self.agent = ActorCriticAgent(
            predictor_checkpoint_path=config.get('predictor_checkpoint', './checkpoints/best_model_100m_1.18.pt'),
            state_dim=config.get('state_dim', 5881),
            hidden_dim=config.get('hidden_dim', 1024),
            action_dim=config.get('action_dim', 5)
        ).to(self.device)

        # Load checkpoint
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # DEBUG: Verify checkpoint contents
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        print(f"   Episode: {checkpoint.get('episode', 'UNKNOWN')}")

        # Load with strict=True to catch any mismatches
        missing_keys, unexpected_keys = self.agent.load_state_dict(checkpoint['agent_state_dict'], strict=False)
        if missing_keys:
            print(f"   ⚠️  MISSING KEYS: {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️  UNEXPECTED KEYS: {unexpected_keys}")

        self.agent.eval()  # Set to evaluation mode
        self.agent.feature_extractor.eval()  # CRITICAL: Ensure predictor dropout is disabled

        # DEBUG: Verify actor weights were actually loaded (check a sample weight)
        sample_weight = self.agent.actor.policy_head[0].weight[0, :5].detach().cpu().numpy()
        print(f"   Sample actor weights: {sample_weight}")  # Should NOT be uniform/zero

        print(f"   ✅ Loaded model from episode {checkpoint['episode']}")

        # Preload features
        print("\n3. Preloading features...")
        cache_path = config.get('feature_cache_path', 'data/rl_feature_cache_4yr.h5')
        if os.path.exists(cache_path):
            self.data_loader.load_feature_cache(cache_path)
        else:
            print("   ⚠️  Cache not found, features will be loaded on-demand")

        # Get inference dates (configurable period)
        print("\n4. Preparing inference dates...")
        inference_months = config.get('inference_months', 4)  # Default: last 4 months
        cutoff_date = (datetime.now() - timedelta(days=inference_months * 30)).strftime('%Y-%m-%d')
        sample_ticker = list(self.data_loader.prices_file.keys())[0]
        prices_dates_bytes = self.data_loader.prices_file[sample_ticker]['dates'][:]
        all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
        self.inference_dates = [d for d in all_trading_days if d >= cutoff_date]

        print(f"   📅 Inference period: {self.inference_dates[0]} to {self.inference_dates[-1]}")
        print(f"   📊 {len(self.inference_dates)} trading days (last ~{inference_months} months)")

        # Initialize environment
        print("\n5. Initializing environment...")
        # Use a longer episode length to allow the environment to select from the range
        self.env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config.get('initial_capital', 100000),
            max_positions=config.get('max_positions', 1),
            episode_length=min(len(self.inference_dates), 250),  # Cap at 250 days or available days
            device=self.device,
            trading_days_filter=self.inference_dates,
            top_k_per_horizon=config.get('top_k_per_horizon', 10)
        )

        # Precompute states
        state_cache_path = config.get('state_cache_path', 'data/rl_state_cache_4yr.h5')
        if os.path.exists(state_cache_path):
            print(f"   Loading state cache: {state_cache_path}")
            self.env.load_state_cache(state_cache_path)
        else:
            print("   Computing states...")
            self.env._precompute_all_states()

        print("\n✅ Inference engine ready!\n")

    def run_inference(self, force_initial_trade: bool = False, epsilon: float = 0.0) -> pd.DataFrame:
        """
        Run inference over the full period.

        Args:
            force_initial_trade: If True, force a random buy on day 1
            epsilon: Exploration rate (0.0 = greedy, >0 = epsilon-greedy exploration)
                    Set to 0.05-0.1 to test if agent can trade with exploration

        Returns:
            DataFrame with trade history and portfolio values
        """
        print("="*80)
        print("RUNNING INFERENCE")
        print("="*80 + "\n")

        if epsilon > 0:
            print(f"🎲 Exploration enabled: epsilon = {epsilon:.3f}")
            print(f"   Agent will take random actions {100*epsilon:.1f}% of the time\n")

        # Reset environment to start from first inference date
        self.env.reset(start_date=self.inference_dates[0])
        current_position = None
        initial_capital = self.config.get('initial_capital', 100000)

        # Track everything
        trades = []
        daily_portfolio_values = []
        daily_cash = []
        daily_positions = []
        all_actions = []  # Track all actions
        total_trades = 0  # Count total trades

        with torch.no_grad():
            for step in range(len(self.inference_dates)):
                # Use environment's current date (it tracks this internally)
                date = self.env.current_date

                # Get current state - USE ALL STOCKS for inference (realistic production conditions)
                all_cached_states = self.env.state_cache[date]
                # Only filter out stocks without price data, but include ALL stocks (not just test set)
                cached_states = {ticker: state for ticker, state in all_cached_states.items()
                               if ticker in self.data_loader.prices_file}

                # Get top stocks (deterministic for inference)
                top_k_per_horizon = self.config.get('top_k_per_horizon_sampling', 3)
                top_k_stocks = get_top_k_stocks_per_horizon(cached_states, k=top_k_per_horizon)
                top_4_stocks = sample_top_4_from_top_k(top_k_stocks, sample_size=4, deterministic=True)

                # Get cached prices and compute portfolio value (needed for both forced and normal trades)
                cached_prices = self.env.price_cache.get(date, {})
                portfolio_value = self.env._portfolio_value_cached(cached_prices)

                # Force an initial random trade if requested (to demonstrate trading behavior)
                if force_initial_trade and step == 0 and current_position is None:
                    print("\n🎲 Forcing initial random trade to demonstrate agent behavior...\n")
                    # Pick a random stock from top-4 that actually has data
                    import random
                    available_stocks = [ticker for ticker, _ in top_4_stocks if ticker in cached_states and ticker in cached_prices]
                    if available_stocks:
                        random_stock = random.choice(available_stocks)
                        action = [i+1 for i, (ticker, _) in enumerate(top_4_stocks) if ticker == random_stock][0]
                        trades_list = [{'action': 'BUY', 'ticker': random_stock}]
                        print(f"   Selected {random_stock} from available stocks: {available_stocks}")
                    else:
                        print("   ⚠️  No available stocks to trade, skipping forced trade")
                        trades_list = []
                else:
                    # Create states for top-4 stocks
                    states = {}

                    for ticker, _ in top_4_stocks:
                        if ticker in cached_states and ticker in cached_prices:
                            cached_state = cached_states[ticker]
                            price = cached_prices[ticker]
                            portfolio_context = self.env._create_portfolio_context_fast(
                                ticker, price, portfolio_value
                            )
                            state = torch.cat([
                                cached_state[:1444].to(self.device),
                                portfolio_context.to(self.device)
                            ])
                            states[ticker] = state

                    # DEBUG: Log policy behavior for first 10 steps
                    if step < 10 or step % 20 == 0:
                        from rl.reduced_action_space import create_global_state
                        import torch.nn.functional as F

                        global_state = create_global_state(
                            top_4_stocks, states, current_position, device=self.device
                        )
                        with torch.no_grad():
                            logits = self.agent.actor(global_state.unsqueeze(0)).squeeze(0)
                            probs = F.softmax(logits, dim=-1)

                            # Get Q-values for debugging
                            q_values1 = self.agent.critic1(global_state.unsqueeze(0)).squeeze(0)
                            q_values2 = self.agent.critic2(global_state.unsqueeze(0)).squeeze(0)
                            q_values = torch.min(q_values1, q_values2)

                        print(f"\n   🔍 DEBUG [Step {step}, Date {date}]:")
                        print(f"      Current position: {current_position}")
                        print(f"      Top-4 stocks: {[t for t, _ in top_4_stocks]}")
                        print(f"      Portfolio value: ${portfolio_value:,.2f}")
                        print(f"      Cash: ${self.env.cash:,.2f}")
                        print(f"      Step index: {self.env.step_idx}")

                        # DEBUG: Print state statistics
                        print(f"      Global state shape: {global_state.shape}")
                        print(f"      Global state mean: {global_state.mean().item():.6f}")
                        print(f"      Global state std: {global_state.std().item():.6f}")
                        print(f"      Global state min/max: {global_state.min().item():.2f} / {global_state.max().item():.2f}")

                        # Show sample of portfolio context (last 25 dims before position encoding)
                        portfolio_context_start = 4 * 1469  # After 4 stocks × 1469 dims
                        sample_portfolio = global_state[portfolio_context_start:portfolio_context_start+10]
                        print(f"      Sample portfolio context: {sample_portfolio.cpu().numpy()}")

                        print(f"      Logits: {logits.cpu().numpy()}")
                        print(f"      Action probs: {probs.cpu().numpy()}")
                        print(f"      Q-values: {q_values.cpu().numpy()}")
                        print(f"      Action 0 (HOLD) prob: {probs[0].item():.4f}")

                    # Select action (with configurable exploration)
                    deterministic = (epsilon == 0.0)  # Only deterministic if no exploration
                    action, _, _, trades_list = self.agent.select_action_reduced(
                        top_4_stocks=top_4_stocks,
                        states=states,
                        current_position=current_position,
                        epsilon=epsilon,
                        deterministic=deterministic
                    )

                    # DEBUG: Log selected action with interpretation
                    if step < 10 or step % 20 == 0:
                        action_meaning = "HOLD" if action == 0 else f"SWITCH TO {[t for t, _ in top_4_stocks][action-1]}"
                        print(f"      Selected action: {action} ({action_meaning})")
                        print(f"      Trades: {trades_list}")
                        if action == 0:
                            print(f"      ⚠️  Agent chose HOLD (action 0)")

                # Track action and trades
                all_actions.append(action)
                total_trades += len(trades_list)

                # Log trades
                for trade in trades_list:
                    ticker = trade['ticker']
                    action_type = trade['action']

                    # Get actual price from prices file
                    if ticker in self.data_loader.prices_file:
                        ticker_data = self.data_loader.prices_file[ticker]
                        dates_bytes = ticker_data['dates'][:]
                        prices_array = ticker_data['prices'][:]
                        date_to_idx = {d.decode('utf-8'): i for i, d in enumerate(dates_bytes)}
                        if date in date_to_idx:
                            price = float(prices_array[date_to_idx[date]])
                        else:
                            price = 0.0
                    else:
                        price = 0.0

                    trade_record = {
                        'date': date,
                        'action': action_type,
                        'ticker': ticker,
                        'price': price,
                        'portfolio_value': portfolio_value,
                        'cash': self.env.cash
                    }
                    trades.append(trade_record)

                    # Print trade
                    print(f"[{date}] {action_type:4s} {ticker:6s} @ ${price:7.2f} | Portfolio: ${portfolio_value:,.0f}")

                # Execute trades in environment
                actions = {}
                for trade in trades_list:
                    ticker = trade['ticker']
                    if trade['action'] == 'BUY':
                        actions[ticker] = 1
                    elif trade['action'] == 'SELL':
                        actions[ticker] = 2

                # Step environment
                _, reward, done, info = self.env.step(actions)

                # Update position tracking
                for trade in trades_list:
                    if trade['action'] == 'SELL':
                        current_position = None
                    elif trade['action'] == 'BUY':
                        current_position = trade['ticker']

                # Record daily metrics
                daily_portfolio_values.append({
                    'date': date,
                    'portfolio_value': info['portfolio_value'],
                    'return': (info['portfolio_value'] - initial_capital) / initial_capital,
                    'position': current_position,
                    'cash': self.env.cash
                })

                if done:
                    break

        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(daily_portfolio_values)

        print("\n" + "="*80)
        print("INFERENCE COMPLETE")
        print("="*80)

        # Log action distribution
        import collections
        action_counts = collections.Counter(all_actions)
        total_steps = len(all_actions)
        print(f"\n📊 Action Distribution ({total_steps} total steps):")
        for action_id in range(5):
            count = action_counts.get(action_id, 0)
            pct = 100.0 * count / total_steps if total_steps > 0 else 0
            action_name = "HOLD" if action_id == 0 else f"SWITCH-{action_id}"
            print(f"   Action {action_id} ({action_name}): {count:4d} ({pct:5.1f}%)")
        print(f"📊 Total trades executed: {total_trades}\n")

        return trades_df, portfolio_df

    def analyze_results(self, trades_df: pd.DataFrame, portfolio_df: pd.DataFrame):
        """
        Analyze and display inference results.

        Args:
            trades_df: DataFrame with all trades
            portfolio_df: DataFrame with daily portfolio values
        """
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS")
        print("="*80 + "\n")

        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        print(f"Initial Capital:    ${initial_value:,.2f}")
        print(f"Final Portfolio:    ${final_value:,.2f}")
        print(f"Total Return:       {total_return:+.2%}")
        print(f"Total P&L:          ${final_value - initial_value:+,.2f}")

        # Calculate daily returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        avg_daily_return = portfolio_df['daily_return'].mean()
        std_daily_return = portfolio_df['daily_return'].std()

        # Sharpe ratio (annualized, assuming 252 trading days)
        if std_daily_return > 0:
            sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        print(f"\nAvg Daily Return:   {avg_daily_return:+.4%}")
        print(f"Std Daily Return:   {std_daily_return:.4%}")
        print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")

        # Max drawdown
        cummax = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        print(f"Max Drawdown:       {max_drawdown:.2%}")

        # Trade statistics
        if len(trades_df) > 0:
            num_buys = len(trades_df[trades_df['action'] == 'BUY'])
            num_sells = len(trades_df[trades_df['action'] == 'SELL'])

            print(f"\nTotal Trades:       {len(trades_df)}")
            print(f"  Buys:             {num_buys}")
            print(f"  Sells:            {num_sells}")

            # Analyze buy trades
            if num_buys > 0:
                print("\nTop 5 Most Traded Stocks:")
                top_stocks = trades_df[trades_df['action'] == 'BUY']['ticker'].value_counts().head()
                for ticker, count in top_stocks.items():
                    print(f"  {ticker:6s}: {count:3d} trades")

        # Save results
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80 + "\n")

        os.makedirs('results', exist_ok=True)

        trades_path = 'results/inference_trades.csv'
        portfolio_path = 'results/inference_portfolio.csv'

        trades_df.to_csv(trades_path, index=False)
        portfolio_df.to_csv(portfolio_path, index=False)

        print(f"✅ Trades saved to:     {trades_path}")
        print(f"✅ Portfolio saved to:  {portfolio_path}")

        # Plot portfolio value
        self.plot_results(portfolio_df, trades_df)

    def plot_results(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame):
        """
        Plot portfolio value over time with trade markers.

        Args:
            portfolio_df: DataFrame with daily portfolio values
            trades_df: DataFrame with trades
        """
        plt.figure(figsize=(14, 8))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], linewidth=2, label='Portfolio Value')
        plt.axhline(y=self.config.get('initial_capital', 100000), color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

        # Mark buy/sell trades
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if len(buy_trades) > 0:
                plt.scatter(buy_trades['date'], buy_trades['portfolio_value'],
                           color='green', marker='^', s=100, alpha=0.7, label='Buy', zorder=5)

            if len(sell_trades) > 0:
                plt.scatter(sell_trades['date'], sell_trades['portfolio_value'],
                           color='red', marker='v', s=100, alpha=0.7, label='Sell', zorder=5)

        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Actor-Critic Trading Agent - Portfolio Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Plot returns
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_df['date'], portfolio_df['return'] * 100, linewidth=2, color='blue')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.title('Cumulative Return Over Time')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_path = 'results/inference_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to:       {plot_path}")

        plt.close()


if __name__ == '__main__':
    # Configuration
    config = {
        # Data
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'num_test_stocks': 100,  # Used by DatasetLoader (legacy), inference uses ALL stocks

        # Model
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',
        'state_dim': 5881,
        'hidden_dim': 1024,
        'action_dim': 5,

        # Inference period
        'inference_months': 1,  # Backtest on last N months of data

        # Inference behavior
        'epsilon': 0.15,  # Exploration rate (0.0 = greedy, 0.05-0.1 = some exploration)
                         # Set to 0.0 for production, >0 to test if agent can trade with exploration

        # Environment
        'initial_capital': 100000,
        'max_positions': 1,
        'top_k_per_horizon': 10,
        'top_k_per_horizon_sampling': 3,

        # Caching
        'feature_cache_path': 'data/rl_feature_cache_4yr.h5',
        'state_cache_path': 'data/rl_state_cache_4yr.h5',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Checkpoint to evaluate (update this path!)
    checkpoint_path = './checkpoints/actor_critic_ep30017.pt'

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        if os.path.exists('./checkpoints'):
            checkpoints = [f for f in os.listdir('./checkpoints') if f.startswith('actor_critic') and f.endswith('.pt')]
            for ckpt in sorted(checkpoints):
                print(f"  - ./checkpoints/{ckpt}")
        sys.exit(1)

    # Run inference
    inference = ActorCriticInference(checkpoint_path, config)

    # Inference parameters:
    # - force_initial_trade: True to force random buy on day 1 (for demonstration)
    # - epsilon: 0.0 for greedy (production), 0.05-0.1 to test with exploration
    trades_df, portfolio_df = inference.run_inference(
        force_initial_trade=True,
        epsilon=config.get('epsilon', 0.0)
    )

    inference.analyze_results(trades_df, portfolio_df)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
