#!/usr/bin/env python3
"""
Generate synthetic training data for critic pre-training.

Uses a programmatic trading strategy with the following rules:
1. Single stock at a time (max_positions=1)
2. Randomly sample time horizon (1d, 3d, 5d, 10d)
3. Buy top-ranked stock for that horizon
4. Sell probabilistically based on:
   - Relative rank (compared to other stocks for same horizon)
   - Time held (approaching target horizon)
5. Random exploration for robustness

This generates a large dataset of (state, action, reward, next_state, done) tuples
that the critic can learn from BEFORE going online with the actor.
"""

import torch
import numpy as np
import sys
import os
import random
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import ActorCriticAgent, ReplayBuffer
from rl.rl_environment import TradingEnvironment
from rl.reduced_action_space import get_top_4_stocks, create_global_state, decode_action_to_trades
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients
torch.set_grad_enabled(True)


class ProgrammaticTradingStrategy:
    """
    Programmatic trading strategy for generating synthetic data.

    Strategy:
    - Single stock at a time (simplified)
    - Buy: Randomly sample horizon, buy top-ranked stock for that horizon
    - Sell: Probabilistic based on relative rank + time held
    - Exploration: ε% random actions

    Hyperparameters are randomized per episode for better distribution coverage.
    """

    def __init__(self,
                 alpha: Optional[float] = None,  # Weight for time factor (randomized if None)
                 beta: Optional[float] = None,   # Weight for rank factor (randomized if None)
                 epsilon: Optional[float] = None,  # Random exploration rate (randomized if None)
                 randomize_per_episode: bool = True):  # Whether to randomize params per episode
        """
        Initialize strategy.

        Args:
            alpha: Weight for time factor in sell probability (randomized if None)
            beta: Weight for rank factor in sell probability (randomized if None)
            epsilon: Probability of taking random action (randomized if None)
            randomize_per_episode: If True, randomize hyperparameters for each episode
        """
        self.randomize_per_episode = randomize_per_episode

        # Set initial hyperparameters (will be randomized if None)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        # Randomize now if not provided
        if self.alpha is None or self.beta is None or self.epsilon is None:
            self._randomize_hyperparameters()

        # Time horizons available (in days)
        self.horizons = [1, 3, 5, 10]
        self.horizon_indices = [0, 1, 2, 3]  # Corresponding indices in predictions

        # Current position tracking
        self.current_position: Optional[str] = None  # Ticker
        self.position_target_horizon: Optional[int] = None  # Days
        self.position_horizon_idx: Optional[int] = None  # Index in [0,1,2,3]
        self.days_held: int = 0

    def _randomize_hyperparameters(self):
        """Randomize strategy hyperparameters for diversity."""
        # Alpha: time factor weight (0.2 to 0.8)
        # Low α = more rank-sensitive (sell fast if rank drops)
        # High α = more patient (wait for horizon)
        self.alpha = np.random.uniform(0.2, 0.8)

        # Beta: rank factor weight (0.2 to 0.8)
        # Low β = hold even if rank drops
        # High β = sell aggressively when rank drops
        self.beta = np.random.uniform(0.2, 0.8)

        # Epsilon: exploration rate (0.05 to 0.25)
        # Low ε = mostly follow strategy
        # High ε = more random exploration
        self.epsilon = np.random.uniform(0.05, 0.25)

    def reset_episode(self):
        """Reset for new episode, potentially randomizing hyperparameters."""
        if self.randomize_per_episode:
            self._randomize_hyperparameters()

        self.reset_position()

    def reset_position(self):
        """Reset position tracking (called when position is closed)."""
        self.current_position = None
        self.position_target_horizon = None
        self.position_horizon_idx = None
        self.days_held = 0

    def get_stock_rankings(self,
                          cached_states: Dict[str, torch.Tensor],
                          available_tickers: set,
                          horizon_idx: int) -> List[Tuple[str, float]]:
        """
        Get stocks ranked by expected return for a specific horizon.

        Args:
            cached_states: Dictionary mapping ticker -> cached state (predictor features + context)
            available_tickers: Set of tickers that are actually available (in filtered states)
            horizon_idx: Index of horizon (0=1d, 1=3d, 2=5d, 3=10d)

        Returns:
            List of (ticker, expected_return) tuples, sorted by return (descending)
            Only includes tickers that are in available_tickers
        """
        stock_returns = []

        for ticker, cached_state in cached_states.items():
            # Only consider stocks that are in the filtered states
            if ticker not in available_tickers:
                continue

            # Extract predictor features (first 1444 dims)
            # Handle both 1D and 2D tensors
            if cached_state.dim() > 1:
                cached_state = cached_state.squeeze()

            pred_features = cached_state[:1444]

            # Expected returns are at positions 1428:1432
            expected_returns = pred_features[1428:1432]

            # Get return for this horizon
            # Handle both scalar and 1-element tensors
            horizon_value = expected_returns[horizon_idx]
            if isinstance(horizon_value, torch.Tensor):
                if horizon_value.numel() == 1:
                    horizon_return = horizon_value.item()
                else:
                    # Multi-element tensor - just take first element
                    horizon_return = horizon_value.flatten()[0].item()
            else:
                horizon_return = float(horizon_value)

            stock_returns.append((ticker, horizon_return))

        # Sort by return (descending)
        stock_returns.sort(key=lambda x: x[1], reverse=True)

        return stock_returns

    def get_percentile_rank(self,
                           ticker: str,
                           cached_states: Dict[str, torch.Tensor],
                           available_tickers: set,
                           horizon_idx: int) -> float:
        """
        Get percentile rank of a stock for a specific horizon.

        Args:
            ticker: Stock ticker
            cached_states: Dictionary mapping ticker -> cached state
            available_tickers: Set of tickers that are actually available
            horizon_idx: Index of horizon

        Returns:
            Percentile rank (0=worst, 1=best)
        """
        rankings = self.get_stock_rankings(cached_states, available_tickers, horizon_idx)

        # Find ticker's position
        for i, (t, _) in enumerate(rankings):
            if t == ticker:
                # Percentile rank: 0 = worst (last), 1 = best (first)
                return 1.0 - (i / len(rankings))

        # Shouldn't happen, but return median if ticker not found
        return 0.5

    def calculate_sell_probability(self,
                                   percentile_rank: float,
                                   days_held: int,
                                   target_horizon: int) -> float:
        """
        Calculate sell probability based on rank and time held.

        Args:
            percentile_rank: Rank among all stocks (0=worst, 1=best)
            days_held: Number of days position has been held
            target_horizon: Target holding period (days)

        Returns:
            Sell probability (0-1)
        """
        # Time factor: increases as we approach target horizon
        time_factor = min(1.0, days_held / target_horizon)

        # Rank factor: 0 if above median, increases as rank drops below median
        # If rank < 0.5 (below median), rank_factor > 0
        rank_factor = max(0.0, (0.5 - percentile_rank) / 0.5)

        # Combined probability
        sell_prob = min(1.0, self.alpha * time_factor + self.beta * rank_factor)

        return sell_prob

    def select_action(self,
                     top_4_stocks: List[Tuple[str, int]],
                     cached_states: Dict[str, torch.Tensor],
                     current_position: Optional[str]) -> int:
        """
        Select action using programmatic strategy (reduced action space).

        Args:
            top_4_stocks: List of (ticker, horizon_idx) for top 4 stocks
            cached_states: Dictionary mapping ticker -> cached state (predictor features)
            current_position: Currently held stock ticker (or None if in cash)

        Returns:
            Discrete action (0-4):
                0 = Hold current position (cash or stock)
                1-4 = Switch to stock 1-4
        """
        # Random exploration
        if random.random() < self.epsilon:
            return random.randint(0, 4)

        # Extract stock tickers from top_4_stocks
        stock_tickers = [ticker for ticker, _ in top_4_stocks]

        # Check if holding position
        if current_position is not None:
            # We're holding a position - decide whether to hold or switch
            self.days_held += 1

            # Check if current position is still in top 4
            current_in_top4 = current_position in stock_tickers

            if not current_in_top4:
                # Current stock not in top 4 anymore - switch to best stock
                # Pick action 1 (top stock for 1d horizon) as default
                self.reset_position()
                return 1

            # Find which action corresponds to current position
            try:
                current_action_idx = stock_tickers.index(current_position) + 1  # 1-4
            except ValueError:
                # Shouldn't happen, but fallback to switching
                self.reset_position()
                return 1

            # Get horizon for current position
            if self.position_horizon_idx is None:
                # Position tracking lost - switch to a new stock
                self.reset_position()
                return 1

            # Calculate switch probability (similar to old sell probability)
            # Get current stock's rank for all horizons
            best_horizon_idx = self.position_horizon_idx
            best_rank = 0.0

            # Check rank across all horizons
            for horizon_idx in self.horizon_indices:
                try:
                    # For top-4 stocks, rank is implicit from position
                    # Stock at index i in top_4_stocks is rank i for its horizon
                    # But we need to check if current stock would be top for other horizons
                    for idx, (ticker, h_idx) in enumerate(top_4_stocks):
                        if ticker == current_position and h_idx == horizon_idx:
                            rank = 1.0 - (idx / 4)  # Approximate percentile
                            if rank > best_rank:
                                best_rank = rank
                                best_horizon_idx = horizon_idx
                except Exception:
                    pass

            # Time factor: increases as we approach target horizon
            if self.position_target_horizon is not None:
                time_factor = min(1.0, self.days_held / self.position_target_horizon)
            else:
                time_factor = 0.5

            # Rank factor: higher if rank is low (below 0.5)
            rank_factor = max(0.0, (0.5 - best_rank) / 0.5)

            # Switch probability (similar to sell probability)
            switch_prob = min(0.8, self.alpha * time_factor + self.beta * rank_factor)

            # Decide whether to switch
            if random.random() < switch_prob:
                # Switch to a different stock (weighted by expected returns)
                # Get expected returns for all 4 stocks
                returns = []
                for ticker, horizon_idx in top_4_stocks:
                    if ticker == current_position:
                        continue  # Don't switch to same stock

                    if ticker in cached_states:
                        cached_state = cached_states[ticker]
                        if cached_state.dim() > 1:
                            cached_state = cached_state.squeeze()
                        pred_features = cached_state[:1444]
                        expected_returns = pred_features[1428:1432]
                        horizon_return = expected_returns[horizon_idx].item() if isinstance(expected_returns[horizon_idx], torch.Tensor) else float(expected_returns[horizon_idx])
                        returns.append((ticker, horizon_return))

                if returns:
                    # Pick best alternative stock
                    best_ticker, _ = max(returns, key=lambda x: x[1])
                    new_action = stock_tickers.index(best_ticker) + 1

                    # Update tracking
                    self.current_position = best_ticker
                    for ticker, horizon_idx in top_4_stocks:
                        if ticker == best_ticker:
                            self.position_horizon_idx = horizon_idx
                            self.position_target_horizon = self.horizons[horizon_idx]
                            break
                    self.days_held = 0

                    return new_action
                else:
                    # No alternatives - hold current
                    return 0
            else:
                # Hold current position
                return 0

        else:
            # Not holding anything - pick a stock to buy
            # Randomly sample a time horizon to target
            horizon_idx = random.choice(self.horizon_indices)
            target_horizon = self.horizons[horizon_idx]

            # Find the stock matching this horizon in top_4_stocks
            target_ticker = None
            for ticker, h_idx in top_4_stocks:
                if h_idx == horizon_idx:
                    target_ticker = ticker
                    break

            if target_ticker is None:
                # Fallback: pick first stock
                target_ticker = stock_tickers[0]
                horizon_idx = top_4_stocks[0][1]
                target_horizon = self.horizons[horizon_idx]

            # Record position info
            self.current_position = target_ticker
            self.position_target_horizon = target_horizon
            self.position_horizon_idx = horizon_idx
            self.days_held = 0

            # Return action for this stock (1-4)
            action = stock_tickers.index(target_ticker) + 1
            return action


class SimpleBatchedEpisodeRunner:
    """
    Runs multiple episodes in a single pass through dates.
    Based on BatchedRandomSimulator pattern from statistical_comparison.py.

    Key difference from BatchedRandomSimulator: Each episode has different portfolio state,
    so states must be computed per-episode (not shared). We update self.env to each episode's
    state (portfolio, cash, step_idx) before calling _get_states() to ensure portfolio context
    is correct. This is critical for the critic to learn proper Q-values.

    One pass through dates, all episodes processed at their respective dates.
    """

    def __init__(
        self,
        num_episodes: int,
        env: TradingEnvironment,
        episode_length: int,
        base_seed: int
    ):
        """
        Initialize batched episode runner.

        Args:
            num_episodes: Number of episodes to run
            env: Initialized TradingEnvironment (shared)
            episode_length: Length of each episode
            base_seed: Base random seed
        """
        self.num_episodes = num_episodes
        self.env = env
        self.episode_length = episode_length
        self.base_seed = base_seed

        # Initialize strategy for each episode (randomized hyperparameters)
        self.strategies = []
        for i in range(num_episodes):
            np.random.seed(base_seed + i)
            random.seed(base_seed + i)
            strategy = ProgrammaticTradingStrategy(randomize_per_episode=False)
            strategy._randomize_hyperparameters()
            self.strategies.append(strategy)

        # State for each episode
        self.episode_portfolios = [{} for _ in range(num_episodes)]
        self.episode_cash = [env.initial_capital] * num_episodes
        self.episode_step = [0] * num_episodes
        self.episode_active = [True] * num_episodes  # Track if episode still running
        self.episode_positions = [None] * num_episodes  # Current position (ticker or None)

        # Storage
        self.episode_transitions = [[] for _ in range(num_episodes)]
        self.episode_returns = [0.0] * num_episodes

    def run_all_episodes(self, trading_days: List[str]) -> Tuple[List[List[Dict]], List[float], List[Dict]]:
        """
        Run all episodes with random start dates (batched by date).

        Each episode gets random start date for state diversity.
        Episodes are processed in batches by date for efficiency.

        Returns:
            Tuple of (all_transitions, episode_returns, hyperparameters)
        """
        print(f"\n  Running {self.num_episodes} episodes with random start dates (batched)...")

        # Assign random start dates to each episode
        max_start_idx = len(trading_days) - self.episode_length - 1
        if max_start_idx < 0:
            raise ValueError("Not enough trading days for episode length")

        episode_start_dates = []
        episode_date_ranges = []
        for i in range(self.num_episodes):
            np.random.seed(self.base_seed + i + 1000)
            start_idx = np.random.randint(0, max_start_idx + 1)
            start_date = trading_days[start_idx]
            episode_dates = trading_days[start_idx:start_idx + self.episode_length]
            episode_start_dates.append(start_date)
            episode_date_ranges.append(episode_dates)

        # Find all unique dates needed (union of all episode date ranges)
        all_dates_needed = set()
        for dates in episode_date_ranges:
            all_dates_needed.update(dates)
        all_dates_sorted = sorted(all_dates_needed, key=lambda d: trading_days.index(d))

        print(f"   Processing {len(all_dates_sorted)} unique dates (covering all episode windows)")

        # Single pass through all dates needed
        for date in tqdm(all_dates_sorted, desc="  Trading days"):
            # Set current date in environment
            self.env.current_date = date
            cached_states = self.env.state_cache.get(date, {})
            cached_prices = self.env.price_cache.get(date, {})

            # PERFORMANCE OPTIMIZATION: Compute filtered tickers ONCE per date instead of per episode
            # With 50k episodes and ~20 active per date, this saves ~20x filtering operations per date
            # (from ~40 calls per date to 1 call per date = 40x speedup on filtering)
            self.env.portfolio = {}  # Temporarily empty for filtering
            self.env.cash = self.env.initial_capital
            filtered_tickers = self.env._filter_top_k_stocks(cached_states)

            # Pre-extract predictor features for all filtered stocks (CPU tensors, cached)
            pred_features_cache = {}
            for ticker in filtered_tickers:
                if ticker in cached_states:
                    pred_features_cache[ticker] = cached_states[ticker][:1444]  # First 1444 dims

            # Process each episode that includes this date
            for ep_idx in range(self.num_episodes):
                if not self.episode_active[ep_idx]:
                    continue

                # Check if this date is in this episode's range
                if date not in episode_date_ranges[ep_idx]:
                    continue

                # Check if episode finished
                if self.episode_step[ep_idx] >= self.episode_length:
                    self.episode_active[ep_idx] = False
                    continue

                # Compute portfolio value for this episode (for portfolio context)
                portfolio_value = self._compute_portfolio_value(
                    self.episode_portfolios[ep_idx],
                    self.episode_cash[ep_idx],
                    cached_prices
                )

                # Get top-4 stocks (one per time horizon)
                top_4_stocks = get_top_4_stocks(cached_states)

                # OPTIMIZED: Create states with correct portfolio context (no GPU transfer, stay on CPU)
                states = self._create_states_for_episode(
                    ep_idx,
                    filtered_tickers,
                    pred_features_cache,
                    cached_prices,
                    portfolio_value
                )

                # Create global state (4 stocks × 1469 + 5 position encoding = 5881 dims)
                current_position = self.episode_positions[ep_idx]
                try:
                    global_state = create_global_state(
                        top_4_stocks=top_4_stocks,
                        states=states,
                        current_position=current_position,
                        device='cpu'
                    )
                except (KeyError, IndexError, ValueError):
                    self.episode_step[ep_idx] += 1
                    continue

                # Select discrete action (0-4)
                try:
                    action = self.strategies[ep_idx].select_action(
                        top_4_stocks, cached_states, current_position
                    )
                except (KeyError, IndexError, ValueError):
                    self.episode_step[ep_idx] += 1
                    continue

                # Decode action to trades
                trades = decode_action_to_trades(action, top_4_stocks, current_position)

                # Execute trades
                reward = 0.0
                next_portfolio = {k: v.copy() for k, v in self.episode_portfolios[ep_idx].items()}
                next_cash = self.episode_cash[ep_idx]
                next_position = current_position

                for trade in trades:
                    ticker = trade['ticker']
                    trade_reward, next_portfolio, next_cash = self._execute_action(
                        ep_idx, ticker, 1 if trade['action'] == 'BUY' else 2, date
                    )
                    reward += trade_reward

                    # Update next position
                    if trade['action'] == 'SELL':
                        next_position = None
                    elif trade['action'] == 'BUY':
                        next_position = ticker

                # If no trades, reward is 0
                if not trades:
                    reward = 0.0

                # Create next global state
                next_portfolio_value = self._compute_portfolio_value(
                    next_portfolio,
                    next_cash,
                    cached_prices
                )
                next_cached_states = self.env.state_cache[date]  # Same date, updated position
                next_top_4_stocks = get_top_4_stocks(next_cached_states)

                # Create next states with updated portfolio
                next_states = {}
                for ticker in filtered_tickers:
                    if ticker in pred_features_cache:
                        next_states[ticker] = self._create_single_state(
                            ticker,
                            pred_features_cache[ticker],
                            cached_prices.get(ticker),
                            next_portfolio,
                            next_cash,
                            next_portfolio_value,
                            self.episode_step[ep_idx] + 1
                        )

                try:
                    next_global_state = create_global_state(
                        top_4_stocks=next_top_4_stocks,
                        states=next_states,
                        current_position=next_position,
                        device='cpu'
                    )
                except (KeyError, IndexError, ValueError):
                    next_global_state = global_state  # Fallback

                # Store transition with global states and discrete action
                transition = {
                    'state': global_state.cpu(),
                    'action': action,  # Discrete action (0-4)
                    'reward': reward,
                    'next_state': next_global_state.cpu(),
                    'done': False,
                    'ticker': next_position if next_position else 'CASH',
                    'portfolio_value': next_cash
                }
                self.episode_transitions[ep_idx].append(transition)
                self.episode_returns[ep_idx] += reward

                # Update episode state
                self.episode_portfolios[ep_idx] = next_portfolio
                self.episode_cash[ep_idx] = next_cash
                self.episode_positions[ep_idx] = next_position
                self.episode_step[ep_idx] += 1

        # Collect hyperparameters
        hyperparameters = []
        for strategy in self.strategies:
            hyperparameters.append({
                'alpha': strategy.alpha,
                'beta': strategy.beta,
                'epsilon': strategy.epsilon
            })

        return self.episode_transitions, self.episode_returns, hyperparameters

    def _compute_portfolio_value(self, portfolio: Dict, cash: float, cached_prices: Dict[str, float]) -> float:
        """
        Compute total portfolio value (positions + cash).

        Args:
            portfolio: Episode portfolio dict
            cash: Available cash
            cached_prices: Price cache for current date

        Returns:
            Total portfolio value
        """
        position_value = 0.0
        for ticker, pos in portfolio.items():
            current_price = cached_prices.get(ticker)
            if current_price is not None:
                position_value += pos['size'] * current_price
        return position_value + cash

    def _create_states_for_episode(
        self,
        ep_idx: int,
        filtered_tickers: List[str],
        pred_features_cache: Dict[str, torch.Tensor],
        cached_prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, torch.Tensor]:
        """
        Create states dict for an episode (CPU tensors, no GPU transfer).

        Args:
            ep_idx: Episode index
            filtered_tickers: Top-k filtered tickers for this date
            pred_features_cache: Cached predictor features (1444 dims) per ticker
            cached_prices: Cached prices per ticker
            portfolio_value: Total portfolio value for this episode

        Returns:
            Dict mapping ticker -> state tensor (CPU, 1469 dims)
        """
        portfolio = self.episode_portfolios[ep_idx]
        cash = self.episode_cash[ep_idx]
        step_idx = self.episode_step[ep_idx]

        states = {}
        for ticker in filtered_tickers:
            pred_features = pred_features_cache.get(ticker)
            current_price = cached_prices.get(ticker)

            if pred_features is None or current_price is None:
                continue

            # Create portfolio context (25 dims)
            portfolio_context = self._create_portfolio_context_cpu(
                ticker, current_price, portfolio, cash, portfolio_value, step_idx
            )

            # Concatenate: 1444 + 25 = 1469 dims (stay on CPU)
            state = torch.cat([pred_features, portfolio_context])
            states[ticker] = state

        return states

    def _create_single_state(
        self,
        ticker: str,
        pred_features: torch.Tensor,
        current_price: float,
        portfolio: Dict,
        cash: float,
        portfolio_value: float,
        step_idx: int
    ) -> torch.Tensor:
        """
        Create single state tensor for one ticker (CPU, no GPU transfer).

        Returns:
            State tensor (CPU, 1469 dims)
        """
        if pred_features is None or current_price is None:
            return None

        portfolio_context = self._create_portfolio_context_cpu(
            ticker, current_price, portfolio, cash, portfolio_value, step_idx
        )

        return torch.cat([pred_features, portfolio_context])

    def _create_portfolio_context_cpu(
        self,
        ticker: str,
        current_price: float,
        portfolio: Dict,
        cash: float,
        portfolio_value: float,
        step_idx: int
    ) -> torch.Tensor:
        """
        Create portfolio context tensor (25 dims, CPU).
        Matches environment's _create_portfolio_context_fast() logic.

        Returns:
            Portfolio context tensor (CPU, 25 dims)
        """
        # Check if we have a position in this stock
        if ticker in portfolio:
            pos = portfolio[ticker]
            unrealized_return = (current_price / pos['entry_price']) - 1.0
            position_value = pos['size'] * current_price
            portfolio_weight = position_value / portfolio_value if portfolio_value > 0 else 0.0

            context = [
                pos['size'],                              # Position size (shares)
                pos['days_held'],                         # Days held
                pos['entry_price'],                       # Entry price
                current_price,                            # Current price
                unrealized_return,                        # Unrealized return (%)
                position_value,                           # Position value ($)
                portfolio_weight,                         # Portfolio weight (0-1)
                1.0,                                      # Has position flag
            ]
        else:
            # No position
            context = [0.0] * 7 + [0.0]  # 7 zeros + no position flag

        # Add global portfolio info
        context.extend([
            cash,                                         # Available cash
            cash / portfolio_value if portfolio_value > 0 else 1.0,  # Cash ratio
            len(portfolio),                               # Number of positions
            self.env.max_positions - len(portfolio),      # Remaining position slots
            step_idx,                                     # Current step in episode
            self.episode_length - step_idx,               # Steps remaining
        ])

        # Convert to tensor on CPU
        context_tensor = torch.tensor(context, dtype=torch.float32, device='cpu')

        # Pad to 25 dims if needed
        if len(context) < 25:
            padding = torch.zeros(25 - len(context), dtype=torch.float32, device='cpu')
            context_tensor = torch.cat([context_tensor, padding])
        elif len(context) > 25:
            context_tensor = context_tensor[:25]

        return context_tensor

    def _execute_action(self, ep_idx: int, ticker: str, action: int, date: str) -> Tuple[float, Dict, float]:
        """
        Execute action for an episode (matches environment's execution logic).

        Returns:
            Tuple of (reward, next_portfolio, next_cash)
        """
        portfolio = self.episode_portfolios[ep_idx]
        cash = self.episode_cash[ep_idx]

        # Get current price
        result = self.env.data_loader.get_features_and_price(ticker, date)
        if result is None:
            return 0.0, portfolio, cash

        _, current_price = result

        reward = 0.0
        next_portfolio = portfolio.copy()
        next_cash = cash

        if action == 1:  # BUY
            if len(portfolio) == 0:  # Not holding
                allocation = cash * 0.5
                shares = allocation / current_price
                # FIX: Use same structure as environment (size, not shares)
                next_portfolio[ticker] = {
                    'size': shares,
                    'entry_price': current_price,
                    'entry_date': date,
                    'days_held': 0
                }
                next_cash = cash - allocation

        elif action == 2:  # SELL
            if ticker in portfolio:  # Holding this stock
                # FIX: Use 'size' to match environment structure
                shares = portfolio[ticker]['size']
                entry_price = portfolio[ticker]['entry_price']
                exit_value = shares * current_price
                profit = exit_value - (shares * entry_price)
                reward = profit / (shares * entry_price)  # Return as fraction
                next_cash = cash + exit_value
                next_portfolio = {}

        # Update days_held for all positions (if HOLD or BUY)
        if action != 2:  # Not selling
            for tick, pos in next_portfolio.items():
                pos['days_held'] = pos.get('days_held', 0) + 1

        return reward, next_portfolio, next_cash


def compute_monte_carlo_returns(episode_transitions: List[Dict], gamma: float = 0.99) -> List[float]:
    """
    Compute full Monte Carlo returns for each transition in an episode.

    For each transition at time t, computes:
        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T

    This gives the actual cumulative discounted return from that point,
    providing zero-bias targets for critic training (vs bootstrapped TD).

    Args:
        episode_transitions: List of transitions from one episode
        gamma: Discount factor

    Returns:
        List of Monte Carlo returns (one per transition)
    """
    T = len(episode_transitions)
    mc_returns = []

    for t in range(T):
        G = 0.0
        for k in range(t, T):
            # Accumulate discounted rewards from t to end
            G += (gamma ** (k - t)) * episode_transitions[k]['reward']
        mc_returns.append(G)

    return mc_returns


def deduplicate_transitions(buffer: ReplayBuffer) -> ReplayBuffer:
    """
    [DEPRECATED - Not used with Monte Carlo approach]

    Deduplicate transitions based on (state, action) pairs.

    With Monte Carlo returns, duplicates are valuable (same state-action
    can have different returns), so this function is no longer used.

    Args:
        buffer: ReplayBuffer with potentially duplicate transitions

    Returns:
        New ReplayBuffer with deduplicated transitions
    """
    print(f"   Deduplicating {len(buffer)} transitions...")

    # Create hash for (state, action) pairs
    seen_pairs = set()
    unique_transitions = []

    for transition in buffer.buffer:
        # Create hashable key from state and action
        # Use state tensor hash + action value
        state_hash = hash(transition['state'].cpu().numpy().tobytes())
        action = transition['action']
        pair_key = (state_hash, action)

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_transitions.append(transition)

    print(f"   Removed {len(buffer) - len(unique_transitions):,} duplicate transitions")
    print(f"   Unique transitions: {len(unique_transitions):,}")

    # Create new buffer with deduplicated transitions
    deduped_buffer = ReplayBuffer(capacity=len(unique_transitions) * 2)
    for transition in unique_transitions:
        deduped_buffer.push(**transition)

    return deduped_buffer


def _run_episode_worker(args):
    """
    Worker function to run a single episode (must be top-level for multiprocessing).

    NOTE: This function is kept for backward compatibility but is no longer used.
    Use BatchedEpisodeRunner for vectorized execution instead.

    Args:
        args: Tuple of (episode_id, config_dict)

    Returns:
        Tuple of (transitions_list, episode_return, hyperparameters)
    """
    episode_id, config = args

    # Set random seed for this worker (ensures different randomness per episode)
    np.random.seed(episode_id + int(datetime.now().timestamp()))
    random.seed(episode_id + int(datetime.now().timestamp()))
    torch.manual_seed(episode_id + int(datetime.now().timestamp()))

    # Unpack config
    dataset_path = config['dataset_path']
    prices_path = config['prices_path']
    predictor_checkpoint = config['predictor_checkpoint']
    num_test_stocks = config['num_test_stocks']
    episode_length = config['episode_length']
    recent_trading_days = config['recent_trading_days']
    device = config.get('device', 'cpu')  # Use CPU for workers

    # Initialize data loader (each worker gets its own)
    data_loader = DatasetLoader(
        dataset_path=dataset_path,
        prices_path=prices_path,
        num_test_stocks=num_test_stocks
    )

    # Load feature cache if available
    cache_path = config.get('feature_cache_path', 'data/rl_feature_cache_4yr.h5')
    if os.path.exists(cache_path):
        data_loader.load_feature_cache(cache_path)
    else:
        # This shouldn't happen if main process already created cache
        data_loader.preload_features(recent_trading_days)

    # Initialize agent (CPU only for workers)
    agent = ActorCriticAgent(
        predictor_checkpoint_path=predictor_checkpoint,
        state_dim=5881,  # 4 stocks × 1469 + 5 position encoding
        hidden_dim=1024,
        action_dim=5  # cash + 4 stocks
    ).to(device)

    agent.feature_extractor.freeze_predictor()
    agent.actor.eval()
    agent.critic.eval()

    # Initialize environment
    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=1,
        episode_length=episode_length,
        device=device,
        trading_days_filter=recent_trading_days,
        top_k_per_horizon=10
    )

    # Load state cache if available
    state_cache_path = config.get('state_cache_path', 'data/rl_state_cache_4yr.h5')
    if os.path.exists(state_cache_path):
        env.load_state_cache(state_cache_path)
    else:
        # This shouldn't happen if main process already created cache
        env._precompute_all_states()

    # Initialize strategy with randomized hyperparameters
    strategy = ProgrammaticTradingStrategy(randomize_per_episode=True)

    # Run episode
    states = env.reset()
    strategy.reset_episode()
    episode_return = 0.0
    transitions = []

    for step in range(episode_length):
        cached_states = env.state_cache[env.current_date]

        try:
            ticker, action = strategy.select_action(states, cached_states, env.portfolio)
        except (KeyError, IndexError, ValueError):
            if step == 0:
                break
            continue

        if ticker not in states:
            continue

        actions = {ticker: action}

        try:
            next_states, reward, done, info = env.step(actions)
        except Exception:
            continue

        if ticker not in states:
            continue

        if ticker in next_states:
            next_state = next_states[ticker]
        else:
            if len(next_states) > 0:
                next_state = list(next_states.values())[0]
            else:
                next_state = states[ticker]

        # Store transition as dict
        transition = {
            'state': states[ticker].cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'ticker': ticker,
            'portfolio_value': info['portfolio_value']
        }
        transitions.append(transition)

        episode_return += reward
        states = next_states if len(next_states) > 0 else states

        if done:
            break

    # Return episode data
    hyperparams = {
        'alpha': strategy.alpha,
        'beta': strategy.beta,
        'epsilon': strategy.epsilon
    }

    return transitions, episode_return, hyperparams


def generate_synthetic_data(
    num_episodes: int = 1000,
    episode_length: int = 20,
    dataset_path: str = 'data/all_complete_dataset.h5',
    prices_path: str = 'data/actual_prices.h5',
    predictor_checkpoint: str = './checkpoints/best_model_100m_1.18.pt',
    num_test_stocks: int = 100,
    save_path: str = 'data/critic_training_data.pkl',
    device: str = 'cpu'
) -> ReplayBuffer:
    """
    Generate synthetic training data with Monte Carlo returns (VECTORIZED BATCHING).

    Uses vectorized batching (like BatchedRandomSimulator) to run episodes efficiently.
    Each transition includes the full Monte Carlo return from that point in the episode.
    No deduplication - duplicate (state, action) pairs are kept since they have different
    returns based on different episode trajectories.

    Args:
        num_episodes: Number of episodes to generate
        episode_length: Length of each episode (trading days)
        dataset_path: Path to dataset HDF5 file
        prices_path: Path to prices HDF5 file
        predictor_checkpoint: Path to predictor checkpoint
        num_test_stocks: Number of stocks to trade
        save_path: Where to save generated data
        device: Device to use (CPU recommended for data generation)

    Returns:
        ReplayBuffer containing transitions with Monte Carlo returns
    """
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC CRITIC TRAINING DATA (Monte Carlo Returns)")
    print("="*80)

    # Initialize data loader
    print("\n1. Loading data...")
    data_loader = DatasetLoader(
        dataset_path=dataset_path,
        prices_path=prices_path,
        num_test_stocks=num_test_stocks
    )
    print(f"   ✅ Loaded {len(data_loader.test_tickers)} stocks")

    # Initialize agent
    print("\n2. Initializing agent...")
    agent = ActorCriticAgent(
        predictor_checkpoint_path=predictor_checkpoint,
        state_dim=5881,  # 4 stocks × 1469 + 5 position encoding
        hidden_dim=1024,
        action_dim=5  # cash + 4 stocks
    ).to(device)

    agent.feature_extractor.freeze_predictor()
    agent.actor.eval()  # Don't need actor for data generation
    agent.critic.eval()  # Don't need critic for data generation

    # Preload features
    print("\n3. Preloading features...")
    cache_path = 'data/rl_feature_cache_4yr.h5'

    if os.path.exists(cache_path):
        print(f"   ⚡ Loading from cache: {cache_path}")
        data_loader.load_feature_cache(cache_path)
    else:
        print("   ⚠️  Cache not found - preloading from HDF5...")
        # Filter to last 4 years
        cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        sample_ticker = list(data_loader.prices_file.keys())[0]
        prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
        all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
        recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]

        data_loader.preload_features(recent_trading_days)
        data_loader.save_feature_cache(cache_path)

    # Initialize environment (single-stock mode)
    print("\n4. Initializing environment (single-stock mode)...")

    # Filter to last 4 years
    cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
    sample_ticker = list(data_loader.prices_file.keys())[0]
    prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
    all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
    recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]

    env = TradingEnvironment(
        data_loader=data_loader,
        agent=agent,
        initial_capital=100000,
        max_positions=1,  # SINGLE STOCK MODE
        episode_length=episode_length,
        device=device,
        trading_days_filter=recent_trading_days,
        top_k_per_horizon=10  # Still filter to top stocks
    )

    # Precompute states or load from cache
    print("\n5. Precomputing states...")
    state_cache_path = 'data/rl_state_cache_4yr.h5'

    if os.path.exists(state_cache_path):
        print(f"   ⚡ Loading state cache: {state_cache_path}")
        env.load_state_cache(state_cache_path)
    else:
        print("   Computing states...")
        env._precompute_all_states()
        env.save_state_cache(state_cache_path)

    # Prepare batched episode runner
    print("\n6. Preparing vectorized batch execution...")
    print(f"   Strategy config:")
    print(f"   - Hyperparameters: RANDOMIZED per episode for better distribution coverage")
    print(f"   - Alpha (time weight): Uniform(0.2, 0.8)")
    print(f"   - Beta (rank weight): Uniform(0.2, 0.8)")
    print(f"   - Epsilon (exploration): Uniform(0.05, 0.25)")

    # Generate data using vectorized batching
    print(f"\n7. Generating data ({num_episodes} episodes using vectorized batching)...")
    print(f"   Running episodes sequentially but reusing environment state")
    print(f"   (No multiprocessing overhead!)")

    runner = SimpleBatchedEpisodeRunner(
        num_episodes=num_episodes,
        env=env,
        episode_length=episode_length,
        base_seed=42
    )

    all_transitions, episode_returns, hyperparameters = runner.run_all_episodes(recent_trading_days)

    # Process results
    print(f"\n8. Computing Monte Carlo returns and collecting transitions...")

    # Collect all transitions with Monte Carlo returns
    buffer = ReplayBuffer(capacity=num_episodes * episode_length * 2)
    hyperparameter_history = {'alpha': [], 'beta': [], 'epsilon': []}

    total_transitions = 0
    for transitions_list, hyperparams in zip(all_transitions, hyperparameters):
        # Compute Monte Carlo returns for this episode
        mc_returns = compute_monte_carlo_returns(transitions_list, gamma=0.99)

        # Store transitions with MC returns
        for transition, mc_return in zip(transitions_list, mc_returns):
            # Replace immediate reward with full Monte Carlo return
            transition['mc_return'] = mc_return
            buffer.push(**transition)
            total_transitions += 1

        hyperparameter_history['alpha'].append(hyperparams['alpha'])
        hyperparameter_history['beta'].append(hyperparams['beta'])
        hyperparameter_history['epsilon'].append(hyperparams['epsilon'])

    print(f"   Total transitions collected: {total_transitions:,}")
    print(f"   (No deduplication - each episode trajectory is unique)")

    # Save buffer
    print(f"\n9. Saving data to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(buffer, f)

    print(f"   ✅ Saved {len(buffer)} transitions")

    # Statistics
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nEpisode statistics:")
    print(f"  - Episodes generated: {num_episodes}")
    print(f"  - Transitions stored: {len(buffer)}")
    print(f"  - Average episode return: {np.mean(episode_returns):.4f}")
    print(f"  - Std episode return: {np.std(episode_returns):.4f}")
    print(f"  - Min episode return: {np.min(episode_returns):.4f}")
    print(f"  - Max episode return: {np.max(episode_returns):.4f}")

    print(f"\nHyperparameter distribution (for diversity):")
    print(f"  - Alpha (time weight):")
    print(f"    Mean: {np.mean(hyperparameter_history['alpha']):.3f}, "
          f"Std: {np.std(hyperparameter_history['alpha']):.3f}, "
          f"Range: [{np.min(hyperparameter_history['alpha']):.3f}, {np.max(hyperparameter_history['alpha']):.3f}]")
    print(f"  - Beta (rank weight):")
    print(f"    Mean: {np.mean(hyperparameter_history['beta']):.3f}, "
          f"Std: {np.std(hyperparameter_history['beta']):.3f}, "
          f"Range: [{np.min(hyperparameter_history['beta']):.3f}, {np.max(hyperparameter_history['beta']):.3f}]")
    print(f"  - Epsilon (exploration):")
    print(f"    Mean: {np.mean(hyperparameter_history['epsilon']):.3f}, "
          f"Std: {np.std(hyperparameter_history['epsilon']):.3f}, "
          f"Range: [{np.min(hyperparameter_history['epsilon']):.3f}, {np.max(hyperparameter_history['epsilon']):.3f}]")

    buffer_stats = buffer.get_stats()
    print(f"\nBuffer statistics:")
    print(f"  - Avg reward: {buffer_stats['avg_reward']:.4f}")
    print(f"  - Std reward: {buffer_stats['std_reward']:.4f}")
    print(f"  - Min reward: {buffer_stats['min_reward']:.4f}")
    print(f"  - Max reward: {buffer_stats['max_reward']:.4f}")

    return buffer


if __name__ == '__main__':
    # Generate synthetic data (VECTORIZED BATCHING - no multiprocessing!)
    buffer = generate_synthetic_data(
        num_episodes=50000,  # Generate 20K episodes
        episode_length=30,  # 30 days each
        save_path='data/critic_training_data.pkl',
        device='cpu',  # CPU recommended for data generation
    )

    print("\n✅ Synthetic data generation complete!")
    print(f"   Data saved to: data/critic_training_data.pkl")
    print(f"   Ready for critic pre-training!")
