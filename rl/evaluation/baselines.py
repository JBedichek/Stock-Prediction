"""
Baseline trading strategies for comparison.

Includes random, hold, momentum, and mean reversion strategies.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional


class BaselineStrategy:
    """Base class for baseline trading strategies."""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """
        Select action given current state.

        Args:
            state: Current state dictionary
            top_4_stocks: List of (ticker, horizon) for top stocks
            bottom_4_stocks: List of (ticker, horizon) for bottom stocks

        Returns:
            Action index (0-8)
        """
        raise NotImplementedError


class RandomStrategy(BaselineStrategy):
    """
    Random agent that selects actions uniformly at random.

    This is the simplest baseline - any reasonable agent should beat this.
    """

    def __init__(self, initial_capital: float = 100000, seed: Optional[int] = None):
        super().__init__(initial_capital)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """Select action uniformly at random from 0-8."""
        return random.randint(0, 8)


class AlwaysHoldStrategy(BaselineStrategy):
    """
    Agent that always holds (never trades).

    Returns should be exactly 0% (stays at initial capital).
    """

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """Always return HOLD action."""
        return 0  # HOLD


class AlwaysLongStrategy(BaselineStrategy):
    """
    Agent that always goes long on the top predicted stock.

    Buys top stock and holds until episode end.
    """

    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.has_position = False

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """Buy top stock if no position, else hold."""
        if not self.has_position and len(top_4_stocks) > 0:
            self.has_position = True
            return 1  # BUY top stock
        return 0  # HOLD

    def reset(self):
        """Reset for new episode."""
        self.has_position = False


class AlwaysShortStrategy(BaselineStrategy):
    """
    Agent that always shorts the bottom predicted stock.

    Shorts worst stock and holds until episode end.
    """

    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.has_position = False

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """Short bottom stock if no position, else hold."""
        if not self.has_position and len(bottom_4_stocks) > 0:
            self.has_position = True
            return 5  # SHORT bottom stock
        return 0  # HOLD

    def reset(self):
        """Reset for new episode."""
        self.has_position = False


class MomentumStrategy(BaselineStrategy):
    """
    Simple momentum strategy.

    If no position:
        - Go LONG if top stock has positive momentum
        - Go SHORT if bottom stock has negative momentum
    """

    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.has_position = False
        self.position_type = None  # 'LONG' or 'SHORT'

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """
        Use simple momentum signal from stock predictions.

        Note: This is a simplified version since we don't have access to
        price history in the state. In practice, you'd calculate momentum
        from recent price changes.
        """
        if self.has_position:
            return 0  # HOLD

        # Simplified: go long on top stock
        if len(top_4_stocks) > 0:
            self.has_position = True
            self.position_type = 'LONG'
            return 1  # BUY top stock

        return 0  # HOLD

    def reset(self):
        """Reset for new episode."""
        self.has_position = False
        self.position_type = None


class LongShortStrategy(BaselineStrategy):
    """
    Market-neutral long-short strategy.

    Alternates between going long top stock and shorting bottom stock.
    """

    def __init__(self, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.trade_count = 0

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """Alternate between long and short positions."""
        if self.trade_count == 0:
            # First trade: go long
            if len(top_4_stocks) > 0:
                self.trade_count += 1
                return 1  # BUY top stock
        elif self.trade_count == 1:
            # Second trade: close long, go short
            if len(bottom_4_stocks) > 0:
                self.trade_count += 1
                return 5  # SHORT bottom stock

        return 0  # HOLD

    def reset(self):
        """Reset for new episode."""
        self.trade_count = 0


class RandomUniverseStrategy(BaselineStrategy):
    """
    Random agent that selects from the ENTIRE stock universe.

    This baseline helps measure the value of the predictor's stock selection.
    It randomly picks any stock, not just from top-k predicted stocks.
    """

    def __init__(self, initial_capital: float = 100000, all_tickers: List[str] = None, seed: Optional[int] = None):
        super().__init__(initial_capital)
        self.all_tickers = all_tickers or []
        self.has_position = False
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def select_action(self, state: Dict, top_4_stocks: List, bottom_4_stocks: List) -> int:
        """
        Randomly decide to buy a random ticker from universe or hold.

        NOTE: Returns a special action indicator and stores selected ticker.
        The evaluation code needs to handle universe-based selection differently.
        """
        if not self.has_position and len(self.all_tickers) > 0 and random.random() < 0.3:
            # 30% chance to enter a position each step
            self.has_position = True
            self.selected_ticker = random.choice(self.all_tickers)
            return -1  # Special flag: use selected_ticker instead of top_4
        elif self.has_position and random.random() < 0.2:
            # 20% chance to exit position each step
            self.has_position = False
            return -2  # Special flag: close position
        return 0  # HOLD

    def reset(self):
        """Reset for new episode."""
        self.has_position = False
        self.selected_ticker = None


def get_baseline_strategy(name: str, initial_capital: float = 100000, all_tickers: List[str] = None) -> BaselineStrategy:
    """
    Factory function to get baseline strategy by name.

    Args:
        name: Strategy name ('random', 'hold', 'long', 'short', 'momentum', 'long_short', 'random_universe')
        initial_capital: Initial capital
        all_tickers: List of all available tickers (for random_universe strategy)

    Returns:
        BaselineStrategy instance
    """
    if name == 'random_universe':
        return RandomUniverseStrategy(initial_capital, all_tickers)

    strategies = {
        'random': RandomStrategy,
        'hold': AlwaysHoldStrategy,
        'long': AlwaysLongStrategy,
        'short': AlwaysShortStrategy,
        'momentum': MomentumStrategy,
        'long_short': LongShortStrategy
    }

    if name not in strategies:
        raise ValueError(f"Unknown baseline strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](initial_capital=initial_capital)


def evaluate_baseline(
    strategy: BaselineStrategy,
    vec_env,
    stock_selections_cache,
    num_episodes: int = 100,
    episode_length: int = 30
) -> List[Dict]:
    """
    Evaluate a baseline strategy over multiple episodes.

    Args:
        strategy: Baseline strategy instance
        vec_env: Vectorized trading environment
        stock_selections_cache: GPU stock selection cache
        num_episodes: Number of episodes to run
        episode_length: Length of each episode

    Returns:
        List of episode results
    """
    from rl.reduced_action_space import create_global_state

    results = []

    for episode in range(num_episodes):
        # Reset strategy
        if hasattr(strategy, 'reset'):
            strategy.reset()

        # Reset environment
        states_list, positions_list = vec_env.reset()

        # Get stock selections for first environment
        if len(states_list) > 0 and len(vec_env.episode_dates) > 0:
            current_date = vec_env.episode_dates[0][0]
            num_samples = stock_selections_cache.get_num_samples(current_date)
            sample_idx = random.randint(0, num_samples - 1)
            top_4_stocks, bottom_4_stocks = stock_selections_cache.get_sample(current_date, sample_idx)
        else:
            top_4_stocks, bottom_4_stocks = [], []

        # Track portfolio over episode
        portfolio_values = [vec_env.initial_capital]
        trades = []

        # Run episode
        for step in range(episode_length):
            # Select action using baseline strategy
            action = strategy.select_action(states_list[0], top_4_stocks, bottom_4_stocks)

            # Convert action to trade
            if action == -1:
                # Special: RandomUniverseStrategy selected a random ticker
                if hasattr(strategy, 'selected_ticker') and strategy.selected_ticker:
                    ticker = strategy.selected_ticker
                    trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
                else:
                    trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            elif action == -2:
                # Special: RandomUniverseStrategy wants to close position
                trade = {'action': 'SELL', 'ticker': None, 'position_type': None}
            elif action == 0:
                trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            elif 1 <= action <= 4:
                stock_idx = action - 1
                if stock_idx < len(top_4_stocks):
                    ticker = top_4_stocks[stock_idx][0]
                    trade = {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
                else:
                    trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            elif 5 <= action <= 8:
                stock_idx = action - 5
                if stock_idx < len(bottom_4_stocks):
                    ticker = bottom_4_stocks[stock_idx][0]
                    trade = {'action': 'SHORT', 'ticker': ticker, 'position_type': 'SHORT'}
                else:
                    trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
            else:
                trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}

            # Step environment
            next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
                [action],
                [[trade]],
                positions_list
            )

            # Record portfolio value
            if len(infos_list) > 0:
                portfolio_values.append(infos_list[0].get('portfolio_value', vec_env.initial_capital))

                # Record trades
                episode_trades = infos_list[0].get('trades', [])
                for t in episode_trades:
                    if isinstance(t, list) and len(t) > 0:
                        trades.append(t[0])
                    elif isinstance(t, dict):
                        trades.append(t)

            # Update state
            states_list = next_states_list
            positions_list = next_positions_list

            # Check if done
            if dones_list[0]:
                break

            # Update stock selections for next step
            if len(vec_env.episode_dates) > 0 and vec_env.step_indices[0] < len(vec_env.episode_dates[0]):
                current_date = vec_env.episode_dates[0][vec_env.step_indices[0].item()]
                num_samples = stock_selections_cache.get_num_samples(current_date)
                sample_idx = random.randint(0, num_samples - 1)
                top_4_stocks, bottom_4_stocks = stock_selections_cache.get_sample(current_date, sample_idx)

        # Compute episode metrics
        from rl.evaluation.metrics import compute_all_metrics
        episode_metrics = compute_all_metrics(portfolio_values, trades)
        results.append(episode_metrics)

    return results
