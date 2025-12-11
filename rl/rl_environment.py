"""
Trading Environment for Reinforcement Learning

Provides a Gym-style interface for RL training. Wraps existing data
infrastructure from backtest_simulation.py and creates an episodic
trading environment where the agent learns to buy/sell stocks.
"""

import torch
import random
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import datetime


class TradingEnvironment:
    """
    RL environment for stock trading.

    The environment:
    - Manages a portfolio of stocks
    - Executes buy/sell actions
    - Computes rewards based on portfolio performance
    - Provides state observations to the agent

    Episode structure:
    - Random start date
    - Fixed length (e.g., 40 trading days)
    - Agent makes decisions daily
    - Reward = change in portfolio value
    """

    def __init__(self,
                 data_loader,
                 agent,
                 initial_capital: float = 100000,
                 max_positions: int = 10,
                 episode_length: int = 40,
                 transaction_cost: float = 0.001,
                 device: str = 'cuda'):
        """
        Initialize trading environment.

        Args:
            data_loader: DatasetLoader from backtest_simulation.py
            agent: TradingAgent for feature extraction
            initial_capital: Starting capital ($)
            max_positions: Maximum number of simultaneous positions
            episode_length: Number of trading days per episode
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            device: Device for computations
        """
        self.data_loader = data_loader
        self.agent = agent
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.device = device

        # Portfolio state
        self.portfolio = {}  # ticker -> {size, entry_price, entry_date, days_held}
        self.cash = initial_capital
        self.current_date = None
        self.dates = []
        self.step_idx = 0

        # Episode tracking
        self.episode_history = []
        self.total_trades = 0
        self.profitable_trades = 0

        print(f"✅ TradingEnvironment initialized")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Max positions: {max_positions}")
        print(f"   Episode length: {episode_length} days")
        print(f"   Transaction cost: {transaction_cost*100:.2f}%")

    def reset(self, start_date: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environment to start new episode.

        Args:
            start_date: Optional specific start date. If None, random.

        Returns:
            Initial states for all available stocks
        """
        # Select episode dates
        if start_date is None:
            # Random start date
            available_dates = self.data_loader.test_dates
            max_start_idx = len(available_dates) - self.episode_length - 1
            start_idx = random.randint(0, max_start_idx)
        else:
            start_idx = self.data_loader.test_dates.index(start_date)

        self.dates = self.data_loader.test_dates[start_idx:start_idx + self.episode_length]
        self.current_date = self.dates[0]
        self.step_idx = 0

        # Reset portfolio
        self.portfolio = {}
        self.cash = self.initial_capital
        self.episode_history = []
        self.total_trades = 0
        self.profitable_trades = 0

        # Get initial states
        states = self._get_states()

        print(f"\n📅 Episode reset: {self.dates[0]} to {self.dates[-1]} ({len(self.dates)} days)")
        print(f"   Available stocks: {len(states)}")

        return states

    def _get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get current state for all available stocks.

        State composition:
        - Predictor features: 1919 dims (transformer activations, distributions, etc.)
        - Portfolio context: Variable dims (position info, capital, etc.)

        Returns:
            Dictionary mapping ticker -> state tensor
        """
        states = {}

        for ticker in self.data_loader.test_tickers:
            state = self._get_state_for_ticker(ticker)
            if state is not None:
                states[ticker] = state

        return states

    def _get_state_for_ticker(self, ticker: str) -> Optional[torch.Tensor]:
        """
        Get state for a single stock.

        Args:
            ticker: Stock ticker

        Returns:
            State tensor or None if data not available
        """
        # Get features and price from data loader
        result = self.data_loader.get_features_and_price(ticker, self.current_date)
        if result is None:
            return None

        features, current_price = result

        # Extract predictor features (1919 dims)
        with torch.no_grad():
            # Move to device
            features = features.to(self.device)

            # Extract features using agent's feature extractor
            pred_features = self.agent.feature_extractor.extract_features(
                features.unsqueeze(0)  # Add batch dimension
            ).squeeze(0)  # Remove batch dimension

        # Create portfolio context
        portfolio_context = self._create_portfolio_context(ticker, current_price)

        # Concatenate: predictor features + portfolio context
        state = torch.cat([pred_features, portfolio_context])

        return state

    def _create_portfolio_context(self, ticker: str, current_price: float) -> torch.Tensor:
        """
        Create portfolio context features for a stock.

        Args:
            ticker: Stock ticker
            current_price: Current price

        Returns:
            Portfolio context tensor (25 dims)
        """
        # Check if we have a position in this stock
        if ticker in self.portfolio:
            pos = self.portfolio[ticker]
            unrealized_return = (current_price / pos['entry_price']) - 1.0
            position_value = pos['size'] * current_price
            portfolio_weight = position_value / self._portfolio_value()

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
        portfolio_value = self._portfolio_value()
        context.extend([
            self.cash,                                    # Available cash
            self.cash / portfolio_value if portfolio_value > 0 else 1.0,  # Cash ratio
            len(self.portfolio),                          # Number of positions
            self.max_positions - len(self.portfolio),     # Remaining position slots
            self.step_idx,                                # Current step in episode
            self.episode_length - self.step_idx,          # Steps remaining
        ])

        # Convert to tensor
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device)

        # Pad to 25 dims if needed
        if len(context) < 25:
            padding = torch.zeros(25 - len(context), dtype=torch.float32, device=self.device)
            context_tensor = torch.cat([context_tensor, padding])
        elif len(context) > 25:
            context_tensor = context_tensor[:25]

        return context_tensor

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """
        Execute actions and advance environment by one day.

        Args:
            actions: Dictionary mapping ticker -> action_id

        Returns:
            (next_states, reward, done, info)
        """
        # Record portfolio value before actions
        portfolio_value_before = self._portfolio_value()

        # Execute all actions
        trade_info = self._execute_actions(actions)

        # Update days held for all positions
        for ticker, pos in self.portfolio.items():
            pos['days_held'] += 1

        # Advance to next day
        self.step_idx += 1

        if self.step_idx >= len(self.dates):
            # Episode ended - close all positions
            self._close_all_positions()
            done = True
            next_states = {}
        else:
            self.current_date = self.dates[self.step_idx]
            done = False

            # Mark-to-market portfolio (revalue positions)
            self._mark_to_market()

            # Get next states
            next_states = self._get_states()

        # Compute reward (percentage change in portfolio value)
        portfolio_value_after = self._portfolio_value()
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before

        # Info dictionary
        info = {
            'portfolio_value': portfolio_value_after,
            'cash': self.cash,
            'num_positions': len(self.portfolio),
            'trades': trade_info,
            'step': self.step_idx,
            'date': self.current_date if not done else self.dates[-1]
        }

        # Record in history
        self.episode_history.append({
            'date': self.current_date if not done else self.dates[-1],
            'portfolio_value': portfolio_value_after,
            'reward': reward,
            'actions': actions,
            **info
        })

        return next_states, reward, done, info

    def _execute_actions(self, actions: Dict[str, int]) -> List[Dict]:
        """
        Execute all actions.

        Args:
            actions: Dictionary mapping ticker -> action_id

        Returns:
            List of trade information dictionaries
        """
        trade_info = []

        for ticker, action_id in actions.items():
            # Get current price
            result = self.data_loader.get_features_and_price(ticker, self.current_date)
            if result is None:
                continue

            _, current_price = result

            # Execute action
            trade = self._execute_single_action(ticker, action_id, current_price)
            if trade is not None:
                trade_info.append(trade)

        return trade_info

    def _execute_single_action(self, ticker: str, action_id: int, current_price: float) -> Optional[Dict]:
        """
        Execute a single action.

        Actions:
        0 = HOLD: Do nothing
        1 = BUY_SMALL: Allocate 25% of available capital
        2 = BUY_MEDIUM: Allocate 50% of available capital
        3 = BUY_LARGE: Allocate 100% of available capital
        4 = SELL: Close position

        Args:
            ticker: Stock ticker
            action_id: Action to execute
            current_price: Current price of stock

        Returns:
            Trade info dictionary or None
        """
        if action_id == 0:  # HOLD
            return None

        elif action_id in [1, 2, 3]:  # BUY
            # Check constraints
            if ticker in self.portfolio:
                return None  # Already have position
            if len(self.portfolio) >= self.max_positions:
                return None  # Too many positions
            if self.cash <= 0:
                return None  # No cash

            # Determine allocation
            allocation_pct = {1: 0.25, 2: 0.5, 3: 1.0}[action_id]
            allocation = self.cash * allocation_pct

            # Account for transaction costs
            allocation_after_costs = allocation * (1 - self.transaction_cost)

            # Calculate shares
            shares = allocation_after_costs / current_price

            if shares <= 0:
                return None

            # Execute buy
            self.portfolio[ticker] = {
                'size': shares,
                'entry_price': current_price,
                'entry_date': self.current_date,
                'days_held': 0
            }
            self.cash -= allocation
            self.total_trades += 1

            return {
                'ticker': ticker,
                'action': 'BUY',
                'action_size': ['SMALL', 'MEDIUM', 'LARGE'][action_id - 1],
                'shares': shares,
                'price': current_price,
                'cost': allocation,
                'date': self.current_date
            }

        elif action_id == 4:  # SELL
            if ticker not in self.portfolio:
                return None  # No position to sell

            # Close position
            pos = self.portfolio[ticker]
            proceeds = pos['size'] * current_price
            proceeds_after_costs = proceeds * (1 - self.transaction_cost)

            # Calculate P&L
            cost_basis = pos['size'] * pos['entry_price']
            pnl = proceeds_after_costs - cost_basis
            pnl_pct = pnl / cost_basis

            # Execute sell
            self.cash += proceeds_after_costs
            del self.portfolio[ticker]
            self.total_trades += 1

            if pnl > 0:
                self.profitable_trades += 1

            return {
                'ticker': ticker,
                'action': 'SELL',
                'shares': pos['size'],
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'proceeds': proceeds_after_costs,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'days_held': pos['days_held'],
                'date': self.current_date
            }

        return None

    def _mark_to_market(self):
        """Update portfolio values to current market prices."""
        # Nothing to do - we'll compute values on-demand in _portfolio_value()
        pass

    def _close_all_positions(self):
        """Close all positions at end of episode."""
        for ticker in list(self.portfolio.keys()):
            result = self.data_loader.get_features_and_price(ticker, self.dates[-1])
            if result is not None:
                _, current_price = result
                self._execute_single_action(ticker, 4, current_price)  # SELL

    def _portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + equity).

        Returns:
            Total portfolio value
        """
        equity = 0.0

        for ticker, pos in self.portfolio.items():
            result = self.data_loader.get_features_and_price(ticker, self.current_date)
            if result is not None:
                _, current_price = result
                equity += pos['size'] * current_price

        return self.cash + equity

    def get_episode_stats(self) -> Dict:
        """
        Get statistics about the completed episode.

        Returns:
            Dictionary of episode statistics
        """
        if len(self.episode_history) == 0:
            return {}

        portfolio_values = [h['portfolio_value'] for h in self.episode_history]
        rewards = [h['reward'] for h in self.episode_history]

        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Compute max drawdown
        peak = portfolio_values[0]
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Compute Sharpe ratio (annualized)
        if len(rewards) > 1:
            mean_return = np.mean(rewards)
            std_return = np.std(rewards)
            sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'total_return': total_return,
            'final_value': portfolio_values[-1],
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'avg_reward': np.mean(rewards),
            'num_steps': len(self.episode_history)
        }
