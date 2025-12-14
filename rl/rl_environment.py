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
from numba import jit
from tqdm import tqdm


@jit(nopython=True, cache=True)
def compute_max_drawdown_numba(portfolio_values: np.ndarray) -> float:
    """
    Compute maximum drawdown using numba JIT compilation.

    Args:
        portfolio_values: Array of portfolio values over time

    Returns:
        Maximum drawdown (0-1)
    """
    if len(portfolio_values) == 0:
        return 0.0

    peak = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


@jit(nopython=True, cache=True)
def compute_sharpe_ratio_numba(rewards: np.ndarray, periods_per_year: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio using numba JIT compilation.

    Args:
        rewards: Array of per-period returns
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(rewards) <= 1:
        return 0.0

    mean_return = np.mean(rewards)
    std_return = np.std(rewards)

    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(periods_per_year)


@jit(nopython=True, cache=True)
def compute_portfolio_equity_numba(shares: np.ndarray, prices: np.ndarray) -> float:
    """
    Compute total equity value using numba JIT compilation.

    Args:
        shares: Array of share quantities
        prices: Array of current prices

    Returns:
        Total equity value
    """
    return np.sum(shares * prices)


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
                 device: str = 'cuda',
                 precompute_all_states: bool = False,
                 trading_days_filter: Optional[List[str]] = None,
                 top_k_per_horizon: int = 10):
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
            precompute_all_states: If True, pre-compute states for ALL dates once at init
                                  (one-time cost, then reused across all episodes)
            trading_days_filter: Optional list of specific trading days to use
                                (e.g., last 4 years only). If None, uses all available days.
            top_k_per_horizon: Number of top stocks to select per time horizon (default: 10)
                              This filters stocks based on predictor expected returns before Q-network
        """
        self.data_loader = data_loader
        self.agent = agent
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.device = device
        self.top_k_per_horizon = top_k_per_horizon

        # Get actual trading days (weekdays only, no weekends/holidays)
        if trading_days_filter is not None:
            # Use provided filtered days (e.g., last 4 years only)
            self.trading_days = trading_days_filter
            print(f"   Using {len(self.trading_days)} filtered trading days")
        elif data_loader.prices_file is not None:
            # Get all trading days from prices file
            sample_ticker = list(data_loader.prices_file.keys())[0]
            prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
            self.trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
            print(f"   Using {len(self.trading_days)} actual trading days from prices file")
        else:
            # Fall back to all dates (may include weekends)
            self.trading_days = data_loader.all_dates
            print(f"   ⚠️  No prices file - using all dates (may include weekends)")

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

        # State cache (computed once per episode, stored in CPU RAM)
        # Structure: {date: {ticker: state_tensor (CPU)}}
        self.state_cache = {}

        # Price cache (avoid repeated HDF5 reads)
        # Structure: {date: {ticker: price}}
        self.price_cache = {}

        print(f"✅ TradingEnvironment initialized")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Max positions: {max_positions}")
        print(f"   Episode length: {episode_length} days")
        print(f"   Transaction cost: {transaction_cost*100:.2f}%")

        # Pre-compute all states if requested (one-time cost, reused for all episodes!)
        if precompute_all_states:
            print(f"\n🔄 Pre-computing states for ALL {len(self.trading_days)} trading days...")
            print(f"   This is a one-time cost - states will be reused across all episodes!")
            self._precompute_all_states()
            print(f"✅ Pre-computation complete! Cached {sum(len(self.state_cache[d]) for d in self.state_cache)} states")
            print(f"   Memory usage: ~{self._estimate_cache_size_mb():.1f} MB")

    def reset(self, start_date: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environment to start new episode.

        Args:
            start_date: Optional specific start date. If None, random.

        Returns:
            Initial states for all available stocks
        """
        # Select episode dates (using actual trading days only)
        if start_date is None:
            # Random start date
            max_start_idx = len(self.trading_days) - self.episode_length - 1
            start_idx = random.randint(0, max_start_idx)
        else:
            start_idx = self.trading_days.index(start_date)

        self.dates = self.trading_days[start_idx:start_idx + self.episode_length]
        self.current_date = self.dates[0]
        self.step_idx = 0

        # Reset portfolio
        self.portfolio = {}
        self.cash = self.initial_capital
        self.episode_history = []
        self.total_trades = 0
        self.profitable_trades = 0

        # Check if states are already cached (from precompute_all_states)
        all_dates_cached = all(date in self.state_cache for date in self.dates)

        if not all_dates_cached:
            # Need to compute states for this episode
            self._precompute_episode_states()

        # Get initial states from cache
        states = self._get_states()

        return states

    def _precompute_all_states(self):
        """
        Pre-compute states for ALL trading days once at initialization.
        This is a one-time cost - states are then reused across all episodes!

        Memory estimate: ~6000 days × 900 stocks × 1469 features × 4 bytes = ~32 GB RAM
        """
        print(f"\n{'='*80}")
        print(f"PRE-COMPUTING STATES FOR ALL TRADING DAYS")
        print(f"{'='*80}\n")

        # Process all trading days
        for date in tqdm(self.trading_days, desc="   Pre-computing all states", unit="day"):
            if date not in self.state_cache:
                self.state_cache[date] = {}

            # Batch extract predictor features for all stocks
            pred_features_dict, prices_dict = self._batch_extract_predictor_features(date)

            # Cache prices
            self.price_cache[date] = prices_dict

            # Create portfolio context (zeros for precomputation, will be updated during episodes)
            portfolio_context = torch.zeros(25, dtype=torch.float32)

            # Store states in CPU cache
            for ticker, pred_features in pred_features_dict.items():
                # Concatenate predictor features + portfolio context
                state = torch.cat([pred_features.cpu(), portfolio_context])
                self.state_cache[date][ticker] = state

        print(f"\n{'='*80}")
        print(f"✅ ALL STATES PRE-COMPUTED")
        print(f"{'='*80}\n")

    def _estimate_cache_size_mb(self) -> float:
        """Estimate memory usage of state cache in MB."""
        total_elements = 0
        for date in self.state_cache:
            for ticker in self.state_cache[date]:
                total_elements += self.state_cache[date][ticker].numel()
        # Each element is 4 bytes (float32)
        total_bytes = total_elements * 4
        return total_bytes / (1024 * 1024)

    def save_state_cache(self, cache_path: str):
        """
        Save precomputed state cache to disk (HDF5 format).
        This allows reusing the 3.5-hour precomputation across runs!

        Args:
            cache_path: Path to save the cache file
        """
        if len(self.state_cache) == 0:
            print(f"  ⚠️  No states cached, nothing to save")
            return

        import h5py
        from tqdm import tqdm

        print(f"\n{'='*80}")
        print("SAVING STATE CACHE")
        print(f"{'='*80}")
        print(f"Saving state cache to: {cache_path}")

        with h5py.File(cache_path, 'w') as f:
            # Group states by date
            for date in tqdm(sorted(self.state_cache.keys()), desc="  Saving states"):
                date_grp = f.create_group(date)

                # Get all tickers and states for this date
                tickers = list(self.state_cache[date].keys())
                states = [self.state_cache[date][ticker].cpu().numpy() for ticker in tickers]
                prices = [self.price_cache.get(date, {}).get(ticker, 0.0) for ticker in tickers]

                # Save to HDF5
                date_grp.create_dataset('tickers', data=np.array(tickers, dtype='S10'))
                date_grp.create_dataset('states', data=np.stack(states), compression='gzip', compression_opts=4)
                date_grp.create_dataset('prices', data=np.array(prices), compression='gzip', compression_opts=4)

        print(f"  ✅ State cache saved successfully")
        print(f"  📁 File: {cache_path}")

        # Print file size
        import os
        file_size_mb = os.path.getsize(cache_path) / 1e6
        print(f"  💾 Size: {file_size_mb:.1f} MB")
        print(f"{'='*80}\n")

    def load_state_cache(self, cache_path: str) -> bool:
        """
        Load precomputed state cache from disk (HDF5 format).
        This skips the 3.5-hour precomputation!

        Args:
            cache_path: Path to load the cache file

        Returns:
            True if successfully loaded, False otherwise
        """
        import os
        import h5py
        from tqdm import tqdm

        if not os.path.exists(cache_path):
            return False

        print(f"\n{'='*80}")
        print("LOADING STATE CACHE")
        print(f"{'='*80}")
        print(f"Loading state cache from: {cache_path}")

        try:
            with h5py.File(cache_path, 'r') as f:
                self.state_cache = {}
                self.price_cache = {}

                for date in tqdm(list(f.keys()), desc="  Loading states"):
                    date_grp = f[date]

                    # Load data
                    tickers = [t.decode('utf-8') for t in date_grp['tickers'][:]]
                    states = date_grp['states'][:]
                    prices = date_grp['prices'][:]

                    # Reconstruct cache
                    self.state_cache[date] = {}
                    if date not in self.price_cache:
                        self.price_cache[date] = {}

                    for i, ticker in enumerate(tickers):
                        self.state_cache[date][ticker] = torch.from_numpy(states[i]).float()
                        self.price_cache[date][ticker] = float(prices[i])

            total_states = sum(len(self.state_cache[d]) for d in self.state_cache)
            print(f"  ✅ Loaded {total_states:,} states across {len(self.state_cache)} dates")
            print(f"  💾 Cache size: ~{self._estimate_cache_size_mb():.1f} MB")
            print(f"{'='*80}\n")
            return True

        except Exception as e:
            print(f"  ❌ Error loading state cache: {e}")
            print(f"{'='*80}\n")
            return False

    def _precompute_episode_states(self):
        """
        Pre-compute states for ALL dates in the episode using batched extraction.
        Stores in CPU RAM cache to avoid recomputation.
        """
        # Clear existing caches
        self.state_cache = {}
        self.price_cache = {}
        total_states = 0

        # Progress bar for dates
        date_pbar = tqdm(self.dates, desc="   Pre-computing states", unit="day", leave=False)

        for date in date_pbar:
            # Batch extract predictor features and prices for all stocks on this date
            pred_features_dict, prices_dict = self._batch_extract_predictor_features(date)

            # Store prices in cache
            self.price_cache[date] = prices_dict

            # Store predictor features in cache (move to CPU to save GPU memory)
            self.state_cache[date] = {}
            for ticker, pred_features in pred_features_dict.items():
                # Create placeholder portfolio context (will be updated dynamically during episode)
                portfolio_context = torch.zeros(25, device='cpu')

                # Concatenate and store in CPU
                state = torch.cat([pred_features.cpu(), portfolio_context])
                self.state_cache[date][ticker] = state
                total_states += 1

            # Update progress bar with stats
            date_pbar.set_postfix({'stocks': len(pred_features_dict), 'cached': total_states})

        date_pbar.close()
        print(f"   ✅ Cached {total_states} states across {len(self.dates)} days (avg {total_states/len(self.dates):.0f} stocks/day)")

    def _batch_extract_predictor_features(self, date: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Extract predictor features for ALL stocks on a given date using batching.

        Args:
            date: Date to extract features for

        Returns:
            Tuple of (pred_features_dict, prices_dict):
                - pred_features_dict: {ticker: predictor features (GPU tensor)}
                - prices_dict: {ticker: current_price}
        """
        # OPTIMIZED: Load all features for this date in one batch (much faster!)
        batch_results = self.data_loader.get_all_features_for_date_batched(date)

        if len(batch_results) == 0:
            return {}, {}

        # Extract tickers, features, and prices from batch results
        tickers = list(batch_results.keys())
        features_list = [batch_results[ticker][0] for ticker in tickers]
        prices = [batch_results[ticker][1] for ticker in tickers]

        # Find max feature dimension (features may have different dims due to varying fundamentals)
        max_feature_dim = max(f.shape[0] for f in features_list)

        # Pad features to same dimension
        padded_features = []
        for features in features_list:
            if features.shape[0] < max_feature_dim:
                # Pad with zeros
                padding = torch.zeros(max_feature_dim - features.shape[0], dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding])
            padded_features.append(features)

        # Process in mini-batches to avoid OOM (can't fit all 925 stocks × 3000 seq_len in GPU memory)
        # Increased from 32 to 128 - batch size 32 was too conservative
        # 128 stocks × 3000 seq_len × 998 features = ~1.4 GB (should fit comfortably)
        mini_batch_size = 128
        seq_len = 3000

        pred_features_dict = {}
        prices_dict = {}

        # Process in mini-batches
        for batch_start in range(0, len(padded_features), mini_batch_size):
            batch_end = min(batch_start + mini_batch_size, len(padded_features))

            # Get mini-batch
            mini_batch_features = padded_features[batch_start:batch_end]
            mini_batch_tickers = tickers[batch_start:batch_end]
            mini_batch_prices = prices[batch_start:batch_end]

            # Stack mini-batch
            features_batch = torch.stack(mini_batch_features).to(self.device)

            # Expand to 3D: (mini_batch, 1, feature_dim) -> (mini_batch, seq_len, feature_dim)
            features_3d = features_batch.unsqueeze(1).expand(-1, seq_len, -1)

            # Extract features for this mini-batch
            with torch.no_grad():
                pred_features_batch = self.agent.feature_extractor.extract_features(features_3d)
                # Shape: (mini_batch, 1444) - predictor features

            # Store results
            for ticker, pred_features, price in zip(mini_batch_tickers, pred_features_batch, mini_batch_prices):
                pred_features_dict[ticker] = pred_features
                prices_dict[ticker] = price

        return pred_features_dict, prices_dict

    def _filter_top_k_stocks(self, cached_states: Dict[str, torch.Tensor]) -> List[str]:
        """
        Filter stocks to top-k per time horizon based on expected returns.

        Extracts expected returns from predictor features and selects top-k stocks
        for each of the 4 time horizons, then returns the union.

        Args:
            cached_states: Dictionary of {ticker: cached_state_tensor}

        Returns:
            List of tickers that are in top-k for at least one time horizon
        """
        if self.top_k_per_horizon == 0 or self.top_k_per_horizon >= len(cached_states):
            # No filtering (return all tickers)
            return list(cached_states.keys())

        # Extract expected returns for all stocks
        # Expected returns are at positions 1428:1432 (4 horizons)
        stock_returns = {}  # {ticker: [ret_h1, ret_h2, ret_h3, ret_h4]}

        for ticker, cached_state in cached_states.items():
            pred_features = cached_state[:1444]
            expected_returns = pred_features[1428:1432]  # Shape: (4,)
            stock_returns[ticker] = expected_returns.cpu().numpy()

        # For each time horizon, select top-k stocks
        selected_tickers = set()
        num_horizons = 4

        for horizon_idx in range(num_horizons):
            # Rank stocks by expected return for this horizon
            stocks_by_return = [(ticker, returns[horizon_idx])
                               for ticker, returns in stock_returns.items()]
            stocks_by_return.sort(key=lambda x: x[1], reverse=True)  # Highest return first

            # Take top-k for this horizon
            top_k_this_horizon = stocks_by_return[:self.top_k_per_horizon]
            for ticker, _ in top_k_this_horizon:
                selected_tickers.add(ticker)

        return list(selected_tickers)

    def _get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get current state for filtered top-k stocks FROM CACHE.

        Filters to top-k stocks per time horizon based on predictor expected returns,
        then returns states only for these filtered stocks.

        State composition:
        - Predictor features: 1444 dims (from cache)
        - Portfolio context: 25 dims (computed dynamically based on current portfolio)

        Returns:
            Dictionary mapping ticker -> state tensor (only for top-k stocks)
        """
        if self.current_date not in self.state_cache:
            # Fallback to old method if cache miss (shouldn't happen)
            return self._get_states_uncached()

        cached_states = self.state_cache[self.current_date]
        cached_prices = self.price_cache.get(self.current_date, {})

        # Filter to top-k stocks per horizon based on expected returns
        filtered_tickers = self._filter_top_k_stocks(cached_states)

        # CRITICAL OPTIMIZATION: Compute portfolio value ONCE instead of 925 times!
        portfolio_value = self._portfolio_value_cached(cached_prices)

        # Batch create portfolio contexts (only for filtered stocks)
        pred_features_list = []
        portfolio_contexts_list = []
        tickers_list = []

        for ticker in filtered_tickers:
            # Get cached predictor features (on CPU)
            cached_state = cached_states[ticker]
            pred_features = cached_state[:1444]  # First 1444 dims are predictor features

            # Get cached price (fast lookup, no HDF5 read!)
            current_price = cached_prices.get(ticker)
            if current_price is None:
                continue

            # Create portfolio context (pass precomputed portfolio_value)
            portfolio_context = self._create_portfolio_context_fast(
                ticker, current_price, portfolio_value
            )

            pred_features_list.append(pred_features)
            portfolio_contexts_list.append(portfolio_context)
            tickers_list.append(ticker)

        # Batch concatenate and move to GPU (much faster than one-by-one)
        states = {}
        if len(pred_features_list) > 0:
            # Stack and concatenate on CPU
            pred_features_batch = torch.stack(pred_features_list)
            portfolio_contexts_batch = torch.stack(portfolio_contexts_list)
            states_batch = torch.cat([pred_features_batch, portfolio_contexts_batch], dim=1)

            # Single GPU transfer for all states (much faster!)
            states_batch_gpu = states_batch.to(self.device)

            # Split back to dictionary
            for i, ticker in enumerate(tickers_list):
                states[ticker] = states_batch_gpu[i]

        return states

    def _get_states_uncached(self) -> Dict[str, torch.Tensor]:
        """
        Fallback: Get states without cache (old sequential method).
        Only used if cache miss occurs.
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
        try:
            result = self.data_loader.get_features_and_price(ticker, self.current_date)
            if result is None:
                return None

            features, current_price = result
        except (ValueError, KeyError):
            # Stock doesn't have data for this date (e.g., not listed yet)
            return None

        # Extract predictor features (1919 dims)
        with torch.no_grad():
            # Move to device
            features = features.to(self.device)

            # Reshape to (batch, seq_len, feature_dim) for the model
            # Model expects sequences, so we expand the single timestep to seq_len
            seq_len = 3000  # Match model's expected sequence length
            features_3d = features.unsqueeze(0).unsqueeze(1).expand(-1, seq_len, -1)
            # Shape: (1, 2000, feature_dim)

            # Extract features using agent's feature extractor
            pred_features = self.agent.feature_extractor.extract_features(
                features_3d  # Pass 3D tensor
            ).squeeze(0)  # Remove batch dimension

        # Create portfolio context
        portfolio_context = self._create_portfolio_context(ticker, current_price)

        # Concatenate: predictor features + portfolio context
        state = torch.cat([pred_features, portfolio_context])

        return state

    def _create_portfolio_context(self, ticker: str, current_price: float) -> torch.Tensor:
        """
        Create portfolio context features for a stock.

        WARNING: Calls _portfolio_value() which makes HDF5 reads!
        Use _create_portfolio_context_fast() when possible.

        Args:
            ticker: Stock ticker
            current_price: Current price

        Returns:
            Portfolio context tensor (25 dims)
        """
        portfolio_value = self._portfolio_value()
        return self._create_portfolio_context_fast(ticker, current_price, portfolio_value)

    def _create_portfolio_context_fast(self, ticker: str, current_price: float,
                                       portfolio_value: float) -> torch.Tensor:
        """
        Create portfolio context features for a stock (OPTIMIZED - no HDF5 reads).

        Args:
            ticker: Stock ticker
            current_price: Current price
            portfolio_value: Precomputed portfolio value (pass from _portfolio_value_cached)

        Returns:
            Portfolio context tensor (25 dims) on CPU
        """
        # Check if we have a position in this stock
        if ticker in self.portfolio:
            pos = self.portfolio[ticker]
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
            self.cash,                                    # Available cash
            self.cash / portfolio_value if portfolio_value > 0 else 1.0,  # Cash ratio
            len(self.portfolio),                          # Number of positions
            self.max_positions - len(self.portfolio),     # Remaining position slots
            self.step_idx,                                # Current step in episode
            self.episode_length - self.step_idx,          # Steps remaining
        ])

        # Convert to tensor on CPU (caller will batch-transfer to GPU)
        context_tensor = torch.tensor(context, dtype=torch.float32, device='cpu')

        # Pad to 25 dims if needed
        if len(context) < 25:
            padding = torch.zeros(25 - len(context), dtype=torch.float32, device='cpu')
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
        # Get cached prices for current date (fast lookup, no HDF5 reads!)
        current_cached_prices = self.price_cache.get(self.current_date, {})

        # Record portfolio value before actions (using cached prices)
        portfolio_value_before = self._portfolio_value_cached(current_cached_prices)

        # Execute all actions (using cached prices)
        trade_info = self._execute_actions(actions, current_cached_prices)

        # Update days held for all positions
        for ticker, pos in self.portfolio.items():
            pos['days_held'] += 1

        # Advance to next day
        self.step_idx += 1

        if self.step_idx >= len(self.dates):
            # Episode ended - close all positions
            final_cached_prices = self.price_cache.get(self.dates[-1], {})
            self._close_all_positions(final_cached_prices)
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
        # Use cached prices for the current date (after advancing)
        next_cached_prices = self.price_cache.get(self.current_date if not done else self.dates[-1], {})
        portfolio_value_after = self._portfolio_value_cached(next_cached_prices)
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

    def _execute_actions(self, actions: Dict[str, int], cached_prices: Dict[str, float]) -> List[Dict]:
        """
        Execute all actions using cached prices (no HDF5 reads!).

        Args:
            actions: Dictionary mapping ticker -> action_id
            cached_prices: Pre-loaded prices for current date

        Returns:
            List of trade information dictionaries
        """
        trade_info = []

        for ticker, action_id in actions.items():
            # Get current price from cache (fast lookup!)
            current_price = cached_prices.get(ticker)
            if current_price is None:
                # Stock doesn't have price data for this date
                continue

            # Execute action
            trade = self._execute_single_action(ticker, action_id, current_price)
            if trade is not None:
                trade_info.append(trade)

        return trade_info

    def _execute_single_action(self, ticker: str, action_id: int, current_price: float) -> Optional[Dict]:
        """
        Execute a single action.

        Simplified action space (3 actions):
        0 = HOLD: Do nothing
        1 = BUY: Allocate 50% of available capital
        2 = SELL: Close position

        Args:
            ticker: Stock ticker
            action_id: Action to execute
            current_price: Current price of stock

        Returns:
            Trade info dictionary or None
        """
        if action_id == 0:  # HOLD
            return None

        elif action_id == 1:  # BUY
            # Check constraints
            if ticker in self.portfolio:
                return None  # Already have position
            if len(self.portfolio) >= self.max_positions:
                return None  # Too many positions
            if self.cash <= 0:
                return None  # No cash

            # Fixed 50% allocation (simpler than learning position sizing)
            allocation_pct = 0.5
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
                'shares': shares,
                'price': current_price,
                'cost': allocation,
                'date': self.current_date
            }

        elif action_id == 2:  # SELL
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

    def _close_all_positions(self, cached_prices: Dict[str, float]):
        """Close all positions at end of episode using cached prices (no HDF5 reads!)."""
        for ticker in list(self.portfolio.keys()):
            current_price = cached_prices.get(ticker)
            if current_price is not None:
                self._execute_single_action(ticker, 4, current_price)  # SELL

    def _portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + equity).

        WARNING: This makes HDF5 reads! Use _portfolio_value_cached() when possible.

        Returns:
            Total portfolio value
        """
        equity = 0.0

        for ticker, pos in self.portfolio.items():
            try:
                result = self.data_loader.get_features_and_price(ticker, self.current_date)
                if result is not None:
                    _, current_price = result
                    equity += pos['size'] * current_price
            except (ValueError, KeyError):
                # Stock doesn't have data for this date, skip (shouldn't happen if we bought it)
                continue

        return self.cash + equity

    def _portfolio_value_cached(self, cached_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value using cached prices (NO HDF5 reads!).

        Args:
            cached_prices: Dictionary of {ticker: price} from price cache

        Returns:
            Total portfolio value
        """
        equity = 0.0

        for ticker, pos in self.portfolio.items():
            current_price = cached_prices.get(ticker)
            if current_price is not None:
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

        # Convert to numpy arrays for fast computation
        portfolio_values = np.array([h['portfolio_value'] for h in self.episode_history], dtype=np.float64)
        rewards = np.array([h['reward'] for h in self.episode_history], dtype=np.float64)

        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        win_rate = self.profitable_trades / self.total_trades if self.total_trades > 0 else 0.0

        # Compute max drawdown using numba (faster)
        max_drawdown = compute_max_drawdown_numba(portfolio_values)

        # Compute Sharpe ratio using numba (faster)
        sharpe_ratio = compute_sharpe_ratio_numba(rewards, periods_per_year=252.0)

        return {
            'total_return': total_return,
            'final_value': float(portfolio_values[-1]),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'avg_reward': float(np.mean(rewards)),
            'num_steps': len(self.episode_history)
        }
