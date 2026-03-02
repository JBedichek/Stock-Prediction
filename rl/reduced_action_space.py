#!/usr/bin/env python3
"""
Reduced action space for actor-critic trading.

Action space: 5 discrete actions
- 0: HOLD CASH (sell current position if any)
- 1: BE IN STOCK 1 (top-ranked for 1d horizon)
- 2: BE IN STOCK 2 (top-ranked for 3d horizon)
- 3: BE IN STOCK 3 (top-ranked for 5d horizon)
- 4: BE IN STOCK 4 (top-ranked for 10d horizon)

Transitions happen automatically:
- Currently in cash, action=1: BUY stock 1
- Currently holding stock 1, action=2: SELL stock 1, BUY stock 2 (same timestep)
- Currently holding stock 1, action=1: HOLD stock 1
- Currently holding stock 1, action=0: SELL stock 1
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional


def get_top_4_stocks(
    cached_states: Dict[str, torch.Tensor],
    sample_fraction: float = 1.0
) -> List[Tuple[str, int]]:
    """
    Get top-1 stock for each of 4 time horizons (VECTORIZED) with optional random sampling.

    Args:
        cached_states: Dict mapping ticker -> cached state (1444 dims)
        sample_fraction: Fraction of stocks to randomly sample before selecting top-4
                        (1.0 = use all stocks, 0.3 = randomly sample 30% of stocks)

    Returns:
        List of (ticker, horizon_idx) tuples, one per horizon
    """
    return get_top_k_stocks_per_horizon(cached_states, k=1, sample_fraction=sample_fraction)


def get_bottom_4_stocks(
    cached_states: Dict[str, torch.Tensor],
    sample_fraction: float = 1.0
) -> List[Tuple[str, int]]:
    """
    Get bottom-1 stock (worst predicted) for each of 4 time horizons for SHORT SELLING.

    Args:
        cached_states: Dict mapping ticker -> cached state (1444 dims)
        sample_fraction: Fraction of stocks to randomly sample before selecting bottom-4
                        (1.0 = use all stocks, 0.3 = randomly sample 30% of stocks)

    Returns:
        List of (ticker, horizon_idx) tuples, one per horizon (worst performers)
    """
    if len(cached_states) == 0:
        return []

    # Vectorized version: much faster than Python loops
    tickers = list(cached_states.keys())

    # Random sampling for diversity (prevents overfitting to specific stocks)
    if sample_fraction < 1.0:
        import random
        sample_size = max(1, int(len(tickers) * sample_fraction))
        # Ensure we have enough stocks to select bottom-4 from
        sample_size = max(sample_size, 4)  # At least 4 stocks total
        if sample_size < len(tickers):
            tickers = random.sample(tickers, sample_size)

    # Stack all cached states: (N_stocks, 1444 or 1469)
    # OPTIMIZED: Pre-allocate tensor instead of list append + stack
    n_stocks = len(tickers)
    device = next(iter(cached_states.values())).device
    states_tensor = torch.empty((n_stocks, 1444), device=device, dtype=torch.float32)

    for i, ticker in enumerate(tickers):
        state = cached_states[ticker]
        if state.dim() > 1:
            state = state.squeeze()
        states_tensor[i] = state[:1444]  # First 1444 dims only

    # Extract expected returns for all stocks at once: (N_stocks, 4)
    expected_returns = states_tensor[:, 1428:1432]

    # Find bottom-1 (worst) for each horizon (0-3) for shorting
    bottom_stocks = []
    for horizon_idx in range(4):
        horizon_returns = expected_returns[:, horizon_idx]  # (N_stocks,)

        # Get BOTTOM-1 index for this horizon (topk with largest=False gets smallest)
        _, bottom_indices = torch.topk(horizon_returns, k=1, largest=False)

        for idx in bottom_indices:
            bottom_stocks.append((tickers[idx.item()], horizon_idx))

    return bottom_stocks


def get_top_k_stocks_per_horizon(
    cached_states: Dict[str, torch.Tensor],
    k: int = 3,
    sample_fraction: float = 1.0
) -> List[Tuple[str, int]]:
    """
    Get top-K stocks for each of 4 time horizons (VECTORIZED) with optional random sampling.

    This function supports random sampling of the stock universe before selecting top-K,
    which prevents overfitting to specific stocks and improves generalization.

    Args:
        cached_states: Dict mapping ticker -> cached state (1444 dims)
        k: Number of top stocks to return per horizon
        sample_fraction: Fraction of stocks to randomly sample before selecting top-K
                        (1.0 = use all stocks, 0.3 = randomly sample 30% of stocks)
                        Set < 1.0 during training for diversity, 1.0 for validation

    Returns:
        List of (ticker, horizon_idx) tuples, k per horizon (4*k total)
    """
    if len(cached_states) == 0:
        return []

    # Vectorized version: much faster than Python loops
    tickers = list(cached_states.keys())

    # Random sampling for diversity (prevents overfitting to specific stocks)
    if sample_fraction < 1.0:
        import random
        sample_size = max(1, int(len(tickers) * sample_fraction))
        # Ensure we have enough stocks to select top-K from
        sample_size = max(sample_size, k * 4)  # At least k per horizon
        if sample_size < len(tickers):
            tickers = random.sample(tickers, sample_size)

    # Stack all cached states: (N_stocks, 1444 or 1469)
    # OPTIMIZED: Pre-allocate tensor instead of list append + stack
    n_stocks = len(tickers)
    device = next(iter(cached_states.values())).device
    states_tensor = torch.empty((n_stocks, 1444), device=device, dtype=torch.float32)

    for i, ticker in enumerate(tickers):
        state = cached_states[ticker]
        if state.dim() > 1:
            state = state.squeeze()
        states_tensor[i] = state[:1444]  # First 1444 dims only

    # Extract expected returns for all stocks at once: (N_stocks, 4)
    expected_returns = states_tensor[:, 1428:1432]

    # Find top-K for each horizon (0-3)
    top_stocks = []
    for horizon_idx in range(4):
        horizon_returns = expected_returns[:, horizon_idx]  # (N_stocks,)

        # Get top-k indices for this horizon
        top_k = min(k, len(tickers))  # Handle case where fewer stocks than k
        _, top_indices = torch.topk(horizon_returns, top_k)

        for idx in top_indices:
            top_stocks.append((tickers[idx.item()], horizon_idx))

    return top_stocks


def sample_top_4_from_top_k(
    top_k_stocks: List[Tuple[str, int]],
    sample_size: int = 4,
    deterministic: bool = False
) -> List[Tuple[str, int]]:
    """
    Sample 4 stocks from top-K stocks (with randomization for training).

    Args:
        top_k_stocks: List of (ticker, horizon_idx) from get_top_k_stocks_per_horizon
        sample_size: Number of stocks to sample (default 4)
        deterministic: If True, always take first 4 (for validation)

    Returns:
        List of 4 (ticker, horizon_idx) tuples
    """
    if len(top_k_stocks) <= sample_size:
        return top_k_stocks

    if deterministic:
        # Take first one from each horizon (deterministic)
        selected = []
        for horizon_idx in range(4):
            for ticker, h_idx in top_k_stocks:
                if h_idx == horizon_idx and (ticker, h_idx) not in selected:
                    selected.append((ticker, h_idx))
                    break
        return selected[:sample_size]
    else:
        # Randomly sample (for training diversity)
        import random
        return random.sample(top_k_stocks, sample_size)


def select_action_epsilon_greedy(
    logits: torch.Tensor,
    epsilon: float = 0.1,
    training: bool = True
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """
    Select action using epsilon-greedy on softmax probabilities.

    Args:
        logits: Action logits from actor (5 dims)
        epsilon: Exploration rate (probability of random action)
        training: If True, use epsilon-greedy; if False, use greedy (argmax)

    Returns:
        Tuple of (action_id, log_prob, entropy)
    """
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    if training and np.random.random() < epsilon:
        # Random action (exploration)
        action = torch.randint(0, logits.shape[-1], (1,)).item()
    else:
        # Greedy action (exploitation)
        action = probs.argmax(dim=-1).item()

    # Compute log_prob and entropy for selected action
    log_prob = dist.log_prob(torch.tensor(action, device=logits.device))
    entropy = dist.entropy()

    return action, log_prob, entropy


def create_global_state(
    top_4_stocks: List[Tuple[str, int]],
    bottom_4_stocks: List[Tuple[str, int]],
    states: Dict[str, torch.Tensor],
    current_position: Optional[str],
    is_short: bool = False,
    device: str = 'cuda',
    allow_short: bool = True
) -> torch.Tensor:
    """
    Create global state representation for actor.

    With short selling (allow_short=True):
    - States for 4 top stocks to LONG (4 × 1469 = 5876 dims)
    - States for 4 bottom stocks to SHORT (4 × 1469 = 5876 dims)
    - One-hot encoding of current position (9 dims: cash, long 1-4, short 1-4)
    Total: 11761 dims

    Without short selling (allow_short=False):
    - States for 4 top stocks to LONG (4 × 1469 = 5876 dims)
    - One-hot encoding of current position (5 dims: cash, long 1-4)
    Total: 5881 dims

    Args:
        top_4_stocks: List of (ticker, horizon_idx) for top stocks (to go LONG)
        bottom_4_stocks: List of (ticker, horizon_idx) for bottom stocks (to go SHORT)
        states: Dict mapping ticker -> state tensor
        current_position: Currently held stock ticker (or None if in cash)
        is_short: True if current position is a short, False if long
        device: Device to create tensor on
        allow_short: Whether short selling is allowed

    Returns:
        Global state tensor (11761 dims if allow_short=True, 5881 dims if allow_short=False)
    """
    # Extract states for top 4 stocks (LONG candidates)
    long_stock_states = []
    long_tickers = [ticker for ticker, _ in top_4_stocks]

    for ticker, _ in top_4_stocks:
        if ticker in states:
            long_stock_states.append(states[ticker])
        else:
            # Fallback: zero state if ticker not found
            long_stock_states.append(torch.zeros(1469, device=device))

    if allow_short:
        # Extract states for bottom 4 stocks (SHORT candidates)
        short_stock_states = []
        short_tickers = [ticker for ticker, _ in bottom_4_stocks]

        for ticker, _ in bottom_4_stocks:
            if ticker in states:
                short_stock_states.append(states[ticker])
            else:
                # Fallback: zero state if ticker not found
                short_stock_states.append(torch.zeros(1469, device=device))

        # Concatenate all 8 stock states (8 × 1469 = 11752 dims)
        stock_states_concat = torch.cat(long_stock_states + short_stock_states)

        # One-hot encode current position (9 dims)
        # [cash, long1, long2, long3, long4, short1, short2, short3, short4]
        position_encoding = torch.zeros(9, device=device)

        if current_position is None:
            position_encoding[0] = 1.0  # Cash
        elif not is_short and current_position in long_tickers:
            # Long position in one of the top stocks
            idx = long_tickers.index(current_position) + 1  # 1-4 for long stocks
            position_encoding[idx] = 1.0
        elif is_short and current_position in short_tickers:
            # Short position in one of the bottom stocks
            idx = short_tickers.index(current_position) + 5  # 5-8 for short stocks
            position_encoding[idx] = 1.0
        else:
            # Holding a stock not in current top/bottom 8 - encode as cash
            position_encoding[0] = 1.0

        # Concatenate: 11752 + 9 = 11761 dims
        global_state = torch.cat([stock_states_concat, position_encoding])

    else:
        # Without short selling: only 4 long stocks
        # Concatenate 4 stock states (4 × 1469 = 5876 dims)
        stock_states_concat = torch.cat(long_stock_states)

        # One-hot encode current position (5 dims)
        # [cash, long1, long2, long3, long4]
        position_encoding = torch.zeros(5, device=device)

        if current_position is None:
            position_encoding[0] = 1.0  # Cash
        elif current_position in long_tickers:
            # Long position in one of the top stocks
            idx = long_tickers.index(current_position) + 1  # 1-4 for long stocks
            position_encoding[idx] = 1.0
        else:
            # Holding a stock not in current top 4 - encode as cash
            position_encoding[0] = 1.0

        # Concatenate: 5876 + 5 = 5881 dims
        global_state = torch.cat([stock_states_concat, position_encoding])

    return global_state


def decode_action_to_trades(
    action: int,
    top_4_stocks: List[Tuple[str, int]],
    bottom_4_stocks: List[Tuple[str, int]],
    current_position: Optional[str],
    current_is_short: bool = False
) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Decode discrete action to trades (buy/sell/short operations) WITH SHORT SELLING.

    Args:
        action: Discrete action (0-8)
                0 = HOLD CURRENT POSITION (cash, long, or short)
                1-4 = GO LONG STOCK 1-4 (top-ranked for each horizon)
                5-8 = GO SHORT STOCK 5-8 (bottom-ranked for each horizon)
        top_4_stocks: List of (ticker, horizon_idx) for top stocks (LONG candidates)
        bottom_4_stocks: List of (ticker, horizon_idx) for bottom stocks (SHORT candidates)
        current_position: Currently held stock ticker (or None for cash)
        current_is_short: True if current position is a short, False if long

    Returns:
        Tuple of (trades, new_position, new_is_short)
        - trades: List of trade dicts with 'action' ('BUY', 'SELL', 'SHORT', 'COVER')
        - new_position: New position ticker (or None for cash)
        - new_is_short: True if new position is short
    """
    trades = []
    long_tickers = [ticker for ticker, _ in top_4_stocks]
    short_tickers = [ticker for ticker, _ in bottom_4_stocks]

    # Action 0: Hold current position (whether cash, long, or short)
    if action == 0:
        return trades, current_position, current_is_short

    # Actions 1-4: Go LONG on stocks 1-4
    if 1 <= action <= 4:
        target_ticker = long_tickers[action - 1]
        target_is_short = False

        # If already long this stock, do nothing
        if current_position == target_ticker and not current_is_short:
            return trades, current_position, current_is_short

        # Close current position if any
        if current_position is not None:
            if current_is_short:
                trades.append({'action': 'COVER', 'ticker': current_position})
            else:
                trades.append({'action': 'SELL', 'ticker': current_position})

        # Go long target stock
        trades.append({'action': 'BUY', 'ticker': target_ticker})

        return trades, target_ticker, False

    # Actions 5-8: Go SHORT on stocks 5-8 (bottom 4)
    if 5 <= action <= 8:
        target_ticker = short_tickers[action - 5]
        target_is_short = True

        # If already short this stock, do nothing
        if current_position == target_ticker and current_is_short:
            return trades, current_position, current_is_short

        # Close current position if any
        if current_position is not None:
            if current_is_short:
                trades.append({'action': 'COVER', 'ticker': current_position})
            else:
                trades.append({'action': 'SELL', 'ticker': current_position})

        # Go short target stock
        trades.append({'action': 'SHORT', 'ticker': target_ticker})

        return trades, target_ticker, True

    # Should never reach here
    return trades, current_position, current_is_short
