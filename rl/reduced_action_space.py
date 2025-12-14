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


def get_top_4_stocks(cached_states: Dict[str, torch.Tensor]) -> List[Tuple[str, int]]:
    """
    Get top-1 stock for each of 4 time horizons (VECTORIZED).

    Args:
        cached_states: Dict mapping ticker -> cached state (1444 dims)

    Returns:
        List of (ticker, horizon_idx) tuples, one per horizon
    """
    if len(cached_states) == 0:
        return []

    # Vectorized version: much faster than Python loops
    tickers = list(cached_states.keys())

    # Stack all cached states: (N_stocks, 1444 or 1469)
    states_list = []
    for ticker in tickers:
        state = cached_states[ticker]
        if state.dim() > 1:
            state = state.squeeze()
        states_list.append(state[:1444])  # First 1444 dims only

    states_tensor = torch.stack(states_list)  # (N_stocks, 1444)

    # Extract expected returns for all stocks at once: (N_stocks, 4)
    expected_returns = states_tensor[:, 1428:1432]

    # Find argmax for each horizon (0-3)
    top_stocks = []
    for horizon_idx in range(4):
        horizon_returns = expected_returns[:, horizon_idx]  # (N_stocks,)
        best_idx = horizon_returns.argmax().item()
        best_ticker = tickers[best_idx]
        top_stocks.append((best_ticker, horizon_idx))

    return top_stocks


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
    states: Dict[str, torch.Tensor],
    current_position: Optional[str],
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create global state representation for actor.

    Concatenates:
    - States for 4 top stocks (4 × 1469 = 5876 dims)
    - One-hot encoding of current position (5 dims: cash or stock 1-4)
    Total: 5881 dims

    Args:
        top_4_stocks: List of (ticker, horizon_idx) for top stocks
        states: Dict mapping ticker -> state tensor
        current_position: Currently held stock ticker (or None if in cash)
        device: Device to create tensor on

    Returns:
        Global state tensor (5881 dims)
    """
    # Extract states for top 4 stocks
    stock_states = []
    stock_tickers = [ticker for ticker, _ in top_4_stocks]

    for ticker, _ in top_4_stocks:
        if ticker in states:
            stock_states.append(states[ticker])
        else:
            # Fallback: zero state if ticker not found
            stock_states.append(torch.zeros(1469, device=device))

    # Concatenate stock states (4 × 1469 = 5876 dims)
    stock_states_concat = torch.cat(stock_states)

    # One-hot encode current position (5 dims)
    position_encoding = torch.zeros(5, device=device)
    if current_position is None:
        position_encoding[0] = 1.0  # Cash
    elif current_position in stock_tickers:
        idx = stock_tickers.index(current_position) + 1  # 1-4 for stocks
        position_encoding[idx] = 1.0
    else:
        # Holding a stock not in top 4 - encode as cash
        position_encoding[0] = 1.0

    # Concatenate: 5876 + 5 = 5881 dims
    global_state = torch.cat([stock_states_concat, position_encoding])

    return global_state


def decode_action_to_trades(
    action: int,
    top_4_stocks: List[Tuple[str, int]],
    current_position: Optional[str]
) -> List[Dict]:
    """
    Decode discrete action to trades (sell/buy operations).

    Args:
        action: Discrete action (0-4)
                0 = HOLD CURRENT POSITION (cash or stock)
                1-4 = SWITCH TO STOCK 1-4 (top-ranked for each horizon)
        top_4_stocks: List of (ticker, horizon_idx) for top stocks
        current_position: Currently held stock ticker (or None for cash)

    Returns:
        List of trade dicts: [{'action': 'SELL', 'ticker': ...}, {'action': 'BUY', 'ticker': ...}]
    """
    trades = []
    stock_tickers = [ticker for ticker, _ in top_4_stocks]

    # Action 0: Hold current position (whether cash or stock)
    if action == 0:
        return trades  # No trades - maintain current position

    # Actions 1-4: Switch to specific stock
    target_position = stock_tickers[action - 1]

    # If already holding the target stock, do nothing
    if current_position == target_position:
        return trades

    # Sell current position if any
    if current_position is not None:
        trades.append({'action': 'SELL', 'ticker': current_position})

    # Buy target position
    trades.append({'action': 'BUY', 'ticker': target_position})

    return trades
