"""
Reinforcement Learning for Stock Trading

This package contains RL-based trading agents that learn when to buy, sell,
and how much to allocate to maximize portfolio returns.

Components:
- rl_components: Core RL neural networks (DQN, feature extractors, replay buffer)
- rl_environment: Trading environment (Gym-style interface)
- train_rl: Training scripts
- rl_backtest: Backtesting and evaluation
"""

from .rl_components import (
    PredictorFeatureExtractor,
    StockDQN,
    TradingAgent,
    ReplayBuffer,
    compute_dqn_loss,
    ACTIONS,
    ACTION_NAMES_TO_IDS
)

from .rl_environment import TradingEnvironment

# Import backtest only if needed (lazy import to avoid circular dependencies)
def get_backtester():
    """Get RLBacktester class (lazy import)."""
    from .rl_backtest import RLBacktester, BaselineComparison
    return RLBacktester, BaselineComparison

__all__ = [
    'PredictorFeatureExtractor',
    'StockDQN',
    'TradingAgent',
    'ReplayBuffer',
    'compute_dqn_loss',
    'ACTIONS',
    'ACTION_NAMES_TO_IDS',
    'TradingEnvironment',
    'get_backtester'
]
