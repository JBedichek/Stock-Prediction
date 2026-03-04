"""
DQN Agent Evaluation Framework

Comprehensive evaluation system for testing trained DQN trading agents.

Main components:
- DQNEvaluator: Run out-of-sample evaluation with multiple episodes
- Metrics: Risk-adjusted performance metrics (Sharpe, Sortino, max drawdown, etc.)
- Baselines: Comparison strategies (random, hold, long, etc.)
"""

from rl.evaluation.evaluator import DQNEvaluator
from rl.evaluation.metrics import (
    compute_all_metrics,
    aggregate_metrics,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor
)
from rl.evaluation.baselines import (
    get_baseline_strategy,
    RandomStrategy,
    AlwaysHoldStrategy,
    AlwaysLongStrategy,
    AlwaysShortStrategy,
    MomentumStrategy,
    LongShortStrategy
)

__all__ = [
    'DQNEvaluator',
    'compute_all_metrics',
    'aggregate_metrics',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'get_baseline_strategy',
    'RandomStrategy',
    'AlwaysHoldStrategy',
    'AlwaysLongStrategy',
    'AlwaysShortStrategy',
    'MomentumStrategy',
    'LongShortStrategy',
]
