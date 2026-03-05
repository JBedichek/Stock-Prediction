"""
Reinforcement Learning for Stock Trading

This module contains RL-based trading strategies:

Submodules:
- rl.dqn: Deep Q-Network based trading
- rl.actor_critic: Actor-Critic based trading
- rl.core: Shared components (environment, networks, backtest)
- rl.utils: Utility functions
- rl.evaluation: Evaluation and metrics
"""

from .core import rl_components, rl_environment
