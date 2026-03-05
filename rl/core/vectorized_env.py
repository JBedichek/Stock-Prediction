#!/usr/bin/env python3
"""
Vectorized trading environment for parallel episode simulation.

Runs N trading environments in parallel to maximize GPU utilization
and speed up data collection.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from rl.rl_environment import TradingEnvironment


class VectorizedTradingEnv:
    """
    Vectorized wrapper for TradingEnvironment.

    Runs N environments in parallel. All environments share the same
    state_cache (read-only) but maintain independent episode state
    (portfolio, cash, position, dates).
    """

    def __init__(
        self,
        num_envs: int,
        data_loader,
        agent,
        initial_capital: float = 100000,
        max_positions: int = 1,
        episode_length: int = 30,
        device: str = 'cuda',
        trading_days_filter: Optional[List[str]] = None,
        top_k_per_horizon: int = 10
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            data_loader: Shared DatasetLoader instance
            agent: Shared agent instance
            initial_capital: Initial cash per environment
            max_positions: Max positions per environment
            episode_length: Episode length
            device: Device for tensors
            trading_days_filter: Trading days to use
            top_k_per_horizon: Top-k stocks per horizon
        """
        self.num_envs = num_envs
        self.device = device
        self.episode_length = episode_length

        # Create N independent environments
        # They all share state_cache (created once in first env, then reused)
        print(f"\n  Creating {num_envs} parallel environments...")

        # Create first environment (state cache will be shared from external env)
        self.envs = [
            TradingEnvironment(
                data_loader=data_loader,
                agent=agent,
                initial_capital=initial_capital,
                max_positions=max_positions,
                episode_length=episode_length,
                device=device,
                precompute_all_states=False,  # Will share cache externally
                trading_days_filter=trading_days_filter,
                top_k_per_horizon=top_k_per_horizon
            )
        ]

        # Create remaining environments (share state cache)
        for i in range(1, num_envs):
            env = TradingEnvironment(
                data_loader=data_loader,
                agent=agent,
                initial_capital=initial_capital,
                max_positions=max_positions,
                episode_length=episode_length,
                device=device,
                precompute_all_states=False,  # Will share cache from first env
                trading_days_filter=trading_days_filter,
                top_k_per_horizon=top_k_per_horizon
            )
            # Share state cache from first env (read-only, safe)
            env.state_cache = self.envs[0].state_cache
            env.price_cache = self.envs[0].price_cache
            self.envs.append(env)

        print(f"  ✅ {num_envs} environments created (sharing state cache)")

        # Track which episodes are done
        self.dones = [False] * num_envs
        self.episode_steps = [0] * num_envs

    def reset(self) -> Tuple[List[Dict[str, torch.Tensor]], List[Optional[str]]]:
        """
        Reset all environments.

        Returns:
            Tuple of (states_list, positions_list)
            - states_list: List of N state dicts
            - positions_list: List of N current positions (None for cash)
        """
        states_list = []
        positions_list = [None] * self.num_envs

        for i, env in enumerate(self.envs):
            states = env.reset()
            states_list.append(states)
            self.dones[i] = False
            self.episode_steps[i] = 0

        return states_list, positions_list

    def step(
        self,
        actions_list: List[int],
        trades_list: List[List[Dict]],
        positions_list: List[Optional[str]]
    ) -> Tuple[List[Dict], List[float], List[bool], List[Dict], List[Optional[str]]]:
        """
        Step all environments in parallel.

        Args:
            actions_list: List of N discrete actions (0-4)
            trades_list: List of N trade lists
            positions_list: List of N current positions

        Returns:
            Tuple of (next_states_list, rewards_list, dones_list, infos_list, next_positions_list)
        """
        next_states_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        next_positions_list = []

        for i, env in enumerate(self.envs):
            # Skip if episode already done
            if self.dones[i]:
                next_states_list.append({})
                rewards_list.append(0.0)
                dones_list.append(True)
                infos_list.append({'portfolio_value': 0, 'num_positions': 0})
                next_positions_list.append(None)
                continue

            # Convert trades to environment actions
            actions = {}
            for trade in trades_list[i]:
                ticker = trade['ticker']
                if trade['action'] == 'BUY':
                    actions[ticker] = 1
                elif trade['action'] == 'SELL':
                    actions[ticker] = 2

            # Step environment
            next_states, reward, done, info = env.step(actions)

            # Update position based on trades
            next_position = positions_list[i]
            for trade in trades_list[i]:
                if trade['action'] == 'SELL':
                    next_position = None
                elif trade['action'] == 'BUY':
                    next_position = trade['ticker']

            # Increment step counter
            self.episode_steps[i] += 1

            # Check if episode done (by length or env done flag)
            if done or self.episode_steps[i] >= self.episode_length:
                done = True
                self.dones[i] = True

            next_states_list.append(next_states)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)
            next_positions_list.append(next_position)

        return next_states_list, rewards_list, dones_list, infos_list, next_positions_list

    def get_active_mask(self) -> List[bool]:
        """Get mask of which environments are still active."""
        return [not done for done in self.dones]

    def get_num_active(self) -> int:
        """Get number of active (not done) environments."""
        return sum(self.get_active_mask())

    def reset_done_envs(self, positions_list: List[Optional[str]]) -> Tuple[List[Dict], List[Optional[str]]]:
        """
        Reset environments that are done.

        Args:
            positions_list: Current positions list

        Returns:
            Tuple of (updated_states_list, updated_positions_list)
        """
        states_list = [None] * self.num_envs
        new_positions_list = list(positions_list)

        for i, env in enumerate(self.envs):
            if self.dones[i]:
                # Reset this environment
                states = env.reset()
                states_list[i] = states
                new_positions_list[i] = None
                self.dones[i] = False
                self.episode_steps[i] = 0
            else:
                # Keep existing state
                states_list[i] = None  # Will be updated in next step

        return states_list, new_positions_list
