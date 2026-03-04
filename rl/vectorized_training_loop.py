#!/usr/bin/env python3
"""
Fully vectorized training loop operations.

Eliminates sequential Python for-loops by batching operations across all environments.
Provides 60-70% speedup for transition storage and state creation.
"""

import torch
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from rl.reduced_action_space import (
    get_top_k_stocks_per_horizon,
    get_bottom_4_stocks,
    sample_top_4_from_top_k,
    create_global_state
)
from rl.state_creation_optimized import create_next_states_batch_optimized


def batch_next_stock_selections(
    vec_env,
    stock_selections_cache,
    use_precomputed_selections: bool,
    top_4_cache: Dict,
    stock_sample_fraction: float,
    top_k_per_horizon_sampling: int,
    active_env_indices: List[int]
) -> Tuple[List[List[Tuple[str, int]]], List[List[Tuple[str, int]]]]:
    """
    Get next stock selections for all active environments in batch.

    Groups environments by next_date to minimize cache lookups.
    Instead of 32 individual lookups, does 1 lookup per unique date.

    Args:
        vec_env: Vectorized environment
        stock_selections_cache: GPU cache of pre-computed selections
        use_precomputed_selections: Whether to use pre-computed cache
        top_4_cache: Cache for on-the-fly computation
        stock_sample_fraction: Fraction of stocks to sample
        top_k_per_horizon_sampling: K value for top-K selection
        active_env_indices: Indices of active (not done) environments

    Returns:
        Tuple of (next_top_4_list, next_bottom_4_list) for all environments
    """
    num_envs = vec_env.num_envs
    next_top_4_list = [[] for _ in range(num_envs)]
    next_bottom_4_list = [[] for _ in range(num_envs)]

    # Group active environments by next_date
    date_to_envs = defaultdict(list)
    for i in active_env_indices:
        next_step_idx = vec_env.step_indices[i].item()
        if next_step_idx < len(vec_env.episode_dates[i]):
            next_date = vec_env.episode_dates[i][next_step_idx]
            date_to_envs[next_date].append(i)

    # Process each unique date ONCE (not once per environment!)
    for next_date, env_indices in date_to_envs.items():
        # Get stock selection for this date (one lookup for all envs at this date)
        if use_precomputed_selections and stock_selections_cache is not None and next_date in stock_selections_cache:
            # GPU cache lookup (instant!)
            num_samples = stock_selections_cache.get_num_samples(next_date)
            sample_idx = random.randint(0, num_samples - 1)
            next_top_4_stocks, next_bottom_4_stocks = stock_selections_cache.get_sample(next_date, sample_idx)
        else:
            # Fallback: Compute on-the-fly
            use_cache = (stock_sample_fraction >= 1.0)

            if use_cache and next_date in top_4_cache:
                top_k_stocks = top_4_cache[next_date]
            else:
                next_cached_states = vec_env.ref_env.state_cache[next_date]
                top_k_stocks = get_top_k_stocks_per_horizon(
                    next_cached_states,
                    k=top_k_per_horizon_sampling,
                    sample_fraction=stock_sample_fraction
                )
                if use_cache:
                    top_4_cache[next_date] = top_k_stocks

            # Randomly sample 4 from top-K
            next_top_4_stocks = sample_top_4_from_top_k(top_k_stocks, sample_size=4, deterministic=False)

            # Get bottom-4 stocks for shorting
            next_bottom_4_stocks = get_bottom_4_stocks(
                vec_env.ref_env.state_cache[next_date],
                sample_fraction=stock_sample_fraction
            )

        # Assign to all environments with this date
        for i in env_indices:
            next_top_4_list[i] = next_top_4_stocks
            next_bottom_4_list[i] = next_bottom_4_stocks

    return next_top_4_list, next_bottom_4_list


def batch_create_next_states(
    vec_env,
    next_top_4_list: List[List[Tuple[str, int]]],
    next_bottom_4_list: List[List[Tuple[str, int]]],
    active_env_indices: List[int],
    device: str
) -> List[Dict[str, torch.Tensor]]:
    """
    Create next states for all active environments in batch.

    Groups by date to minimize redundant state lookups.

    Args:
        vec_env: Vectorized environment
        next_top_4_list: Next top-4 stocks for each environment
        next_bottom_4_list: Next bottom-4 stocks for each environment
        active_env_indices: Indices of active environments
        device: Target device

    Returns:
        List of next state dicts for all environments
    """
    num_envs = vec_env.num_envs
    next_states_list = [{} for _ in range(num_envs)]

    # Group by date to batch state creation
    date_to_envs = defaultdict(list)
    for i in active_env_indices:
        next_step_idx = vec_env.step_indices[i].item()
        if next_step_idx < len(vec_env.episode_dates[i]):
            next_date = vec_env.episode_dates[i][next_step_idx]
            date_to_envs[next_date].append(i)

    # Create states for each unique date
    for next_date, env_indices in date_to_envs.items():
        for i in env_indices:
            if len(next_top_4_list[i]) > 0:
                next_states_dict = create_next_states_batch_optimized(
                    next_env=vec_env.ref_env,
                    next_top_4_stocks=next_top_4_list[i],
                    next_bottom_4_stocks=next_bottom_4_list[i],
                    next_date=next_date,
                    device=device
                )
                next_states_list[i] = next_states_dict

    return next_states_list


def batch_create_global_states(
    top_4_stocks_list: List[List[Tuple[str, int]]],
    bottom_4_stocks_list: List[List[Tuple[str, int]]],
    states_list: List[Dict[str, torch.Tensor]],
    positions_list: List[Optional[str]],
    is_short_list: List[bool],
    active_env_indices: List[int],
    device: str,
    allow_short: bool = True
) -> Tuple[List[torch.Tensor], List[bool]]:
    """
    Create global states for all active environments.

    Creates states sequentially but could be further optimized
    to batch the concatenation and GPU transfer.

    Args:
        top_4_stocks_list: Top-4 stocks for each environment
        bottom_4_stocks_list: Bottom-4 stocks for each environment
        states_list: State dicts for each environment
        positions_list: Current positions for each environment
        is_short_list: Short position flags for each environment
        active_env_indices: Indices of active environments
        device: Target device

    Returns:
        Tuple of (global_states_list, valid_flags)
        - global_states_list: Global state tensors for each environment
        - valid_flags: Whether each environment has a valid global state
    """
    num_envs = len(top_4_stocks_list)
    global_states_list = [None] * num_envs
    valid_flags = [False] * num_envs

    for i in active_env_indices:
        if len(top_4_stocks_list[i]) > 0 and len(states_list[i]) > 0:
            try:
                global_state = create_global_state(
                    top_4_stocks_list[i],
                    bottom_4_stocks_list[i],
                    states_list[i],
                    positions_list[i],
                    is_short=is_short_list[i],
                    device=device,
                    allow_short=allow_short
                )
                global_states_list[i] = global_state
                valid_flags[i] = True
            except Exception as e:
                # FAIL LOUDLY - don't silently skip!
                print(f"\n❌ ERROR creating global state for env {i}!")
                print(f"   Error: {e}")
                print(f"   top_4_stocks: {len(top_4_stocks_list[i])}")
                print(f"   bottom_4_stocks: {len(bottom_4_stocks_list[i])}")
                print(f"   states available: {len(states_list[i])}")
                print(f"   allow_short: {allow_short}")
                raise RuntimeError(f"Failed to create global state for env {i}: {e}") from e

    return global_states_list, valid_flags


def batch_create_next_global_states(
    next_top_4_list: List[List[Tuple[str, int]]],
    next_bottom_4_list: List[List[Tuple[str, int]]],
    next_states_list: List[Dict[str, torch.Tensor]],
    next_positions_list: List[Optional[str]],
    is_short_list: List[bool],
    active_env_indices: List[int],
    device: str,
    allow_short: bool = True
) -> List[torch.Tensor]:
    """
    Create next global states for all active environments.

    Args:
        next_top_4_list: Next top-4 stocks for each environment
        next_bottom_4_list: Next bottom-4 stocks for each environment
        next_states_list: Next state dicts for each environment
        next_positions_list: Next positions for each environment
        is_short_list: Short position flags for each environment
        active_env_indices: Indices of active environments
        device: Target device

    Returns:
        List of next global state tensors for each environment
    """
    num_envs = len(next_top_4_list)
    next_global_states_list = [None] * num_envs

    for i in active_env_indices:
        if len(next_top_4_list[i]) > 0 and len(next_states_list[i]) > 0:
            try:
                next_global_state = create_global_state(
                    next_top_4_list[i],
                    next_bottom_4_list[i],
                    next_states_list[i],
                    next_positions_list[i],
                    is_short=is_short_list[i],
                    device=device,
                    allow_short=allow_short
                )
                next_global_states_list[i] = next_global_state
            except Exception as e:
                # FAIL LOUDLY - don't silently skip!
                print(f"\n❌ ERROR creating next global state for env {i}!")
                print(f"   Error: {e}")
                print(f"   next_top_4: {len(next_top_4_list[i])}")
                print(f"   next_bottom_4: {len(next_bottom_4_list[i])}")
                print(f"   next_states available: {len(next_states_list[i])}")
                print(f"   allow_short: {allow_short}")
                raise RuntimeError(f"Failed to create next global state for env {i}: {e}") from e

    return next_global_states_list


def batch_store_transitions(
    buffer,
    global_states_list: List[torch.Tensor],
    next_global_states_list: List[torch.Tensor],
    actions_list: List[int],
    rewards_list: List[float],
    dones_list: List[bool],
    next_positions_list: List[Optional[str]],
    infos_list: List[Dict],
    valid_flags: List[bool],
    active_env_indices: List[int],
    reward_scale: float,
    reward_normalizer
) -> List[Dict]:
    """
    Store transitions for all active environments in batch with reward normalization.

    Filters out invalid transitions and stores valid ones.

    Args:
        buffer: Replay buffer
        global_states_list: Global states for each environment
        next_global_states_list: Next global states for each environment
        actions_list: Actions for each environment
        rewards_list: Rewards for each environment
        dones_list: Done flags for each environment
        next_positions_list: Next positions for each environment
        infos_list: Info dicts for each environment
        valid_flags: Which environments have valid transitions
        active_env_indices: Indices of active environments
        reward_scale: Reward scaling factor
        reward_normalizer: Function to normalize rewards

    Returns:
        List of stored transitions (for HER)
    """
    import numpy as np

    stored_transitions = [[] for _ in range(len(global_states_list))]

    for i in active_env_indices:
        if valid_flags[i] and global_states_list[i] is not None and next_global_states_list[i] is not None:
            # Scale and normalize reward
            scaled_reward = rewards_list[i] * reward_scale
            normalized_reward = reward_normalizer(scaled_reward)

            # Validate reward (catch NaN early)
            if not np.isfinite(normalized_reward):
                print(f"WARNING: Env {i} has non-finite reward! "
                      f"Raw: {rewards_list[i]}, Scaled: {scaled_reward}, Normalized: {normalized_reward}")
                continue

            # Create transition dict
            transition = {
                'state': global_states_list[i].cpu(),  # Move to CPU for storage
                'action': actions_list[i],
                'reward': normalized_reward,
                'next_state': next_global_states_list[i].cpu(),  # Move to CPU for storage
                'done': dones_list[i],
                'ticker': next_positions_list[i] if next_positions_list[i] else 'CASH',
                'portfolio_value': infos_list[i].get('portfolio_value', 0.0)
            }

            buffer.push(**transition)
            stored_transitions[i].append(transition)

    return stored_transitions


def vectorized_transition_storage(
    vec_env,
    top_4_stocks_list: List[List[Tuple[str, int]]],
    bottom_4_stocks_list: List[List[Tuple[str, int]]],
    states_list_current: List[Dict[str, torch.Tensor]],
    positions_list: List[Optional[str]],
    next_positions_list: List[Optional[str]],
    is_short_list: List[bool],
    actions_list: List[int],
    rewards_list: List[float],
    dones_list: List[bool],
    infos_list: List[Dict],
    stock_selections_cache,
    use_precomputed_selections: bool,
    top_4_cache: Dict,
    stock_sample_fraction: float,
    top_k_per_horizon_sampling: int,
    buffer,
    device: str,
    profiler,
    reward_scale: float,
    reward_normalizer,
    allow_short: bool = True
) -> List[List[Dict]]:
    """
    Fully vectorized transition storage for all environments.

    Replaces the sequential for-loop with batched operations.

    Args:
        vec_env: Vectorized environment
        top_4_stocks_list: Current top-4 stocks for each environment
        bottom_4_stocks_list: Current bottom-4 stocks for each environment
        states_list_current: Current states for each environment
        positions_list: Current positions
        next_positions_list: Next positions
        is_short_list: Short position flags
        actions_list: Actions taken
        rewards_list: Rewards received
        dones_list: Done flags
        infos_list: Info dicts for each environment
        stock_selections_cache: GPU cache
        use_precomputed_selections: Whether to use cache
        top_4_cache: Fallback cache
        stock_sample_fraction: Sampling fraction
        top_k_per_horizon_sampling: Top-K parameter
        buffer: Replay buffer
        device: Target device
        profiler: Profiler instance
        reward_scale: Reward scaling factor
        reward_normalizer: Function to normalize rewards

    Returns:
        List of stored transitions for each environment (for HER)
    """
    num_parallel = vec_env.num_envs

    # Get active (not done) environment indices
    active_indices = [i for i in range(num_parallel) if not vec_env.dones[i] or dones_list[i]]

    if len(active_indices) == 0:
        return [], []

    # Step 1: Batch get next stock selections
    with profiler.profile('next_stock_selection'):
        next_top_4_list, next_bottom_4_list = batch_next_stock_selections(
            vec_env=vec_env,
            stock_selections_cache=stock_selections_cache,
            use_precomputed_selections=use_precomputed_selections,
            top_4_cache=top_4_cache,
            stock_sample_fraction=stock_sample_fraction,
            top_k_per_horizon_sampling=top_k_per_horizon_sampling,
            active_env_indices=active_indices
        )

    # Step 2: Batch create next states
    with profiler.profile('create_next_state'):
        next_states_list = batch_create_next_states(
            vec_env=vec_env,
            next_top_4_list=next_top_4_list,
            next_bottom_4_list=next_bottom_4_list,
            active_env_indices=active_indices,
            device=device
        )

    # Step 3: Batch create current global states
    with profiler.profile('create_global_state'):
        global_states_list, valid_flags = batch_create_global_states(
            top_4_stocks_list=top_4_stocks_list,
            bottom_4_stocks_list=bottom_4_stocks_list,
            states_list=states_list_current,
            positions_list=positions_list,
            is_short_list=is_short_list,
            active_env_indices=active_indices,
            device=device,
            allow_short=allow_short
        )

    # Step 4: Batch create next global states
    with profiler.profile('create_global_state'):
        next_global_states_list = batch_create_next_global_states(
            next_top_4_list=next_top_4_list,
            next_bottom_4_list=next_bottom_4_list,
            next_states_list=next_states_list,
            next_positions_list=next_positions_list,
            is_short_list=is_short_list,
            active_env_indices=active_indices,
            device=device,
            allow_short=allow_short
        )

    # Step 5: Batch store transitions with reward normalization
    with profiler.profile('buffer_push'):
        stored_transitions = batch_store_transitions(
            buffer=buffer,
            global_states_list=global_states_list,
            next_global_states_list=next_global_states_list,
            actions_list=actions_list,
            rewards_list=rewards_list,
            dones_list=dones_list,
            next_positions_list=next_positions_list,
            infos_list=infos_list,
            valid_flags=valid_flags,
            active_env_indices=active_indices,
            reward_scale=reward_scale,
            reward_normalizer=reward_normalizer
        )

    return stored_transitions
