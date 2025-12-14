"""
Worker Process for Multi-GPU RL Training

Workers run on individual GPUs collecting episodes asynchronously
while the main process trains the Q-network.
"""

import torch
import random
import numpy as np
import multiprocessing as mp
import sys
import os
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, StockDQN, PredictorFeatureExtractor
from rl.rl_environment import TradingEnvironment
from rl.shared_buffer import SharedReplayBuffer
from rl.shared_weights import SharedWeights, WorkerWeightSyncer
from inference.backtest_simulation import DatasetLoader


def worker_process(
    worker_id: int,
    gpu_id: int,
    config: Dict,
    shared_buffer: SharedReplayBuffer,
    shared_weights: SharedWeights,
    global_step: mp.Value,
    epsilon: mp.Value,
    stop_event: mp.Event,
    stats_queue: mp.Queue
):
    """
    Worker process that collects episodes on a specific GPU.

    Args:
        worker_id: Worker ID (0 or 1)
        gpu_id: GPU to use (0 or 1)
        config: Configuration dictionary
        shared_buffer: Shared replay buffer for experiences
        shared_weights: Shared Q-network weights
        global_step: Shared global step counter
        epsilon: Shared exploration rate
        stop_event: Event to signal worker to stop
        stats_queue: Queue for sending statistics to main process
    """
    try:
        # Set device
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*60}")
        print(f"Worker {worker_id} starting on {device}")
        print(f"{'='*60}\n")

        # Set random seeds for reproducibility (per-worker seeds)
        seed = config['seed'] + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load data
        print(f"[Worker {worker_id}] Loading data...")
        data_loader = DatasetLoader(
            dataset_path=config['dataset_path'],
            prices_path=config.get('prices_path'),
            cluster_path=config.get('cluster_path'),
            scaler_path=config.get('scaler_path')
        )

        if config.get('features_cache'):
            print(f"[Worker {worker_id}] Loading feature cache...")
            data_loader.load_feature_cache(config['features_cache'])
        else:
            print(f"[Worker {worker_id}] Preloading features...")
            data_loader.preload_features(num_workers=2)

        # Subsample stocks if needed (Phase 1)
        if config.get('num_stocks') and config['num_stocks'] < len(data_loader.test_tickers):
            original_tickers = data_loader.test_tickers
            data_loader.test_tickers = random.sample(
                original_tickers,
                config['num_stocks']
            )
            print(f"[Worker {worker_id}] Using {len(data_loader.test_tickers)} stocks")

        # Initialize local agent (for feature extraction)
        print(f"[Worker {worker_id}] Initializing agent...")
        agent = TradingAgent(
            predictor_checkpoint_path=config['predictor_checkpoint'],
            state_dim=config['state_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=config['action_dim'],
            device=device
        )

        # Initialize local environment
        print(f"[Worker {worker_id}] Initializing environment...")
        env = TradingEnvironment(
            data_loader=data_loader,
            agent=agent,
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            episode_length=config['episode_length'],
            transaction_cost=config['transaction_cost'],
            device=device
        )

        # Initialize local Q-network (for action selection)
        local_q_network = StockDQN(
            state_dim=config['state_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=config['action_dim']
        ).to(device)
        local_q_network.eval()  # Always in eval mode for collection

        # Wait for shared weights to be initialized
        print(f"[Worker {worker_id}] Waiting for initial weights...")
        while not shared_weights.is_initialized():
            if stop_event.is_set():
                return
            import time
            time.sleep(0.1)

        # Initialize weight syncer
        weight_syncer = WorkerWeightSyncer(
            shared_weights=shared_weights,
            q_network=local_q_network,
            sync_frequency=config.get('weight_sync_frequency', 500)
        )

        # Sync initial weights
        weight_syncer.maybe_sync(force=True)
        print(f"[Worker {worker_id}] Initial weights loaded")

        print(f"[Worker {worker_id}] Starting episode collection...\n")

        # Episode collection loop
        episode_count = 0
        phase = config.get('phase', 1)  # Phase 1 or Phase 2

        while not stop_event.is_set():
            # Run episode
            episode_reward, episode_steps = run_episode(
                worker_id=worker_id,
                env=env,
                local_q_network=local_q_network,
                shared_buffer=shared_buffer,
                global_step=global_step,
                epsilon=epsilon,
                weight_syncer=weight_syncer,
                stop_event=stop_event,
                device=device,
                phase=phase,
                config=config
            )

            if stop_event.is_set():
                break

            # Get episode statistics
            episode_stats = env.get_episode_stats()

            # Send statistics to main process
            stats_queue.put({
                'worker_id': worker_id,
                'episode': episode_count,
                'reward': episode_reward,
                'steps': episode_steps,
                'stats': episode_stats,
                'sync_info': weight_syncer.get_sync_info()
            })

            episode_count += 1

        print(f"\n[Worker {worker_id}] Shutting down. Collected {episode_count} episodes.")

    except Exception as e:
        print(f"\n[Worker {worker_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_episode(
    worker_id: int,
    env: TradingEnvironment,
    local_q_network: StockDQN,
    shared_buffer: SharedReplayBuffer,
    global_step: mp.Value,
    epsilon: mp.Value,
    weight_syncer: WorkerWeightSyncer,
    stop_event: mp.Event,
    device: torch.device,
    phase: int,
    config: Dict
) -> tuple:
    """
    Run a single episode and collect experiences.

    Args:
        worker_id: Worker ID
        env: Trading environment
        local_q_network: Local Q-network for action selection
        shared_buffer: Shared replay buffer
        global_step: Shared step counter
        epsilon: Shared exploration rate
        weight_syncer: Weight synchronization manager
        stop_event: Stop signal
        device: Device to use
        phase: Training phase (1 or 2)
        config: Configuration dict

    Returns:
        (episode_reward, episode_steps)
    """
    # Reset environment
    states = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = 0

    while not done and not stop_event.is_set():
        # Select actions based on phase
        current_epsilon = epsilon.value

        if phase == 1:
            actions = select_actions_phase1(
                states, local_q_network, env, current_epsilon, device, config
            )
        else:  # Phase 2
            actions = select_actions_phase2(
                states, local_q_network, env, current_epsilon, device
            )

        # Execute actions
        next_states, reward, done, info = env.step(actions)

        # Store transitions in shared buffer
        for ticker in actions.keys():
            if ticker in states and ticker in next_states:
                shared_buffer.push(
                    state=states[ticker],
                    action=actions[ticker],
                    reward=reward,
                    next_state=next_states[ticker],
                    done=done
                )

        # Increment global step
        with global_step.get_lock():
            global_step.value += 1

        # Periodic weight synchronization
        weight_syncer.maybe_sync()

        states = next_states
        episode_reward += reward
        episode_steps += 1

    return episode_reward, episode_steps


def select_actions_phase1(
    states: Dict[str, torch.Tensor],
    q_network: StockDQN,
    env: TradingEnvironment,
    epsilon: float,
    device: torch.device,
    config: Dict
) -> Dict[str, int]:
    """
    Select actions for Phase 1 (2 actions: HOLD or BUY_LARGE).

    Phase 1 mapping:
    - Agent action 0 (HOLD) -> Environment action 0 (HOLD)
    - Agent action 1 (BUY) -> Environment action 3 (BUY_LARGE)

    Args:
        states: Dictionary of ticker -> state tensor
        q_network: Q-network for action selection
        env: Environment (for portfolio state)
        epsilon: Exploration rate
        device: Device
        config: Configuration

    Returns:
        Dictionary of ticker -> environment action id
    """
    if len(states) == 0:
        return {}

    # Stack states for batch inference
    tickers = list(states.keys())
    state_tensors = torch.stack([states[ticker] for ticker in tickers])

    # Get Q-values for all stocks
    with torch.no_grad():
        q_values_batch = q_network(state_tensors)  # (num_stocks, 2 or 5)
        # For Phase 1, only use first 2 actions
        q_values_batch = q_values_batch[:, :2]

    # Epsilon-greedy action selection
    actions_phase1 = {}
    for i, ticker in enumerate(tickers):
        if random.random() < epsilon:
            # Explore: random action (0 or 1)
            action = random.randint(0, 1)
        else:
            # Exploit: best action
            action = q_values_batch[i].argmax().item()

        actions_phase1[ticker] = action

    # Filter to top-K buy actions
    buy_candidates = []
    for i, ticker in enumerate(tickers):
        action = actions_phase1[ticker]

        if action == 1:  # BUY
            # Check constraints
            if ticker not in env.portfolio and len(env.portfolio) < env.max_positions:
                q_value_buy = q_values_batch[i, 1].item()
                buy_candidates.append((ticker, q_value_buy))

    # Keep top-K buys
    buy_candidates.sort(key=lambda x: x[1], reverse=True)
    top_k_buys = buy_candidates[:config.get('top_k_buys', 3)]

    # Map to environment actions
    actions_env = {}
    for ticker, _ in top_k_buys:
        actions_env[ticker] = 3  # BUY_LARGE in environment

    return actions_env


def select_actions_phase2(
    states: Dict[str, torch.Tensor],
    q_network: StockDQN,
    env: TradingEnvironment,
    epsilon: float,
    device: torch.device
) -> Dict[str, int]:
    """
    Select actions for Phase 2 (5 actions: HOLD, BUY_SMALL/MED/LARGE, SELL).

    Args:
        states: Dictionary of ticker -> state tensor
        q_network: Q-network for action selection
        env: Environment (for portfolio state)
        epsilon: Exploration rate
        device: Device

    Returns:
        Dictionary of ticker -> action id (0-4)
    """
    if len(states) == 0:
        return {}

    # Stack states for batch inference
    tickers = list(states.keys())
    state_tensors = torch.stack([states[ticker] for ticker in tickers])

    # Get Q-values for all stocks
    with torch.no_grad():
        q_values_batch = q_network(state_tensors)  # (num_stocks, 5)

    # Epsilon-greedy action selection
    actions = {}
    for i, ticker in enumerate(tickers):
        if random.random() < epsilon:
            # Explore: random action
            action = random.randint(0, 4)
        else:
            # Exploit: best action
            action = q_values_batch[i].argmax().item()

        # Apply constraints based on current state
        action = apply_action_constraints(ticker, action, q_values_batch[i], env)

        if action is not None:
            actions[ticker] = action

    return actions


def apply_action_constraints(
    ticker: str,
    action: int,
    q_values: torch.Tensor,
    env: TradingEnvironment
) -> Optional[int]:
    """
    Apply portfolio constraints to actions.

    Args:
        ticker: Stock ticker
        action: Proposed action
        q_values: Q-values for all actions
        env: Environment

    Returns:
        Valid action or None if no valid action
    """
    # Check if we have a position
    has_position = ticker in env.portfolio

    # Constraint 1: Can't buy if already have position
    if action in [1, 2, 3] and has_position:
        # Find next best action (either HOLD or SELL)
        valid_actions = [0, 4]  # HOLD or SELL
        valid_q_values = q_values[valid_actions]
        best_idx = valid_q_values.argmax().item()
        action = valid_actions[best_idx]

    # Constraint 2: Can't buy if portfolio is full
    if action in [1, 2, 3] and len(env.portfolio) >= env.max_positions:
        action = 0  # HOLD

    # Constraint 3: Can't sell if no position
    if action == 4 and not has_position:
        action = 0  # HOLD

    # Constraint 4: Can't buy if no cash
    if action in [1, 2, 3] and env.cash <= 0:
        action = 0  # HOLD

    return action
