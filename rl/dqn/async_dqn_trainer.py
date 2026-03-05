#!/usr/bin/env python3
"""
Asynchronous DQN Training with Parallel Experience Collection.

This solves the GPU utilization problem by:
1. Running multiple environment workers in parallel (CPU)
2. Having a dedicated GPU training thread that trains continuously
3. Workers and trainer communicate via shared replay buffer

This keeps the GPU saturated with training work while CPUs collect experience.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
import numpy as np
import random
import time
from queue import Empty
from typing import Dict, List, Optional
from tqdm import tqdm
import wandb

from rl.train_dqn_simple import SimpleDQNTrainer
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.rl_components import TradingAgent
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader


class SharedReplayBuffer:
    """
    Thread-safe replay buffer that can be shared between processes.

    Uses multiprocessing Manager to share data between worker processes.
    """

    def __init__(self, capacity: int, manager):
        self.capacity = capacity
        self.buffer = manager.list()
        self.lock = manager.Lock()

    def add(self, transition: Dict):
        """Add transition to buffer."""
        with self.lock:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)  # Remove oldest
            self.buffer.append(transition)

    def add_batch(self, transitions: List[Dict]):
        """Add multiple transitions at once (more efficient)."""
        with self.lock:
            for transition in transitions:
                if len(self.buffer) >= self.capacity:
                    self.buffer.pop(0)
                self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return []
            indices = random.sample(range(len(self.buffer)), batch_size)
            return [self.buffer[i] for i in indices]

    def __len__(self):
        with self.lock:
            return len(self.buffer)


def experience_worker(
    worker_id: int,
    shared_buffer,
    q_network_state_dict_queue,
    stop_event,
    config: Dict,
    train_days: List[str],
    device: str = 'cpu'  # Workers use CPU
):
    """
    Experience collection worker process.

    Runs environment episodes and adds transitions to shared buffer.
    Periodically syncs Q-network weights from main training process.

    Args:
        worker_id: Unique worker ID
        shared_buffer: Shared replay buffer
        q_network_state_dict_queue: Queue for receiving Q-network updates
        stop_event: Signal to stop worker
        config: Training configuration
        train_days: List of training dates
        device: Device for this worker (CPU)
    """

    # Only main worker prints (worker 0)
    is_main_worker = (worker_id == 0)

    if is_main_worker:
        print(f"Worker {worker_id}: Starting (main reporting worker) on device {device}")

    # Suppress prints from non-main workers
    def worker_print(*args, **kwargs):
        if is_main_worker:
            print(*args, **kwargs)

    # Set random seed for this worker
    random.seed(config['seed'] + worker_id)
    np.random.seed(config['seed'] + worker_id)
    torch.manual_seed(config['seed'] + worker_id)

    # Load data (each worker has its own copy)
    data_loader = DatasetLoader(
        dataset_path=config['dataset_path'],
        prices_path=config['prices_path'],
        num_test_stocks=100
    )

    # Load feature cache
    if os.path.exists(config['feature_cache']):
        data_loader.load_feature_cache(config['feature_cache'])

    # Create dummy agent for feature extraction
    dummy_agent = TradingAgent(
        predictor_checkpoint_path=config['predictor_checkpoint'],
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(device)
    dummy_agent.train()

    # Load stock cache first to filter dates
    import h5py
    with h5py.File(config['prices_path'], 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path=config['cache_file'],
        prices_file_keys=prices_file_keys,
        device=device
    )

    # Get available dates from stock cache
    stock_cache_dates = set(stock_cache.selections.keys())
    worker_print(f"Worker {worker_id}: Stock cache has {len(stock_cache_dates)} dates")

    # Filter train_days to only include dates in stock cache
    filtered_train_days = [d for d in train_days if d in stock_cache_dates]
    if len(filtered_train_days) < len(train_days):
        worker_print(f"Worker {worker_id}: Filtered out {len(train_days) - len(filtered_train_days)} days not in stock cache")
    train_days = filtered_train_days

    # Create temporary environment for state cache (with filtered dates)
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        device=device,
        trading_days_filter=train_days,
        top_k_per_horizon=config['top_k_per_horizon']
    )

    # Load state cache
    if os.path.exists(config['state_cache']):
        temp_env.load_state_cache(config['state_cache'])
        worker_print(f"Worker {worker_id}: State cache has {len(temp_env.state_cache)} dates")

        # Also verify train_days are in state cache
        state_cache_dates = set(temp_env.state_cache.keys())
        filtered_train_days = [d for d in train_days if d in state_cache_dates]
        if len(filtered_train_days) < len(train_days):
            worker_print(f"Worker {worker_id}: Filtered out {len(train_days) - len(filtered_train_days)} days not in state cache")
        train_days = filtered_train_days

    worker_print(f"Worker {worker_id}: Using {len(train_days)} valid training days")

    if len(train_days) == 0:
        print(f"Worker {worker_id}: ERROR - No valid training days after filtering!")  # Always print errors
        return

    # Create vectorized environment with fully filtered train_days
    vec_env = GPUVectorizedTradingEnv(
        num_envs=config['worker_num_envs'],
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        transaction_cost=0.0,
        device=device,
        trading_days_filter=train_days,  # Use fully filtered list
        top_k_per_horizon=config['top_k_per_horizon']
    )
    vec_env.ref_env.state_cache = temp_env.state_cache
    vec_env.ref_env.price_cache = temp_env.price_cache

    # Create local DQN for action selection (will be synced from main process)
    from rl.rl_components import StockDQN
    from rl.reduced_action_space import create_global_state
    from rl.state_creation_optimized import create_next_states_batch_optimized

    state_dim = config['state_dim']
    action_dim = 9 if config['allow_short'] else 5

    local_q_network = StockDQN(
        state_dim=state_dim,
        hidden_dim=config['hidden_dim'],
        action_dim=action_dim
    ).to(device)
    local_q_network.eval()  # Workers only do inference

    # Worker stats
    episodes_collected = 0
    transitions_collected = 0
    last_sync_time = time.time()
    last_report_time = time.time()

    # EMA tracking for portfolio value (large alpha for smooth trend)
    ema_portfolio_value = config['initial_capital']
    ema_pv_alpha = 0.999

    worker_print(f"Worker {worker_id}: Ready to collect experience")

    # Get initial epsilon from config
    epsilon = config['epsilon_start']

    try:
        while not stop_event.is_set():
            # Periodically sync Q-network from main process
            if time.time() - last_sync_time > 5.0:  # Sync every 5 seconds
                try:
                    new_state_dict = q_network_state_dict_queue.get_nowait()
                    local_q_network.load_state_dict(new_state_dict)
                    last_sync_time = time.time()
                except Empty:
                    pass

            # Run one episode
            vec_env.reset()

            # Get initial stock selections
            top_4_stocks_list = []
            bottom_4_stocks_list = []
            for i in range(config['worker_num_envs']):
                current_date = vec_env.episode_dates[i][0]
                num_samples = stock_cache.get_num_samples(current_date)
                sample_idx = random.randint(0, num_samples - 1)
                top_4, bottom_4 = stock_cache.get_sample(current_date, sample_idx)
                top_4_stocks_list.append(top_4)
                bottom_4_stocks_list.append(bottom_4)

            # Create initial states
            from rl.state_creation_optimized import create_next_states_batch_optimized
            states_list = []
            for i in range(config['worker_num_envs']):
                current_date = vec_env.episode_dates[i][0]
                if len(top_4_stocks_list[i]) > 0:
                    states_dict = create_next_states_batch_optimized(
                        next_env=vec_env.ref_env,
                        next_top_4_stocks=top_4_stocks_list[i],
                        next_bottom_4_stocks=bottom_4_stocks_list[i],
                        next_date=current_date,
                        device=device
                    )
                    states_list.append(states_dict)
                else:
                    states_list.append({})

            # Create initial global states
            global_states_list = []
            positions_list = [None] * config['worker_num_envs']
            is_short_list = [False] * config['worker_num_envs']

            for i in range(config['worker_num_envs']):
                if len(states_list[i]) > 0:
                    global_state = create_global_state(
                        top_4_stocks_list[i],
                        bottom_4_stocks_list[i],
                        states_list[i],
                        positions_list[i],
                        is_short=is_short_list[i],
                        device=device,
                        allow_short=config['allow_short']
                    )
                    global_states_list.append(global_state)
                else:
                    global_states_list.append(None)

            # Collect episode transitions
            episode_transitions = []

            for step in range(30):  # episode_length
                # Select actions using local Q-network
                actions_list = []
                trades_list = []

                for i in range(config['worker_num_envs']):
                    if not vec_env.dones[i] and global_states_list[i] is not None:
                        # ε-greedy action selection
                        if random.random() < epsilon:
                            action = random.randint(0, action_dim - 1)
                        else:
                            with torch.no_grad():
                                q_values = local_q_network(global_states_list[i].unsqueeze(0))
                                action = q_values.argmax(dim=1).item()

                        # Convert to trade
                        from rl.train_dqn_simple import SimpleDQNTrainer
                        trainer_temp = SimpleDQNTrainer(config)
                        trade = trainer_temp.action_to_trade(action, top_4_stocks_list[i], bottom_4_stocks_list[i])
                    else:
                        action = 0
                        trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}

                    actions_list.append(action)
                    trades_list.append([trade])

                # Step environment
                next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
                    actions_list, trades_list, positions_list
                )

                # Create next global states
                next_global_states_list = []

                # Update stock selections
                for i in range(config['worker_num_envs']):
                    if not vec_env.dones[i]:
                        current_step = vec_env.step_indices[i].item()
                        if current_step < len(vec_env.episode_dates[i]):
                            current_date = vec_env.episode_dates[i][current_step]
                            num_samples = stock_cache.get_num_samples(current_date)
                            sample_idx = random.randint(0, num_samples - 1)
                            top_4, bottom_4 = stock_cache.get_sample(current_date, sample_idx)
                            top_4_stocks_list[i] = top_4
                            bottom_4_stocks_list[i] = bottom_4

                # Create next states
                for i in range(config['worker_num_envs']):
                    if not vec_env.dones[i] and len(top_4_stocks_list[i]) > 0:
                        current_step = vec_env.step_indices[i].item()
                        if current_step < len(vec_env.episode_dates[i]):
                            current_date = vec_env.episode_dates[i][current_step]
                            next_states_dict = create_next_states_batch_optimized(
                                next_env=vec_env.ref_env,
                                next_top_4_stocks=top_4_stocks_list[i],
                                next_bottom_4_stocks=bottom_4_stocks_list[i],
                                next_date=current_date,
                                device=device
                            )
                            next_global_state = create_global_state(
                                top_4_stocks_list[i],
                                bottom_4_stocks_list[i],
                                next_states_dict,
                                next_positions_list[i],
                                is_short=is_short_list[i],
                                device=device,
                                allow_short=config['allow_short']
                            )
                            next_global_states_list.append(next_global_state)
                        else:
                            next_global_states_list.append(None)
                    else:
                        next_global_states_list.append(None)

                # Store transitions
                for i in range(config['worker_num_envs']):
                    if global_states_list[i] is not None:
                        transition = {
                            'state': global_states_list[i].cpu().numpy(),
                            'action': actions_list[i],
                            'reward': rewards_list[i],
                            'next_state': next_global_states_list[i].cpu().numpy() if next_global_states_list[i] is not None else np.zeros_like(global_states_list[i].cpu().numpy()),
                            'done': dones_list[i]
                        }
                        episode_transitions.append(transition)

                # Update for next step
                global_states_list = next_global_states_list
                positions_list = next_positions_list

                # Update is_short flags
                for i in range(config['worker_num_envs']):
                    trade = trades_list[i][0]
                    if trade['position_type'] == 'SHORT':
                        is_short_list[i] = True
                    elif trade['position_type'] == 'LONG':
                        is_short_list[i] = False
                    elif trade['action'] == 'HOLD' and next_positions_list[i] is None:
                        is_short_list[i] = False

                if all(vec_env.dones):
                    break

            # Track final portfolio values and update EMA
            for i in range(config['worker_num_envs']):
                if len(infos_list) > i and 'portfolio_value' in infos_list[i]:
                    pv = infos_list[i]['portfolio_value']
                    # Update EMA: ema_new = alpha * ema_old + (1 - alpha) * new_value
                    ema_portfolio_value = ema_pv_alpha * ema_portfolio_value + (1 - ema_pv_alpha) * pv

            # Add transitions to shared buffer (batch add is more efficient)
            if len(episode_transitions) > 0:
                shared_buffer.add_batch(episode_transitions)
                transitions_collected += len(episode_transitions)
                episodes_collected += config['worker_num_envs']

            # Periodic reporting (every 30 seconds for main worker only)
            if is_main_worker and time.time() - last_report_time > 30.0:
                buffer_size = len(shared_buffer)
                pv_gain_pct = (ema_portfolio_value - config['initial_capital']) / config['initial_capital'] * 100
                worker_print(f"[Worker {worker_id}] Episodes: {episodes_collected:,} | Transitions: {transitions_collected:,} | Buffer: {buffer_size:,} | PV_ema: ${ema_portfolio_value:,.0f} ({pv_gain_pct:+.2f}%) | ε: {epsilon:.3f}")
                last_report_time = time.time()

            # Decay epsilon locally (approximate - will get better sync from main process later)
            epsilon = max(
                config['epsilon_end'],
                epsilon - (config['epsilon_start'] - config['epsilon_end']) / config['epsilon_decay_episodes']
            )

    except KeyboardInterrupt:
        print(f"Worker {worker_id}: Interrupted by user")  # Always print interrupts
    except Exception as e:
        print(f"Worker {worker_id}: ERROR - {e}")  # Always print errors
        import traceback
        traceback.print_exc()
    finally:
        worker_print(f"Worker {worker_id}: Shutting down. Collected {episodes_collected} episodes, {transitions_collected} transitions")


def async_training_loop(config: Dict):
    """
    Main asynchronous training loop.

    Spawns multiple worker processes for experience collection while
    main process continuously trains on GPU.
    """

    print(f"\n{'='*80}")
    print("ASYNCHRONOUS DQN TRAINING")
    print(f"{'='*80}")
    print(f"GPU Trainer: {config['device']}")
    print(f"Experience Workers: {config['num_workers']} workers × {config['worker_num_envs']} envs = {config['num_workers'] * config['worker_num_envs']} parallel envs")
    print(f"Replay Buffer: {config['buffer_capacity']:,} transitions")
    print(f"Training: Batch size {config['batch_size']}, {config['train_iterations_per_step']} updates per collection cycle")
    print(f"{'='*80}\n")

    # Create shared replay buffer using multiprocessing Manager
    mp_manager = mp.Manager()
    shared_buffer = SharedReplayBuffer(config['buffer_capacity'], mp_manager)

    # Queue for sending Q-network updates to workers
    q_network_queue = mp.Manager().Queue(maxsize=config['num_workers'])

    # Event to signal workers to stop
    stop_event = mp.Event()

    # Load train/val dates
    print("Loading dataset and caches...")
    data_loader = DatasetLoader(
        dataset_path=config['dataset_path'],
        prices_path=config['prices_path'],
        num_test_stocks=100
    )

    # Load feature cache
    if os.path.exists(config['feature_cache']):
        print(f"  Loading feature cache: {config['feature_cache']}")
        data_loader.load_feature_cache(config['feature_cache'])
    else:
        raise FileNotFoundError(f"Feature cache not found: {config['feature_cache']}")

    # Create dummy agent for loading state cache
    dummy_agent = TradingAgent(
        predictor_checkpoint_path=config['predictor_checkpoint'],
        state_dim=1469,
        hidden_dim=512,
        action_dim=5
    ).to(config['device'])
    dummy_agent.train()

    # Get all trading days
    sample_ticker = list(data_loader.prices_file.keys())[0]
    prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
    all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])

    # Define train/val/test split
    train_start = '2020-01-01'
    train_end = '2022-12-31'
    val_start = '2023-01-01'
    val_end = '2023-06-30'

    train_days = [d for d in all_trading_days if train_start <= d <= train_end]
    val_days = [d for d in all_trading_days if val_start <= d <= val_end]

    print(f"  Initial train days: {len(train_days)} ({train_start} to {train_end})")
    print(f"  Initial val days: {len(val_days)} ({val_start} to {val_end})")

    # Load state cache to filter dates
    print(f"  Loading state cache: {config['state_cache']}")
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        device=config['device'],
        trading_days_filter=all_trading_days,
        top_k_per_horizon=config['top_k_per_horizon']
    )

    if os.path.exists(config['state_cache']):
        temp_env.load_state_cache(config['state_cache'])
        cached_dates = sorted(temp_env.state_cache.keys())
        print(f"  ✅ State cache loaded: {len(cached_dates)} dates")
    else:
        raise FileNotFoundError(f"State cache not found: {config['state_cache']}")

    # Load stock selection cache to filter dates
    print(f"  Loading stock selection cache: {config['cache_file']}")
    import h5py
    with h5py.File(config['prices_path'], 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path=config['cache_file'],
        prices_file_keys=prices_file_keys,
        device=config['device']
    )
    stock_cache_dates = set(stock_cache.selections.keys())
    print(f"  ✅ Stock cache loaded: {len(stock_cache_dates)} dates")

    # Filter train/val days to only dates that exist in BOTH caches
    cached_dates_set = set(cached_dates)
    valid_dates = cached_dates_set.intersection(stock_cache_dates)

    train_days_original = len(train_days)
    val_days_original = len(val_days)
    train_days = [d for d in train_days if d in valid_dates]
    val_days = [d for d in val_days if d in valid_dates]

    if len(train_days) < train_days_original:
        print(f"  ⚠️  Filtered {train_days_original - len(train_days)} training days not in both caches")
    if len(val_days) < val_days_original:
        print(f"  ⚠️  Filtered {val_days_original - len(val_days)} validation days not in both caches")

    print(f"  ✅ Final train days: {len(train_days)}")
    print(f"  ✅ Final val days: {len(val_days)}")

    if len(train_days) == 0:
        raise ValueError("No valid training days after filtering! Check your cache date ranges.")

    print(f"\nTraining on {len(train_days)} days: {min(train_days)} to {max(train_days)}")
    print(f"Date range: {train_days[0]} to {train_days[-1]}\n")

    # Verify dates are valid before starting workers
    print("Verifying all dates are in both caches...")
    for date in train_days[:5]:  # Check first 5 as sample
        if date not in stock_cache_dates:
            print(f"  ❌ ERROR: {date} not in stock cache!")
        if date not in cached_dates_set:
            print(f"  ❌ ERROR: {date} not in state cache!")
    print("  ✅ Sample dates verified\n")

    # Start worker processes
    print("Starting worker processes...")
    workers = []
    for worker_id in range(config['num_workers']):
        worker = mp.Process(
            target=experience_worker,
            args=(worker_id, shared_buffer, q_network_queue, stop_event, config, train_days, 'cpu')
        )
        worker.start()
        workers.append(worker)
        time.sleep(0.2)  # Stagger worker startup

    print(f"✅ Started {len(workers)} worker processes")
    print(f"   - Worker 0: Main reporting worker (will print updates)")
    print(f"   - Workers 1-{len(workers)-1}: Silent (only print errors)\n")

    # Create DQN trainer (GPU)
    trainer = SimpleDQNTrainer(config)

    # Replace trainer's buffer with our shared buffer
    # We'll sample from shared buffer instead

    # Initialize WandB
    if config.get('use_wandb', False):
        wandb.init(
            project='stock-trading-async-dqn',
            config=config,
            name=f'async_{config["num_workers"]}workers_{config["batch_size"]}bs'
        )

    # Training loop
    print("Waiting for replay buffer to fill...\n")
    iteration = 0
    last_log_time = time.time()
    last_checkpoint_time = time.time()

    try:
        # Use tqdm with custom format to keep it clean
        pbar = tqdm(desc="GPU Training", unit=" iter", dynamic_ncols=True, leave=True, position=0)
        while iteration < config['max_iterations']:
            # Wait for buffer to have enough samples
            buffer_size = len(shared_buffer)
            if buffer_size < config['min_buffer_size']:
                time.sleep(1.0)
                continue

            # Train for multiple iterations
            for _ in range(config['train_iterations_per_step']):
                # Sample from shared buffer
                batch = shared_buffer.sample(config['batch_size'])
                if len(batch) == 0:
                    break

                # Train
                loss, info = trainer.compute_td_loss(batch)

                # Backprop
                trainer.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainer.q_network.parameters(), max_norm=10.0)
                trainer.optimizer.step()

                # Update target network
                if trainer.use_soft_update:
                    trainer.update_target_network()
                else:
                    if trainer.global_step % config.get('target_update_freq', 1000) == 0:
                        trainer.update_target_network()

                trainer.global_step += 1

                # Update EMAs
                if trainer.global_step == 1:
                    trainer.ema_loss = info['loss']
                    trainer.ema_q_value = info['avg_q_value']
                    trainer.ema_grad_norm = grad_norm.item()
                else:
                    trainer.ema_loss = 0.01 * info['loss'] + 0.99 * trainer.ema_loss
                    trainer.ema_q_value = 0.01 * info['avg_q_value'] + 0.99 * trainer.ema_q_value
                    trainer.ema_grad_norm = 0.01 * grad_norm.item() + 0.99 * trainer.ema_grad_norm

            # Periodically sync Q-network to workers
            if iteration % 10 == 0:
                state_dict = trainer.q_network.state_dict()
                # Send to all workers (non-blocking)
                for _ in range(config['num_workers']):
                    try:
                        q_network_queue.put_nowait(state_dict)
                    except:
                        pass  # Queue full, skip

            # Update LR scheduler
            trainer.scheduler.step()

            # Logging
            if time.time() - last_log_time > 2.0:
                buffer_size = len(shared_buffer)
                lr = trainer.optimizer.param_groups[0]['lr']

                pbar.set_postfix_str(
                    f"Step={trainer.global_step:,} | "
                    f"Buf={buffer_size:,} | "
                    f"L={trainer.ema_loss:.3f} | "
                    f"Q={trainer.ema_q_value:+.2f} | "
                    f"G={trainer.ema_grad_norm:.2f} | "
                    f"LR={lr:.1e}"
                )
                pbar.update(1)

                if config.get('use_wandb', False):
                    wandb.log({
                        'global_step': trainer.global_step,
                        'buffer_size': buffer_size,
                        'loss_ema': trainer.ema_loss,
                        'q_value_ema': trainer.ema_q_value,
                        'grad_norm_ema': trainer.ema_grad_norm,
                        'learning_rate': lr
                    })

                last_log_time = time.time()

            iteration += 1

            # Save checkpoint periodically (time-based to avoid spamming)
            if time.time() - last_checkpoint_time > 600:  # Every 10 minutes
                checkpoint_path = f"checkpoints/async_dqn_step{trainer.global_step}.pt"
                os.makedirs("checkpoints", exist_ok=True)

                # Suppress the print from save_checkpoint
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                trainer.save_checkpoint(checkpoint_path)
                sys.stdout = old_stdout

                pbar.write(f"✅ Checkpoint saved: {checkpoint_path}")
                last_checkpoint_time = time.time()

    except KeyboardInterrupt:
        pbar.write("\n\nTraining interrupted by user")
    finally:
        # Stop workers
        pbar.write("\nStopping worker processes...")
        stop_event.set()
        for worker in workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

        # Save final checkpoint
        final_path = f"checkpoints/async_dqn_final_step{trainer.global_step}.pt"
        os.makedirs("checkpoints", exist_ok=True)

        # Suppress the print from save_checkpoint
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        trainer.save_checkpoint(final_path)
        sys.stdout = old_stdout

        pbar.write(f"✅ Final checkpoint saved: {final_path}")

        if config.get('use_wandb', False):
            wandb.finish()

        pbar.close()

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total steps: {trainer.global_step:,}")
    print(f"Final buffer size: {len(shared_buffer):,}")


if __name__ == '__main__':
    import argparse

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Async DQN training with parallel experience collection')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of experience collection workers')
    parser.add_argument('--worker-envs', type=int, default=8, help='Number of parallel envs per worker')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--train-iters', type=int, default=16, help='Training iterations per collection cycle')
    parser.add_argument('--max-iters', type=int, default=100000, help='Maximum training iterations')
    parser.add_argument('--buffer-size', type=int, default=200000, help='Replay buffer capacity')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:1', help='Training device')
    parser.add_argument('--wandb', action='store_true', default=True, help='Enable WandB')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Configuration
    allow_short = False
    state_dim = 11761 if allow_short else 5881

    config = {
        # Async training
        'num_workers': args.num_workers,
        'worker_num_envs': args.worker_envs,
        'train_iterations_per_step': args.train_iters,
        'max_iterations': args.max_iters,

        # DQN
        'device': args.device,
        'state_dim': state_dim,
        'hidden_dim': 1024,
        'action_dim': 9 if allow_short else 5,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'buffer_capacity': args.buffer_size,
        'batch_size': args.batch_size,
        'gamma': 0.99,
        'n_step': 1,
        'use_per': False,
        'use_soft_update': True,
        'tau': 0.005,
        'target_update_freq': 1000,
        'min_buffer_size': 10000,
        'num_episodes': 100000,  # For epsilon decay
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_episodes': 20000,

        # Environment
        'initial_capital': 100000,
        'top_k_per_horizon': 50,
        'allow_short': allow_short,

        # Paths
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'cache_file': 'data/rl_stock_selections_4yr.h5',
        'feature_cache': 'data/rl_feature_cache_4yr.h5',
        'state_cache': 'data/rl_state_cache_4yr.h5',
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',

        # Logging
        'use_wandb': args.wandb,
        'save_freq': 500,
        'seed': args.seed
    }

    # Run async training
    async_training_loop(config)
