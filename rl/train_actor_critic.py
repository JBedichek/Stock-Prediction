#!/usr/bin/env python3
"""
Online Actor-Critic Training (Phase 3).

Trains both actor and critic online using the environment.
Loads pre-trained critic from Phase 2 to jumpstart learning.

Actor-Critic algorithm:
- Actor: Learns policy π(a|s) to maximize expected return
- Critic: Learns Q(s,a) to estimate value of actions
- Training: Actor uses policy gradient, Critic uses TD learning
"""

import torch
import torch.optim as optim
import numpy as np
import sys
import os
import pickle
import pandas as pd
import wandb
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import ActorCriticAgent, ReplayBuffer, compute_critic_loss, compute_actor_loss
from rl.rl_environment import TradingEnvironment
from rl.vectorized_env import VectorizedTradingEnv
from rl.reduced_action_space import get_top_4_stocks, create_global_state, decode_action_to_trades
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients
torch.set_grad_enabled(True)


class ActorCriticTrainer:
    """
    Online actor-critic trainer.

    Manages training loop, optimizers, logging, and checkpointing.
    """

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get('device', 'cuda')

        # Initialize data loader
        print("\n1. Loading data...")
        self.data_loader = DatasetLoader(
            dataset_path=config.get('dataset_path', 'data/all_complete_dataset.h5'),
            prices_path=config.get('prices_path', 'data/actual_prices.h5'),
            num_test_stocks=config.get('num_test_stocks', 100)
        )
        print(f"   ✅ Loaded {len(self.data_loader.test_tickers)} stocks")

        # Initialize agent
        print("\n2. Initializing agent...")
        self.agent = ActorCriticAgent(
            predictor_checkpoint_path=config.get('predictor_checkpoint', './checkpoints/best_model_100m_1.18.pt'),
            state_dim=config.get('state_dim', 1469),
            hidden_dim=config.get('hidden_dim', 1024),
            action_dim=config.get('action_dim', 3)
        ).to(self.device)

        # Load pre-trained critic if available (only for critic1, critic2 starts random)
        critic_checkpoint_path = config.get('pretrained_critic_path', './checkpoints/pretrained_critic.pt')
        if os.path.exists(critic_checkpoint_path):
            print(f"\n   📂 Loading pre-trained critic from {critic_checkpoint_path}...")
            checkpoint = torch.load(critic_checkpoint_path, map_location=self.device, weights_only=False)
            self.agent.critic1.load_state_dict(checkpoint['critic_state_dict'])
            self.agent.target_critic1.load_state_dict(checkpoint['target_critic_state_dict'])
            print(f"   ✅ Pre-trained critic1 loaded (trained for {checkpoint['epoch']+1} epochs)")
            print(f"   ⚠️  Critic2 initialized randomly (twin critics)")
        else:
            print(f"   ⚠️  No pre-trained critic found at {critic_checkpoint_path}")
            print(f"      Starting with random critic weights (both critics)")

        # Freeze predictor
        self.agent.feature_extractor.freeze_predictor()

        # Optimizers
        self.actor_optimizer = optim.AdamW(
            self.agent.actor.parameters(),
            lr=config.get('actor_lr', 1e-4),
            weight_decay=1e-4
        )

        # Twin critic optimizers (one for each critic network)
        self.critic1_optimizer = optim.AdamW(
            self.agent.critic1.parameters(),
            lr=config.get('critic_lr', 3e-4),
            weight_decay=1e-4
        )

        self.critic2_optimizer = optim.AdamW(
            self.agent.critic2.parameters(),
            lr=config.get('critic_lr', 3e-4),
            weight_decay=1e-4
        )

        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=config.get('num_episodes', 1000),
            eta_min=config.get('actor_lr', 1e-4) * 0.1
        )

        self.critic1_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic1_optimizer,
            T_max=config.get('num_episodes', 1000),
            eta_min=config.get('critic_lr', 3e-4) * 0.1
        )

        self.critic2_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic2_optimizer,
            T_max=config.get('num_episodes', 1000),
            eta_min=config.get('critic_lr', 3e-4) * 0.1
        )

        # Preload features
        print("\n3. Preloading features...")
        cache_path = config.get('feature_cache_path', 'data/rl_feature_cache_4yr.h5')

        if os.path.exists(cache_path):
            print(f"   ⚡ Loading from cache: {cache_path}")
            self.data_loader.load_feature_cache(cache_path)
        else:
            print("   ⚠️  Cache not found - preloading from HDF5...")
            cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            sample_ticker = list(self.data_loader.prices_file.keys())[0]
            prices_dates_bytes = self.data_loader.prices_file[sample_ticker]['dates'][:]
            all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
            self.recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]

            self.data_loader.preload_features(self.recent_trading_days)
            self.data_loader.save_feature_cache(cache_path)

        # Filter to last 4 years
        cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        sample_ticker = list(self.data_loader.prices_file.keys())[0]
        prices_dates_bytes = self.data_loader.prices_file[sample_ticker]['dates'][:]
        all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
        self.recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]

        print(f"   📊 Using {len(self.recent_trading_days)} trading days (last 4 years)")

        # Initialize environment (vectorized for parallel simulation)
        print("\n4. Initializing vectorized environment...")
        num_parallel = config.get('num_parallel_envs', 8)

        # Create a single env first to load/precompute state cache
        temp_env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config.get('initial_capital', 100000),
            max_positions=config.get('max_positions', 1),
            episode_length=config.get('episode_length', 50),
            device=self.device,
            trading_days_filter=self.recent_trading_days,
            top_k_per_horizon=config.get('top_k_per_horizon', 10)
        )

        # Precompute/load states into temp environment
        print("\n5. Precomputing states...")
        state_cache_path = config.get('state_cache_path', 'data/rl_state_cache_4yr.h5')

        if os.path.exists(state_cache_path):
            print(f"   ⚡ Loading state cache: {state_cache_path}")
            temp_env.load_state_cache(state_cache_path)
        else:
            print("   Computing states...")
            temp_env._precompute_all_states()
            temp_env.save_state_cache(state_cache_path)

        # Now create vectorized environment (shares state cache from temp_env)
        print(f"\n6. Creating vectorized environment ({num_parallel} parallel envs)...")
        self.vec_env = VectorizedTradingEnv(
            num_envs=num_parallel,
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config.get('initial_capital', 100000),
            max_positions=config.get('max_positions', 1),
            episode_length=config.get('episode_length', 30),
            device=self.device,
            trading_days_filter=self.recent_trading_days,
            top_k_per_horizon=config.get('top_k_per_horizon', 10)
        )

        # Share state cache from temp_env to all vectorized envs
        for env in self.vec_env.envs:
            env.state_cache = temp_env.state_cache
            env.price_cache = temp_env.price_cache

        print(f"   ✅ Vectorized environment ready ({num_parallel} parallel episodes)")

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=config.get('buffer_capacity', 100000))

        # Training state
        self.episode = 0
        self.global_step = 0

        # Reward normalization (running statistics)
        self.reward_scale = config.get('reward_scale', 100.0)  # Scale rewards by 100x
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Logging
        self.episode_history = []

        # EMA tracking (alpha = 0.99)
        self.ema_alpha = 0.99
        self.ema_critic_loss = None
        self.ema_actor_loss = None
        self.ema_return = None
        self.ema_portfolio_value = None

        # Resume from checkpoint if specified
        resume_checkpoint = config.get('resume_checkpoint', None)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"\n📂 Resuming training from checkpoint: {resume_checkpoint}")
            self._load_checkpoint(resume_checkpoint)
            print(f"   ✅ Resumed from episode {self.episode}, global step {self.global_step}")

            # Advance schedulers to match resumed episode count
            for _ in range(self.episode):
                self.actor_scheduler.step()
                self.critic1_scheduler.step()
                self.critic2_scheduler.step()
            print(f"   Adjusted learning rate schedulers to episode {self.episode}")
        elif resume_checkpoint:
            print(f"\n⚠️  Resume checkpoint not found: {resume_checkpoint}")
            print(f"   Starting training from scratch")

        # Cache top-4 stocks per date (same across all parallel envs)
        self.top_4_cache = {}

        # Initialize WandB
        use_wandb = config.get('use_wandb', True)
        if use_wandb:
            run_name = config.get('wandb_run_name', None)
            if run_name is None:
                # Auto-generate name, add "resume_" prefix if resuming
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                prefix = "resume_" if resume_checkpoint and os.path.exists(resume_checkpoint) else "run_"
                run_name = f"{prefix}{timestamp}"

            wandb.init(
                project=config.get('wandb_project', 'stock-rl-trading'),
                name=run_name,
                config=config
            )

        print("\n✅ Trainer initialized!")
        print(f"   Device: {self.device}")
        print(f"   Actor LR: {config.get('actor_lr', 1e-4)}")
        print(f"   Critic LR: {config.get('critic_lr', 3e-4)}")
        print(f"   Buffer capacity: {config.get('buffer_capacity', 100000)}")
        print(f"   WandB: {'Enabled' if use_wandb else 'Disabled'}")
        freeze_eps = config.get('freeze_critic_episodes', 0)
        if freeze_eps > 0:
            print(f"   🔒 Critic FROZEN for first {freeze_eps} episodes (preserve pretrained knowledge)")

    def train_episode_vectorized(self) -> List[Dict]:
        """
        Run one batch of parallel training episodes (vectorized).

        Returns:
            List of episode statistics (one per completed episode)
        """
        num_parallel = self.config.get('num_parallel_envs', 8)

        # Reset all environments
        states_list, positions_list = self.vec_env.reset()

        # Track statistics for all environments
        episode_returns = [0.0] * num_parallel
        episode_critic_losses = [[] for _ in range(num_parallel)]
        episode_actor_losses = [[] for _ in range(num_parallel)]
        episode_entropies = [[] for _ in range(num_parallel)]
        episode_advantages = [[] for _ in range(num_parallel)]
        episode_q_values = [[] for _ in range(num_parallel)]
        episode_grad_norms = [[] for _ in range(num_parallel)]
        episode_actions = [[] for _ in range(num_parallel)]
        episode_q1_values = [[] for _ in range(num_parallel)]
        episode_q2_values = [[] for _ in range(num_parallel)]
        episode_q_diffs = [[] for _ in range(num_parallel)]

        # Run episodes until all complete
        max_steps = self.config.get('episode_length', 30)

        for step in range(max_steps):
            # Check if all envs are done
            if self.vec_env.get_num_active() == 0:
                break

            # Compute top-4 stocks per unique date (cached - same across envs at same date)
            unique_dates = {}
            for i, env in enumerate(self.vec_env.envs):
                if not self.vec_env.dones[i]:
                    date = env.current_date
                    if date not in unique_dates:
                        unique_dates[date] = []
                    unique_dates[date].append(i)

            # Get top-4 stocks for each unique date (only compute once per date)
            date_to_top4 = {}
            for date in unique_dates:
                if date not in self.top_4_cache:
                    # First time seeing this date - compute and cache
                    env_idx = unique_dates[date][0]
                    cached_states = self.vec_env.envs[env_idx].state_cache[date]
                    self.top_4_cache[date] = get_top_4_stocks(cached_states)
                date_to_top4[date] = self.top_4_cache[date]

            # Assign top-4 stocks to each environment
            top_4_stocks_list = []
            for i, env in enumerate(self.vec_env.envs):
                if not self.vec_env.dones[i]:
                    top_4_stocks_list.append(date_to_top4[env.current_date])
                else:
                    top_4_stocks_list.append([])

            # Get states for each environment (OPTIMIZED: only create states for top-4 stocks)
            states_list_current = []
            for i, env in enumerate(self.vec_env.envs):
                if not self.vec_env.dones[i]:
                    # Get only the 4 stocks we need (not all filtered stocks)
                    top_4_tickers = [ticker for ticker, _ in top_4_stocks_list[i]]
                    date = env.current_date

                    # Get cached states for this date
                    cached_states = env.state_cache[date]
                    cached_prices = env.price_cache.get(date, {})

                    # Create states only for top-4 stocks (skip filtering, we know what we need)
                    states = {}
                    portfolio_value = env._portfolio_value_cached(cached_prices)

                    for ticker in top_4_tickers:
                        if ticker in cached_states and ticker in cached_prices:
                            # Get cached state (predictor features + placeholder context)
                            cached_state = cached_states[ticker]
                            price = cached_prices[ticker]

                            # Create portfolio context (25 dims)
                            portfolio_context = env._create_portfolio_context_fast(
                                ticker, price, portfolio_value
                            )

                            # Concatenate: predictor features (1444) + portfolio context (25) = 1469
                            state = torch.cat([
                                cached_state[:1444].to(self.device),
                                portfolio_context.to(self.device)
                            ])
                            states[ticker] = state

                    states_list_current.append(states)
                else:
                    states_list_current.append({})

            # Batch action selection (SINGLE GPU forward pass for all envs)
            epsilon = self.config.get('epsilon', 0.1)
            results = self.agent.select_actions_reduced_batch(
                top_4_stocks_list=top_4_stocks_list,
                states_list=states_list_current,
                positions_list=positions_list,
                epsilon=epsilon,
                deterministic=False
            )

            # Extract actions and trades
            actions_list = [r[0] for r in results]
            trades_list = [r[3] for r in results]

            # Track actions
            for i, action in enumerate(actions_list):
                if not self.vec_env.dones[i]:
                    episode_actions[i].append(action)

            # Step all environments
            next_states_list, rewards_list, dones_list, infos_list, next_positions_list = self.vec_env.step(
                actions_list, trades_list, positions_list
            )

            # Store transitions and update statistics
            for i in range(num_parallel):
                if not self.vec_env.dones[i] or dones_list[i]:  # Just finished or still active
                    # Create global states for storage
                    if len(top_4_stocks_list[i]) > 0 and len(states_list_current[i]) > 0:
                        try:
                            global_state = create_global_state(
                                top_4_stocks_list[i],
                                states_list_current[i],
                                positions_list[i],
                                device=self.device
                            )

                            # Create next global state (OPTIMIZED: use cached top-4)
                            next_env = self.vec_env.envs[i]
                            next_date = next_env.current_date

                            # Get cached top-4 for next date
                            if next_date not in self.top_4_cache:
                                next_cached_states = next_env.state_cache[next_date]
                                self.top_4_cache[next_date] = get_top_4_stocks(next_cached_states)
                            next_top_4_stocks = self.top_4_cache[next_date]

                            # Create states for next top-4 stocks only
                            next_top_4_tickers = [ticker for ticker, _ in next_top_4_stocks]
                            next_cached_states = next_env.state_cache[next_date]
                            next_cached_prices = next_env.price_cache.get(next_date, {})
                            next_portfolio_value = next_env._portfolio_value_cached(next_cached_prices)

                            next_states_dict = {}
                            for ticker in next_top_4_tickers:
                                if ticker in next_cached_states and ticker in next_cached_prices:
                                    cached_state = next_cached_states[ticker]
                                    price = next_cached_prices[ticker]
                                    portfolio_context = next_env._create_portfolio_context_fast(
                                        ticker, price, next_portfolio_value
                                    )
                                    state = torch.cat([
                                        cached_state[:1444].to(self.device),
                                        portfolio_context.to(self.device)
                                    ])
                                    next_states_dict[ticker] = state

                            next_global_state = create_global_state(
                                next_top_4_stocks,
                                next_states_dict,
                                next_positions_list[i],
                                device=self.device
                            )

                            # Scale reward (raw rewards are ~0.001-0.02, scale to ~0.1-2.0)
                            scaled_reward = rewards_list[i] * self.reward_scale

                            # Store transition
                            self.buffer.push(
                                state=global_state,
                                action=actions_list[i],
                                reward=scaled_reward,
                                next_state=next_global_state,
                                done=dones_list[i],
                                ticker=next_positions_list[i] if next_positions_list[i] else 'CASH',
                                portfolio_value=infos_list[i]['portfolio_value']
                            )

                            episode_returns[i] += rewards_list[i]  # Track unscaled for logging
                        except (KeyError, IndexError, ValueError):
                            pass  # Skip if state construction fails

            # Update positions
            positions_list = next_positions_list

            self.global_step += 1

        # ============================================================================
        # TRAINING PHASE: Do gradient updates AFTER all experience is collected
        # ============================================================================
        # This prevents Q-value overestimation by ensuring we collect data before
        # updating the networks. Standard practice in modern RL (SAC, TD3).
        # ============================================================================

        if len(self.buffer) >= self.config.get('batch_size', 256):
            # Calculate number of updates based on data collected
            # Use 1 update per N transitions (configurable ratio)
            transitions_collected = num_parallel * max_steps
            updates_per_transition = self.config.get('updates_per_transition', 1.0)
            num_critic_updates = int(transitions_collected * updates_per_transition)
            num_actor_updates = num_critic_updates // self.config.get('actor_update_freq', 2)

            # Perform gradient updates
            freeze_critic_episodes = self.config.get('freeze_critic_episodes', 0)

            for update_idx in range(num_critic_updates):
                # Train critic
                if self.episode >= freeze_critic_episodes:
                    critic_loss, critic_info = self._train_critic()
                    # Add to all episodes (average across parallel envs)
                    for i in range(num_parallel):
                        episode_critic_losses[i].append(critic_loss)
                        episode_q1_values[i].append(critic_info['q1_mean'])
                        episode_q2_values[i].append(critic_info['q2_mean'])
                        episode_q_diffs[i].append(critic_info['q_diff'])
                else:
                    # Critic frozen - just compute loss for logging
                    batch = self.buffer.sample(self.config.get('batch_size', 256))
                    with torch.no_grad():
                        loss1, loss2 = compute_critic_loss(
                            agent=self.agent,
                            batch=batch,
                            gamma=self.config.get('gamma', 0.99),
                            device=self.device
                        )
                        critic_loss = (loss1.item() + loss2.item()) / 2.0
                    for i in range(num_parallel):
                        episode_critic_losses[i].append(critic_loss)

                # Train actor (less frequently)
                if update_idx % self.config.get('actor_update_freq', 2) == 0:
                    actor_loss, actor_info = self._train_actor()
                    for i in range(num_parallel):
                        episode_actor_losses[i].append(actor_loss)
                        episode_entropies[i].append(actor_info['entropy'])
                        episode_advantages[i].append(actor_info['avg_advantage'])
                        episode_q_values[i].append(actor_info['avg_q_value'])
                        episode_grad_norms[i].append(actor_info['grad_norm'])

                # Update both target critics
                self.agent.update_target_critics(tau=self.config.get('tau', 0.005))

        # Collect statistics for all completed episodes
        all_stats = []
        for i in range(num_parallel):
            avg_critic_loss = np.mean(episode_critic_losses[i]) if episode_critic_losses[i] else 0.0
            avg_actor_loss = np.mean(episode_actor_losses[i]) if episode_actor_losses[i] else 0.0

            # Compute action distribution (0-4)
            action_counts = {j: 0 for j in range(5)}
            for action in episode_actions[i]:
                action_counts[action] = action_counts.get(action, 0) + 1
            total_actions = len(episode_actions[i]) if episode_actions[i] else 1
            action_dist = {k: v/total_actions for k, v in action_counts.items()}

            # Update EMA values (use first env's stats)
            if i == 0:
                portfolio_val = infos_list[i]['portfolio_value'] if infos_list else 100000
                if self.ema_critic_loss is None:
                    self.ema_critic_loss = avg_critic_loss
                    self.ema_actor_loss = avg_actor_loss
                    self.ema_return = episode_returns[i]
                    self.ema_portfolio_value = portfolio_val
                else:
                    self.ema_critic_loss = self.ema_alpha * self.ema_critic_loss + (1 - self.ema_alpha) * avg_critic_loss
                    self.ema_actor_loss = self.ema_alpha * self.ema_actor_loss + (1 - self.ema_alpha) * avg_actor_loss
                    self.ema_return = self.ema_alpha * self.ema_return + (1 - self.ema_alpha) * episode_returns[i]
                    self.ema_portfolio_value = self.ema_alpha * self.ema_portfolio_value + (1 - self.ema_alpha) * portfolio_val

            stats = {
                'episode': self.episode + i,
                'return': episode_returns[i],
                'portfolio_value': infos_list[i]['portfolio_value'] if infos_list else 100000,
                'num_positions': infos_list[i]['num_positions'] if infos_list else 0,
                'avg_critic_loss': avg_critic_loss,
                'avg_actor_loss': avg_actor_loss,
                'avg_entropy': np.mean(episode_entropies[i]) if episode_entropies[i] else 0.0,
                'avg_advantage': np.mean(episode_advantages[i]) if episode_advantages[i] else 0.0,
                'avg_q_value': np.mean(episode_q_values[i]) if episode_q_values[i] else 0.0,
                'avg_q1_value': np.mean(episode_q1_values[i]) if episode_q1_values[i] else 0.0,
                'avg_q2_value': np.mean(episode_q2_values[i]) if episode_q2_values[i] else 0.0,
                'avg_q_diff': np.mean(episode_q_diffs[i]) if episode_q_diffs[i] else 0.0,
                'avg_grad_norm': np.mean(episode_grad_norms[i]) if episode_grad_norms[i] else 0.0,
                'buffer_size': len(self.buffer),
                'ema_critic_loss': self.ema_critic_loss,
                'ema_actor_loss': self.ema_actor_loss,
                'ema_return': self.ema_return,
                'ema_portfolio_value': self.ema_portfolio_value,
                'pct_action_0': action_dist.get(0, 0),
                'pct_action_1': action_dist.get(1, 0),
                'pct_action_2': action_dist.get(2, 0),
                'pct_action_3': action_dist.get(3, 0),
                'pct_action_4': action_dist.get(4, 0)
            }
            all_stats.append(stats)

        self.episode += num_parallel
        return all_stats

    def train_episode_old(self) -> Dict:
        """
        Run one training episode using reduced action space.

        Returns:
            Dictionary of episode statistics
        """
        # Reset environment
        _ = self.env.reset()

        # Track current position across episode
        current_position = None  # Start in cash

        episode_return = 0.0
        episode_critic_loss = []
        episode_actor_loss = []
        episode_entropy = []
        episode_advantages = []
        episode_q_values = []
        episode_grad_norms = []
        episode_actions = []  # Track action distribution (0=HOLD, 1-4=STOCKS)

        for step in range(self.config.get('episode_length', 20)):
            # Get current cached states from environment
            cached_states = self.env.state_cache[self.env.current_date]

            # Get top-4 stocks (one per time horizon)
            top_4_stocks = get_top_4_stocks(cached_states)

            # Get individual stock states from environment (for state construction)
            states = self.env._get_states()

            # Create global state (4 stocks × 1469 + 5 position encoding = 5881 dims)
            global_state = create_global_state(
                top_4_stocks=top_4_stocks,
                states=states,
                current_position=current_position,
                device=self.device
            )

            # Select single discrete action (0-4) using epsilon-greedy
            epsilon = self.config.get('epsilon', 0.1)
            action, log_prob, entropy, trades = self.agent.select_action_reduced(
                top_4_stocks=top_4_stocks,
                states=states,
                current_position=current_position,
                epsilon=epsilon,
                deterministic=False
            )

            # Track action taken
            episode_actions.append(action)

            # Execute trades in environment
            # Convert trades to environment format: {ticker: action_id}
            actions = {}
            for trade in trades:
                ticker = trade['ticker']
                if trade['action'] == 'BUY':
                    actions[ticker] = 1  # BUY
                elif trade['action'] == 'SELL':
                    actions[ticker] = 2  # SELL

            # Take step in environment
            next_states_dict, reward, done, info = self.env.step(actions)

            # Update current position based on trades
            for trade in trades:
                if trade['action'] == 'SELL':
                    current_position = None  # Sold, now in cash
                elif trade['action'] == 'BUY':
                    current_position = trade['ticker']  # Bought, now holding this stock

            # Get next global state
            next_cached_states = self.env.state_cache[self.env.current_date]
            next_top_4_stocks = get_top_4_stocks(next_cached_states)
            next_states = self.env._get_states()
            next_global_state = create_global_state(
                top_4_stocks=next_top_4_stocks,
                states=next_states,
                current_position=current_position,
                device=self.device
            )

            # Store transition with global states and discrete action
            self.buffer.push(
                state=global_state,
                action=action,  # Discrete action (0-4)
                reward=reward,
                next_state=next_global_state,
                done=done,
                ticker=current_position if current_position else 'CASH',
                portfolio_value=info['portfolio_value']
            )

            episode_return += reward
            self.global_step += 1

            # Train networks if enough data (multiple updates per step for better GPU utilization)
            if len(self.buffer) >= self.config.get('batch_size', 256):
                num_updates = self.config.get('num_updates_per_step', 1)

                for _ in range(num_updates):
                    # Train critic
                    freeze_critic_episodes = self.config.get('freeze_critic_episodes', 0)
                    if self.episode >= freeze_critic_episodes:
                        critic_loss = self._train_critic()
                        episode_critic_loss.append(critic_loss)
                    else:
                        # Critic frozen - just compute loss for logging
                        batch = self.buffer.sample(self.config.get('batch_size', 256))
                        with torch.no_grad():
                            loss1, loss2 = compute_critic_loss(
                                agent=self.agent,
                                batch=batch,
                                gamma=self.config.get('gamma', 0.99),
                                device=self.device
                            )
                            critic_loss = (loss1.item() + loss2.item()) / 2.0
                        episode_critic_loss.append(critic_loss)

                    # Train actor (less frequently)
                    if self.global_step % self.config.get('actor_update_freq', 2) == 0:
                        actor_loss, actor_info = self._train_actor()
                        episode_actor_loss.append(actor_loss)
                        episode_entropy.append(actor_info['entropy'])
                        episode_advantages.append(actor_info['avg_advantage'])
                        episode_q_values.append(actor_info['avg_q_value'])
                        episode_grad_norms.append(actor_info['grad_norm'])

                    # Update both target critics
                    self.agent.update_target_critics(tau=self.config.get('tau', 0.005))

            if done:
                break

        # Episode statistics
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0.0
        avg_actor_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0.0

        # Compute action distribution (0=HOLD, 1-4=STOCKS)
        action_counts = {i: 0 for i in range(5)}
        for action in episode_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        total_actions = len(episode_actions) if episode_actions else 1
        action_dist = {k: v/total_actions for k, v in action_counts.items()}

        # Update EMA values
        if self.ema_critic_loss is None:
            self.ema_critic_loss = avg_critic_loss
            self.ema_actor_loss = avg_actor_loss
            self.ema_return = episode_return
        else:
            self.ema_critic_loss = self.ema_alpha * self.ema_critic_loss + (1 - self.ema_alpha) * avg_critic_loss
            self.ema_actor_loss = self.ema_alpha * self.ema_actor_loss + (1 - self.ema_alpha) * avg_actor_loss
            self.ema_return = self.ema_alpha * self.ema_return + (1 - self.ema_alpha) * episode_return

        stats = {
            'episode': self.episode,
            'return': episode_return,
            'portfolio_value': info['portfolio_value'],
            'num_positions': info['num_positions'],
            'avg_critic_loss': avg_critic_loss,
            'avg_actor_loss': avg_actor_loss,
            'avg_entropy': np.mean(episode_entropy) if episode_entropy else 0.0,
            'avg_advantage': np.mean(episode_advantages) if episode_advantages else 0.0,
            'avg_q_value': np.mean(episode_q_values) if episode_q_values else 0.0,
            'avg_grad_norm': np.mean(episode_grad_norms) if episode_grad_norms else 0.0,
            'buffer_size': len(self.buffer),
            'ema_critic_loss': self.ema_critic_loss,
            'ema_actor_loss': self.ema_actor_loss,
            'ema_return': self.ema_return,
            'pct_action_0': action_dist.get(0, 0),
            'pct_action_1': action_dist.get(1, 0),
            'pct_action_2': action_dist.get(2, 0),
            'pct_action_3': action_dist.get(3, 0),
            'pct_action_4': action_dist.get(4, 0)
        }

        self.episode += 1

        return stats

    def _train_critic(self) -> Tuple[float, Dict]:
        """Train both critics for one step (twin critics)."""
        batch = self.buffer.sample(self.config.get('batch_size', 1024))

        loss1, loss2 = compute_critic_loss(
            agent=self.agent,
            batch=batch,
            gamma=self.config.get('gamma', 0.99),
            device=self.device
        )

        # Train critic 1
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), max_norm=10.0)
        self.critic1_optimizer.step()

        # Train critic 2
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), max_norm=10.0)
        self.critic2_optimizer.step()

        # Compute Q-values for diagnostics
        with torch.no_grad():
            states = torch.stack([t['state'] for t in batch]).to(self.device)
            actions = torch.tensor([t['action'] for t in batch], dtype=torch.long).unsqueeze(1).to(self.device)

            q1_vals = self.agent.critic1(states).gather(1, actions).squeeze(1)
            q2_vals = self.agent.critic2(states).gather(1, actions).squeeze(1)

            info = {
                'loss1': loss1.item(),
                'loss2': loss2.item(),
                'q1_mean': q1_vals.mean().item(),
                'q2_mean': q2_vals.mean().item(),
                'q_diff': (q1_vals - q2_vals).abs().mean().item()
            }

        # Return average loss for logging
        return (loss1.item() + loss2.item()) / 2.0, info

    def _train_actor(self) -> Tuple[float, Dict]:
        """Train actor for one step."""
        batch = self.buffer.sample(self.config.get('batch_size', 1024))

        loss, info = compute_actor_loss(
            agent=self.agent,
            batch=batch,
            device=self.device,
            entropy_coef=self.config.get('entropy_coef', 0.05)
        )

        self.actor_optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=10.0)
        info['grad_norm'] = total_norm.item()
        self.actor_optimizer.step()

        return loss.item(), info

    def train(self):
        """Main training loop (vectorized)."""
        print("\n" + "="*80)
        print("STARTING ONLINE ACTOR-CRITIC TRAINING (VECTORIZED)")
        print("="*80)

        num_episodes = self.config.get('num_episodes', 1000)
        num_parallel = self.config.get('num_parallel_envs', 8)
        use_wandb = self.config.get('use_wandb', True)

        # Calculate number of iterations (each iteration runs N parallel episodes)
        num_iterations = num_episodes // num_parallel

        pbar = tqdm(range(num_iterations), desc="Training")
        for iteration in pbar:
            # Train batch of parallel episodes
            stats_list = self.train_episode_vectorized()

            # Add all stats to history
            for stats in stats_list:
                self.episode_history.append(stats)

            # Use first episode for logging
            stats = stats_list[0]

            # Update progress bar with key metrics
            pbar.set_postfix({
                'Ep': f"{self.episode-num_parallel+1}-{self.episode}",
                'R_ema': f"{stats['ema_return']:+.4f}",
                'Port_ema': f"${stats['ema_portfolio_value']/1000:.1f}k",
                'C_ema': f"{stats['ema_critic_loss']:.3f}",
                'A_ema': f"{stats['ema_actor_loss']:.3f}",
                'H': f"{stats['avg_entropy']:.3f}",
                'Q': f"{stats['avg_q_value']:+.2f}",
                'Buf': f"{stats['buffer_size']:,}"
            })

            # Log to WandB
            if use_wandb:
                wandb.log({
                    'episode': self.episode,
                    'return': stats['return'],
                    'return_ema': stats['ema_return'],
                    'portfolio_value': stats['portfolio_value'],
                    'portfolio_value_ema': stats['ema_portfolio_value'],
                    'critic_loss': stats['avg_critic_loss'],
                    'critic_loss_ema': stats['ema_critic_loss'],
                    'actor_loss': stats['avg_actor_loss'],
                    'actor_loss_ema': stats['ema_actor_loss'],
                    'entropy': stats['avg_entropy'],
                    'q_value': stats['avg_q_value'],
                    'q1_value': stats['avg_q1_value'],
                    'q2_value': stats['avg_q2_value'],
                    'q_diff': stats['avg_q_diff'],
                    'advantage': stats['avg_advantage'],
                    'grad_norm': stats['avg_grad_norm'],
                    'buffer_size': stats['buffer_size'],
                    'action_0_pct': stats['pct_action_0'],
                    'action_1_pct': stats['pct_action_1'],
                    'action_2_pct': stats['pct_action_2'],
                    'action_3_pct': stats['pct_action_3'],
                    'action_4_pct': stats['pct_action_4'],
                }, step=self.episode)

            # Save checkpoint
            if self.episode % self.config.get('save_interval', 1000) < num_parallel:
                self._save_checkpoint(self.episode)

            # Step schedulers
            for _ in range(num_parallel):
                self.actor_scheduler.step()
                self.critic1_scheduler.step()
                self.critic2_scheduler.step()

        pbar.close()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)

        # Final save
        self._save_checkpoint(self.episode - 1, final=True)
        print(f"✅ Checkpoint saved: ./checkpoints/actor_critic_final.pt")

        # Save episode history
        self._save_episode_history()
        print(f"✅ Training history saved: ./checkpoints/actor_critic_training_history.csv")

        # Finish WandB run
        if use_wandb:
            wandb.finish()
            print(f"✅ WandB run finished")

    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint."""
        suffix = '_final' if final else f'_ep{episode+1}'
        checkpoint_path = f"./checkpoints/actor_critic{suffix}.pt"

        checkpoint = {
            'episode': episode,
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'config': self.config,
            'episode_history': self.episode_history
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def _save_episode_history(self):
        """Save episode history to CSV."""
        df = pd.DataFrame(self.episode_history)
        csv_path = './checkpoints/actor_critic_training_history.csv'
        df.to_csv(csv_path, index=False)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        print(f"   Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore agent networks
        self.agent.load_state_dict(checkpoint['agent_state_dict'])

        # Restore optimizers
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])

        # Restore training state
        self.episode = checkpoint['episode']
        self.global_step = checkpoint.get('global_step', 0)

        # Restore episode history
        self.episode_history = checkpoint.get('episode_history', [])

        # Restore EMA values if available
        if self.episode_history:
            last_stats = self.episode_history[-1]
            self.ema_critic_loss = last_stats.get('ema_critic_loss', None)
            self.ema_actor_loss = last_stats.get('ema_actor_loss', None)
            self.ema_return = last_stats.get('ema_return', None)
            self.ema_portfolio_value = last_stats.get('ema_portfolio_value', None)

        print(f"   Loaded {len(self.episode_history)} historical episodes")


if __name__ == '__main__':
    # Configuration
    config = {
        # Data
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'num_test_stocks': 100,

        # Model
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',
        #'pretrained_critic_path': './checkpoints/pretrained_critic.pt',
        'state_dim': 5881,  # 4 stocks × 1469 + 5 position encoding
        'hidden_dim': 1024,
        'action_dim': 5,  # cash + 4 stocks

        # Environment
        'initial_capital': 100000,
        'max_positions': 1,  # Single stock mode
        'episode_length': 30,  # Shorter episodes for faster opportunity switching
        'top_k_per_horizon': 10,

        # Training
        'num_episodes': 500000,
        'num_parallel_envs': 16,  # Parallel environments for faster data collection
        'batch_size': 1024,  # Large batch for stable gradients
        'updates_per_transition': 0.05,  # 0.25 = 1 update per 4 env steps (conservative, prevents overestimation)
        'buffer_capacity': 50000,  # Smaller buffer = fresher data
        'reward_scale': 100.0,  # Scale rewards (0.01 → 1.0) for better gradients
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,  # Increased from 5e-5 for faster learning
        'gamma': 0.99,
        'tau': 0.01,  # Increased from 0.005 for faster target network updates
        'entropy_coef': 0.05,  # Entropy regularization (prevent collapse)
        'epsilon': 0.1,  # Epsilon-greedy exploration rate
        'actor_update_freq': 2,  # Update actor every 2 critic updates
        'freeze_critic_episodes': 0,  # No critic freezing

        # Logging
        'log_interval': 50,
        'save_interval': 1000,
        'use_wandb': True,
        'wandb_project': 'stock-rl-trading',
        'wandb_run_name': None,  # Auto-generated if None

        # Resume training (set to checkpoint path to continue from)
        'resume_checkpoint': None,  # e.g., './checkpoints/actor_critic_ep10000.pt'

        # Caching
        'feature_cache_path': 'data/rl_feature_cache_4yr.h5',
        'state_cache_path': 'data/rl_state_cache_4yr.h5',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create trainer and train
    trainer = ActorCriticTrainer(config)
    trainer.train()
