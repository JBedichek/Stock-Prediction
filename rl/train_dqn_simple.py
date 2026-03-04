#!/usr/bin/env python3
"""
Simplified DQN Trainer for Stock Trading with Short Selling.

Much simpler than actor-critic:
- One Q-network (instead of actor + critic)
- Direct Q-value prediction
- Simple TD learning
- ε-greedy exploration

Action Space (9 actions):
- Action 0: HOLD (stay in cash)
- Actions 1-4: LONG (buy top_4 stocks)
- Actions 5-8: SHORT (short bottom_4 stocks)
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import wandb

from rl.rl_components import StockDQN, ReplayBuffer
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.reduced_action_space import create_global_state
from rl.state_creation_optimized import create_next_states_batch_optimized
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.vectorized_training_loop import vectorized_transition_storage
from rl.profiling_utils import TrainingProfiler


class SimpleDQNTrainer:
    """
    Simplified DQN trainer with configurable action space.

    Action space:
    - allow_short=False: 5 actions (HOLD + 4 LONG)
    - allow_short=True: 9 actions (HOLD + 4 LONG + 4 SHORT)

    Much simpler than actor-critic:
    - One network to train (Q-network)
    - One loss function (TD error)
    - One optimizer
    - Direct action selection (argmax Q or ε-greedy)
    """

    def __init__(self, config: Dict):
        """
        Initialize DQN trainer.

        Args:
            config: Training configuration dict
        """
        self.config = config
        self.device = config.get('device', 'cuda')
        self.allow_short = config.get('allow_short', False)  # Default: no short selling

        # Networks (Q-network + target network)
        state_dim = config.get('state_dim', 1469)
        self.action_dim = 9 if self.allow_short else 5  # 9 with shorts, 5 without

        self.q_network = StockDQN(
            state_dim=state_dim,
            hidden_dim=config.get('hidden_dim', 1024),
            action_dim=self.action_dim
        ).to(self.device)

        self.target_network = StockDQN(
            state_dim=state_dim,
            hidden_dim=config.get('hidden_dim', 1024),
            action_dim=self.action_dim
        ).to(self.device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode

        # Optimizer (only need one!)
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler (cosine annealing for smooth decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_episodes', 100000),
            eta_min=config.get('learning_rate', 1e-4) * 0.1  # Decay to 10% of initial LR
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=config.get('buffer_capacity', 100000),
            n_step=config.get('n_step', 1),
            gamma=config.get('gamma', 0.99),
            use_per=config.get('use_per', False)
        )

        # Exploration (ε-greedy)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay_episodes', 10000)

        # Training counters
        self.episode = 0
        self.global_step = 0

        # Target network update method
        self.use_soft_update = config.get('use_soft_update', True)
        self.tau = config.get('tau', 0.005)  # Polyak averaging coefficient

        # Profiler
        self.profiler = TrainingProfiler()

        # EMA tracking for smooth metrics
        self.ema_return = 0.0
        self.ema_q_value = 0.0
        self.ema_loss = 0.0
        self.ema_grad_norm = 0.0
        self.ema_alpha = 0.01

        # EMA for portfolio value with large alpha to see long-term trends
        self.ema_portfolio_value = config.get('initial_capital', 100000)
        self.ema_pv_alpha = 0.999

        # Store initial capital for easy access
        self.initial_capital = config.get('initial_capital', 100000)

        print(f"\n{'='*80}")
        print("SIMPLIFIED DQN TRAINER")
        print(f"{'='*80}")
        print(f"  Q-Network: {state_dim} → {self.action_dim} actions")
        if self.allow_short:
            print(f"  Action space: HOLD + 4 LONG + 4 SHORT = {self.action_dim}")
        else:
            print(f"  Action space: HOLD + 4 LONG = {self.action_dim} (SHORT SELLING DISABLED)")
        print(f"  Replay buffer: {config.get('buffer_capacity', 100000):,} transitions")
        print(f"  Exploration: ε = {self.epsilon:.2f} → {self.epsilon_end:.2f} over {self.epsilon_decay:,} episodes")
        print(f"  Target updates: {'Soft (Polyak, τ=' + str(self.tau) + ')' if self.use_soft_update else 'Hard (every ' + str(config.get('target_update_freq', 1000)) + ' steps)'}")
        print(f"  Learning rate: {config.get('learning_rate', 1e-4):.1e} (cosine annealing)")
        print(f"{'='*80}\n")

    def select_action(self, state: torch.Tensor, epsilon: float = 0.0, has_position: bool = False) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: State tensor (batch_size, state_dim) or (state_dim,)
            epsilon: Exploration probability
            has_position: Whether agent currently holds a position (if False, HOLD is not allowed)

        Returns:
            Action index (0-8 or 1-8 if no position)
        """
        if random.random() < epsilon:
            # Explore: random action
            if has_position:
                # Can choose any action including HOLD (0)
                return random.randint(0, self.action_dim - 1)
            else:
                # Cannot HOLD when no position - choose from actions 1 to action_dim-1
                return random.randint(1, self.action_dim - 1)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)  # Add batch dim
                q_values = self.q_network(state)

                # Mask out HOLD action (0) if no position
                if not has_position:
                    q_values = q_values.clone()  # Don't modify original
                    q_values[0, 0] = -float('inf')  # Make HOLD have -inf Q-value

                action = q_values.argmax(dim=1).item()
                return action

    def action_to_trade(self, action: int, top_4_stocks: List, bottom_4_stocks: List) -> Dict:
        """
        Convert discrete action to trade dict.

        Action mapping (allow_short=False, 5 actions):
        - 0: HOLD (no trade)
        - 1-4: LONG top_4_stocks[0-3]

        Action mapping (allow_short=True, 9 actions):
        - 0: HOLD (no trade)
        - 1-4: LONG top_4_stocks[0-3]
        - 5-8: SHORT bottom_4_stocks[0-3]

        Args:
            action: Action index (0-4 or 0-8 depending on allow_short)
            top_4_stocks: List of (ticker, horizon) for long positions
            bottom_4_stocks: List of (ticker, horizon) for short positions

        Returns:
            Trade dict with 'action', 'ticker', 'position_type'
        """
        if action == 0:
            # HOLD
            return {'action': 'HOLD', 'ticker': None, 'position_type': None}
        elif 1 <= action <= 4:
            # LONG
            stock_idx = action - 1
            if stock_idx < len(top_4_stocks):
                ticker = top_4_stocks[stock_idx][0]
                return {'action': 'BUY', 'ticker': ticker, 'position_type': 'LONG'}
            else:
                return {'action': 'HOLD', 'ticker': None, 'position_type': None}
        elif 5 <= action <= 8 and self.allow_short:
            # SHORT (only if short selling is allowed)
            stock_idx = action - 5
            if stock_idx < len(bottom_4_stocks):
                ticker = bottom_4_stocks[stock_idx][0]
                return {'action': 'SHORT', 'ticker': ticker, 'position_type': 'SHORT'}
            else:
                return {'action': 'HOLD', 'ticker': None, 'position_type': None}
        else:
            # Invalid action or short selling disabled
            return {'action': 'HOLD', 'ticker': None, 'position_type': None}

    def compute_td_loss(self, batch: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """
        Compute TD loss for DQN.

        Loss = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]

        Args:
            batch: List of transition dicts

        Returns:
            Tuple of (loss, info_dict)
        """
        # CRITICAL: Ensure gradients are enabled (might be disabled globally somewhere)
        torch.set_grad_enabled(True)

        # Set networks to correct mode
        self.q_network.train()
        self.target_network.eval()

        # Extract batch
        # States are already on CPU as tensors from buffer - just move to device
        state_list = []
        next_state_list = []
        for t in batch:
            state = t['state']
            next_state = t['next_state']
            # Ensure they're tensors (not already on GPU)
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            state_list.append(state)
            next_state_list.append(next_state)

        states = torch.stack(state_list).to(self.device)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack(next_state_list).to(self.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32, device=self.device)

        # Debug: Check state dimensions
        expected_dim = self.config.get('state_dim', 11761)
        actual_dim = states.shape[1]
        if actual_dim != expected_dim:
            print(f"\n⚠️  STATE DIMENSION MISMATCH!")
            print(f"  Expected: {expected_dim}")
            print(f"  Actual: {actual_dim}")
            print(f"  Q-network input dim: {self.q_network.fc1.in_features}")
            print(f"  allow_short: {self.allow_short}")
            print(f"  This will cause training to fail!\n")
            # Skip this batch
            return torch.tensor(0.0, device=self.device), {
                'loss': 0.0, 'avg_q_value': 0.0, 'avg_target_q': 0.0, 'avg_reward': 0.0
            }

        # Current Q-values: Q(s, a) - this should have gradients!
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.config.get('gamma', 0.99) * next_q_values * (1 - dones)

        # MSE loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Debug: Check if loss has gradients
        if not loss.requires_grad:
            print(f"⚠️  WARNING: Loss doesn't require grad!")
            print(f"  current_q_values.requires_grad: {current_q_values.requires_grad}")
            print(f"  states.requires_grad: {states.requires_grad}")
            # Check if network parameters require grad
            has_grad = any(p.requires_grad for p in self.q_network.parameters())
            print(f"  Q-network has grad params: {has_grad}")
            if not has_grad:
                print(f"  ❌ Q-network parameters don't require grad - enabling...")
                for p in self.q_network.parameters():
                    p.requires_grad = True

        # Info for logging
        info = {
            'loss': loss.item(),
            'avg_q_value': current_q_values.mean().item(),
            'avg_target_q': target_q_values.mean().item(),
            'avg_reward': rewards.mean().item()
        }

        return loss, info

    def update_target_network(self, soft=None):
        """
        Update target network (hard or soft update).

        Args:
            soft: If True, use soft update (Polyak averaging). If None, use config setting.
        """
        use_soft = soft if soft is not None else self.use_soft_update

        if use_soft:
            # Soft update: θ_target = τ * θ_local + (1 - τ) * θ_target
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            # Hard update: copy all weights
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train_step(self) -> Tuple[float, Dict]:
        """
        Perform one training step (sample batch and update Q-network).

        Returns:
            Tuple of (loss, info_dict)
        """
        batch_size = min(self.config.get('batch_size', 1024), len(self.buffer))
        if batch_size < 32:
            # Buffer too small
            return 0.0, {'loss': 0.0, 'avg_q_value': 0.0, 'avg_target_q': 0.0, 'avg_reward': 0.0}

        # Sample batch
        batch = self.buffer.sample(batch_size)

        # Compute loss
        loss, info = self.compute_td_loss(batch)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients and track norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Add gradient norm to info
        info['grad_norm'] = grad_norm.item()

        return loss.item(), info

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_end + (self.config.get('epsilon_start', 1.0) - self.epsilon_end) *
            (1 - self.episode / self.epsilon_decay)
        )

    def train_episode_vectorized(
        self,
        vec_env: GPUVectorizedTradingEnv,
        stock_selections_cache: GPUStockSelectionCache,
        use_precomputed_selections: bool
    ) -> List[Dict]:
        """
        Train one batch of parallel episodes using vectorized environment.

        Args:
            vec_env: Vectorized environment
            stock_selections_cache: GPU cache of stock selections
            use_precomputed_selections: Whether to use cache

        Returns:
            List of episode stats (one per parallel env)
        """
        num_parallel = vec_env.num_envs
        max_steps = vec_env.episode_length

        # Reset environments
        with self.profiler.profile('env_reset'):
            vec_env.reset()

        # Initial stock selections
        with self.profiler.profile('initial_stock_selection'):
            top_4_stocks_list = []
            bottom_4_stocks_list = []

            for i in range(num_parallel):
                current_date = vec_env.episode_dates[i][0]
                if use_precomputed_selections and current_date in stock_selections_cache:
                    num_samples = stock_selections_cache.get_num_samples(current_date)
                    sample_idx = random.randint(0, num_samples - 1)
                    top_4, bottom_4 = stock_selections_cache.get_sample(current_date, sample_idx)
                else:
                    # Fallback (shouldn't happen often)
                    top_4 = []
                    bottom_4 = []

                top_4_stocks_list.append(top_4)
                bottom_4_stocks_list.append(bottom_4)

        # Create initial states
        with self.profiler.profile('create_initial_states'):
            states_list = []
            for i in range(num_parallel):
                current_date = vec_env.episode_dates[i][0]
                if len(top_4_stocks_list[i]) > 0:
                    states_dict = create_next_states_batch_optimized(
                        next_env=vec_env.ref_env,
                        next_top_4_stocks=top_4_stocks_list[i],
                        next_bottom_4_stocks=bottom_4_stocks_list[i],
                        next_date=current_date,
                        device=self.device
                    )
                    states_list.append(states_dict)
                else:
                    states_list.append({})

        # Create initial global states
        global_states_list = []
        positions_list = [None] * num_parallel
        is_short_list = [False] * num_parallel

        for i in range(num_parallel):
            if len(states_list[i]) > 0:
                global_state = create_global_state(
                    top_4_stocks_list[i],
                    bottom_4_stocks_list[i],
                    states_list[i],
                    positions_list[i],
                    is_short=is_short_list[i],
                    device=self.device,
                    allow_short=self.allow_short
                )
                # Debug: Check state dimension
                expected_dim = self.config.get('state_dim', 11761)
                if global_state.shape[0] != expected_dim:
                    print(f"\n⚠️  GLOBAL STATE DIMENSION MISMATCH (env {i})!")
                    print(f"  Expected: {expected_dim}")
                    print(f"  Actual: {global_state.shape[0]}")
                    print(f"  allow_short: {self.allow_short}")
                    print(f"  top_4_stocks: {len(top_4_stocks_list[i])}")
                    print(f"  bottom_4_stocks: {len(bottom_4_stocks_list[i])}")
                global_states_list.append(global_state)
            else:
                global_states_list.append(None)

        # Episode tracking
        episode_returns = [0.0] * num_parallel
        episode_steps = [0] * num_parallel
        episode_losses = [[] for _ in range(num_parallel)]
        episode_q_values = [[] for _ in range(num_parallel)]
        episode_grad_norms = [[] for _ in range(num_parallel)]

        # Main episode loop
        for step in range(max_steps):
            with self.profiler.profile('episode_step'):
                # Select actions for all environments
                actions_list = []
                trades_list = []

                for i in range(num_parallel):
                    if not vec_env.dones[i] and global_states_list[i] is not None:
                        # Check if agent currently has a position
                        has_position = (positions_list[i] is not None)
                        action = self.select_action(global_states_list[i], epsilon=self.epsilon, has_position=has_position)
                        trade = self.action_to_trade(action, top_4_stocks_list[i], bottom_4_stocks_list[i])
                    else:
                        action = 0  # HOLD
                        trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}

                    actions_list.append(action)
                    trades_list.append([trade])  # Wrap in list - vec_env expects List[List[Dict]]

                # Environment step
                with self.profiler.profile('environment_step'):
                    next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
                        actions_list,
                        trades_list,
                        positions_list
                    )

                # Update is_short flags based on trades
                for i in range(num_parallel):
                    trade = trades_list[i][0]  # Get first trade from list
                    if trade['position_type'] == 'SHORT':
                        is_short_list[i] = True
                    elif trade['position_type'] == 'LONG':
                        is_short_list[i] = False
                    elif trade['action'] == 'HOLD' and next_positions_list[i] is None:
                        is_short_list[i] = False

                # Store transitions and get next states (vectorized!)
                stored_transitions = vectorized_transition_storage(
                    vec_env=vec_env,
                    top_4_stocks_list=top_4_stocks_list,
                    bottom_4_stocks_list=bottom_4_stocks_list,
                    states_list_current=states_list,
                    positions_list=positions_list,
                    next_positions_list=next_positions_list,
                    is_short_list=is_short_list,
                    actions_list=actions_list,
                    rewards_list=rewards_list,
                    dones_list=dones_list,
                    infos_list=infos_list,
                    stock_selections_cache=stock_selections_cache,
                    use_precomputed_selections=use_precomputed_selections,
                    top_4_cache={},
                    stock_sample_fraction=1.0,
                    top_k_per_horizon_sampling=self.config.get('top_k_per_horizon', 50),
                    buffer=self.buffer,
                    device=self.device,
                    profiler=self.profiler,
                    reward_scale=self.config.get('reward_scale', 1.0),
                    reward_normalizer=lambda x: x,  # No normalization
                    allow_short=self.allow_short
                )

                # Train Q-network
                min_buffer_size = self.config.get('min_buffer_size', 1000)
                buffer_size = len(self.buffer)

                # Debug: Print buffer status once
                if self.global_step == 0 and buffer_size > 0:
                    print(f"\n📊 Buffer Status:")
                    print(f"  Buffer size: {buffer_size}")
                    print(f"  Min required: {min_buffer_size}")
                    print(f"  Training will start when buffer >= {min_buffer_size}\n")

                if buffer_size >= min_buffer_size:
                    num_updates = self.config.get('updates_per_step', 1)
                    for _ in range(num_updates):
                        loss, info = self.train_step()
                        self.global_step += 1

                        # Track metrics
                        for i in range(num_parallel):
                            if not vec_env.dones[i]:
                                episode_losses[i].append(loss)
                                episode_q_values[i].append(info['avg_q_value'])
                                episode_grad_norms[i].append(info['grad_norm'])

                        # Update target network
                        if self.use_soft_update:
                            # Soft update: update every step with small tau
                            self.update_target_network()
                        else:
                            # Hard update: update periodically
                            if self.global_step % self.config.get('target_update_freq', 1000) == 0:
                                self.update_target_network()

                # Update episode stats
                for i in range(num_parallel):
                    if not vec_env.dones[i]:
                        episode_returns[i] += rewards_list[i]
                        episode_steps[i] += 1

                # Update for next step
                positions_list = next_positions_list

                # Check if all done
                if all(vec_env.dones):
                    break

        # Compile episode stats
        stats_list = []
        for i in range(num_parallel):
            portfolio_value = infos_list[i].get('portfolio_value', self.config.get('initial_capital', 100000))

            # Update EMAs
            if self.episode == 0:
                self.ema_return = episode_returns[i]
                self.ema_q_value = np.mean(episode_q_values[i]) if episode_q_values[i] else 0.0
                self.ema_loss = np.mean(episode_losses[i]) if episode_losses[i] else 0.0
                self.ema_grad_norm = np.mean(episode_grad_norms[i]) if episode_grad_norms[i] else 0.0
                self.ema_portfolio_value = portfolio_value
            else:
                self.ema_return = self.ema_alpha * episode_returns[i] + (1 - self.ema_alpha) * self.ema_return
                self.ema_q_value = self.ema_alpha * (np.mean(episode_q_values[i]) if episode_q_values[i] else 0.0) + (1 - self.ema_alpha) * self.ema_q_value
                self.ema_loss = self.ema_alpha * (np.mean(episode_losses[i]) if episode_losses[i] else 0.0) + (1 - self.ema_alpha) * self.ema_loss
                self.ema_grad_norm = self.ema_alpha * (np.mean(episode_grad_norms[i]) if episode_grad_norms[i] else 0.0) + (1 - self.ema_alpha) * self.ema_grad_norm
                self.ema_portfolio_value = self.ema_pv_alpha * self.ema_portfolio_value + (1 - self.ema_pv_alpha) * portfolio_value

            stats = {
                'return': episode_returns[i],
                'steps': episode_steps[i],
                'portfolio_value': portfolio_value,
                'avg_loss': np.mean(episode_losses[i]) if episode_losses[i] else 0.0,
                'avg_q_value': np.mean(episode_q_values[i]) if episode_q_values[i] else 0.0,
                'avg_grad_norm': np.mean(episode_grad_norms[i]) if episode_grad_norms[i] else 0.0,
                'epsilon': self.epsilon,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'buffer_size': len(self.buffer),
                'ema_return': self.ema_return,
                'ema_q_value': self.ema_q_value,
                'ema_loss': self.ema_loss,
                'ema_grad_norm': self.ema_grad_norm,
                'ema_portfolio_value': self.ema_portfolio_value
            }
            stats_list.append(stats)

        # Decay epsilon and learning rate
        self.decay_epsilon()
        self.scheduler.step()
        self.episode += num_parallel

        return stats_list

    def train(
        self,
        vec_env: GPUVectorizedTradingEnv,
        stock_selections_cache: GPUStockSelectionCache,
        use_precomputed_selections: bool = True,
        val_env: Optional[GPUVectorizedTradingEnv] = None,
        val_stock_cache: Optional[GPUStockSelectionCache] = None,
        val_freq: int = 100,
        val_episodes: int = 10,
        early_stopping_patience: int = 10
    ):
        """
        Main training loop with validation and early stopping.

        Args:
            vec_env: Training vectorized environment
            stock_selections_cache: GPU cache for training
            use_precomputed_selections: Whether to use cache
            val_env: Validation vectorized environment (optional)
            val_stock_cache: GPU cache for validation (optional)
            val_freq: Run validation every N iterations
            val_episodes: Number of validation episodes
            early_stopping_patience: Stop if no improvement for N validations
        """
        print(f"\n{'='*80}")
        print("STARTING DQN TRAINING")
        print(f"{'='*80}\n")

        num_episodes = self.config.get('num_episodes', 10000)
        num_parallel = vec_env.num_envs
        num_iterations = num_episodes // num_parallel
        use_wandb = self.config.get('use_wandb', False)

        # Early stopping tracking
        best_val_sharpe = -np.inf
        patience_counter = 0
        use_validation = val_env is not None and val_stock_cache is not None

        if use_validation:
            print(f"✅ Validation enabled:")
            print(f"   - Running every {val_freq} iterations")
            print(f"   - Early stopping patience: {early_stopping_patience}")
            print(f"   - Validation episodes: {val_episodes}\n")

        pbar = tqdm(range(num_iterations), desc="Training")
        for iteration in pbar:
            # Train batch of parallel episodes
            with self.profiler.profile('full_episode'):
                stats_list = self.train_episode_vectorized(
                    vec_env,
                    stock_selections_cache,
                    use_precomputed_selections
                )

            # Record episodes
            for _ in range(num_parallel):
                self.profiler.record_episode()

            # Use first episode for logging
            stats = stats_list[0]

            # Compute mean portfolio value across all parallel environments
            portfolio_values = [s['portfolio_value'] for s in stats_list]
            mean_pv = np.mean(portfolio_values)

            # Update progress bar
            pbar.set_postfix_str(
                f"Ep={self.episode-num_parallel+1}-{self.episode} | "
                f"R_ema={stats['ema_return']:+.4f} | "
                f"PV_ema=${stats['ema_portfolio_value']:,.0f} | "
                f"Q_ema={stats['ema_q_value']:+.2f} | "
                f"L_ema={stats['ema_loss']:.3f} | "
                f"G_ema={stats['ema_grad_norm']:.2f} | "
                f"LR={stats['learning_rate']:.1e} | "
                f"ε={stats['epsilon']:.3f}"
            )

            # Log to WandB
            if use_wandb:
                wandb.log({
                    'episode': self.episode,
                    'return': stats['return'],
                    'return_ema': stats['ema_return'],
                    'portfolio_value': stats['portfolio_value'],
                    'portfolio_value_mean': np.mean(portfolio_values),
                    'portfolio_value_std': np.std(portfolio_values),
                    'portfolio_value_ema': stats['ema_portfolio_value'],
                    'loss': stats['avg_loss'],
                    'loss_ema': stats['ema_loss'],
                    'q_value': stats['avg_q_value'],
                    'q_value_ema': stats['ema_q_value'],
                    'grad_norm': stats['avg_grad_norm'],
                    'grad_norm_ema': stats['ema_grad_norm'],
                    'learning_rate': stats['learning_rate'],
                    'epsilon': stats['epsilon'],
                    'buffer_size': stats['buffer_size']
                })

            # Run validation periodically
            if use_validation and (iteration + 1) % val_freq == 0:
                print(f"\n{'='*80}")
                print(f"RUNNING VALIDATION (iteration {iteration + 1})")
                print(f"{'='*80}")

                val_metrics = self.validate(
                    val_env,
                    val_stock_cache,
                    num_episodes=val_episodes,
                    use_precomputed_selections=True
                )

                print(f"\nValidation Results:")
                print(f"  Mean Return:  {val_metrics['mean_return']:+.4f} ± {val_metrics['std_return']:.4f}")
                print(f"  Mean PV:      ${val_metrics['mean_pv']:,.0f} ± ${val_metrics['std_pv']:,.0f}")
                print(f"  Sharpe Ratio: {val_metrics['sharpe_ratio']:.4f}")
                print(f"{'='*80}\n")

                # Log validation metrics to WandB
                if use_wandb:
                    wandb.log({
                        'val/mean_return': val_metrics['mean_return'],
                        'val/std_return': val_metrics['std_return'],
                        'val/mean_pv': val_metrics['mean_pv'],
                        'val/sharpe_ratio': val_metrics['sharpe_ratio'],
                        'episode': self.episode
                    })

                # Early stopping check
                current_val_sharpe = val_metrics['sharpe_ratio']
                if current_val_sharpe > best_val_sharpe:
                    best_val_sharpe = current_val_sharpe
                    patience_counter = 0

                    # Save best model
                    best_model_path = "checkpoints/dqn_best.pt"
                    os.makedirs("checkpoints", exist_ok=True)
                    self.save_checkpoint(best_model_path, val_metrics)
                    print(f"✅ New best model! Sharpe: {best_val_sharpe:.4f} (saved to {best_model_path})\n")
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement. Patience: {patience_counter}/{early_stopping_patience}\n")

                    if patience_counter >= early_stopping_patience:
                        print(f"\n{'='*80}")
                        print(f"EARLY STOPPING TRIGGERED")
                        print(f"{'='*80}")
                        print(f"No improvement for {early_stopping_patience} validations.")
                        print(f"Best validation Sharpe: {best_val_sharpe:.4f}")
                        print(f"{'='*80}\n")
                        break

            # Save checkpoint periodically
            if (iteration + 1) % self.config.get('save_freq', 100) == 0:
                self.save_checkpoint(f"checkpoints/dqn_checkpoint_ep{self.episode}.pt")

        # Print profiler summary
        print("\n" + "="*80)
        print("TRAINING PROFILER SUMMARY")
        print("="*80)
        self.profiler.print_summary()

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")

    def validate(
        self,
        vec_env: GPUVectorizedTradingEnv,
        stock_selections_cache: GPUStockSelectionCache,
        num_episodes: int = 10,
        use_precomputed_selections: bool = True
    ) -> Dict:
        """
        Run validation episodes (greedy policy, no exploration).

        Args:
            vec_env: Validation vectorized environment
            stock_selections_cache: GPU cache
            num_episodes: Number of validation episodes to run
            use_precomputed_selections: Whether to use cache

        Returns:
            Dict with validation metrics
        """
        self.q_network.eval()  # Set to eval mode

        num_parallel = vec_env.num_envs
        num_iterations = max(1, num_episodes // num_parallel)

        all_returns = []
        all_portfolio_values = []
        all_steps = []

        # Run validation episodes with greedy policy (epsilon=0)
        for ep_iter in range(num_iterations):
            # Reset environment (returns tuple of states_list and positions_list)
            states_list, positions_list = vec_env.reset()
            is_short_list = [False] * num_parallel

            episode_returns = [0.0] * num_parallel
            episode_steps = [0] * num_parallel

            # Get initial stock selections
            top_4_stocks_list = []
            bottom_4_stocks_list = []

            print(f"\n{'='*100}")
            print(f"VALIDATION EPISODES {ep_iter * num_parallel + 1} - {(ep_iter + 1) * num_parallel}")
            print(f"{'='*100}")

            for env_idx in range(num_parallel):
                current_date = vec_env.episode_dates[env_idx][0]  # First date of episode
                num_samples = stock_selections_cache.get_num_samples(current_date)
                sample_idx = random.randint(0, num_samples - 1)
                top_4, bottom_4 = stock_selections_cache.get_sample(current_date, sample_idx)
                top_4_stocks_list.append(top_4)
                bottom_4_stocks_list.append(bottom_4)

                print(f"\n📊 Episode {ep_iter * num_parallel + env_idx + 1} - Start Date: {current_date}")
                print(f"   Top 4 LONG:  {[f'{t}(H{h})' for t, h in top_4[:4]]}")
                print(f"   Bottom 4 SHORT: {[f'{t}(H{h})' for t, h in bottom_4[:4]]}")
                print(f"   Initial Capital: ${self.initial_capital:,.0f}")

            # Run episode
            max_steps = vec_env.episode_length
            for step in range(max_steps):
                # Create global states
                global_states_list = []
                for i in range(num_parallel):
                    if not vec_env.dones[i]:
                        global_state = create_global_state(
                            top_4_stocks=top_4_stocks_list[i],
                            bottom_4_stocks=bottom_4_stocks_list[i],
                            states=states_list[i],  # Correct parameter name is 'states'
                            current_position=positions_list[i],
                            is_short=is_short_list[i],
                            device=self.device,
                            allow_short=self.allow_short
                        )
                        global_states_list.append(global_state)
                    else:
                        global_states_list.append(None)

                # Select actions (GREEDY - no exploration)
                actions_list = []
                trades_list = []
                q_values_list = []

                for i in range(num_parallel):
                    if not vec_env.dones[i] and global_states_list[i] is not None:
                        # Get Q-values for logging
                        with torch.no_grad():
                            q_vals = self.q_network(global_states_list[i].unsqueeze(0)).squeeze(0)
                            q_values_list.append(q_vals.cpu().numpy())

                        # Check if agent currently has a position
                        has_position = (positions_list[i] is not None)
                        action = self.select_action(global_states_list[i], epsilon=0.0, has_position=has_position)
                        trade = self.action_to_trade(action, top_4_stocks_list[i], bottom_4_stocks_list[i])
                    else:
                        action = 0  # HOLD
                        trade = {'action': 'HOLD', 'ticker': None, 'position_type': None}
                        q_values_list.append(None)

                    actions_list.append(action)
                    trades_list.append([trade])  # Wrap in list - vec_env expects List[List[Dict]]

                # Step environment
                next_states_list, rewards_list, dones_list, infos_list, next_positions_list = vec_env.step(
                    actions_list,
                    trades_list,
                    positions_list
                )

                # Log each environment's decision
                for i in range(num_parallel):
                    if not vec_env.dones[i] or step == 0:
                        current_step = vec_env.step_indices[i].item() - 1  # -1 because step_indices was incremented
                        if current_step >= 0 and current_step < len(vec_env.episode_dates[i]):
                            current_date = vec_env.episode_dates[i][current_step]

                            # Position info
                            pos_str = f"{positions_list[i]} ({'SHORT' if is_short_list[i] else 'LONG'})" if positions_list[i] else "CASH"

                            # Action info
                            action_names = {0: 'HOLD', 1: 'LONG-1', 2: 'LONG-2', 3: 'LONG-3', 4: 'LONG-4',
                                          5: 'SHORT-1', 6: 'SHORT-2', 7: 'SHORT-3', 8: 'SHORT-4'}
                            action_name = action_names.get(actions_list[i], f'ACT-{actions_list[i]}')

                            # Trade info
                            trade = trades_list[i][0]
                            trade_str = f"{trade['action']}"
                            if trade['ticker']:
                                trade_str += f" {trade['ticker']} ({trade['position_type']})"

                            # Q-values info
                            if q_values_list[i] is not None:
                                q_str = ", ".join([f"{q:.2f}" for q in q_values_list[i][:5]])
                                if len(q_values_list[i]) > 5:
                                    q_str += f", ... (9 total)"
                            else:
                                q_str = "N/A"

                            # Portfolio value
                            pv = infos_list[i].get('portfolio_value', 0)

                            print(f"   Ep{ep_iter * num_parallel + i + 1} Step {step:2d} | {current_date} | Pos: {pos_str:20s} | "
                                  f"Action: {action_name:8s} | Trade: {trade_str:25s} | "
                                  f"Reward: {rewards_list[i]:+7.2f} | PV: ${pv:>10,.0f} | Q: [{q_str}]")

                # Update is_short tracking based on position_type
                for i in range(num_parallel):
                    trade = trades_list[i][0]
                    if trade['position_type'] == 'SHORT':
                        is_short_list[i] = True
                    elif trade['position_type'] == 'LONG':
                        is_short_list[i] = False
                    elif trade['action'] == 'HOLD' and next_positions_list[i] is None:
                        is_short_list[i] = False

                # Update episode stats
                for i in range(num_parallel):
                    if not vec_env.dones[i]:
                        episode_returns[i] += rewards_list[i]
                        episode_steps[i] += 1

                # Update for next step
                positions_list = next_positions_list
                states_list = next_states_list

                # Get next stock selections (use current step index for next date)
                for env_idx in range(num_parallel):
                    if not vec_env.dones[env_idx]:
                        current_step = vec_env.step_indices[env_idx].item()
                        if current_step < len(vec_env.episode_dates[env_idx]):
                            current_date = vec_env.episode_dates[env_idx][current_step]
                            num_samples = stock_selections_cache.get_num_samples(current_date)
                            sample_idx = random.randint(0, num_samples - 1)
                            top_4, bottom_4 = stock_selections_cache.get_sample(current_date, sample_idx)
                            top_4_stocks_list[env_idx] = top_4
                            bottom_4_stocks_list[env_idx] = bottom_4

                if all(vec_env.dones):
                    break

            # Collect metrics and print episode summaries
            print(f"\n{'─'*100}")
            print(f"EPISODE SUMMARIES:")
            print(f"{'─'*100}")
            for i in range(num_parallel):
                pv = infos_list[i].get('portfolio_value', 100000)
                ret = episode_returns[i]
                ret_pct = (pv - self.initial_capital) / self.initial_capital * 100
                all_returns.append(ret)
                all_portfolio_values.append(pv)
                all_steps.append(episode_steps[i])

                print(f"   Episode {ep_iter * num_parallel + i + 1}: "
                      f"Return={ret:+8.2f} | PV=${pv:>10,.0f} | Gain={ret_pct:+6.2f}% | Steps={episode_steps[i]:2d}")

        # Compute validation metrics
        val_metrics = {
            'mean_return': float(np.mean(all_returns)),
            'std_return': float(np.std(all_returns)),
            'mean_pv': float(np.mean(all_portfolio_values)),
            'std_pv': float(np.std(all_portfolio_values)),
            'mean_steps': float(np.mean(all_steps)),
            'num_episodes': len(all_returns)
        }

        # Compute Sharpe ratio (annualized)
        if len(all_returns) > 1 and np.std(all_returns) > 0:
            sharpe = np.mean(all_returns) / np.std(all_returns) * np.sqrt(252 / 30)  # Assuming 30-day episodes
            val_metrics['sharpe_ratio'] = float(sharpe)
        else:
            val_metrics['sharpe_ratio'] = 0.0

        # Print aggregate statistics
        print(f"\n{'='*100}")
        print(f"VALIDATION AGGREGATE STATISTICS ({len(all_returns)} episodes)")
        print(f"{'='*100}")

        # Calculate additional stats
        gains_pct = [(pv - self.initial_capital) / self.initial_capital * 100 for pv in all_portfolio_values]
        win_rate = sum(1 for g in gains_pct if g > 0) / len(gains_pct) if len(gains_pct) > 0 else 0

        print(f"   Mean Return:      {val_metrics['mean_return']:+8.2f} ± {val_metrics['std_return']:.2f}")
        print(f"   Mean PV:          ${val_metrics['mean_pv']:>10,.0f} ± ${val_metrics['std_pv']:,.0f}")
        print(f"   Mean Gain:        {np.mean(gains_pct):+6.2f}% ± {np.std(gains_pct):.2f}%")
        print(f"   Best Episode:     +{max(gains_pct):.2f}% (${max(all_portfolio_values):,.0f})")
        print(f"   Worst Episode:    {min(gains_pct):+.2f}% (${min(all_portfolio_values):,.0f})")
        print(f"   Win Rate:         {win_rate*100:.1f}% ({sum(1 for g in gains_pct if g > 0)}/{len(gains_pct)} profitable)")
        print(f"   Sharpe Ratio:     {val_metrics['sharpe_ratio']:.4f}")
        print(f"   Mean Steps:       {val_metrics['mean_steps']:.1f}")
        print(f"{'='*100}\n")

        self.q_network.train()  # Set back to train mode

        return val_metrics

    def save_checkpoint(self, path: str, val_metrics: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode,
            'global_step': self.global_step,
            'epsilon': self.epsilon,
            'config': self.config
        }
        if val_metrics is not None:
            checkpoint['val_metrics'] = val_metrics
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode = checkpoint['episode']
        self.global_step = checkpoint['global_step']
        self.epsilon = checkpoint['epsilon']
        print(f"Checkpoint loaded: {path} (Episode {self.episode})")


if __name__ == '__main__':
    import argparse
    import h5py
    from datetime import datetime, timedelta
    from inference.backtest_simulation import DatasetLoader
    from rl.rl_components import TradingAgent
    from rl.rl_environment import TradingEnvironment

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DQN agent for stock trading')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes to train')
    parser.add_argument('--parallel', type=int, default=32, help='Number of parallel environments')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--updates-per-step', type=int, default=4, help='Gradient updates per environment step')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epsilon-decay', type=int, default=10000, help='Episodes to decay epsilon')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient (tau)')
    parser.add_argument('--n-step', type=int, default=10, help='N-step return lookahead')
    parser.add_argument('--reward-scale', type=float, default=1.0, help='Reward scaling factor')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device (cuda or cpu)')
    parser.add_argument('--wandb', action='store_true', default=True, help='Enable WandB logging')
    parser.add_argument('--dataset-path', type=str, default='data/all_complete_dataset.h5', help='Path to dataset HDF5 file')
    parser.add_argument('--prices-path', type=str, default='data/actual_prices_clean.h5', help='Path to prices HDF5 (use _clean for validated data)')
    parser.add_argument('--cache-file', type=str, default='data/rl_stock_selections_4yr.h5', help='Path to stock selections cache')
    parser.add_argument('--feature-cache', type=str, default='data/rl_feature_cache_4yr.h5', help='Path to feature cache')
    parser.add_argument('--state-cache', type=str, default='data/rl_state_cache_4yr.h5', help='Path to state cache')
    parser.add_argument('--predictor-checkpoint', type=str, default='./checkpoints/best_model_100m_1.18.pt', help='Path to predictor checkpoint')
    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print(f"\n{'='*80}")
    print("STARTING DQN TRAINING")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes:,}")
    print(f"Parallel envs: {args.parallel}")
    print(f"Batch size: {args.batch_size}")
    print(f"Updates per step: {args.updates_per_step}")
    print(f"Learning rate: {args.lr} (with cosine annealing)")
    print(f"N-step returns: {args.n_step} (looks {args.n_step} steps ahead)")
    print(f"Reward scale: {args.reward_scale}")
    print(f"Epsilon decay: {args.epsilon_decay} episodes")
    print(f"Target update: Soft (Polyak) with τ={args.tau}")
    print(f"No-position rule: HOLD disabled when not holding stocks")
    print(f"WandB: {'enabled' if args.wandb else 'disabled'}")
    print(f"{'='*80}\n")

    # Config
    allow_short = False  # DISABLE SHORT SELLING (prevents negative portfolio values)

    # State dimension depends on whether short selling is enabled:
    # - With shorts: 4 longs × 1469 + 4 shorts × 1469 + 9 position = 11761
    # - Without shorts: 4 longs × 1469 + 5 position = 5881
    state_dim = 11761 if allow_short else 5881

    config = {
        'device': args.device,
        'state_dim': state_dim,
        'hidden_dim': 4096,
        'learning_rate': args.lr,
        'weight_decay': 1e-2,
        'buffer_capacity': 100000,
        'batch_size': args.batch_size,
        'gamma': 0.99,
        'n_step': args.n_step,  # N-step return lookahead
        'use_per': False,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_episodes': args.epsilon_decay,
        'use_soft_update': True,  # Use soft (Polyak) target updates
        'tau': args.tau,  # Soft update coefficient
        'target_update_freq': 1000,  # Only used if use_soft_update=False
        'updates_per_step': args.updates_per_step,
        'min_buffer_size': 1000,
        'num_episodes': args.episodes,
        'num_parallel_envs': args.parallel,
        'initial_capital': 100000,
        'reward_scale': args.reward_scale,
        'use_wandb': args.wandb,
        'save_freq': 100,
        'top_k_per_horizon': 50,
        'allow_short': allow_short
    }

    # Initialize WandB if enabled
    if args.wandb:
        wandb.init(
            project='stock-trading-dqn',
            config=config,
            name=f'dqn_{args.parallel}env_{args.batch_size}bs'
        )

    # Load data
    print("Loading dataset...")
    data_loader = DatasetLoader(
        dataset_path=args.dataset_path,
        prices_path=args.prices_path,
        num_test_stocks=100
    )

    # Count stocks
    all_tickers = list(data_loader.h5_file.keys()) if data_loader.is_hdf5 else list(data_loader.data.keys())
    stocks_with_prices = sum(1 for ticker in all_tickers if ticker in data_loader.prices_file)
    print(f"  ✅ Loaded {len(all_tickers)} total stocks")
    print(f"  ✅ {stocks_with_prices} stocks have price data")

    # Load feature cache
    print("\nLoading feature cache...")
    if os.path.exists(args.feature_cache):
        print(f"  ⚡ Loading from cache: {args.feature_cache}")
        data_loader.load_feature_cache(args.feature_cache)
        print(f"  ✅ Feature cache loaded!")
    else:
        print(f"  ⚠️  Feature cache not found: {args.feature_cache}")
        print(f"  ⚠️  Run 'python rl/precompute_stock_selections.py' to create it")
        print(f"  ⚠️  Training will be MUCH slower without the cache!")
        # Could fallback to preloading, but it's better to tell user to create cache
        raise FileNotFoundError(f"Feature cache not found: {args.feature_cache}")

    # Create a dummy agent for feature extraction (required by environment)
    # We won't use this agent for action selection - DQN network handles that
    print("Creating feature extraction agent...")
    dummy_agent = TradingAgent(
        predictor_checkpoint_path=args.predictor_checkpoint,
        state_dim=1469,
        hidden_dim=512,
        action_dim=5  # Doesn't matter, we use DQN network
    ).to(config['device'])

    # CRITICAL: Set to training mode so feature extraction doesn't disable gradients globally!
    # The feature extractor uses: with torch.set_grad_enabled(self.predictor.training)
    # So we need the predictor in training mode, even though we won't train it
    dummy_agent.train()

    # Create proper train/validation/test split
    print("\nCreating train/validation/test split...")
    sample_ticker = list(data_loader.prices_file.keys())[0]
    prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
    all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])

    # Define split dates
    # Train: 2020-01-01 to 2022-12-31 (3 years)
    # Val:   2023-01-01 to 2023-06-30 (6 months)
    # Test:  2023-07-01 to present (held out for final evaluation)
    train_start = '2020-01-01'
    train_end = '2022-12-31'
    val_start = '2023-01-01'
    val_end = '2023-06-30'

    train_days = [d for d in all_trading_days if train_start <= d <= train_end]
    val_days = [d for d in all_trading_days if val_start <= d <= val_end]
    test_days = [d for d in all_trading_days if d > val_end]

    print(f"  Train: {train_start} to {train_end} ({len(train_days)} days)")
    print(f"  Val:   {val_start} to {val_end} ({len(val_days)} days)")
    print(f"  Test:  {val_end} to present ({len(test_days)} days) [held out]")
    print(f"  ✅ Using {len(train_days)} training days with {len(val_days)} validation days")

    # Create temporary environment to load state cache with ALL trading days
    # The vectorized envs will have filters to limit episode sampling to train/val only
    print("\nCreating temporary environment for state cache...")
    temp_env = TradingEnvironment(
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        device=config['device'],
        trading_days_filter=all_trading_days,  # Use all available trading days
        top_k_per_horizon=config['top_k_per_horizon']
    )

    # Load or precompute state cache
    print("\nLoading state cache...")
    if os.path.exists(args.state_cache):
        print(f"  ⚡ Loading from cache: {args.state_cache}")
        temp_env.load_state_cache(args.state_cache)
        print(f"  ✅ State cache loaded ({len(temp_env.state_cache)} dates)")

        # Check what dates are actually in the cache
        cached_dates = sorted(temp_env.state_cache.keys())
        cache_start = cached_dates[0] if cached_dates else "N/A"
        cache_end = cached_dates[-1] if cached_dates else "N/A"
        print(f"  📅 Cache date range: {cache_start} to {cache_end}")

        # Filter train/val days to only include dates that exist in the cache
        cached_dates_set = set(cached_dates)
        train_days_original = len(train_days)
        val_days_original = len(val_days)
        train_days = [d for d in train_days if d in cached_dates_set]
        val_days = [d for d in val_days if d in cached_dates_set]

        if len(train_days) < train_days_original:
            print(f"  ⚠️  Warning: {train_days_original - len(train_days)} training days not in cache")
        if len(val_days) < val_days_original:
            print(f"  ⚠️  Warning: {val_days_original - len(val_days)} validation days not in cache")

        print(f"  ✅ Using {len(train_days)} training days, {len(val_days)} validation days (filtered to cache)")
    else:
        print(f"  Computing states for all trading days...")
        temp_env._precompute_all_states()
        print(f"  Saving cache to: {args.state_cache}")
        temp_env.save_state_cache(args.state_cache)
        print(f"  ✅ State cache saved ({len(temp_env.state_cache)} dates)")

    # Initialize DQN trainer
    print("\nInitializing DQN trainer...")
    trainer = SimpleDQNTrainer(config)

    # Create vectorized environment for TRAINING (uses filtered train_days)
    print(f"\nCreating GPU-vectorized TRAINING environment ({args.parallel} parallel envs, {len(train_days)} days)...")
    vec_env = GPUVectorizedTradingEnv(
        num_envs=args.parallel,
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        transaction_cost=0.0,  # No transaction costs
        device=config['device'],
        trading_days_filter=train_days,  # Filtered training days (only those in cache)
        top_k_per_horizon=config['top_k_per_horizon']
    )

    # Share state cache and price cache from temp_env to vectorized env
    vec_env.ref_env.state_cache = temp_env.state_cache
    vec_env.ref_env.price_cache = temp_env.price_cache
    print(f"  ✅ Training environment ready (shared state cache: {len(temp_env.state_cache)} dates)")

    # Create validation environment (uses filtered val_days)
    print(f"\nCreating validation environment ({len(val_days)} validation days)...")
    val_vec_env = GPUVectorizedTradingEnv(
        num_envs=min(args.parallel, 8),  # Use fewer envs for validation (faster)
        data_loader=data_loader,
        agent=dummy_agent,
        initial_capital=config['initial_capital'],
        max_positions=1,
        episode_length=30,
        transaction_cost=0.0,
        device=config['device'],
        trading_days_filter=val_days,  # Filtered validation days (only those in cache)
        top_k_per_horizon=config['top_k_per_horizon']
    )

    # Share state cache and price cache
    val_vec_env.ref_env.state_cache = temp_env.state_cache
    val_vec_env.ref_env.price_cache = temp_env.price_cache
    print(f"  ✅ Validation environment ready")

    # Load GPU stock selection cache
    print("\nLoading GPU stock selection cache...")
    with h5py.File(args.prices_path, 'r') as f:
        prices_file_keys = list(f.keys())

    stock_cache = GPUStockSelectionCache(
        h5_path=args.cache_file,
        prices_file_keys=prices_file_keys,
        device=config['device']
    )

    # Start training with validation
    print("\nStarting training...\n")
    try:
        trainer.train(
            vec_env=vec_env,
            stock_selections_cache=stock_cache,
            use_precomputed_selections=True,
            val_env=val_vec_env,
            val_stock_cache=stock_cache,  # Same cache, but val_env uses val_days
            val_freq=100,  # Run validation every 100 iterations
            val_episodes=32,  # Number of validation episodes
            early_stopping_patience=10  # Stop if no improvement for 10 validations
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        os.makedirs("checkpoints", exist_ok=True)
        trainer.save_checkpoint(f"checkpoints/dqn_checkpoint_interrupted_ep{trainer.episode}.pt")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total episodes: {trainer.episode}")
    print(f"Final epsilon: {trainer.epsilon:.4f}")
    print(f"Buffer size: {len(trainer.buffer):,}")

    # Save final checkpoint
    final_checkpoint_path = f"checkpoints/dqn_final_ep{trainer.episode}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_checkpoint(final_checkpoint_path)

    if args.wandb:
        wandb.finish()
