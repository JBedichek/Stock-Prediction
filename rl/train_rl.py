"""
RL Trading Agent Training Script

Trains a DQN agent to learn stock trading strategies.

Phase 1 (Simplified - this file):
- 2 actions: HOLD or BUY_FULL
- Fixed exit: 5-day holding period
- 100 stocks subsample
- Frozen predictor

Phase 2 (future):
- 5 actions: HOLD, BUY_SMALL/MED/LARGE, SELL
- Dynamic exit timing
- Full stock universe

Phase 3 (future):
- Joint training with predictor
"""

import torch
import torch.optim as optim
import argparse
import random
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, ReplayBuffer, compute_dqn_loss, ACTIONS
from rl.rl_environment import TradingEnvironment
from inference.backtest_simulation import DatasetLoader

# Optional: W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not available. Install with: pip install wandb")


class Phase1TrainingLoop:
    """
    Phase 1: Simplified training loop for proof of concept.

    Simplifications:
    - 2 actions only: HOLD (0) or BUY_FULL (maps to action 3)
    - Fixed 5-day holding period (auto-sell after 5 days)
    - 100 stocks subsample
    - Frozen predictor
    """

    def __init__(self, config: Dict):
        """
        Initialize training loop.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config['device']

        # Set random seeds for reproducibility
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

        # Initialize W&B if available and requested
        if WANDB_AVAILABLE and config['use_wandb']:
            wandb.init(
                project=config['wandb_project'],
                name=config['run_name'],
                config=config
            )

        # Load data
        print("\n" + "="*60)
        print("PHASE 1: SIMPLIFIED RL TRAINING (PROOF OF CONCEPT)")
        print("="*60)
        print("\nInitializing data loader...")
        self.data_loader = self._load_data()

        # Subsample stocks for Phase 1
        print(f"\n📊 Subsampling {config['num_stocks']} stocks from universe...")
        original_tickers = self.data_loader.test_tickers
        self.data_loader.test_tickers = random.sample(
            original_tickers,
            min(config['num_stocks'], len(original_tickers))
        )
        print(f"   Selected {len(self.data_loader.test_tickers)} stocks")

        # Initialize agent
        print("\n🤖 Initializing RL agent...")
        self.agent = TradingAgent(
            predictor_checkpoint_path=config['predictor_checkpoint'],
            state_dim=config['state_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=2  # Phase 1: Only 2 actions
        ).to(self.device)

        # Freeze predictor (Phase 1)
        self.agent.feature_extractor.freeze_predictor()

        # Initialize environment
        print("\n🏪 Initializing trading environment...")
        self.env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            episode_length=config['episode_length'],
            device=self.device
        )

        # Initialize replay buffer
        self.buffer = ReplayBuffer(capacity=config['buffer_capacity'])

        # Initialize optimizer (Q-network only in Phase 1)
        self.optimizer = optim.Adam(
            self.agent.q_network.parameters(),
            lr=config['lr_q_network']
        )

        # Training state
        self.global_step = 0
        self.epsilon = config['epsilon_start']
        self.episode_rewards = []
        self.best_avg_reward = -float('inf')

        # Phase 1 specific: Track positions for auto-sell
        self.position_entry_dates = {}  # ticker -> entry_step

        print("\n✅ Initialization complete!")
        print(f"\nConfiguration:")
        print(f"  Stocks: {len(self.data_loader.test_tickers)}")
        print(f"  Actions: 2 (HOLD, BUY_FULL)")
        print(f"  Holding period: {config['holding_period']} days (auto-sell)")
        print(f"  Episodes: {config['num_episodes']}")
        print(f"  Episode length: {config['episode_length']} days")
        print(f"  Initial capital: ${config['initial_capital']:,.0f}")
        print(f"  Predictor: FROZEN")

    def _load_data(self) -> DatasetLoader:
        """Load dataset using existing infrastructure."""
        data_loader = DatasetLoader(
            dataset_path=self.config['dataset_path'],
            num_test_stocks=self.config.get('num_test_stocks', 1000),
            prices_path=self.config.get('prices_path')
        )
        print(f"✅ Data loaded:")
        print(f"   Stocks: {len(data_loader.test_tickers)}")
        print(f"   Dates: {len(data_loader.test_dates)}")
        return data_loader

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60 + "\n")

        for episode in range(self.config['num_episodes']):
            episode_reward = self._run_episode(episode)
            self.episode_rewards.append(episode_reward)

            # Logging
            if (episode + 1) % self.config['log_frequency'] == 0:
                self._log_progress(episode)

            # Evaluation
            if (episode + 1) % self.config['eval_frequency'] == 0:
                self._evaluate(episode)

            # Checkpoint
            if (episode + 1) % self.config['checkpoint_frequency'] == 0:
                self._save_checkpoint(episode)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        self._final_evaluation()

    def _run_episode(self, episode: int) -> float:
        """
        Run a single episode.

        Args:
            episode: Episode number

        Returns:
            Total episode reward
        """
        # Reset environment
        states = self.env.reset()
        done = False
        episode_reward = 0.0
        self.position_entry_dates = {}  # Reset position tracking

        while not done:
            # Select actions (Phase 1: only HOLD or BUY_FULL)
            actions = self._select_actions_phase1(states)

            # Auto-sell positions held for 5 days (Phase 1 constraint)
            actions = self._add_auto_sell_actions(actions)

            # Execute actions
            next_states, reward, done, info = self.env.step(actions)

            # Store transitions in replay buffer
            self._store_transitions(states, actions, reward, next_states, done)

            # Train Q-network
            if len(self.buffer) >= self.config['batch_size'] and \
               self.global_step % self.config['train_frequency'] == 0:
                loss = self._train_step()

                if WANDB_AVAILABLE and self.config['use_wandb']:
                    wandb.log({'loss': loss, 'global_step': self.global_step})

            # Update target network
            if self.global_step % self.config['target_update_frequency'] == 0:
                self.agent.update_target_network(tau=self.config['tau'])

            # Decay epsilon
            self._decay_epsilon()

            states = next_states
            episode_reward += reward
            self.global_step += 1

        return episode_reward

    def _select_actions_phase1(self, states: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """
        Select actions for Phase 1 (2 actions only).

        Phase 1 mapping:
        - Agent action 0 (HOLD) -> Environment action 0 (HOLD)
        - Agent action 1 (BUY) -> Environment action 3 (BUY_LARGE)

        Args:
            states: Dictionary mapping ticker -> state tensor

        Returns:
            Dictionary mapping ticker -> environment action id
        """
        # Stack states for batch inference
        tickers = list(states.keys())
        if len(tickers) == 0:
            return {}

        state_tensors = torch.stack([states[ticker] for ticker in tickers])

        # Get Q-values for all stocks
        with torch.no_grad():
            q_values_batch = self.agent.q_network(state_tensors)  # (num_stocks, 2)

        # Epsilon-greedy action selection
        actions_phase1 = {}
        for i, ticker in enumerate(tickers):
            if random.random() < self.epsilon:
                # Explore: random action (0 or 1)
                action = random.randint(0, 1)
            else:
                # Exploit: best action
                action = q_values_batch[i].argmax().item()

            actions_phase1[ticker] = action

        # Filter to top-K buy actions (Phase 1: limit concurrent positions)
        actions_env = self._filter_to_top_k(actions_phase1, q_values_batch, tickers)

        return actions_env

    def _filter_to_top_k(self, actions_phase1: Dict[str, int],
                        q_values_batch: torch.Tensor,
                        tickers: List[str]) -> Dict[str, int]:
        """
        Filter actions to top-K buys based on Q-values.

        Args:
            actions_phase1: Agent actions (0 or 1)
            q_values_batch: Q-values for all stocks
            tickers: List of tickers

        Returns:
            Environment actions (0 or 3)
        """
        # Separate buy and hold actions
        buy_candidates = []

        for i, ticker in enumerate(tickers):
            action = actions_phase1[ticker]

            if action == 1:  # BUY
                # Check constraints
                if ticker not in self.env.portfolio and len(self.env.portfolio) < self.env.max_positions:
                    q_value_buy = q_values_batch[i, 1].item()
                    buy_candidates.append((ticker, q_value_buy))

        # Keep top-K buys
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_buys = buy_candidates[:self.config['top_k_buys']]

        # Map to environment actions
        actions_env = {}
        for ticker, _ in top_k_buys:
            actions_env[ticker] = 3  # BUY_LARGE in environment

        return actions_env

    def _add_auto_sell_actions(self, actions: Dict[str, int]) -> Dict[str, int]:
        """
        Add auto-sell actions for positions held >= 5 days (Phase 1 constraint).

        Args:
            actions: Current actions

        Returns:
            Actions with auto-sells added
        """
        holding_period = self.config['holding_period']

        for ticker, pos in self.env.portfolio.items():
            if pos['days_held'] >= holding_period:
                # Auto-sell after holding period
                actions[ticker] = 4  # SELL

        return actions

    def _store_transitions(self, states: Dict[str, torch.Tensor],
                          actions: Dict[str, int],
                          reward: float,
                          next_states: Dict[str, torch.Tensor],
                          done: bool):
        """Store transitions in replay buffer."""
        for ticker in actions.keys():
            if ticker in states and ticker in next_states:
                # Map environment action back to agent action (for learning)
                env_action = actions[ticker]
                if env_action == 0:
                    agent_action = 0  # HOLD
                elif env_action == 3:
                    agent_action = 1  # BUY
                elif env_action == 4:
                    # Auto-sell - treat as HOLD for learning (not agent's decision)
                    continue
                else:
                    continue

                self.buffer.push(
                    state=states[ticker],
                    action=agent_action,  # Store agent action (0 or 1)
                    reward=reward,
                    next_state=next_states[ticker],
                    done=done,
                    ticker=ticker
                )

    def _train_step(self) -> float:
        """Perform one training step."""
        batch = self.buffer.sample(self.config['batch_size'])

        loss = compute_dqn_loss(
            agent=self.agent,
            batch=batch,
            gamma=self.config['gamma'],
            device=self.device
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon - (self.config['epsilon_start'] - self.config['epsilon_end']) / self.config['epsilon_decay_steps']
        )

    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_rewards = self.episode_rewards[-self.config['log_frequency']:]
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        buffer_stats = self.buffer.get_stats()

        print(f"\nEpisode {episode + 1}/{self.config['num_episodes']}")
        print(f"  Avg Reward (last {self.config['log_frequency']}): {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"  Epsilon: {self.epsilon:.3f}")
        print(f"  Buffer: {buffer_stats['size']}/{self.config['buffer_capacity']} ({buffer_stats['capacity_used']*100:.1f}%)")
        print(f"  Global Step: {self.global_step}")

        if WANDB_AVAILABLE and self.config['use_wandb']:
            wandb.log({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'epsilon': self.epsilon,
                'buffer_size': buffer_stats['size'],
                'global_step': self.global_step
            })

    def _evaluate(self, episode: int):
        """Run evaluation episodes (epsilon=0)."""
        print(f"\n{'='*60}")
        print(f"EVALUATION at Episode {episode + 1}")
        print(f"{'='*60}")

        eval_rewards = []
        eval_stats = []

        for eval_ep in range(self.config['eval_episodes']):
            # Run episode with epsilon=0 (greedy)
            old_epsilon = self.epsilon
            self.epsilon = 0.0

            states = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                actions = self._select_actions_phase1(states)
                actions = self._add_auto_sell_actions(actions)
                next_states, reward, done, info = self.env.step(actions)
                states = next_states
                episode_reward += reward

            eval_rewards.append(episode_reward)
            eval_stats.append(self.env.get_episode_stats())

            self.epsilon = old_epsilon

        # Compute statistics
        avg_eval_reward = np.mean(eval_rewards)
        avg_return = np.mean([s['total_return'] for s in eval_stats])
        avg_sharpe = np.mean([s['sharpe_ratio'] for s in eval_stats])
        avg_win_rate = np.mean([s['win_rate'] for s in eval_stats])

        print(f"\nEvaluation Results ({self.config['eval_episodes']} episodes):")
        print(f"  Avg Reward: {avg_eval_reward:.4f}")
        print(f"  Avg Return: {avg_return*100:.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Avg Win Rate: {avg_win_rate*100:.1f}%")

        if WANDB_AVAILABLE and self.config['use_wandb']:
            wandb.log({
                'eval/avg_reward': avg_eval_reward,
                'eval/avg_return': avg_return,
                'eval/avg_sharpe': avg_sharpe,
                'eval/win_rate': avg_win_rate,
                'episode': episode + 1
            })

        # Save best model
        if avg_eval_reward > self.best_avg_reward:
            self.best_avg_reward = avg_eval_reward
            self._save_checkpoint(episode, prefix='best')
            print(f"  🎉 New best model! (Reward: {avg_eval_reward:.4f})")

    def _save_checkpoint(self, episode: int, prefix: str = 'checkpoint'):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'{prefix}_episode_{episode + 1}.pt'

        torch.save({
            'episode': episode + 1,
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_avg_reward': self.best_avg_reward,
            'config': self.config
        }, checkpoint_path)

        print(f"  💾 Checkpoint saved: {checkpoint_path}")

    def _final_evaluation(self):
        """Final evaluation after training."""
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)

        self._evaluate(self.config['num_episodes'] - 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train RL trading agent (Phase 1)')

    # Data
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset (HDF5 or pickle)')
    parser.add_argument('--prices-path', type=str, help='Path to prices HDF5 (if features are normalized)')
    parser.add_argument('--num-test-stocks', type=int, default=1000, help='Number of test stocks')
    parser.add_argument('--predictor-checkpoint', type=str, required=True, help='Path to predictor checkpoint')

    # Phase 1 settings
    parser.add_argument('--num-stocks', type=int, default=100, help='Number of stocks (Phase 1 subsample)')
    parser.add_argument('--holding-period', type=int, default=5, help='Fixed holding period (days)')

    # Training
    parser.add_argument('--num-episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=40, help='Episode length (days)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--buffer-capacity', type=int, default=100000, help='Replay buffer capacity')
    parser.add_argument('--lr-q-network', type=float, default=1e-4, help='Learning rate for Q-network')

    # RL hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft target update rate')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='Final epsilon')
    parser.add_argument('--epsilon-decay-steps', type=int, default=50000, help='Epsilon decay steps')

    # Environment
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=10, help='Max simultaneous positions')
    parser.add_argument('--top-k-buys', type=int, default=10, help='Top-K buy actions per step')

    # Architecture
    parser.add_argument('--state-dim', type=int, default=1920, help='State dimension')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden dimension')

    # Logging
    parser.add_argument('--log-frequency', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--eval-frequency', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of eval episodes')
    parser.add_argument('--checkpoint-frequency', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--checkpoint-dir', type=str, default='./rl_checkpoints', help='Checkpoint directory')

    # W&B
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='rl-trading-phase1', help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None, help='W&B run name')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-frequency', type=int, default=4, help='Train every N steps')
    parser.add_argument('--target-update-frequency', type=int, default=1000, help='Update target every N steps')

    args = parser.parse_args()

    # Convert to config dict
    config = vars(args)

    # Train
    trainer = Phase1TrainingLoop(config)
    trainer.train()


if __name__ == '__main__':
    main()
