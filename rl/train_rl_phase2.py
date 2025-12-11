"""
RL Trading Agent Training Script - Phase 2

Phase 2: Full action space with dynamic exit timing
- 5 actions: HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL
- Agent learns when to exit (no fixed holding period)
- 1,000 stocks (full test set)
- Frozen predictor (can enable joint training with flag)

Key differences from Phase 1:
- Uses all 5 actions instead of 2
- No auto-sell - agent decides when to exit
- More stocks for better generalization
- Optional position sizing learning
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


class Phase2TrainingLoop:
    """
    Phase 2: Full action space training loop.

    Enhancements over Phase 1:
    - Full 5-action space (HOLD, BUY_SMALL/MED/LARGE, SELL)
    - Dynamic exit timing (agent learns when to sell)
    - More stocks (1,000 instead of 100)
    - Optional risk management (position limits, diversification)
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
                config=config,
                tags=['phase2', 'full-actions']
            )

        # Load data
        print("\n" + "="*80)
        print("PHASE 2: FULL ACTION SPACE TRAINING")
        print("="*80)
        print("\nInitializing data loader...")
        self.data_loader = self._load_data()

        # Subsample stocks if requested (Phase 2 uses more stocks than Phase 1)
        if config.get('num_stocks') and config['num_stocks'] < len(self.data_loader.test_tickers):
            print(f"\n📊 Subsampling {config['num_stocks']} stocks from universe...")
            original_tickers = self.data_loader.test_tickers
            self.data_loader.test_tickers = random.sample(
                original_tickers,
                config['num_stocks']
            )
            print(f"   Selected {len(self.data_loader.test_tickers)} stocks")
        else:
            print(f"\n📊 Using all {len(self.data_loader.test_tickers)} test stocks")

        # Initialize agent
        print("\n🤖 Initializing RL agent...")
        self.agent = TradingAgent(
            predictor_checkpoint_path=config['predictor_checkpoint'],
            state_dim=config['state_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=5  # Phase 2: Full 5 actions
        ).to(self.device)

        # Freeze/unfreeze predictor
        if config.get('freeze_predictor', True):
            self.agent.feature_extractor.freeze_predictor()
        else:
            self.agent.feature_extractor.unfreeze_predictor()
            print("🔥 Joint training enabled (predictor unfrozen)")

        # Initialize environment
        print("\n🏪 Initializing trading environment...")
        self.env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            episode_length=config['episode_length'],
            transaction_cost=config.get('transaction_cost', 0.001),
            device=self.device
        )

        # Initialize replay buffer
        self.buffer = ReplayBuffer(capacity=config['buffer_capacity'])

        # Initialize optimizer
        if config.get('freeze_predictor', True):
            # Q-network only
            self.optimizer = optim.Adam(
                self.agent.q_network.parameters(),
                lr=config['lr_q_network']
            )
        else:
            # Joint training: Q-network + predictor
            self.optimizer = optim.Adam([
                {'params': self.agent.q_network.parameters(), 'lr': config['lr_q_network']},
                {'params': self.agent.feature_extractor.predictor.parameters(), 'lr': config.get('lr_predictor', 1e-5)}
            ])

        # Training state
        self.global_step = 0
        self.epsilon = config['epsilon_start']
        self.episode_rewards = []
        self.best_avg_reward = -float('inf')

        print("\n✅ Initialization complete!")
        print(f"\nConfiguration:")
        print(f"  Stocks: {len(self.data_loader.test_tickers)}")
        print(f"  Actions: 5 (HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL)")
        print(f"  Exit timing: DYNAMIC (agent learns when to sell)")
        print(f"  Episodes: {config['num_episodes']}")
        print(f"  Episode length: {config['episode_length']} days")
        print(f"  Initial capital: ${config['initial_capital']:,.0f}")
        print(f"  Max positions: {config['max_positions']}")
        print(f"  Predictor: {'FROZEN' if config.get('freeze_predictor', True) else 'JOINT TRAINING'}")

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
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")

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

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
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

        while not done:
            # Select actions (Phase 2: full 5-action space)
            actions = self._select_actions_phase2(states)

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

    def _select_actions_phase2(self, states: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """
        Select actions for Phase 2 (full 5-action space).

        Phase 2 actions:
        - 0: HOLD - Do nothing
        - 1: BUY_SMALL - Allocate 25% of available capital
        - 2: BUY_MEDIUM - Allocate 50% of available capital
        - 3: BUY_LARGE - Allocate 100% of available capital
        - 4: SELL - Close position

        Args:
            states: Dictionary mapping ticker -> state tensor

        Returns:
            Dictionary mapping ticker -> action_id (0-4)
        """
        if len(states) == 0:
            return {}

        # Stack states for batch inference
        tickers = list(states.keys())
        state_tensors = torch.stack([states[ticker] for ticker in tickers])

        # Get Q-values for all stocks
        with torch.no_grad():
            q_values_batch = self.agent.q_network(state_tensors)  # (num_stocks, 5)

        # Epsilon-greedy action selection
        actions = {}
        for i, ticker in enumerate(tickers):
            if random.random() < self.epsilon:
                # Explore: random action
                action = random.randint(0, 4)
            else:
                # Exploit: best action
                action = q_values_batch[i].argmax().item()

            # Apply constraints based on current state
            action = self._apply_action_constraints(ticker, action, q_values_batch[i])

            if action is not None:
                actions[ticker] = action

        return actions

    def _apply_action_constraints(self, ticker: str, action: int, q_values: torch.Tensor) -> int:
        """
        Apply portfolio constraints to actions.

        Args:
            ticker: Stock ticker
            action: Proposed action
            q_values: Q-values for all actions

        Returns:
            Valid action or None if no valid action
        """
        # Check if we have a position
        has_position = ticker in self.env.portfolio

        # Constraint 1: Can't buy if already have position
        if action in [1, 2, 3] and has_position:
            # Find next best action (either HOLD or SELL)
            valid_actions = [0, 4]  # HOLD, SELL
            valid_q_values = [q_values[a].item() for a in valid_actions]
            action = valid_actions[np.argmax(valid_q_values)]

        # Constraint 2: Can't sell if don't have position
        if action == 4 and not has_position:
            return 0  # Default to HOLD

        # Constraint 3: Can't buy if at max positions
        if action in [1, 2, 3]:
            if len(self.env.portfolio) >= self.env.max_positions:
                return 0  # Default to HOLD

        # Constraint 4: Can't buy if no cash
        if action in [1, 2, 3]:
            if self.env.cash <= 0:
                return 0  # Default to HOLD

        return action

    def _store_transitions(self, states: Dict[str, torch.Tensor],
                          actions: Dict[str, int],
                          reward: float,
                          next_states: Dict[str, torch.Tensor],
                          done: bool):
        """
        Store transitions in replay buffer.

        Phase 2: Store ALL actions including SELL (no filtering like Phase 1).
        """
        for ticker in actions.keys():
            if ticker in states and ticker in next_states:
                self.buffer.push(
                    state=states[ticker],
                    action=actions[ticker],
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 1.0)
        if not self.config.get('freeze_predictor', True):
            torch.nn.utils.clip_grad_norm_(self.agent.feature_extractor.predictor.parameters(), 1.0)

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
        print(f"\n{'='*80}")
        print(f"EVALUATION at Episode {episode + 1}")
        print(f"{'='*80}")

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
                actions = self._select_actions_phase2(states)
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
        avg_max_dd = np.mean([s['max_drawdown'] for s in eval_stats])

        print(f"\nEvaluation Results ({self.config['eval_episodes']} episodes):")
        print(f"  Avg Reward:       {avg_eval_reward:.4f}")
        print(f"  Avg Return:       {avg_return*100:.2f}%")
        print(f"  Avg Sharpe:       {avg_sharpe:.2f}")
        print(f"  Avg Max Drawdown: {avg_max_dd*100:.2f}%")
        print(f"  Avg Win Rate:     {avg_win_rate*100:.1f}%")

        if WANDB_AVAILABLE and self.config['use_wandb']:
            wandb.log({
                'eval/avg_reward': avg_eval_reward,
                'eval/avg_return': avg_return,
                'eval/avg_sharpe': avg_sharpe,
                'eval/avg_max_drawdown': avg_max_dd,
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

        checkpoint = {
            'episode': episode + 1,
            'global_step': self.global_step,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_avg_reward': self.best_avg_reward,
            'config': self.config
        }

        # Add predictor state if joint training
        if not self.config.get('freeze_predictor', True):
            checkpoint['predictor_state_dict'] = self.agent.feature_extractor.predictor.state_dict()

        torch.save(checkpoint, checkpoint_path)

        print(f"  💾 Checkpoint saved: {checkpoint_path}")

    def _final_evaluation(self):
        """Final evaluation after training."""
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)

        self._evaluate(self.config['num_episodes'] - 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train RL trading agent (Phase 2)')

    # Data
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset (HDF5 or pickle)')
    parser.add_argument('--prices-path', type=str, help='Path to prices HDF5 (if features are normalized)')
    parser.add_argument('--num-test-stocks', type=int, default=1000, help='Number of test stocks')
    parser.add_argument('--predictor-checkpoint', type=str, required=True, help='Path to predictor checkpoint')

    # Phase 2 settings
    parser.add_argument('--num-stocks', type=int, default=1000, help='Number of stocks to use (Phase 2: 1000)')
    parser.add_argument('--freeze-predictor', type=bool, default=True, help='Freeze predictor (False for joint training)')

    # Training
    parser.add_argument('--num-episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=60, help='Episode length (days)')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--buffer-capacity', type=int, default=200000, help='Replay buffer capacity')
    parser.add_argument('--lr-q-network', type=float, default=5e-5, help='Learning rate for Q-network')
    parser.add_argument('--lr-predictor', type=float, default=1e-6, help='Learning rate for predictor (joint training)')

    # RL hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft target update rate')
    parser.add_argument('--epsilon-start', type=float, default=0.5, help='Initial epsilon (Phase 2: lower start)')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay-steps', type=int, default=100000, help='Epsilon decay steps')

    # Environment
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=20, help='Max simultaneous positions (Phase 2: 20)')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost (0.1%)')

    # Architecture
    parser.add_argument('--state-dim', type=int, default=1920, help='State dimension')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden dimension')

    # Logging
    parser.add_argument('--log-frequency', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--eval-frequency', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of eval episodes')
    parser.add_argument('--checkpoint-frequency', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--checkpoint-dir', type=str, default='./rl_checkpoints_phase2', help='Checkpoint directory')

    # W&B
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='rl-trading-phase2', help='W&B project name')
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
    trainer = Phase2TrainingLoop(config)
    trainer.train()


if __name__ == '__main__':
    main()
