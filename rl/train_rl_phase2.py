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
import pickle
import pandas as pd
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

# CRITICAL: Re-enable gradients for RL training
# backtest_simulation.py disables gradients globally for inference,
# but we need them for RL training
torch.set_grad_enabled(True)

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
            action_dim=3  # Simplified: HOLD, BUY, SELL
        ).to(self.device)

        # Freeze/unfreeze predictor
        if config.get('freeze_predictor', True):
            self.agent.feature_extractor.freeze_predictor()
        else:
            self.agent.feature_extractor.unfreeze_predictor()
            print("🔥 Joint training enabled (predictor unfrozen)")

        # Set Q-network to training mode
        self.agent.q_network.train()
        print("🎯 Q-network set to training mode")

        # CRITICAL: Preload features for LAST 4 YEARS only (much faster!)
        # This eliminates HDF5 reads during state precomputation (MASSIVE speedup!)
        print("\n💾 Preloading features into RAM cache...")
        print("   📅 Using last 4 years of data only (recent market dynamics)")

        # Get trading days from prices file and filter to last 4 years
        sample_ticker = list(self.data_loader.prices_file.keys())[0]
        prices_dates_bytes = self.data_loader.prices_file[sample_ticker]['dates'][:]
        all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])

        # Filter to last 4 years (~1000 trading days)
        from datetime import datetime, timedelta
        cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]
        print(f"   📊 Filtered from {len(all_trading_days)} to {len(recent_trading_days)} trading days (last 4 years)")

        # Store for later use
        self.recent_trading_days = recent_trading_days

        # Define cache path (include date range in filename to avoid conflicts)
        cache_path = f'data/rl_feature_cache_4yr.h5'

        # Try to load from cache first (instant if exists!)
        cache_loaded = False
        if os.path.exists(cache_path):
            print(f"   📂 Found existing cache: {cache_path}")
            print(f"   ⚡ Loading from cache (instant!)...")
            cache_loaded = self.data_loader.load_feature_cache(cache_path)

        # If cache not loaded, preload from HDF5 and save cache
        if not cache_loaded:
            print("   ⚠️  No cache found - preloading from HDF5...")
            print("   This will take ~1 minute but only happens once!")
            self.data_loader.preload_features(recent_trading_days)

            # Save cache for next time
            print(f"\n   💾 Saving cache to: {cache_path}")
            print(f"   (Next run will load instantly!)")
            self.data_loader.save_feature_cache(cache_path)

        # Initialize environment (without precomputation first)
        print("\n🏪 Initializing trading environment...")
        self.env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            episode_length=config['episode_length'],
            transaction_cost=config.get('transaction_cost', 0.001),
            device=self.device,
            precompute_all_states=False,  # We'll handle caching manually below
            trading_days_filter=self.recent_trading_days,  # Use last 4 years only!
            top_k_per_horizon=config.get('top_k_per_horizon', 10)  # Filter to top-10 per horizon
        )
        print(f"   🎯 Action space filtering: Top-{config.get('top_k_per_horizon', 10)} stocks per time horizon")
        print(f"   (Reduces from ~900 stocks to ~{min(40, 4 * config.get('top_k_per_horizon', 10))} stocks per day)")

        # Try to load state cache (skips ~20-min precomputation!)
        state_cache_path = 'data/rl_state_cache_4yr.h5'
        state_cache_loaded = False
        if os.path.exists(state_cache_path):
            print(f"\n💾 Found state cache: {state_cache_path}")
            print(f"   ⚡ Loading precomputed states (instant!)...")
            state_cache_loaded = self.env.load_state_cache(state_cache_path)

        # If cache not loaded, precompute all states and save
        if not state_cache_loaded:
            print("\n⚠️  No state cache found - precomputing states for last 4 years...")
            print(f"   This takes ~20 minutes but only happens once!")
            print(f"   Computing states for {len(self.recent_trading_days)} trading days...")
            self.env._precompute_all_states()

            # Save cache for next time
            print(f"\n💾 Saving state cache to: {state_cache_path}")
            print(f"   (Next run will load instantly!)")
            self.env.save_state_cache(state_cache_path)

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
        self.episode_stats = []  # Track episode statistics (returns, etc.)
        self.best_avg_reward = -float('inf')

        # Metrics tracking (for CSV export)
        self.metrics_history = []  # List of dicts with per-episode metrics

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
        print(f"   Dates: {len(data_loader.all_dates)}")
        return data_loader

    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")

        # Progress bar for episodes
        episodes_pbar = tqdm(range(self.config['num_episodes']), desc="Training Episodes", unit="ep")

        for episode in episodes_pbar:
            episode_reward = self._run_episode(episode)
            episode_stats = self.env.get_episode_stats()

            self.episode_rewards.append(episode_reward)
            self.episode_stats.append(episode_stats)

            # Calculate stats for display
            final_value = episode_stats.get('final_value', self.config['initial_capital'])
            profit = final_value - self.config['initial_capital']
            return_pct = episode_stats.get('total_return', 0.0) * 100

            # Track metrics for CSV export
            episode_metrics = {
                'episode': episode + 1,
                'global_step': self.global_step,
                'reward': episode_reward,
                'return_pct': return_pct,
                'profit': profit,
                'final_value': final_value,
                'sharpe_ratio': episode_stats.get('sharpe_ratio', 0.0),
                'max_drawdown': episode_stats.get('max_drawdown', 0.0),
                'win_rate': episode_stats.get('win_rate', 0.0),
                'num_trades': episode_stats.get('num_trades', 0),
                'epsilon': self.epsilon,
                'buffer_size': len(self.buffer)
            }
            self.metrics_history.append(episode_metrics)

            # Update progress bar with current stats
            episodes_pbar.set_postfix({
                'Return': f'{return_pct:+.2f}%',
                'Profit': f'${profit:+,.0f}',
                'Value': f'${final_value:,.0f}',
                'ε': f'{self.epsilon:.3f}'
            })

            # Print immediate feedback after each episode
            print(f"Episode {episode + 1}: Return={return_pct:+.2f}% | Profit=${profit:+,.0f} | Final=${final_value:,.0f} | ε={self.epsilon:.3f}")

            # Save data every episode
            self._save_episode_data(episode)

            # Logging
            if (episode + 1) % self.config['log_frequency'] == 0:
                self._log_progress(episode)

            # Evaluation
            if (episode + 1) % self.config['eval_frequency'] == 0:
                self._evaluate(episode)

            # Checkpoint
            if (episode + 1) % self.config['checkpoint_frequency'] == 0:
                self._save_checkpoint(episode)

        episodes_pbar.close()

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

        # Progress bar for episode steps
        steps_pbar = tqdm(total=self.config['episode_length'],
                         desc=f"Episode {episode+1}",
                         unit="step",
                         leave=False)

        step = 0
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
            step += 1

            # Update step progress bar
            steps_pbar.update(1)
            steps_pbar.set_postfix({
                'Reward': f'{reward:+.4f}',
                'Value': f'${info.get("portfolio_value", 0):,.0f}',
                'Pos': info.get('num_positions', 0)
            })

        steps_pbar.close()
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
                # Explore: random action (3 actions: HOLD, BUY, SELL)
                action = random.randint(0, 2)
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
        recent_stats = self.episode_stats[-self.config['log_frequency']:]

        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        # Calculate money-based metrics
        avg_return = np.mean([s['total_return'] for s in recent_stats]) * 100
        avg_final_value = np.mean([s['final_value'] for s in recent_stats])
        avg_profit = avg_final_value - self.config['initial_capital']
        avg_win_rate = np.mean([s['win_rate'] for s in recent_stats]) * 100

        buffer_stats = self.buffer.get_stats()

        print(f"\n{'='*80}")
        print(f"PROGRESS: Episode {episode + 1}/{self.config['num_episodes']}")
        print(f"{'='*80}")
        print(f"  Avg Return (last {self.config['log_frequency']}):  {avg_return:+.2f}%")
        print(f"  Avg Profit (last {self.config['log_frequency']}):  ${avg_profit:+,.0f}")
        print(f"  Avg Final Value:              ${avg_final_value:,.0f}")
        print(f"  Avg Win Rate:                 {avg_win_rate:.1f}%")
        print(f"  Avg Reward:                   {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"  Epsilon:                      {self.epsilon:.3f}")
        print(f"  Buffer:                       {buffer_stats['size']}/{self.config['buffer_capacity']} ({buffer_stats['capacity_used']*100:.1f}%)")
        print(f"  Global Step:                  {self.global_step}")

        if WANDB_AVAILABLE and self.config['use_wandb']:
            wandb.log({
                'episode': episode + 1,
                'avg_reward': avg_reward,
                'avg_return': avg_return / 100,
                'avg_profit': avg_profit,
                'avg_final_value': avg_final_value,
                'avg_win_rate': avg_win_rate / 100,
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
        avg_final_value = np.mean([s['final_value'] for s in eval_stats])
        avg_profit = avg_final_value - self.config['initial_capital']
        avg_sharpe = np.mean([s['sharpe_ratio'] for s in eval_stats])
        avg_win_rate = np.mean([s['win_rate'] for s in eval_stats])
        avg_max_dd = np.mean([s['max_drawdown'] for s in eval_stats])

        print(f"\nEvaluation Results ({self.config['eval_episodes']} episodes):")
        print(f"  Avg Return:       {avg_return*100:+.2f}%")
        print(f"  Avg Profit:       ${avg_profit:+,.0f}")
        print(f"  Avg Final Value:  ${avg_final_value:,.0f}")
        print(f"  Avg Reward:       {avg_eval_reward:.4f}")
        print(f"  Avg Sharpe:       {avg_sharpe:.2f}")
        print(f"  Avg Max Drawdown: {avg_max_dd*100:.2f}%")
        print(f"  Avg Win Rate:     {avg_win_rate*100:.1f}%")

        if WANDB_AVAILABLE and self.config['use_wandb']:
            wandb.log({
                'eval/avg_reward': avg_eval_reward,
                'eval/avg_return': avg_return,
                'eval/avg_profit': avg_profit,
                'eval/avg_final_value': avg_final_value,
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

    def _save_episode_data(self, episode: int):
        """Save episode data (replay buffer, history, metrics) every episode."""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save replay buffer
        buffer_path = checkpoint_dir / f'replay_buffer_episode_{episode + 1}.pkl'
        with open(buffer_path, 'wb') as f:
            pickle.dump(self.buffer, f)

        # Save episode history (all episodes so far)
        history_path = checkpoint_dir / f'episode_history.pkl'
        episode_data = {
            'episode_rewards': self.episode_rewards,
            'episode_stats': self.episode_stats
        }
        with open(history_path, 'wb') as f:
            pickle.dump(episode_data, f)

        # Save metrics as CSV (cumulative)
        metrics_path = checkpoint_dir / 'training_metrics.csv'
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(metrics_path, index=False)

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
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_stats': self.episode_stats,
            'metrics_history': self.metrics_history
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
    parser.add_argument('--dataset-path', type=str, default="data/all_complete_dataset.h5", help='Path to dataset (HDF5 or pickle)')
    parser.add_argument('--prices-path', type=str, default="data/actual_prices.h5", help='Path to prices HDF5 (if features are normalized)')
    parser.add_argument('--num-test-stocks', type=int, default=1000, help='Number of test stocks')
    parser.add_argument('--predictor-checkpoint', type=str, default="./checkpoints/best_model.pt", help='Path to predictor checkpoint')

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
    parser.add_argument('--state-dim', type=int, default=1469, help='State dimension (1444 predictor + 25 portfolio context)')
    parser.add_argument('--hidden-dim', type=int, default=1024, help='Hidden dimension')

    # Logging
    parser.add_argument('--log-frequency', type=int, default=10, help='Log every N episodes')
    parser.add_argument('--eval-frequency', type=int, default=50, help='Evaluate every N episodes')
    parser.add_argument('--eval-episodes', type=int, default=5, help='Number of eval episodes')
    parser.add_argument('--checkpoint-frequency', type=int, default=100, help='Save checkpoint every N episodes')
    parser.add_argument('--checkpoint-dir', type=str, default='./rl_checkpoints_phase2', help='Checkpoint directory')

    # W&B
    parser.add_argument('--use-wandb', action='store_true', default=True, help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='rl-trading-phase2', help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None, help='W&B run name')

    # System
    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-frequency', type=int, default=4, help='Train every N steps')
    parser.add_argument('--target-update-frequency', type=int, default=1000, help='Update target every N steps')

    # Multi-GPU
    parser.add_argument('--multi-gpu', action='store_true', help='Enable multi-GPU asynchronous episode collection')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of worker processes (default: 2)')
    parser.add_argument('--worker-gpus', type=str, default='0,1', help='Comma-separated GPU IDs for workers (default: 0,1)')
    parser.add_argument('--weight-sync-frequency', type=int, default=500, help='Sync weights every N steps (default: 500)')

    args = parser.parse_args()

    # Convert to config dict
    config = vars(args)

    # Add additional config for multi-GPU mode
    if args.multi_gpu:
        config['phase'] = 2  # Phase 2
        config['action_dim'] = 5  # Phase 2 has 5 actions
        config['learning_rate'] = config['lr_q_network']
        config['transaction_cost'] = 0.001
        config['epsilon_decay'] = (config['epsilon_start'] - config['epsilon_end']) / config['epsilon_decay_steps']
        config['max_training_steps'] = config['num_episodes'] * config['episode_length']

    # Train
    if args.multi_gpu:
        from rl.train_rl_multigpu import MultiGPUTrainingLoop
        print("\n🚀 Using Multi-GPU training mode (Phase 2)")
        trainer = MultiGPUTrainingLoop(config)
        trainer.train()
    else:
        print("\n🚀 Using Single-GPU training mode (Phase 2)")
        trainer = Phase2TrainingLoop(config)
        trainer.train()


if __name__ == '__main__':
    main()
