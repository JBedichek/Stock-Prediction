"""
Multi-GPU RL Training Loop

Orchestrates asynchronous episode collection across multiple GPUs
while training Q-network on GPU 0.

Architecture:
- Worker 0 (GPU 0): Episode collection
- Worker 1 (GPU 1): Episode collection
- Main process (GPU 0): Network training

Workers collect experiences in parallel -> Shared replay buffer -> Main process trains
"""

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import time
import sys
import os
from pathlib import Path
from typing import Dict, List
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import TradingAgent, compute_dqn_loss
from rl.shared_buffer import SharedReplayBuffer
from rl.shared_weights import SharedWeights
from rl.worker import worker_process
from inference.backtest_simulation import DatasetLoader

# Optional: W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MultiGPUTrainingLoop:
    """
    Multi-GPU asynchronous training loop for RL.

    Workers collect episodes on GPU 0 and GPU 1 in parallel.
    Main process trains Q-network on GPU 0.
    """

    def __init__(self, config: Dict):
        """
        Initialize multi-GPU training loop.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda:0')  # Main process uses GPU 0

        # Set random seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

        # Parse worker GPUs
        worker_gpu_str = config.get('worker_gpus', '0,1')
        self.worker_gpus = [int(x.strip()) for x in worker_gpu_str.split(',')]
        self.num_workers = len(self.worker_gpus)

        print(f"\n{'='*80}")
        print(f"MULTI-GPU RL TRAINING")
        print(f"{'='*80}")
        print(f"Main process: GPU 0 (training)")
        print(f"Workers: {self.num_workers} workers on GPUs {self.worker_gpus}")
        print(f"{'='*80}\n")

        # Initialize W&B if available and requested
        if WANDB_AVAILABLE and config.get('use_wandb'):
            wandb.init(
                project=config['wandb_project'],
                name=config['run_name'],
                config=config,
                tags=['multi-gpu']
            )

        # Initialize shared components
        self._setup_shared_components()

        # Load data (main process)
        print("Initializing data loader (main process)...")
        self.data_loader = self._load_data()

        # Initialize agent (main process, GPU 0)
        print("Initializing RL agent (main process, GPU 0)...")
        self.agent = TradingAgent(
            predictor_checkpoint_path=config['predictor_checkpoint'],
            state_dim=config['state_dim'],
            hidden_dim=config['hidden_dim'],
            action_dim=config['action_dim'],
            device=self.device
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.agent.q_network.parameters(),
            lr=config['learning_rate']
        )

        # Initialize shared weights with initial Q-network weights
        print("Sharing initial Q-network weights...")
        self.shared_weights.update(self.agent.q_network)

        # Tracking
        self.training_step = 0
        self.last_weight_sync = 0
        self.worker_stats = deque(maxlen=100)

        print("✅ Multi-GPU training loop initialized\n")

    def _setup_shared_components(self):
        """Setup multiprocessing shared components."""
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        # Create manager for shared objects
        self.manager = mp.Manager()

        # Shared replay buffer
        print(f"Creating shared replay buffer (capacity: {self.config['buffer_capacity']:,})...")
        self.shared_buffer = SharedReplayBuffer(
            capacity=self.config['buffer_capacity'],
            state_dim=self.config['state_dim']
        )

        # Shared weights
        self.shared_weights = SharedWeights(self.manager)

        # Shared counters and flags
        self.global_step = mp.Value('i', 0)
        self.epsilon = mp.Value('d', self.config['epsilon_start'])
        self.stop_event = mp.Event()

        # Statistics queue
        self.stats_queue = self.manager.Queue()

        print("✅ Shared components created\n")

    def _load_data(self) -> DatasetLoader:
        """Load dataset (main process)."""
        data_loader = DatasetLoader(
            dataset_path=self.config['dataset_path'],
            prices_path=self.config.get('prices_path'),
            cluster_path=self.config.get('cluster_path'),
            scaler_path=self.config.get('scaler_path')
        )

        # Load feature cache if available
        if self.config.get('features_cache'):
            print("Loading feature cache...")
            data_loader.load_feature_cache(self.config['features_cache'])

        return data_loader

    def start_workers(self):
        """Spawn worker processes."""
        print(f"\n{'='*80}")
        print(f"STARTING {self.num_workers} WORKER PROCESSES")
        print(f"{'='*80}\n")

        self.workers = []

        for worker_id, gpu_id in enumerate(self.worker_gpus):
            worker = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    gpu_id,
                    self.config,
                    self.shared_buffer,
                    self.shared_weights,
                    self.global_step,
                    self.epsilon,
                    self.stop_event,
                    self.stats_queue
                ),
                daemon=False
            )
            worker.start()
            self.workers.append(worker)
            print(f"✅ Worker {worker_id} started on GPU {gpu_id} (PID: {worker.pid})")

        print(f"\n{'='*80}")
        print(f"ALL WORKERS STARTED")
        print(f"{'='*80}\n")

        # Give workers time to initialize
        time.sleep(5)

    def train(self):
        """Main training loop."""
        try:
            # Start workers
            self.start_workers()

            print(f"\n{'='*80}")
            print(f"STARTING TRAINING LOOP")
            print(f"{'='*80}\n")

            # Wait for buffer to have minimum samples
            print(f"Waiting for replay buffer to reach {self.config['batch_size']} samples...")
            while len(self.shared_buffer) < self.config['batch_size']:
                time.sleep(1)
                print(f"  Buffer size: {len(self.shared_buffer):,}/{self.config['batch_size']:,}", end='\r')

            print(f"\n✅ Buffer ready! Starting training...\n")

            # Main training loop
            max_training_steps = self.config.get('max_training_steps', 100000)
            last_log_step = 0

            while self.training_step < max_training_steps and not self.stop_event.is_set():
                # Check if buffer has enough samples
                if not self.shared_buffer.is_ready(self.config['batch_size']):
                    time.sleep(0.01)
                    continue

                # Training step (every train_frequency steps)
                if self.training_step % self.config['train_frequency'] == 0:
                    loss = self._train_step()

                    # Update target network
                    if self.training_step % self.config['target_update_frequency'] == 0:
                        self.agent.update_target_network(tau=self.config['tau'])

                    # Sync weights to workers
                    if self.training_step - self.last_weight_sync >= self.config.get('weight_sync_frequency', 500):
                        self.shared_weights.update(self.agent.q_network)
                        self.last_weight_sync = self.training_step

                # Decay epsilon
                with self.epsilon.get_lock():
                    self.epsilon.value = max(
                        self.config['epsilon_end'],
                        self.epsilon.value * self.config['epsilon_decay']
                    )

                # Process worker statistics
                self._process_worker_stats()

                # Logging
                if self.training_step - last_log_step >= self.config['log_frequency']:
                    self._log_progress()
                    last_log_step = self.training_step

                # Checkpointing
                if self.training_step % self.config.get('checkpoint_frequency', 5000) == 0 and self.training_step > 0:
                    self._save_checkpoint()

                self.training_step += 1

            print(f"\n{'='*80}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*80}\n")

            # Save final checkpoint
            self._save_checkpoint(final=True)

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")

        except Exception as e:
            print(f"\n\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Stop workers
            self._stop_workers()

    def _train_step(self) -> float:
        """
        Perform one training step.

        Returns:
            Loss value
        """
        # Sample batch from shared buffer
        batch_dict = self.shared_buffer.sample(
            batch_size=self.config['batch_size'],
            device=self.device
        )

        # Convert to list of dicts format expected by compute_dqn_loss
        batch = []
        for i in range(len(batch_dict['states'])):
            batch.append({
                'state': batch_dict['states'][i],
                'action': batch_dict['actions'][i],
                'reward': batch_dict['rewards'][i],
                'next_state': batch_dict['next_states'][i],
                'done': batch_dict['dones'][i]
            })

        # Compute loss and update
        loss = compute_dqn_loss(
            agent=self.agent,
            batch=batch,
            gamma=self.config['gamma'],
            device=self.device
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(
                self.agent.q_network.parameters(),
                self.config['grad_clip']
            )

        self.optimizer.step()

        return loss.item()

    def _process_worker_stats(self):
        """Process statistics from workers."""
        # Drain queue
        while not self.stats_queue.empty():
            try:
                stats = self.stats_queue.get_nowait()
                self.worker_stats.append(stats)
            except:
                break

    def _log_progress(self):
        """Log training progress."""
        # Get recent worker stats
        if len(self.worker_stats) == 0:
            return

        recent_stats = list(self.worker_stats)[-self.config.get('log_frequency', 10):]

        # Compute averages
        avg_reward = np.mean([s['reward'] for s in recent_stats])
        avg_return = np.mean([s['stats']['total_return'] for s in recent_stats]) * 100
        avg_final_value = np.mean([s['stats']['final_value'] for s in recent_stats])
        avg_profit = avg_final_value - self.config['initial_capital']
        avg_sharpe = np.mean([s['stats']['sharpe_ratio'] for s in recent_stats])
        avg_win_rate = np.mean([s['stats']['win_rate'] for s in recent_stats]) * 100

        # Buffer stats
        buffer_stats = self.shared_buffer.get_stats()

        print(f"\n{'='*80}")
        print(f"Training Step: {self.training_step:,} | Global Step: {self.global_step.value:,} | ε: {self.epsilon.value:.3f}")
        print(f"{'='*80}")
        print(f"  Episodes collected:  {len(self.worker_stats):,}")
        print(f"  Avg Reward:          {avg_reward:+.6f}")
        print(f"  Avg Return:          {avg_return:+.2f}%")
        print(f"  Avg Profit:          ${avg_profit:+,.0f}")
        print(f"  Avg Final Value:     ${avg_final_value:,.0f}")
        print(f"  Avg Sharpe Ratio:    {avg_sharpe:.3f}")
        print(f"  Avg Win Rate:        {avg_win_rate:.1f}%")
        print(f"")
        print(f"  Buffer size:         {buffer_stats['size']:,}/{buffer_stats['capacity']:,} ({buffer_stats['capacity_used']*100:.1f}%)")
        print(f"  Buffer avg reward:   {buffer_stats['avg_reward']:+.6f}")
        print(f"{'='*80}\n")

        # W&B logging
        if WANDB_AVAILABLE and self.config.get('use_wandb'):
            wandb.log({
                'training_step': self.training_step,
                'global_step': self.global_step.value,
                'epsilon': self.epsilon.value,
                'avg_reward': avg_reward,
                'avg_return': avg_return,
                'avg_profit': avg_profit,
                'avg_final_value': avg_final_value,
                'avg_sharpe': avg_sharpe,
                'avg_win_rate': avg_win_rate,
                'buffer_size': buffer_stats['size'],
                'buffer_capacity_used': buffer_stats['capacity_used']
            })

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        suffix = 'final' if final else f'step_{self.training_step}'
        checkpoint_path = checkpoint_dir / f"multigpu_rl_{suffix}.pt"

        checkpoint = {
            'training_step': self.training_step,
            'global_step': self.global_step.value,
            'epsilon': self.epsilon.value,
            'q_network_state_dict': self.agent.q_network.state_dict(),
            'target_network_state_dict': self.agent.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def _stop_workers(self):
        """Stop all worker processes."""
        print(f"\n{'='*80}")
        print(f"STOPPING WORKERS")
        print(f"{'='*80}\n")

        # Signal workers to stop
        self.stop_event.set()

        # Wait for workers to finish
        for i, worker in enumerate(self.workers):
            print(f"Waiting for worker {i} to finish...")
            worker.join(timeout=10)

            if worker.is_alive():
                print(f"⚠️  Worker {i} did not stop gracefully, terminating...")
                worker.terminate()
                worker.join(timeout=5)

            if worker.is_alive():
                print(f"❌ Worker {i} did not terminate, killing...")
                worker.kill()

        print(f"\n✅ All workers stopped\n")
