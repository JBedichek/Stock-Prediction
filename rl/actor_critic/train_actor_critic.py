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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import (
    ActorCriticAgent, ReplayBuffer, compute_critic_loss, compute_actor_loss,
    compute_cql_loss, augment_state, post_episode_processing, analyze_return_risk_relationship
)
from rl.attention_networks import AttentionActorCriticAgent
from rl.state_creation_optimized import create_states_batch_optimized, create_next_states_batch_optimized
from rl.profiling_utils import TrainingProfiler
from rl.gpu_stock_cache import GPUStockSelectionCache
from rl.vectorized_training_loop import vectorized_transition_storage
from rl.rl_environment import TradingEnvironment
from rl.gpu_vectorized_env import GPUVectorizedTradingEnv
from rl.reduced_action_space import (
    get_top_4_stocks, get_bottom_4_stocks, get_top_k_stocks_per_horizon, sample_top_4_from_top_k,
    create_global_state, decode_action_to_trades
)
from inference.backtest_simulation import DatasetLoader

# Re-enable gradients
torch.set_grad_enabled(True)


def create_portfolio_histogram(portfolio_values: List[float], num_bins: int = 10) -> str:
    """
    Create a compact text-based histogram of portfolio values.

    Args:
        portfolio_values: List of portfolio values from parallel episodes
        num_bins: Number of bins for the histogram (default 10 for 10% quantiles)

    Returns:
        Compact string representation like "Port: [▁▂▃▅█▇▅▃▂▁] $95k-$105k (med: $102k)"
    """
    if not portfolio_values or len(portfolio_values) == 0:
        return "Port: [──────────] N/A"

    values = np.array(portfolio_values)

    # Remove any non-finite values
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "Port: [─────] N/A"

    # Compute quantiles for bins
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(values, quantiles)

    # Count values in each bin
    counts, _ = np.histogram(values, bins=bin_edges)

    # Normalize counts to fit in histogram (max height = 8)
    max_count = max(counts) if max(counts) > 0 else 1
    normalized_counts = (counts / max_count * 8).astype(int)

    # Block characters for histogram (increasing height)
    blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    # Build histogram string
    hist_str = ''.join([blocks[min(h, 8)] for h in normalized_counts])

    # Add value range
    min_val = np.min(values)
    max_val = np.max(values)
    median_val = np.median(values)

    # Format values in thousands
    range_str = f"${min_val/1000:.0f}k-${max_val/1000:.0f}k (med: ${median_val/1000:.0f}k)"

    return f"Port: [{hist_str}] {range_str}"


def save_portfolio_histogram(portfolio_values: List[float],
                             episode: int,
                             output_path: str = './results/portfolio_distribution.png',
                             mode: str = 'training') -> None:
    """
    Create and save a detailed, aesthetically pleasing histogram of portfolio values.

    Args:
        portfolio_values: List of portfolio values from parallel episodes
        episode: Current episode number for labeling
        output_path: Path to save the figure
        mode: 'training' or 'validation' for labeling
    """
    if not portfolio_values or len(portfolio_values) == 0:
        return

    # Filter out non-finite values
    values = np.array([v for v in portfolio_values if np.isfinite(v)])
    if len(values) == 0:
        return

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)

    # Main histogram
    ax_main = fig.add_subplot(gs[0, 0])

    # Compute statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25

    # Create histogram with optimal bin count (Freedman-Diaconis rule)
    if iqr > 0:
        bin_width = 2 * iqr / (len(values) ** (1/3))
        num_bins = max(10, min(50, int((max_val - min_val) / bin_width)))
    else:
        num_bins = 20

    # Plot histogram with gradient colors
    n, bins, patches = ax_main.hist(values, bins=num_bins, alpha=0.7,
                                     color='steelblue', edgecolor='black', linewidth=0.5)

    # Color gradient based on density
    cm = plt.cm.viridis
    norm = plt.Normalize(vmin=n.min(), vmax=n.max())
    for i, (count, patch) in enumerate(zip(n, patches)):
        color = cm(norm(count))
        patch.set_facecolor(color)

    # Add vertical lines for key statistics
    ax_main.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: ${mean_val/1000:.1f}k', alpha=0.8)
    ax_main.axvline(median_val, color='green', linestyle='--', linewidth=2,
                    label=f'Median: ${median_val/1000:.1f}k', alpha=0.8)
    ax_main.axvline(q25, color='orange', linestyle=':', linewidth=1.5,
                    label=f'Q25: ${q25/1000:.1f}k', alpha=0.7)
    ax_main.axvline(q75, color='orange', linestyle=':', linewidth=1.5,
                    label=f'Q75: ${q75/1000:.1f}k', alpha=0.7)

    # Fill IQR region
    ax_main.axvspan(q25, q75, alpha=0.1, color='orange', label='IQR')

    # Labels and title
    ax_main.set_xlabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    mode_str = mode.capitalize()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax_main.set_title(f'Portfolio Value Distribution - {mode_str} (Episode {episode})\n{timestamp}',
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)

    # Format x-axis as currency
    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

    # Box plot on the right
    ax_box = fig.add_subplot(gs[0, 1])
    bp = ax_box.boxplot(values, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
    ax_box.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax_box.set_title('Box Plot', fontsize=11, fontweight='bold')
    ax_box.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    ax_box.grid(True, alpha=0.3, axis='y')
    ax_box.set_xticklabels([''])

    # Statistics table
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')

    stats_data = [
        ['Statistic', 'Value', '', 'Statistic', 'Value'],
        ['─────────', '─────────', '', '─────────', '─────────'],
        ['Count', f'{len(values)}', '', 'Range', f'${(max_val - min_val)/1000:.1f}k'],
        ['Mean', f'${mean_val/1000:.2f}k', '', 'Std Dev', f'${std_val/1000:.2f}k'],
        ['Median', f'${median_val/1000:.2f}k', '', 'CV', f'{(std_val/mean_val)*100:.1f}%'],
        ['Min', f'${min_val/1000:.2f}k', '', 'IQR', f'${iqr/1000:.2f}k'],
        ['Q25', f'${q25/1000:.2f}k', '', 'Skewness', f'{((mean_val - median_val) / std_val):.3f}'],
        ['Q75', f'${q75/1000:.2f}k', '', 'P90', f'${np.percentile(values, 90)/1000:.2f}k'],
        ['Max', f'${max_val/1000:.2f}k', '', 'P10', f'${np.percentile(values, 10)/1000:.2f}k'],
    ]

    table = ax_stats.table(cellText=stats_data, loc='center', cellLoc='left',
                          colWidths=[0.15, 0.15, 0.05, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    # Style separator row
    for i in range(5):
        cell = table[(1, i)]
        cell.set_facecolor('#E0E0E0')

    # Alternate row colors
    for i in range(2, len(stats_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')

    # Overall figure title
    fig.suptitle(f'Actor-Critic RL Training - Portfolio Performance Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   💾 Saved portfolio histogram to {output_path}")


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
        # Count stocks with price data
        all_tickers = list(self.data_loader.h5_file.keys()) if self.data_loader.is_hdf5 else list(self.data_loader.data.keys())
        stocks_with_prices = sum(1 for ticker in all_tickers
                                if ticker in self.data_loader.prices_file)
        print(f"   ✅ Loaded {len(all_tickers)} total stocks")
        print(f"   ✅ {stocks_with_prices} stocks have price data (used for training/validation)")

        # Initialize agent
        print("\n2. Initializing agent...")
        use_attention = config.get('use_attention_architecture', False)

        if use_attention:
            print("   🔬 Using ATTENTION architecture (lightweight self-attention over stocks)")
            self.agent = AttentionActorCriticAgent(
                predictor_checkpoint_path=config.get('predictor_checkpoint', './checkpoints/best_model_100m_1.18.pt'),
                state_dim=config.get('state_dim', 11761),
                hidden_dim=config.get('attention_hidden_dim', 256),  # Smaller for attention
                action_dim=config.get('action_dim', 9)
            ).to(self.device)
        else:
            print("   🧠 Using MLP architecture (fully-connected networks)")
            self.agent = ActorCriticAgent(
                predictor_checkpoint_path=config.get('predictor_checkpoint', './checkpoints/best_model_100m_1.18.pt'),
                state_dim=config.get('state_dim', 11761),
                hidden_dim=config.get('hidden_dim', 1024),
                action_dim=config.get('action_dim', 9)
            ).to(self.device)

        # Load pre-trained critics if available (loads both critic1 and critic2)
        critic_checkpoint_path = config.get('pretrained_critic_path', './checkpoints/pretrained_critic.pt')
        if os.path.exists(critic_checkpoint_path):
            print(f"\n   📂 Loading pre-trained critics from {critic_checkpoint_path}...")
            checkpoint = torch.load(critic_checkpoint_path, map_location=self.device, weights_only=False)

            # Load both critics (they were trained identically with MC returns)
            self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            self.agent.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            self.agent.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])

            print(f"   ✅ Pre-trained critics loaded (both trained for {checkpoint['epoch']+1} epochs)")
        else:
            print(f"   ⚠️  No pre-trained critic found at {critic_checkpoint_path}")
            print(f"      Starting with random critic weights (both critics)")

        # Freeze predictor and ensure it stays in eval mode (critical for consistent dropout behavior)
        self.agent.feature_extractor.freeze_predictor()
        self.agent.feature_extractor.eval()  # Ensure entire feature extractor is in eval mode

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
        recent_trading_days = [d for d in all_trading_days if d >= cutoff_date]

        # Train/Validation split: Use most recent 2 months for validation
        print(f"   📅 Total trading days in last 4 years: {len(recent_trading_days)}")
        print(f"   📅 Date range: {recent_trading_days[0]} to {recent_trading_days[-1]}")

        # Find the most recent date with data
        most_recent_date = datetime.strptime(recent_trading_days[-1], '%Y-%m-%d')
        print(f"   📅 Most recent date: {most_recent_date.strftime('%Y-%m-%d')}")

        # Calculate validation cutoff (2 months back from most recent)
        val_months_back = config.get('val_months_back', 2)
        val_cutoff_date = (most_recent_date - timedelta(days=val_months_back * 30)).strftime('%Y-%m-%d')

        # Split into train and validation
        self.train_dates = [d for d in recent_trading_days if d < val_cutoff_date]
        self.val_dates = [d for d in recent_trading_days if d >= val_cutoff_date]

        print(f"   📊 Train: {len(self.train_dates)} days ({self.train_dates[0]} to {self.train_dates[-1]})")
        print(f"   📊 Val:   {len(self.val_dates)} days ({self.val_dates[0]} to {self.val_dates[-1]})")
        print(f"   📊 Validation uses most recent {val_months_back} months of data")

        # Validate we have enough data
        if len(self.val_dates) < 20:
            print(f"   ⚠️  Warning: Only {len(self.val_dates)} validation days - may need more data")
        if len(self.train_dates) < 100:
            print(f"   ⚠️  Warning: Only {len(self.train_dates)} training days - may need more data")

        # Verify validation dates have actual price data
        # Check a few validation dates to ensure they have data for test stocks
        print(f"   🔍 Verifying validation data availability...")
        test_dates = self.val_dates[::max(1, len(self.val_dates)//5)]  # Sample 5 dates
        missing_data_dates = []
        for date in test_dates:
            # Check if we have prices for at least some test stocks on this date
            has_data = False
            for ticker in list(self.data_loader.test_tickers)[:10]:  # Check first 10 test stocks
                if ticker in self.data_loader.prices_file:
                    ticker_data = self.data_loader.prices_file[ticker]
                    dates_bytes = ticker_data['dates'][:]
                    ticker_dates = [d.decode('utf-8') for d in dates_bytes]
                    if date in ticker_dates:
                        has_data = True
                        break
            if not has_data:
                missing_data_dates.append(date)

        if missing_data_dates:
            print(f"   ⚠️  Warning: Some validation dates may have limited data: {missing_data_dates[:3]}")
        else:
            print(f"   ✅ Validation dates have price data")

        # Use train dates for training
        self.recent_trading_days = self.train_dates

        print(f"\n   🌍 Training/Validation/Inference will use ALL {stocks_with_prices} stocks")
        print(f"   📈 Agent will select top-K from entire stock universe each day")

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

        # Now create GPU-vectorized environment (shares state cache from temp_env)
        print(f"\n6. Creating GPU-vectorized environment ({num_parallel} parallel envs)...")
        self.vec_env = GPUVectorizedTradingEnv(
            num_envs=num_parallel,
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=config.get('initial_capital', 100000),
            max_positions=config.get('max_positions', 1),
            episode_length=config.get('episode_length', 30),
            transaction_cost=config.get('transaction_cost', 0.0),
            device=self.device,
            trading_days_filter=self.recent_trading_days,
            top_k_per_horizon=config.get('top_k_per_horizon', 10)
        )

        # Share state cache and price cache from temp_env to reference env
        self.vec_env.ref_env.state_cache = temp_env.state_cache
        self.vec_env.ref_env.price_cache = temp_env.price_cache

        print(f"   ✅ Vectorized environment ready ({num_parallel} parallel episodes)")

        # Replay buffer with n-step returns and Prioritized Experience Replay
        # n_step > 1 propagates rewards over multiple timesteps
        # PER samples important transitions more frequently
        n_step = config.get('n_step', 3)  # Default: 3-step returns
        gamma = config.get('gamma', 0.99)
        use_per = config.get('use_per', True)  # Enable PER by default
        self.buffer = ReplayBuffer(
            capacity=config.get('buffer_capacity', 100000),
            n_step=n_step,
            gamma=gamma,
            use_per=use_per,
            per_alpha=config.get('per_alpha', 0.6),
            per_beta=config.get('per_beta', 0.4),
            per_beta_increment=config.get('per_beta_increment', 0.001)
        )
        print(f"   ✅ Replay buffer initialized (n_step={n_step}, gamma={gamma}, PER={'enabled' if use_per else 'disabled'})")

        # Training state
        self.episode = 0
        self.global_step = 0

        # Epsilon decay for exploration
        self.epsilon_start = config.get('epsilon_start', 0.3)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay_episodes = config.get('epsilon_decay_episodes', 100000)
        self.current_epsilon = self.epsilon_start

        # Reward normalization (running statistics using Welford's algorithm)
        self.reward_scale = config.get('reward_scale', 100.0)  # Scale rewards by 100x
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_m2 = 0.0  # Sum of squared deviations for Welford's algorithm
        self.reward_count = 0

        # Curriculum learning (progressive stock difficulty)
        # Start with only top stocks (easiest), gradually include more (harder)
        self.curriculum_start_k = config.get('curriculum_start_k', 5)  # Start with top 5 per horizon
        self.curriculum_end_k = config.get('curriculum_end_k', 15)  # End with top 15 per horizon
        self.curriculum_episodes = config.get('curriculum_episodes', 50000)  # Anneal over 50k episodes
        self.current_top_k = self.curriculum_start_k

        # Stock sampling diversity (prevents overfitting to specific stocks)
        # Randomly sample a fraction of stocks before selecting top-K
        # This ensures the agent sees different stocks each episode
        self.stock_sample_fraction = config.get('stock_sample_fraction', 0.3)  # Sample 30% of stocks

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

        # Load pre-computed stock selections (if available)
        self.stock_selections_cache = None
        self.use_precomputed_selections = config.get('use_precomputed_selections', False)
        stock_selections_path = config.get('stock_selections_cache', None)

        if self.use_precomputed_selections and stock_selections_path and os.path.exists(stock_selections_path):
            print(f"\n7. Loading pre-computed stock selections to GPU...")
            print(f"   Path: {stock_selections_path}")
            # Load entire cache to GPU memory for instant access (no disk I/O during training!)
            all_tickers = list(self.data_loader.prices_file.keys())
            self.stock_selections_cache = GPUStockSelectionCache(
                h5_path=stock_selections_path,
                prices_file_keys=all_tickers,
                device=self.device
            )
            print(f"   ✅ All stock selections cached in GPU memory - zero I/O during training!")
        elif self.use_precomputed_selections:
            print(f"\n⚠️  Pre-computed selections requested but not found: {stock_selections_path}")
            print(f"   Run: python rl/precompute_stock_selections.py")
            print(f"   Falling back to on-the-fly computation...")
            self.use_precomputed_selections = False
            self.stock_selections_cache = None

        # Cache top-4 stocks per date (only used if not using pre-computed selections)
        self.top_4_cache = {}

        # Validation tracking
        self.best_val_return = -float('inf')
        self.best_val_portfolio_value = 0.0
        self.val_history = []
        self.epochs_without_improvement = 0

        # Initialize profiler
        enable_profiling = config.get('enable_profiling', False)
        self.profiler = TrainingProfiler(enabled=enable_profiling)
        if enable_profiling:
            print(f"\n📊 Training profiler ENABLED")
            print(f"   Profiling results will be shown every {config.get('profiling_report_interval', 100)} episodes")

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

    def update_reward_stats(self, reward: float):
        """
        Update running reward statistics using Welford's online algorithm.

        This allows us to compute mean and variance in an online manner
        without storing all rewards, which is memory-efficient.

        Args:
            reward: Raw reward to incorporate into statistics
        """
        # Validate input
        if not np.isfinite(reward):
            print(f"WARNING: Attempted to update reward stats with non-finite reward: {reward}. Skipping.")
            return

        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean

        # Update variance (use M2 = sum of squared deviations)
        if self.reward_count == 1:
            self.reward_m2 = 0.0
        else:
            if not hasattr(self, 'reward_m2'):
                self.reward_m2 = 0.0
            self.reward_m2 += delta * delta2

        # Compute std from variance
        if self.reward_count > 1:
            variance = self.reward_m2 / (self.reward_count - 1)
            # Safety: ensure variance is non-negative and finite
            if np.isfinite(variance) and variance > 0:
                self.reward_std = np.sqrt(variance)
            else:
                self.reward_std = 1.0
        else:
            self.reward_std = 1.0

    def normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using running statistics.

        Normalization helps stabilize Q-value estimates by ensuring
        rewards have consistent scale throughout training.

        Args:
            reward: Raw reward

        Returns:
            Normalized reward: (reward - mean) / (std + epsilon)
        """
        # Update statistics first
        self.update_reward_stats(reward)

        # Normalize (avoid division by zero)
        if self.reward_count < 10:
            # Don't normalize until we have enough samples
            return reward

        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized

    def train_episode_vectorized(self) -> List[Dict]:
        """
        Run one batch of parallel training episodes (vectorized).

        Returns:
            List of episode statistics (one per completed episode)
        """
        num_parallel = self.config.get('num_parallel_envs', 8)

        # Reset all environments
        states_list, positions_list = self.vec_env.reset()
        is_short_list = [False] * num_parallel  # Track if each position is a short

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
        episode_action_diversity_losses = [[] for _ in range(num_parallel)]
        episode_action0_probs = [[] for _ in range(num_parallel)]

        # HINDSIGHT EXPERIENCE REPLAY: Track transitions for each episode
        episode_transitions = [[] for _ in range(num_parallel)]

        # Run episodes until all complete
        max_steps = self.config.get('episode_length', 30)

        for step in range(max_steps):
            # Check if all envs are done
            if self.vec_env.get_num_active() == 0:
                break

            # Compute top-4 stocks per unique date (cached - same across envs at same date)
            unique_dates = {}
            for i in range(self.vec_env.num_envs):
                if not self.vec_env.dones[i]:
                    # Get current date for this environment
                    step_idx = self.vec_env.step_indices[i].item()
                    if step_idx < len(self.vec_env.episode_dates[i]):
                        date = self.vec_env.episode_dates[i][step_idx]
                        if date not in unique_dates:
                            unique_dates[date] = []
                        unique_dates[date].append(i)

            # Get top-K stocks for each unique date, then randomly sample 4
            with self.profiler.profile('initial_stock_selection'):
                date_to_top4 = {}
                date_to_bottom4 = {}

                for date in unique_dates:
                    # OPTIMIZED: Use pre-computed selections if available
                    if self.use_precomputed_selections and self.stock_selections_cache is not None and date in self.stock_selections_cache:
                        # Get from GPU cache (instant - no I/O!)
                        import random
                        num_samples = self.stock_selections_cache.get_num_samples(date)
                        sample_idx = random.randint(0, num_samples - 1)

                        top_4_stocks, bottom_4_stocks = self.stock_selections_cache.get_sample(date, sample_idx)

                        date_to_top4[date] = top_4_stocks
                        date_to_bottom4[date] = bottom_4_stocks
                    else:
                        # DEBUG: Print why fallback is happening (only first time)
                        if not hasattr(self, '_logged_fallback'):
                            self._logged_fallback = True
                            if not self.use_precomputed_selections:
                                print(f"\n⚠️  Stock selection fallback: use_precomputed_selections=False")
                            elif self.stock_selections_cache is None:
                                print(f"\n⚠️  Stock selection fallback: cache is None")
                            elif date not in self.stock_selections_cache:
                                cache_dates = sorted(list(self.stock_selections_cache.selections.keys()))
                                print(f"\n⚠️  Stock selection fallback: date '{date}' not in cache")
                                print(f"   Cache: {len(cache_dates)} dates from {cache_dates[0]} to {cache_dates[-1]}")
                        # FALLBACK: Compute on-the-fly (slower but works without pre-computation)
                        top_k_per_horizon = self.config.get('top_k_per_horizon_sampling', 3)

                        # When using stock sampling (sample_fraction < 1.0), recompute each time for diversity
                        # When sample_fraction == 1.0 (no sampling), cache results for efficiency
                        use_cache = (self.stock_sample_fraction >= 1.0)

                        if use_cache and date in self.top_4_cache:
                            # Use cached result (deterministic, no sampling)
                            top_k_stocks = self.top_4_cache[date]
                        else:
                            # Compute top-K stocks (with or without random sampling)
                            # Access state cache through reference environment
                            all_cached_states = self.vec_env.ref_env.state_cache[date]
                            # TRAINING: Use ALL stocks with price data (same as validation/inference)
                            # This ensures the agent trains on the full stock universe it will see in production
                            cached_states = {ticker: state for ticker, state in all_cached_states.items()
                                           if ticker in self.data_loader.prices_file}
                            # Get top-K per horizon with random sampling for diversity
                            # sample_fraction < 1.0 gives different stocks each time (no caching)
                            # sample_fraction == 1.0 uses all stocks (cached for efficiency)
                            top_k_stocks = get_top_k_stocks_per_horizon(
                                cached_states,
                                k=top_k_per_horizon,
                                sample_fraction=self.stock_sample_fraction
                            )

                            # Only cache if not using random sampling
                            if use_cache:
                                self.top_4_cache[date] = top_k_stocks

                        # Randomly sample 4 from top-K for each environment (training diversity)
                        date_to_top4[date] = sample_top_4_from_top_k(top_k_stocks, sample_size=4, deterministic=False)

                        # Get bottom-4 stocks for shorting
                        bottom_4_stocks = get_bottom_4_stocks(cached_states, sample_fraction=self.stock_sample_fraction)
                        date_to_bottom4[date] = bottom_4_stocks

            # Assign top-4 and bottom-4 stocks to each environment
            top_4_stocks_list = []
            bottom_4_stocks_list = []
            for i in range(self.vec_env.num_envs):
                if not self.vec_env.dones[i]:
                    # Get current date for this environment
                    step_idx = self.vec_env.step_indices[i].item()
                    current_date = self.vec_env.episode_dates[i][step_idx]
                    top_4_stocks_list.append(date_to_top4[current_date])
                    bottom_4_stocks_list.append(date_to_bottom4[current_date])
                else:
                    top_4_stocks_list.append([])
                    bottom_4_stocks_list.append([])

            # Get states for each environment (OPTIMIZED: batched GPU transfer for 70x speedup!)
            with self.profiler.profile('state_creation_current'):
                states_list_current = create_states_batch_optimized(
                    vec_env=self.vec_env,
                    top_4_stocks_list=top_4_stocks_list,
                    bottom_4_stocks_list=bottom_4_stocks_list,
                    device=self.device
                )

            # Batch action selection (SINGLE GPU forward pass for all envs)
            # Use decaying epsilon for exploration
            with self.profiler.profile('action_selection'):
                results = self.agent.select_actions_reduced_batch(
                    top_4_stocks_list=top_4_stocks_list,
                    bottom_4_stocks_list=bottom_4_stocks_list,
                    states_list=states_list_current,
                    positions_list=positions_list,
                    is_short_list=is_short_list,
                    epsilon=self.current_epsilon,
                    deterministic=False
                )

            # Extract actions, trades, new positions, and new is_short flags
            actions_list = [r[0] for r in results]
            trades_list = [r[3] for r in results]
            new_positions_from_agent = [r[4] for r in results]
            new_is_short_from_agent = [r[5] for r in results]

            # Update is_short_list based on agent's action decoding
            for i in range(num_parallel):
                if not self.vec_env.dones[i]:
                    is_short_list[i] = new_is_short_from_agent[i]

            # Track actions
            for i, action in enumerate(actions_list):
                if not self.vec_env.dones[i]:
                    episode_actions[i].append(action)

            # Step all environments
            with self.profiler.profile('environment_step'):
                next_states_list, rewards_list, dones_list, infos_list, next_positions_list = self.vec_env.step(
                    actions_list, trades_list, positions_list
                )

            # VECTORIZED transition storage (replaces sequential for-loop)
            # This batches stock selection, state creation, and buffer storage across all environments
            stored_transitions = vectorized_transition_storage(
                vec_env=self.vec_env,
                top_4_stocks_list=top_4_stocks_list,
                bottom_4_stocks_list=bottom_4_stocks_list,
                states_list_current=states_list_current,
                positions_list=positions_list,
                next_positions_list=next_positions_list,
                is_short_list=is_short_list,
                actions_list=actions_list,
                rewards_list=rewards_list,
                dones_list=dones_list,
                infos_list=infos_list,
                stock_selections_cache=self.stock_selections_cache,
                use_precomputed_selections=self.use_precomputed_selections,
                top_4_cache=self.top_4_cache,
                stock_sample_fraction=self.stock_sample_fraction,
                top_k_per_horizon_sampling=self.config.get('top_k_per_horizon_sampling', 3),
                buffer=self.buffer,
                device=self.device,
                profiler=self.profiler,
                reward_scale=self.reward_scale,
                reward_normalizer=self.normalize_reward
            )

            # Track transitions for HER (if enabled)
            if self.config.get('use_her', False):
                for i in range(num_parallel):
                    episode_transitions[i].extend(stored_transitions[i])

            # Update episode returns
            for i in range(num_parallel):
                if not self.vec_env.dones[i] or dones_list[i]:
                    episode_returns[i] += rewards_list[i]

            # Update positions
            positions_list = next_positions_list

            self.global_step += 1
            self.profiler.record_step()

        # ============================================================================
        # HINDSIGHT EXPERIENCE REPLAY: Add hindsight experiences from failed trades
        # ============================================================================
        # Process episodes that completed (learn from failures)
        if self.config.get('use_her', False):
            for i in range(num_parallel):
                if len(episode_transitions[i]) > 0:
                    her_stats = post_episode_processing(
                        buffer=self.buffer,
                        episode_transitions=episode_transitions[i],
                        use_her=True,
                        hindsight_ratio=self.config.get('her_ratio', 0.5)
                    )
                    # Optional: log HER stats
                    # print(f"   Env {i}: Added {her_stats['hindsight_added']} hindsight experiences")

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
                    with self.profiler.profile('critic_training'):
                        critic_loss, critic_info = self._train_critic()
                    # Add to all episodes (average across parallel envs)
                    for i in range(num_parallel):
                        episode_critic_losses[i].append(critic_loss)
                        episode_q1_values[i].append(critic_info['q1_mean'])
                        episode_q2_values[i].append(critic_info['q2_mean'])
                        episode_q_diffs[i].append(critic_info['q_diff'])
                else:
                    # Critic frozen - just compute loss for logging
                    batch_size = min(self.config.get('batch_size', 256), len(self.buffer))
                    if batch_size >= 32:
                        batch = self.buffer.sample(batch_size)
                    else:
                        # Buffer too small, skip
                        continue
                    with torch.no_grad():
                        loss1, loss2, _ = compute_critic_loss(  # Discard td_errors (diagnostic only)
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
                    with self.profiler.profile('actor_training'):
                        actor_loss, actor_info = self._train_actor()
                    for i in range(num_parallel):
                        episode_actor_losses[i].append(actor_loss)
                        episode_entropies[i].append(actor_info['entropy'])
                        episode_advantages[i].append(actor_info['avg_advantage'])
                        episode_q_values[i].append(actor_info['avg_q_value'])
                        episode_grad_norms[i].append(actor_info['grad_norm'])
                        episode_action_diversity_losses[i].append(actor_info.get('action_diversity_loss', 0.0))
                        episode_action0_probs[i].append(actor_info.get('action0_prob', 0.0))

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
                'avg_action_diversity_loss': np.mean(episode_action_diversity_losses[i]) if episode_action_diversity_losses[i] else 0.0,
                'avg_action0_prob': np.mean(episode_action0_probs[i]) if episode_action0_probs[i] else 0.0,
                'avg_advantage': np.mean(episode_advantages[i]) if episode_advantages[i] else 0.0,
                'avg_q_value': np.mean(episode_q_values[i]) if episode_q_values[i] else 0.0,
                'avg_q1_value': np.mean(episode_q1_values[i]) if episode_q1_values[i] else 0.0,
                'avg_q2_value': np.mean(episode_q2_values[i]) if episode_q2_values[i] else 0.0,
                'avg_q_diff': np.mean(episode_q_diffs[i]) if episode_q_diffs[i] else 0.0,
                'avg_grad_norm': np.mean(episode_grad_norms[i]) if episode_grad_norms[i] else 0.0,
                'buffer_size': len(self.buffer),
                'epsilon': self.current_epsilon,  # Track current exploration rate
                'curriculum_top_k': self.current_top_k,  # Track curriculum difficulty
                'reward_mean': self.reward_mean,  # Running reward mean
                'reward_std': self.reward_std,  # Running reward std
                'reward_count': self.reward_count,  # Number of rewards seen
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

        # Decay epsilon for exploration (linear decay)
        self.current_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.episode / self.epsilon_decay_episodes)
        )

        # Update curriculum (progressive stock difficulty)
        # Increase top_k from start to end over curriculum_episodes
        self.current_top_k = min(
            self.curriculum_end_k,
            self.curriculum_start_k + (self.curriculum_end_k - self.curriculum_start_k) * (self.episode / self.curriculum_episodes)
        )
        # Apply to reference environment (round to nearest int)
        top_k_int = int(round(self.current_top_k))
        self.vec_env.ref_env.top_k_per_horizon = top_k_int

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
        current_is_short = False  # Track if position is short (False = long or cash)

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

            # Get top-4 stocks to LONG and bottom-4 stocks to SHORT
            top_4_stocks = get_top_4_stocks(cached_states, sample_fraction=self.stock_sample_fraction)
            bottom_4_stocks = get_bottom_4_stocks(cached_states, sample_fraction=self.stock_sample_fraction)

            # Get individual stock states from environment (for state construction)
            states = self.env._get_states()

            # Create global state (8 stocks × 1469 + 9 position encoding = 11761 dims)
            global_state = create_global_state(
                top_4_stocks=top_4_stocks,
                bottom_4_stocks=bottom_4_stocks,
                states=states,
                current_position=current_position,
                is_short=current_is_short,
                device=self.device
            )

            # Select single discrete action (0-8) using epsilon-greedy
            epsilon = self.config.get('epsilon', 0.1)
            action, log_prob, entropy, trades, new_position, new_is_short = self.agent.select_action_reduced(
                top_4_stocks=top_4_stocks,
                bottom_4_stocks=bottom_4_stocks,
                states=states,
                current_position=current_position,
                current_is_short=current_is_short,
                epsilon=epsilon,
                deterministic=False
            )

            # Track action taken
            episode_actions.append(action)

            # Execute trades in environment
            # Convert trades to environment format: {ticker: action_id}
            # TODO: Environment needs to be updated to handle SHORT (3) and COVER (4) actions
            actions = {}
            for trade in trades:
                ticker = trade['ticker']
                if trade['action'] == 'BUY':
                    actions[ticker] = 1  # BUY
                elif trade['action'] == 'SELL':
                    actions[ticker] = 2  # SELL
                elif trade['action'] == 'SHORT':
                    actions[ticker] = 3  # SHORT (TODO: implement in environment)
                elif trade['action'] == 'COVER':
                    actions[ticker] = 4  # COVER (TODO: implement in environment)

            # Take step in environment
            next_states_dict, reward, done, info = self.env.step(actions)

            # Update current position from agent's decode_action_to_trades
            current_position = new_position
            current_is_short = new_is_short

            # Get next global state
            next_cached_states = self.env.state_cache[self.env.current_date]
            next_top_4_stocks = get_top_4_stocks(next_cached_states, sample_fraction=self.stock_sample_fraction)
            next_bottom_4_stocks = get_bottom_4_stocks(next_cached_states, sample_fraction=self.stock_sample_fraction)
            next_states = self.env._get_states()
            next_global_state = create_global_state(
                top_4_stocks=next_top_4_stocks,
                bottom_4_stocks=next_bottom_4_stocks,
                states=next_states,
                current_position=current_position,
                is_short=current_is_short,
                device=self.device
            )

            # Scale and normalize reward
            # 1. Scale: raw rewards are ~0.001-0.02, scale to ~0.1-2.0
            # 2. Normalize: zero-mean, unit-variance using running statistics
            scaled_reward = reward * self.reward_scale
            normalized_reward = self.normalize_reward(scaled_reward)

            # Store transition with global states and discrete action
            self.buffer.push(
                state=global_state,
                action=action,  # Discrete action (0-4)
                reward=normalized_reward,
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
                            loss1, loss2, _ = compute_critic_loss(  # Discard td_errors (diagnostic only)
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
            'reward_mean': self.reward_mean,  # Running reward mean
            'reward_std': self.reward_std,  # Running reward std
            'reward_count': self.reward_count,  # Number of rewards seen
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
        """Train both critics for one step (twin critics) with optional CQL and data augmentation."""
        # Ensure we don't sample more than available (early in training)
        batch_size = min(self.config.get('batch_size', 1024), len(self.buffer))
        if batch_size < 32:  # Don't train with tiny batches
            return 0.0, {'loss1': 0.0, 'loss2': 0.0, 'q1_mean': 0.0, 'q2_mean': 0.0, 'q_diff': 0.0}

        batch = self.buffer.sample(batch_size)

        # DATA AUGMENTATION: Add noise to states for robustness to noisy stock data
        if self.config.get('use_data_augmentation', False):
            noise_level = self.config.get('augmentation_noise_level', 0.01)
            for trans in batch:
                trans['state'] = augment_state(trans['state'], noise_level=noise_level, device=self.device)
                trans['next_state'] = augment_state(trans['next_state'], noise_level=noise_level, device=self.device)

        # CONSERVATIVE Q-LEARNING: Prevent overestimation on unseen actions
        use_cql = self.config.get('use_cql', False)
        if use_cql:
            # Use CQL loss instead of standard critic loss
            cql_loss, cql_info = compute_cql_loss(
                agent=self.agent,
                batch=batch,
                gamma=self.config.get('gamma', 0.99),
                device=self.device,
                cql_alpha=self.config.get('cql_alpha', 1.0)
            )
            # CQL returns combined loss for both critics
            loss1 = cql_loss
            loss2 = cql_loss
            td_errors = cql_info.get('td_errors', np.zeros(len(batch)))
        else:
            # Standard twin critic loss
            loss1, loss2, td_errors = compute_critic_loss(
                agent=self.agent,
                batch=batch,
                gamma=self.config.get('gamma', 0.99),
                device=self.device
            )

        # Check if losses are valid before backprop
        if not torch.isfinite(loss1) or not torch.isfinite(loss2):
            print(f"🔴 WARNING: Non-finite critic loss detected! Loss1: {loss1.item()}, Loss2: {loss2.item()}")
            return 0.0, {'loss1': 0.0, 'loss2': 0.0, 'q1_mean': 0.0, 'q2_mean': 0.0, 'q_diff': 0.0}

        # Train critic 1
        self.critic1_optimizer.zero_grad()
        loss1.backward()
        grad_norm1 = torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), max_norm=10.0)
        if grad_norm1 > 100.0:
            print(f"⚠️ Large gradient norm in critic1: {grad_norm1:.2f}")
        self.critic1_optimizer.step()

        # Train critic 2
        self.critic2_optimizer.zero_grad()
        loss2.backward()
        grad_norm2 = torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), max_norm=10.0)
        if grad_norm2 > 100.0:
            print(f"⚠️ Large gradient norm in critic2: {grad_norm2:.2f}")
        self.critic2_optimizer.step()

        # Update priorities for PER with CURRICULUM LEARNING (if enabled)
        if self.buffer.use_per:
            indices = [t['index'] for t in batch]
            use_curriculum = self.config.get('use_curriculum_learning', True)
            self.buffer.update_priorities(
                indices,
                td_errors,
                use_curriculum=use_curriculum,
                curriculum_percentiles=(0.3, 0.95)  # Focus on medium-difficulty samples
            )

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
        # Ensure we don't sample more than available (early in training)
        batch_size = min(self.config.get('batch_size', 1024), len(self.buffer))
        if batch_size < 32:  # Don't train with tiny batches
            return 0.0, {'entropy': 0.0, 'avg_advantage': 0.0, 'avg_q_value': 0.0, 'grad_norm': 0.0}

        batch = self.buffer.sample(batch_size)

        loss, info = compute_actor_loss(
            agent=self.agent,
            batch=batch,
            device=self.device,
            entropy_coef=self.config.get('entropy_coef', 0.05),
            action_diversity_coef=self.config.get('action_diversity_coef', 0.0),
            gamma=self.config.get('gamma', 0.99)
        )

        self.actor_optimizer.zero_grad()
        loss.backward()
        # Clip actor gradients more aggressively for stability (reduced from 10.0 to 0.5)
        total_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), max_norm=0.5)
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
            with self.profiler.profile('full_episode'):
                stats_list = self.train_episode_vectorized()

            # Record episode completions for profiler
            for _ in range(num_parallel):
                self.profiler.record_episode()

            # Add all stats to history
            for stats in stats_list:
                self.episode_history.append(stats)

            # Use first episode for logging
            stats = stats_list[0]

            # Collect portfolio values from all parallel episodes for histogram
            portfolio_values = [s['portfolio_value'] for s in stats_list]
            portfolio_hist = create_portfolio_histogram(portfolio_values, num_bins=10)

            # Update progress bar with key metrics including histogram
            pbar.set_postfix_str(
                f"Ep={self.episode-num_parallel+1}-{self.episode} | "
                f"R_ema={stats['ema_return']:+.4f} | "
                f"{portfolio_hist} | "
                f"C_ema={stats['ema_critic_loss']:.3f} | "
                f"A_ema={stats['ema_actor_loss']:.3f} | "
                f"H={stats['avg_entropy']:.3f} | "
                f"Q={stats['avg_q_value']:+.2f} | "
                f"Buf={stats['buffer_size']:,}"
            )

            # Log to WandB
            if use_wandb:
                # Compute portfolio distribution statistics
                portfolio_vals = np.array([s['portfolio_value'] for s in stats_list if np.isfinite(s['portfolio_value'])])
                if len(portfolio_vals) > 0:
                    port_min = np.min(portfolio_vals)
                    port_max = np.max(portfolio_vals)
                    port_median = np.median(portfolio_vals)
                    port_q25 = np.percentile(portfolio_vals, 25)
                    port_q75 = np.percentile(portfolio_vals, 75)
                else:
                    port_min = port_max = port_median = port_q25 = port_q75 = stats['portfolio_value']

                wandb.log({
                    'episode': self.episode,
                    'return': stats['return'],
                    'return_ema': stats['ema_return'],
                    'portfolio_value': stats['portfolio_value'],
                    'portfolio_value_ema': stats['ema_portfolio_value'],
                    # Portfolio distribution stats
                    'portfolio_min': port_min,
                    'portfolio_max': port_max,
                    'portfolio_median': port_median,
                    'portfolio_q25': port_q25,
                    'portfolio_q75': port_q75,
                    'portfolio_range': port_max - port_min,
                    'portfolio_iqr': port_q75 - port_q25,
                    # Losses and metrics
                    'critic_loss': stats['avg_critic_loss'],
                    'critic_loss_ema': stats['ema_critic_loss'],
                    'actor_loss': stats['avg_actor_loss'],
                    'actor_loss_ema': stats['ema_actor_loss'],
                    'entropy': stats['avg_entropy'],
                    'action_diversity_loss': stats.get('avg_action_diversity_loss', 0.0),
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

            # Save detailed portfolio histogram periodically
            save_histogram_interval = self.config.get('save_histogram_interval', 100)
            if save_histogram_interval > 0 and self.episode % save_histogram_interval < num_parallel:
                # Save episode-numbered version
                save_portfolio_histogram(
                    portfolio_values=portfolio_values,
                    episode=self.episode,
                    output_path=f'./results/portfolio_distribution_ep{self.episode}.png',
                    mode='training'
                )
                # Also save as "latest" for easy access
                save_portfolio_histogram(
                    portfolio_values=portfolio_values,
                    episode=self.episode,
                    output_path='./results/portfolio_distribution_latest.png',
                    mode='training'
                )

            # RETURN/RISK ANALYSIS: Periodically analyze buffer for risk patterns
            analysis_interval = self.config.get('risk_analysis_interval', 1000)
            if self.config.get('use_risk_analysis', False) and analysis_interval > 0 and self.episode % analysis_interval < num_parallel:
                print(f"\n{'='*80}")
                print(f"RETURN/RISK ANALYSIS (Episode {self.episode})")
                print(f"{'='*80}")
                if len(self.buffer) >= 5000:
                    try:
                        analysis = analyze_return_risk_relationship(self.buffer, num_samples=5000)

                        # Log summary statistics
                        print(f"\n1-Day Horizon Analysis:")
                        for quintile in range(1, 6):
                            key = f'1d_quintile_{quintile}'
                            if key in analysis:
                                stats = analysis[key]
                                print(f"   Q{quintile}: Sharpe={stats['sharpe_ratio']:+.2f}, "
                                      f"Loss Prob={stats['prob_loss']:.1%}, "
                                      f"Mean={stats['actual_mean_return']:+.4f}")

                        # Log to WandB if enabled
                        if use_wandb:
                            wandb.log({
                                'risk_analysis/1d_q5_sharpe': analysis.get('1d_quintile_5', {}).get('sharpe_ratio', 0),
                                'risk_analysis/1d_q5_prob_loss': analysis.get('1d_quintile_5', {}).get('prob_loss', 0),
                                'risk_analysis/1d_q1_sharpe': analysis.get('1d_quintile_1', {}).get('sharpe_ratio', 0),
                            }, step=self.episode)
                    except Exception as e:
                        print(f"   ⚠️  Risk analysis failed: {e}")
                else:
                    print(f"   Buffer too small ({len(self.buffer)} samples), skipping analysis")
                print(f"{'='*80}\n")

            # Run validation
            val_interval = self.config.get('val_interval', 500)
            if val_interval > 0 and self.episode % val_interval < num_parallel:
                val_stats = self.validate(
                    num_episodes=self.config.get('val_episodes', 10),
                    force_initial_trade=self.config.get('val_force_initial_trade', True)
                )

                # Log validation to WandB
                if use_wandb:
                    wandb.log({
                        'val_mean_return': val_stats['val_mean_return'],
                        'val_std_return': val_stats['val_std_return'],
                        'val_mean_portfolio_value': val_stats['val_mean_portfolio_value'],
                        'val_std_portfolio_value': val_stats['val_std_portfolio_value'],
                        'val_min_return': val_stats['val_min_return'],
                        'val_max_return': val_stats['val_max_return'],
                        'best_val_return': self.best_val_return,
                        'best_val_portfolio_value': self.best_val_portfolio_value,
                        'epochs_without_improvement': self.epochs_without_improvement,
                    }, step=self.episode)

                # Early stopping check
                early_stop_patience = self.config.get('early_stop_patience', 0)
                if early_stop_patience > 0 and self.epochs_without_improvement >= early_stop_patience:
                    print(f"\n⚠️  Early stopping: No improvement for {early_stop_patience} validation checks")
                    print(f"   Best val return: {self.best_val_return:+.4f}")
                    print(f"   Best val portfolio: ${self.best_val_portfolio_value/1000:.1f}k")
                    break

            # Save checkpoint
            if self.episode % self.config.get('save_interval', 1000) < num_parallel:
                self._save_checkpoint(self.episode)

            # Print profiling report
            if self.profiler.enabled:
                report_interval = self.config.get('profiling_report_interval', 100)
                if self.episode % report_interval < num_parallel:
                    self.profiler.print_summary(top_n=15)
                    # Save to CSV
                    os.makedirs('./profiling_results', exist_ok=True)
                    self.profiler.save_to_csv(f'./profiling_results/profile_ep{self.episode}.csv')

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
            'episode_history': self.episode_history,
            'val_history': self.val_history,
            'best_val_return': self.best_val_return,
            'best_val_portfolio_value': self.best_val_portfolio_value,
            # Reward normalization statistics
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_m2': self.reward_m2,
            'reward_count': self.reward_count,
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

    def _save_episode_history(self):
        """Save episode history to CSV."""
        df = pd.DataFrame(self.episode_history)
        csv_path = './checkpoints/actor_critic_training_history.csv'
        df.to_csv(csv_path, index=False)

    def validate(self, num_episodes: int = 10, force_initial_trade: bool = True) -> Dict:
        """
        Run validation episodes on held-out dates (no training).

        Validation simulates realistic production conditions:
        - Uses held-out DATES (most recent 2 months)
        - Uses ALL STOCKS (not just test subset) - realistic stock universe
        - Tests temporal generalization (train on past, validate on future)

        Args:
            num_episodes: Number of validation episodes to run
            force_initial_trade: If True, force a random buy on first step to ensure trading activity

        Returns:
            Dictionary of validation statistics
        """
        print(f"\n🔍 Running validation ({num_episodes} episodes on {len(self.val_dates)} val dates)...")
        if force_initial_trade:
            print(f"   🎲 Forced initial trade enabled for validation")
        print(f"   🌍 Using ALL stocks for validation (production-realistic conditions)")

        # Create temporary validation environment
        # Cap episode length to available validation days (avoid randint error)
        requested_episode_length = self.config.get('episode_length', 30)
        val_episode_length = min(len(self.val_dates) - 1, requested_episode_length)
        if val_episode_length < requested_episode_length:
            print(f"   ⚠️  Episode length capped at {val_episode_length} days (validation period only has {len(self.val_dates)} days)")

        val_env = TradingEnvironment(
            data_loader=self.data_loader,
            agent=self.agent,
            initial_capital=self.config.get('initial_capital', 100000),
            max_positions=self.config.get('max_positions', 1),
            episode_length=val_episode_length,
            device=self.device,
            trading_days_filter=self.val_dates,  # Use validation dates
            top_k_per_horizon=self.config.get('top_k_per_horizon', 10)
        )

        # Share state cache if available
        if hasattr(self.vec_env.ref_env, 'state_cache'):
            val_env.state_cache = self.vec_env.ref_env.state_cache
            val_env.price_cache = self.vec_env.ref_env.price_cache

        # Set networks to eval mode (CRITICAL: Must include feature_extractor!)
        self.agent.actor.eval()
        self.agent.critic1.eval()
        self.agent.critic2.eval()
        self.agent.feature_extractor.eval()  # FIX: Ensure predictor dropout is disabled

        val_returns = []
        val_portfolio_values = []
        all_actions = []  # Track all actions across episodes
        all_trades_count = 0  # Count total trades

        with torch.no_grad():
            for ep in range(num_episodes):
                val_env.reset()
                current_position = None
                current_is_short = False
                episode_return = 0.0

                for step in range(val_episode_length):
                    # Get current state - USE ALL STOCKS for validation (realistic production conditions)
                    all_cached_states = val_env.state_cache[val_env.current_date]
                    # Only filter out stocks without price data, but include ALL stocks (not just test set)
                    cached_states = {ticker: state for ticker, state in all_cached_states.items()
                                   if ticker in self.data_loader.prices_file}

                    if step == 0 and ep == 0:  # Debug first step of first episode
                        print(f"   📊 Validation debug:")
                        print(f"      Total cached states: {len(all_cached_states)}")
                        print(f"      Filtered states (with price data): {len(cached_states)}")
                        print(f"      Sample filtered tickers: {sorted(list(cached_states.keys()))[:20]}")

                        # Show ticker diversity
                        first_letters = {}
                        for ticker in cached_states.keys():
                            letter = ticker[0]
                            first_letters[letter] = first_letters.get(letter, 0) + 1
                        print(f"      Ticker diversity (first letter): {dict(sorted(first_letters.items())[:10])}")

                    # Use random sampling for validation to get diverse signal
                    # Same sampling fraction as training for robust performance measurement
                    top_k_per_horizon = self.config.get('top_k_per_horizon_sampling', 3)
                    top_k_stocks = get_top_k_stocks_per_horizon(
                        cached_states,
                        k=top_k_per_horizon,
                        sample_fraction=self.stock_sample_fraction  # 0.3 = sample diverse stocks
                    )
                    # Use deterministic=False to sample different stocks each validation episode
                    top_4_stocks = sample_top_4_from_top_k(top_k_stocks, sample_size=4, deterministic=False)

                    # Get bottom-4 stocks for shorting
                    bottom_4_stocks = get_bottom_4_stocks(cached_states, sample_fraction=self.stock_sample_fraction)

                    if step == 0 and ep == 0:
                        print(f"      Top-4 stocks: {[ticker for ticker, _ in top_4_stocks]}")

                    # Get cached prices and compute portfolio value
                    cached_prices = val_env.price_cache.get(val_env.current_date, {})
                    portfolio_value = val_env._portfolio_value_cached(cached_prices)

                    # Force an initial trade if requested (to ensure trading activity)
                    if force_initial_trade and step == 0 and current_position is None:
                        # Pick a random stock from top-4 that has data
                        import random
                        available_stocks = [ticker for ticker, _ in top_4_stocks if ticker in cached_states and ticker in cached_prices]
                        if available_stocks:
                            random_stock = random.choice(available_stocks)
                            action = [i+1 for i, (ticker, _) in enumerate(top_4_stocks) if ticker == random_stock][0]
                            trades = [{'action': 'BUY', 'ticker': random_stock}]
                            new_position = random_stock
                            new_is_short = False
                            if ep == 0:
                                print(f"      🎲 Forced initial trade: BUY {random_stock}")
                        else:
                            if ep == 0:
                                print(f"      ⚠️  No available stocks to trade, skipping forced trade")
                            trades = []
                            action = 0
                            new_position = None
                            new_is_short = False
                    else:
                        # Create states for top-4 stocks
                        states = {}

                        for ticker, _ in top_4_stocks:
                            if ticker in cached_states and ticker in cached_prices:
                                cached_state = cached_states[ticker]
                                price = cached_prices[ticker]
                                portfolio_context = val_env._create_portfolio_context_fast(
                                    ticker, price, portfolio_value
                                )
                                state = torch.cat([
                                    cached_state[:1444].to(self.device),
                                    portfolio_context.to(self.device)
                                ])
                                states[ticker] = state

                        # DEBUG: Log policy behavior for first episode, first 10 steps
                        if ep == 0 and (step < 10 or step % 10 == 0):
                            from rl.reduced_action_space import create_global_state
                            import torch.nn.functional as F

                            global_state = create_global_state(
                                top_4_stocks, bottom_4_stocks, states, current_position, is_short=current_is_short, device=self.device
                            )
                            with torch.no_grad():
                                logits = self.agent.actor(global_state.unsqueeze(0)).squeeze(0)
                                probs = F.softmax(logits, dim=-1)

                                # Get Q-values for debugging
                                q_values1 = self.agent.critic1(global_state.unsqueeze(0)).squeeze(0)
                                q_values2 = self.agent.critic2(global_state.unsqueeze(0)).squeeze(0)
                                q_values = torch.min(q_values1, q_values2)

                            print(f"\n      🔍 VALIDATION DEBUG [Ep {ep}, Step {step}, Date {val_env.current_date}]:")
                            print(f"         Current position: {current_position}")
                            print(f"         Top-4 stocks: {[t for t, _ in top_4_stocks]}")
                            print(f"         Portfolio value: ${portfolio_value:,.2f}")
                            print(f"         Cash: ${val_env.cash:,.2f}")
                            print(f"         Step index: {val_env.step_idx}")

                            # DEBUG: Print state statistics
                            print(f"         Global state shape: {global_state.shape}")
                            print(f"         Global state mean: {global_state.mean().item():.6f}")
                            print(f"         Global state std: {global_state.std().item():.6f}")
                            print(f"         Global state min/max: {global_state.min().item():.2f} / {global_state.max().item():.2f}")

                            # Show sample of portfolio context
                            portfolio_context_start = 4 * 1469
                            sample_portfolio = global_state[portfolio_context_start:portfolio_context_start+10]
                            print(f"         Sample portfolio context: {sample_portfolio.cpu().numpy()}")

                            print(f"         Logits: {logits.cpu().numpy()}")
                            print(f"         Action probs: {probs.cpu().numpy()}")
                            print(f"         Q-values: {q_values.cpu().numpy()}")
                            print(f"         Action 0 (HOLD) prob: {probs[0].item():.4f}")

                        # Select action deterministically (no exploration)
                        action, _, _, trades, new_position, new_is_short = self.agent.select_action_reduced(
                            top_4_stocks=top_4_stocks,
                            bottom_4_stocks=bottom_4_stocks,
                            states=states,
                            current_position=current_position,
                            current_is_short=current_is_short,
                            epsilon=0.0,  # No exploration
                            deterministic=True
                        )

                        if ep == 0 and (step < 10 or step % 10 == 0):
                            # Decode action meaning for new 9-action space
                            if action == 0:
                                action_meaning = "HOLD"
                            elif 1 <= action <= 4:
                                stock = [t for t, _ in top_4_stocks][action-1]
                                action_meaning = f"GO LONG {stock}"
                            elif 5 <= action <= 8:
                                stock = [t for t, _ in bottom_4_stocks][action-5]
                                action_meaning = f"GO SHORT {stock}"
                            else:
                                action_meaning = f"UNKNOWN ACTION {action}"

                            print(f"         Selected action: {action} ({action_meaning})")
                            print(f"         Trades: {trades}")
                            if action == 0:
                                print(f"         ⚠️  Agent chose HOLD (action 0)")

                    # Track actions and trades
                    all_actions.append(action)
                    all_trades_count += len(trades)

                    # Execute trades
                    # TODO: Environment needs to be updated to handle SHORT (3) and COVER (4) actions
                    actions = {}
                    for trade in trades:
                        ticker = trade['ticker']
                        if trade['action'] == 'BUY':
                            actions[ticker] = 1
                        elif trade['action'] == 'SELL':
                            actions[ticker] = 2
                        elif trade['action'] == 'SHORT':
                            actions[ticker] = 3  # SHORT (TODO: implement in environment)
                        elif trade['action'] == 'COVER':
                            actions[ticker] = 4  # COVER (TODO: implement in environment)

                    # Step environment
                    _, reward, done, info = val_env.step(actions)

                    # Update position from agent's decode_action_to_trades
                    current_position = new_position
                    current_is_short = new_is_short

                    episode_return += reward

                    if done:
                        break

                val_returns.append(episode_return)
                val_portfolio_values.append(info['portfolio_value'])

        # Restore training mode for actor/critics
        self.agent.actor.train()
        self.agent.critic1.train()
        self.agent.critic2.train()
        # NOTE: feature_extractor stays in eval mode (predictor is frozen and should not have dropout enabled)

        # Compute statistics
        val_stats = {
            'val_mean_return': np.mean(val_returns),
            'val_std_return': np.std(val_returns),
            'val_mean_portfolio_value': np.mean(val_portfolio_values),
            'val_std_portfolio_value': np.std(val_portfolio_values),
            'val_min_return': np.min(val_returns),
            'val_max_return': np.max(val_returns),
        }

        # Create histogram of validation portfolio values
        val_portfolio_hist = create_portfolio_histogram(val_portfolio_values, num_bins=10)

        print(f"   ✅ Val Return: {val_stats['val_mean_return']:+.4f} ± {val_stats['val_std_return']:.4f}")
        print(f"   ✅ Val Portfolio: ${val_stats['val_mean_portfolio_value']/1000:.1f}k ± ${val_stats['val_std_portfolio_value']/1000:.1f}k")
        print(f"   ✅ {val_portfolio_hist}")

        # Save detailed validation portfolio histogram
        # Save episode-numbered version
        save_portfolio_histogram(
            portfolio_values=val_portfolio_values,
            episode=self.episode,
            output_path=f'./results/portfolio_distribution_validation_ep{self.episode}.png',
            mode='validation'
        )
        # Also save as "latest" for easy access
        save_portfolio_histogram(
            portfolio_values=val_portfolio_values,
            episode=self.episode,
            output_path='./results/portfolio_distribution_validation_latest.png',
            mode='validation'
        )

        # Log action distribution
        import collections
        action_counts = collections.Counter(all_actions)
        total_steps = len(all_actions)
        print(f"   📊 Action Distribution ({total_steps} total steps):")
        for action_id in range(5):
            count = action_counts.get(action_id, 0)
            pct = 100.0 * count / total_steps if total_steps > 0 else 0
            action_name = "HOLD" if action_id == 0 else f"SWITCH-{action_id}"
            print(f"      Action {action_id} ({action_name}): {count:4d} ({pct:5.1f}%)")
        print(f"   📊 Total trades executed: {all_trades_count} (avg {all_trades_count/num_episodes:.1f} per episode)")

        # Track best validation performance
        if val_stats['val_mean_return'] > self.best_val_return:
            self.best_val_return = val_stats['val_mean_return']
            self.best_val_portfolio_value = val_stats['val_mean_portfolio_value']
            self.epochs_without_improvement = 0
            print(f"   🎯 New best validation performance!")
        else:
            self.epochs_without_improvement += 1

        self.val_history.append(val_stats)

        return val_stats

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

        # Restore epsilon (recalculate based on episode count)
        self.current_epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.episode / self.epsilon_decay_episodes)
        )

        # Restore curriculum (recalculate based on episode count)
        self.current_top_k = min(
            self.curriculum_end_k,
            self.curriculum_start_k + (self.curriculum_end_k - self.curriculum_start_k) * (self.episode / self.curriculum_episodes)
        )
        # Apply to reference environment
        top_k_int = int(round(self.current_top_k))
        self.vec_env.ref_env.top_k_per_horizon = top_k_int

        # Restore reward normalization statistics
        self.reward_mean = checkpoint.get('reward_mean', 0.0)
        self.reward_std = checkpoint.get('reward_std', 1.0)
        self.reward_m2 = checkpoint.get('reward_m2', 0.0)
        self.reward_count = checkpoint.get('reward_count', 0)

        # Restore episode history
        self.episode_history = checkpoint.get('episode_history', [])

        # Restore EMA values if available
        if self.episode_history:
            last_stats = self.episode_history[-1]
            self.ema_critic_loss = last_stats.get('ema_critic_loss', None)
            self.ema_actor_loss = last_stats.get('ema_actor_loss', None)
            self.ema_return = last_stats.get('ema_return', None)
            self.ema_portfolio_value = last_stats.get('ema_portfolio_value', None)

        # Restore validation tracking
        self.val_history = checkpoint.get('val_history', [])
        self.best_val_return = checkpoint.get('best_val_return', -float('inf'))
        self.best_val_portfolio_value = checkpoint.get('best_val_portfolio_value', 0.0)

        print(f"   Loaded {len(self.episode_history)} historical episodes")
        if self.val_history:
            print(f"   Loaded {len(self.val_history)} validation checkpoints")
            print(f"   Best validation return: {self.best_val_return:+.4f}")


if __name__ == '__main__':
    """
    ============================================================================
    ACTOR-CRITIC TRAINING WITH NOISE-ROBUST IMPROVEMENTS
    ============================================================================

    This configuration includes all improvements for handling noisy stock data:

    ✅ 0. Architecture Choice (use_attention_architecture=True)
       - Lightweight attention over stocks (permutation-invariant, sample-efficient)
       - ~10x fewer parameters than MLP, explicitly models stock relationships
       - Alternative: MLP architecture (fully-connected, larger networks)

    ✅ 1. Risk-Aware Rewards (automatic in environment)
       - Penalizes volatility and drawdowns
       - Makes agent risk-averse in noisy markets

    ✅ 2. Data Augmentation (use_data_augmentation=True)
       - Adds Gaussian noise to states during training
       - Improves robustness to measurement noise

    ✅ 3. Conservative Q-Learning / CQL (use_cql=True)
       - Prevents Q-value overestimation on unseen actions
       - Useful when training heavily offline

    ✅ 4. Hindsight Experience Replay / HER (use_her=True)
       - Learns from failed trades by relabeling goals
       - Teaches agent to avoid losses

    ✅ 5. Curriculum Learning (use_curriculum_learning=True)
       - Focuses on medium-difficulty samples
       - Avoids noise outliers and already-learned samples

    ✅ 6. Return/Risk Analysis (use_risk_analysis=True)
       - Periodic analysis of predicted returns vs realized risk
       - Helps understand which predictions are reliable

    All features can be enabled/disabled independently via config flags below.
    ============================================================================
    """

    # Configuration
    config = {
        # Data
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'num_test_stocks': 100,

        # Model
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',
        #'pretrained_critic_path': None,
        'state_dim': 11761,  # 8 stocks × 1469 + 9 position encoding (4 long + 4 short + position)
        'hidden_dim': 1024,  # Hidden dim for MLP architecture
        'attention_hidden_dim': 256,  # Hidden dim for attention architecture (smaller, more efficient)
        'action_dim': 9,  # HOLD + 4 long stocks + 4 short stocks

        # Architecture choice
        'use_attention_architecture': True,  # Use lightweight attention over stocks (recommended)
                                              # False = use MLP (larger, less sample-efficient)

        # Environment
        'initial_capital': 100000,
        'max_positions': 1,  # Single stock mode
        'episode_length': 30,  # Longer episodes to expose holding behavior (was 30)
        'top_k_per_horizon': 10,

        # Training
        'num_episodes': 500000,
        'num_parallel_envs': 128,  # Parallel environments for faster data collection
        'batch_size': 256,  # MASSIVELY increased for 100% GPU utilization (was 1024)
        'updates_per_transition': 0.10,  # Fewer but MUCH larger updates (was 0.25) - compensates for 8x batch size
        'buffer_capacity': 50000,  # Smaller buffer = fresher data
        'reward_scale': 100.0,  # Scale rewards (0.01 → 1.0) for better gradients
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,  # Increased from 5e-5 for faster learning
        'gamma': 0.99,
        'n_step': 3,  # N-step returns for better credit assignment (1=standard TD, 3-5=typical)
        'tau': 0.01,  # Increased from 0.005 for faster target network updates

        # Prioritized Experience Replay (PER)
        'use_per': True,  # Enable PER for better sample efficiency
        'per_alpha': 0.6,  # Prioritization exponent (0=uniform, 1=fully prioritized)
        'per_beta': 0.4,  # Importance sampling exponent (annealed to 1.0)
        'per_beta_increment': 0.001,  # Beta increment per sample

        # ========================================================================
        # NEW FEATURES FOR NOISY STOCK DATA (all improvements)
        # ========================================================================
        # 1. Risk-aware rewards (automatically enabled in environment)
        # 2. Data augmentation
        'use_data_augmentation': False,  # DISABLED: Can destabilize training
        'augmentation_noise_level': 0.01,  # 1% Gaussian noise
        # 3. Conservative Q-Learning (CQL)
        'use_cql': False,  # DISABLED: Can slow convergence
        'cql_alpha': 1.0,  # CQL penalty strength
        # 4. Hindsight Experience Replay (HER)
        'use_her': False,  # DISABLED: Can destabilize training with relabeled experiences
        'her_ratio': 0.5,  # Fraction of negative-reward transitions to relabel
        # 5. Curriculum learning (focus on medium-difficulty samples)
        'use_curriculum_learning': False,  # DISABLED: Can interfere with PER
        # 6. Return/Risk analysis
        'use_risk_analysis': True,  # Periodically analyze buffer (analysis only, doesn't affect training)
        'risk_analysis_interval': 1000,  # Analyze every N episodes
        # ========================================================================

        # Curriculum learning (progressive stock difficulty)
        'curriculum_start_k': 5,  # Start with top 5 stocks per horizon (easiest)
        'curriculum_end_k': 15,  # End with top 15 stocks per horizon (harder)
        'curriculum_episodes': 50000,  # Anneal difficulty over 50k episodes

        # Stock sampling diversity (prevents overfitting to specific stocks)
        'stock_sample_fraction': 0.3,  # Randomly sample 30% of stocks before selecting top-K
                                       # This ensures agent sees diverse stocks each episode
                                       # Set to 1.0 to disable (use all stocks)
        'entropy_coef': 0.0,  # Entropy regularization (DISABLED: can destabilize training)
        'action_diversity_coef': 0.0,  # Action diversity regularization (DISABLED: can destabilize training)

        # Exploration schedule (decaying epsilon-greedy)
        'epsilon_start': 0.2,  # Initial exploration rate (30% random actions)
        'epsilon_end': 0.01,  # Final exploration rate (1% random actions)
        'epsilon_decay_episodes': 100000,  # Decay over 100k episodes

        'actor_update_freq': 3,  # Update actor every 2 critic updates
        'freeze_critic_episodes': 0,  # No critic freezing

        # Logging
        'log_interval': 500,
        'save_interval': 5000,
        'use_wandb': True,
        'wandb_project': 'stock-rl-trading',
        'wandb_run_name': None,  # Auto-generated if None

        # Validation
        'val_months_back': 3,  # Use most recent N months for validation (increased from 2 to accommodate 60-day episodes)
        'val_interval': 1500,  # Run validation every N episodes (0 = disable)
        'val_episodes': 50,  # Number of validation episodes to run (increased for robust signal with stock sampling)
        'val_force_initial_trade': True,  # Force an initial buy in validation to ensure trading
        'early_stop_patience': 0,  # Stop if no improvement for N validation checks (0 = disable)

        # Stock selection randomization (prevents overfitting)
        'top_k_per_horizon_sampling': 3,  # Get top-3 per horizon (12 total), randomly sample 4

        # Resume training (set to checkpoint path to continue from)
        'resume_checkpoint': None,  # e.g., './checkpoints/actor_critic_ep10000.pt'

        # Caching
        'feature_cache_path': 'data/rl_feature_cache_4yr.h5',
        'state_cache_path': 'data/rl_state_cache_4yr.h5',

        # ========================================================================
        # PRE-COMPUTED STOCK SELECTIONS (MAJOR SPEEDUP)
        # ========================================================================
        # Pre-compute diverse stock selections once, reuse across training runs
        # Run: python rl/precompute_stock_selections.py
        # This eliminates CPU-intensive stock selection during training
        'use_precomputed_selections': True,  # Enable to use pre-computed selections
        'stock_selections_cache': 'data/rl_stock_selections_4yr.h5',
        # ========================================================================

        # ========================================================================
        # PROFILING (for identifying bottlenecks)
        # ========================================================================
        'enable_profiling': False,  # Enable detailed timing profiler
        'profiling_report_interval': 100,  # Show profiling stats every N episodes (frequent for debugging)
        # ========================================================================

        # Device
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu'
    }

    # Create trainer and train
    trainer = ActorCriticTrainer(config)
    trainer.train()
