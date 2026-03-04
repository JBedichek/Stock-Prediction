#!/usr/bin/env python3
"""
Walk-Forward Portfolio Training with Differentiable Selection

Combines:
1. Walk-forward validation (temporal folds, no look-ahead bias)
2. Differentiable top-k portfolio selection (end-to-end training from returns)

Each fold:
- Train on historical data with differentiable portfolio selection
- Evaluate on future data with actual portfolio returns
- Aggregate results across folds for robust performance estimate

Usage:
    python -m training.walk_forward_portfolio \
        --data all_complete_dataset.h5 \
        --prices actual_prices.h5 \
        --num-folds 5 \
        --top-k 5 \
        --selection gumbel
"""

import gc
import os
import sys
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleTransformerPredictor
from training.train_portfolio_differentiable import (
    DifferentiableTopKSelector,
    PortfolioDataset,
    collate_portfolio_batch,
    PortfolioModel,
    PortfolioLoss,
    gumbel_top_k,
    neuralsort_top_k,
    soft_attention_selection,
    StraightThroughTopK,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    get_rank,
    get_world_size
)


# =============================================================================
# Fold Result Dataclass
# =============================================================================

@dataclass
class PortfolioFoldResult:
    """Results from training and evaluating a single fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_days: int
    test_days: int

    # Training metrics
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int
    best_epoch: int

    # Test evaluation metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_daily_return_pct: float
    num_trading_days: int


# =============================================================================
# Walk-Forward Portfolio Trainer
# =============================================================================

class WalkForwardPortfolioTrainer:
    """
    Walk-forward training with differentiable portfolio selection.
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        # Date range (optional filtering)
        start_date: str = None,
        end_date: str = None,
        # Walk-forward config
        num_folds: int = 5,
        mode: str = 'expanding',
        min_train_months: int = 12,
        test_months: int = 3,
        val_ratio: float = 0.1,
        gap_days: int = 5,
        # Model config
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        seq_len: int = 60,
        # Portfolio config
        top_k: int = 5,
        selection_method: str = 'gumbel',
        initial_temperature: float = 1.0,
        min_temperature: float = 0.2,  # Increased default
        horizon_days: int = 1,
        horizon_idx: int = 0,
        # Gumbel-softmax improvements
        annealing_schedule: str = 'exponential',
        top_m_filter: int = 0,
        decoupled_backward_temp: bool = False,
        backward_temp_ratio: float = 2.0,
        learnable_temperature: bool = False,
        # Training config
        epochs_per_fold: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accumulation_steps: int = 1,
        early_stopping_patience: int = 5,
        max_stocks_per_day: int = 300,
        # Loss weights
        return_weight: float = 1.0,
        sharpe_weight: float = 0.1,
        concentration_weight: float = 0.01,
        confidence_weight: float = 0.1,
        batch_entropy_weight: float = 0.01,
        auxiliary_loss_weight: float = 0.1,
        auxiliary_loss_type: str = 'ranking',
        ranking_only: bool = False,  # Use ONLY ranking loss (more data-efficient)
        # Evaluation
        transaction_cost: float = 0.001,  # 0.1% round-trip cost
        eval_batch_size: int = 8,  # Larger batch for faster evaluation
        eval_every_n_steps: int = 100,  # Evaluate on val set every N optimizer steps (0=disable)
        loss_smoothing_window: int = 50,  # EMA window for smoothed loss curves
        save_loss_curves_every: int = 50,  # Save loss curve PNG every N steps (0=only at end)
        # Performance
        use_amp: bool = True,  # Automatic mixed precision
        use_compile: bool = False,  # torch.compile() - requires PyTorch 2.0+
        use_gradient_checkpointing: bool = False,  # Trade compute for memory
        # Other
        device: str = 'cuda',
        seed: int = 42,
        checkpoint_dir: str = 'checkpoints/walk_forward_portfolio'
    ):
        self.dataset_path = dataset_path
        self.prices_path = prices_path

        # Date range
        self.start_date = start_date
        self.end_date = end_date

        # Walk-forward config
        self.num_folds = num_folds
        self.mode = mode
        self.min_train_months = min_train_months
        self.test_months = test_months
        self.val_ratio = val_ratio
        self.gap_days = gap_days

        # Model config
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = seq_len

        # Portfolio config
        self.top_k = top_k
        self.selection_method = selection_method
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.horizon_days = horizon_days
        self.horizon_idx = horizon_idx
        # Gumbel-softmax improvements
        self.annealing_schedule = annealing_schedule
        self.top_m_filter = top_m_filter
        self.decoupled_backward_temp = decoupled_backward_temp
        self.backward_temp_ratio = backward_temp_ratio
        self.learnable_temperature = learnable_temperature

        # Training config
        self.epochs_per_fold = epochs_per_fold
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.max_stocks_per_day = max_stocks_per_day

        # Loss weights
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.concentration_weight = concentration_weight
        self.confidence_weight = confidence_weight
        self.batch_entropy_weight = batch_entropy_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_loss_type = auxiliary_loss_type
        self.ranking_only = ranking_only

        # Evaluation
        self.transaction_cost = transaction_cost
        self.eval_batch_size = eval_batch_size
        self.eval_every_n_steps = eval_every_n_steps
        self.loss_smoothing_window = loss_smoothing_window
        self.save_loss_curves_every = save_loss_curves_every

        # Performance
        self.use_amp = use_amp and device != 'cpu'  # AMP only works on CUDA
        self.use_compile = use_compile
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Other
        self.device = device
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.cache_dir = os.path.join(checkpoint_dir, 'data_cache')
        self.ddp = False  # Will be set externally if using DDP

        # Results storage
        self.fold_results: List[PortfolioFoldResult] = []

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load date information
        self._load_date_info()

    def _load_date_info(self):
        """Load available dates and determine valid range."""
        if is_main_process():
            print(f"\nLoading date information...")

        with h5py.File(self.dataset_path, 'r') as f:
            sample_ticker = list(f.keys())[0]
            dates_bytes = f[sample_ticker]['dates'][:]
            self.feature_dates = sorted([d.decode('utf-8') if isinstance(d, bytes) else d
                                         for d in dates_bytes])
            self.input_dim = f[sample_ticker]['features'].shape[1]

        with h5py.File(self.prices_path, 'r') as f:
            sample_ticker = list(f.keys())[0]
            dates_bytes = f[sample_ticker]['dates'][:]
            self.price_dates = sorted([d.decode('utf-8') if isinstance(d, bytes) else d
                                       for d in dates_bytes])

        # Use intersection of dates
        self.all_dates = sorted(set(self.feature_dates) & set(self.price_dates))

        # Calculate usable range (need seq_len history + horizon_days future)
        first_usable_idx = self.seq_len
        last_usable_idx = len(self.all_dates) - self.horizon_days - 1

        if first_usable_idx >= last_usable_idx:
            raise ValueError(f"Not enough data for seq_len={self.seq_len}")

        self.usable_dates = self.all_dates[first_usable_idx:last_usable_idx]

        # Filter by date range if specified
        if self.start_date is not None:
            self.usable_dates = [d for d in self.usable_dates if d >= self.start_date]
        if self.end_date is not None:
            self.usable_dates = [d for d in self.usable_dates if d <= self.end_date]

        if len(self.usable_dates) == 0:
            raise ValueError(f"No usable dates in range {self.start_date} to {self.end_date}")

        if is_main_process():
            print(f"  Feature dates: {self.feature_dates[0]} to {self.feature_dates[-1]} ({len(self.feature_dates)})")
            print(f"  Price dates: {self.price_dates[0]} to {self.price_dates[-1]} ({len(self.price_dates)})")
            if self.start_date or self.end_date:
                print(f"  Date filter: {self.start_date or 'start'} to {self.end_date or 'end'}")
            print(f"  Usable dates: {self.usable_dates[0]} to {self.usable_dates[-1]} ({len(self.usable_dates)})")
            print(f"  Input dimension: {self.input_dim}")

    def _get_folds(self) -> List[Tuple[List[str], List[str]]]:
        """Generate temporal folds."""
        days_per_month = 21
        min_train_days = self.min_train_months * days_per_month
        test_days = self.test_months * days_per_month

        total_days = len(self.usable_dates)

        # Check if we have enough data
        min_required = min_train_days + self.gap_days + test_days
        if total_days < min_required:
            print(f"  WARNING: Only {total_days} usable days, need {min_required}")
            print(f"  Using single fold with 70/30 split")
            split_idx = int(total_days * 0.7)
            train_dates = self.usable_dates[:split_idx]
            test_dates = self.usable_dates[split_idx + self.gap_days:]
            if len(train_dates) > 0 and len(test_dates) > 0:
                return [(train_dates, test_dates)]
            raise ValueError("Not enough data for even a single fold")

        # Calculate max possible folds if num_folds is 0 (auto mode)
        if self.num_folds == 0:
            # Auto: calculate max folds to span entire dataset
            first_test_start = min_train_days + self.gap_days
            remaining_days = total_days - first_test_start
            max_folds = max(1, remaining_days // test_days)
            if is_main_process():
                print(f"  Auto mode: {max_folds} folds to span entire dataset")
        else:
            max_folds = self.num_folds

        folds = []

        if self.mode == 'expanding':
            first_test_start = min_train_days + self.gap_days

            for fold_idx in range(max_folds):
                test_start_idx = first_test_start + fold_idx * test_days
                test_end_idx = min(test_start_idx + test_days, total_days)

                if test_start_idx >= total_days:
                    break

                train_end_idx = test_start_idx - self.gap_days
                train_start_idx = 0

                train_dates = self.usable_dates[train_start_idx:train_end_idx]
                test_dates = self.usable_dates[test_start_idx:test_end_idx]

                if len(test_dates) > 0 and len(train_dates) > 0:
                    folds.append((train_dates, test_dates))

        elif self.mode == 'sliding':
            train_days = min_train_days
            stride = test_days

            for fold_idx in range(max_folds):
                test_start_idx = train_days + self.gap_days + fold_idx * stride
                test_end_idx = min(test_start_idx + test_days, total_days)

                if test_start_idx >= total_days:
                    break

                train_end_idx = test_start_idx - self.gap_days
                train_start_idx = max(0, train_end_idx - train_days)

                train_dates = self.usable_dates[train_start_idx:train_end_idx]
                test_dates = self.usable_dates[test_start_idx:test_end_idx]

                if len(test_dates) > 0 and len(train_dates) > 0:
                    folds.append((train_dates, test_dates))

        return folds

    def _create_model(self, wrap_ddp: bool = False) -> PortfolioModel:
        """Create a fresh portfolio model.

        Args:
            wrap_ddp: If True and self.ddp is True, wrap with DDP
        """
        encoder = SimpleTransformerPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            num_pred_days=4,
            pred_mode='regression',
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        model = PortfolioModel(
            encoder=encoder,
            k=self.top_k,
            selection_method=self.selection_method,
            initial_temperature=self.initial_temperature,
            min_temperature=self.min_temperature,
            horizon_idx=self.horizon_idx,
            annealing_schedule=self.annealing_schedule,
            top_m_filter=self.top_m_filter,
            decoupled_backward_temp=self.decoupled_backward_temp,
            backward_temp_ratio=self.backward_temp_ratio,
            learnable_temperature=self.learnable_temperature,
        )

        model = model.to(self.device)

        # Compile model for faster execution (PyTorch 2.0+)
        if self.use_compile:
            try:
                model = torch.compile(model)
                if is_main_process():
                    print("  Model compiled with torch.compile()")
            except Exception as e:
                if is_main_process():
                    print(f"  Warning: torch.compile() failed: {e}")

        # Wrap with DDP if requested
        if wrap_ddp and self.ddp:
            # find_unused_parameters=True needed for ranking_only mode (confidence head unused)
            model = DDP(model, device_ids=[get_rank()], output_device=get_rank(),
                       find_unused_parameters=self.ranking_only)

        return model

    def _train_fold(
        self,
        fold_idx: int,
        train_dates: List[str],
        model: PortfolioModel
    ) -> Tuple[float, float, int, int, List[float], List[float]]:
        """
        Train model on a single fold.

        Returns:
            (final_train_loss, best_val_loss, epochs_trained, best_epoch,
             train_losses, val_losses)
        """
        if is_main_process():
            print(f"\n  Training on {len(train_dates)} days...")

        # Split into train/val
        split_idx = int(len(train_dates) * (1 - self.val_ratio))
        actual_train_dates = train_dates[:split_idx]
        val_dates = train_dates[split_idx:]

        if is_main_process():
            print(f"    Train: {actual_train_dates[0]} to {actual_train_dates[-1]} ({len(actual_train_dates)} days)")
            print(f"    Val: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")

        # Create datasets with shared memory-mapped cache
        # Rank 0 builds cache first, others wait and load from it
        rank = get_rank()
        world_size = get_world_size()

        if rank == 0:
            train_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=actual_train_dates[0],
                end_date=actual_train_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )
            val_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=val_dates[0],
                end_date=val_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )

        # Sync before other ranks load
        if self.ddp:
            dist.barrier()

        if rank != 0:
            train_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=actual_train_dates[0],
                end_date=actual_train_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )
            val_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=val_dates[0],
                end_date=val_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )

        # Sync after all ranks loaded
        if self.ddp:
            dist.barrier()

        # Create samplers for DDP
        train_sampler = None
        val_sampler = None
        if self.ddp:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_portfolio_batch,
            num_workers=0,  # Avoid memory duplication with DDP
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_portfolio_batch,
            num_workers=0,  # Avoid memory duplication with DDP
            pin_memory=True
        )

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler with warmup
        num_training_steps = len(train_loader) * self.epochs_per_fold // self.grad_accumulation_steps
        warmup_steps = min(100, num_training_steps // 10)  # 10% warmup, max 100 steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))  # Don't go below 10% of initial LR

        scheduler = LambdaLR(optimizer, lr_lambda)
        scheduler_step_count = 0

        # Loss function
        loss_fn = PortfolioLoss(
            return_weight=self.return_weight,
            sharpe_weight=self.sharpe_weight,
            concentration_weight=self.concentration_weight,
            confidence_weight=self.confidence_weight,
            batch_entropy_weight=self.batch_entropy_weight,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
            auxiliary_loss_type=self.auxiliary_loss_type,
            ranking_only=self.ranking_only,
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        best_step = 0
        epochs_without_improvement = 0
        best_model_state = None

        train_losses = []  # Epoch-level
        val_losses = []    # Epoch-level

        # Step-level tracking for detailed loss curves
        step_train_losses = []  # Raw losses at each optimizer step
        step_val_losses = []    # Val losses at eval checkpoints
        step_eval_points = []   # Which steps we evaluated at
        global_step = 0         # Total optimizer steps across all epochs

        # Setup AMP scaler
        scaler = GradScaler('cuda') if self.use_amp else None
        amp_context = lambda: autocast('cuda') if self.use_amp else nullcontext()

        def compute_smoothed_losses(losses, window):
            """Compute EMA smoothed losses."""
            if len(losses) == 0:
                return []
            smoothed = []
            alpha = 2.0 / (window + 1)
            ema = losses[0]
            for loss in losses:
                ema = alpha * loss + (1 - alpha) * ema
                smoothed.append(ema)
            return smoothed

        def quick_val_loss(model, val_loader, loss_fn, amp_ctx, max_batches=10):
            """Quick validation loss on subset of data."""
            model.eval()
            val_losses_tmp = []
            with torch.no_grad():
                for i, (features, returns, masks) in enumerate(val_loader):
                    if i >= max_batches:
                        break
                    features = features.to(self.device, non_blocking=True)
                    returns = returns.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    with amp_ctx():
                        scores, weights, confidence = model(features, masks)
                        base_model = model.module if hasattr(model, 'module') else model
                        portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)
                        loss, _ = loss_fn(portfolio_returns, weights, confidence, returns, masks, scores)

                    val_losses_tmp.append(loss.item())
            model.train()
            return np.mean(val_losses_tmp) if val_losses_tmp else float('inf')

        for epoch in range(self.epochs_per_fold):
            # Set epoch for distributed sampler (ensures proper shuffling)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train
            model.train()
            epoch_train_losses = []
            optimizer.zero_grad()

            train_pbar = tqdm(
                train_loader,
                desc=f"    Epoch {epoch+1}/{self.epochs_per_fold} [Train]",
                leave=False,
                disable=not is_main_process(),
                mininterval=1.0  # Reduce update frequency for speed
            )

            num_steps = len(train_loader)
            for step_idx, (features, returns, masks) in enumerate(train_pbar):
                features = features.to(self.device, non_blocking=True)
                returns = returns.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # Determine if we should sync gradients (only on last accumulation step)
                is_accumulating = (step_idx + 1) % self.grad_accumulation_steps != 0 and (step_idx + 1) != num_steps
                sync_context = model.no_sync() if (self.ddp and is_accumulating) else nullcontext()

                with sync_context:
                    with amp_context():
                        scores, weights, confidence = model(features, masks)
                        base_model = model.module if hasattr(model, 'module') else model
                        portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)
                        loss, _ = loss_fn(portfolio_returns, weights, confidence, returns, masks, scores)

                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.grad_accumulation_steps

                    # Backward pass (outside autocast but inside no_sync)
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                epoch_train_losses.append(loss.item())

                # Step optimizer every grad_accumulation_steps or at end of epoch
                if not is_accumulating:
                    if self.grad_clip > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # Step the learning rate scheduler
                    scheduler.step()
                    scheduler_step_count += 1
                    global_step += 1

                    # Track step-level train loss (average of accumulated steps)
                    recent_losses = epoch_train_losses[-self.grad_accumulation_steps:]
                    step_train_losses.append(np.mean(recent_losses))

                    # Periodic validation evaluation
                    if self.eval_every_n_steps > 0 and global_step % self.eval_every_n_steps == 0:
                        val_loss = quick_val_loss(model, val_loader, loss_fn, amp_context)
                        step_val_losses.append(val_loss)
                        step_eval_points.append(global_step)

                        # Check if this is best
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = global_step
                            model_to_save = model.module if hasattr(model, 'module') else model
                            best_model_state = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}

                        if is_main_process():
                            smoothed = compute_smoothed_losses(step_train_losses, self.loss_smoothing_window)
                            print(f"      Step {global_step}: train={smoothed[-1]:.4f}, val={val_loss:.4f}")

                    # Save loss curve periodically
                    if (self.save_loss_curves_every > 0 and
                        global_step % self.save_loss_curves_every == 0 and
                        is_main_process()):
                        self._save_step_loss_curves(
                            fold_idx, step_train_losses, step_val_losses,
                            step_eval_points, best_step, global_step
                        )

                    # Clear CUDA cache to prevent memory fragmentation
                    if self.device != 'cpu' and global_step % (self.grad_accumulation_steps * 4) == 0:
                        torch.cuda.empty_cache()

                current_lr = optimizer.param_groups[0]['lr']
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.2e}', 'step': global_step})

            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)

            # Validate
            model.eval()
            epoch_val_losses = []

            with torch.no_grad():
                for features, returns, masks in val_loader:
                    features = features.to(self.device, non_blocking=True)
                    returns = returns.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    with amp_context():
                        scores, weights, confidence = model(features, masks)
                        base_model = model.module if hasattr(model, 'module') else model
                        portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)
                        loss, _ = loss_fn(portfolio_returns, weights, confidence, returns, masks, scores)

                    epoch_val_losses.append(loss.item())

            avg_val_loss = np.mean(epoch_val_losses)
            val_losses.append(avg_val_loss)

            # Temperature annealing (get underlying model if DDP-wrapped)
            progress = (epoch + 1) / self.epochs_per_fold
            base_model = model.module if hasattr(model, 'module') else model
            base_model.anneal_temperature(progress)

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                # Get state dict from base model (unwrap DDP if needed)
                model_to_save = model.module if hasattr(model, 'module') else model
                best_model_state = {k: v.cpu().clone() for k, v in model_to_save.state_dict().items()}
                if is_main_process():
                    print(f"    Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} [BEST]")
            else:
                epochs_without_improvement += 1
                if is_main_process():
                    print(f"    Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} "
                          f"(no improvement for {epochs_without_improvement})")

            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                if is_main_process():
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best model (load into base model, unwrap DDP if needed)
        if best_model_state is not None:
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if is_main_process():
                print(f"    Restored best model from epoch {best_epoch} (step {best_step})")

        # Plot epoch-level loss curves
        self._plot_loss_curves(fold_idx, train_losses, val_losses, best_epoch)

        # Save final step-level loss curves with smoothing
        if is_main_process() and len(step_train_losses) > 0:
            self._save_step_loss_curves(
                fold_idx, step_train_losses, step_val_losses,
                step_eval_points, best_step, global_step, final=True
            )

        # Clean up datasets and loaders
        del train_loader, val_loader, train_dataset, val_dataset
        del optimizer, loss_fn
        if best_model_state is not None:
            del best_model_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_train_loss, best_val_loss, len(train_losses), best_epoch, train_losses, val_losses

    def _evaluate_fold(
        self,
        fold_idx: int,
        test_dates: List[str],
        model: PortfolioModel
    ) -> Dict:
        """Evaluate model on test dates."""
        if is_main_process():
            print(f"\n  Evaluating on {len(test_dates)} days...")

        # Create test dataset with shared memory-mapped cache
        rank = get_rank()
        world_size = get_world_size()

        if rank == 0:
            test_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=test_dates[0],
                end_date=test_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )

        if self.ddp:
            dist.barrier()

        if rank != 0:
            test_dataset = PortfolioDataset(
                dataset_path=self.dataset_path,
                prices_path=self.prices_path,
                start_date=test_dates[0],
                end_date=test_dates[-1],
                seq_len=self.seq_len,
                horizon_days=self.horizon_days,
                max_stocks_per_day=self.max_stocks_per_day,
                cache_dir=self.cache_dir,
                rank=rank,
                world_size=world_size
            )

        if self.ddp:
            dist.barrier()

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,  # Larger batch for faster evaluation
            shuffle=False,
            collate_fn=collate_portfolio_batch,
            num_workers=0  # Avoid memory duplication
        )

        model.eval()

        daily_returns = []
        capital = 100000.0
        capital_history = [capital]

        # AMP context for evaluation
        amp_context = lambda: autocast('cuda') if self.use_amp else nullcontext()

        with torch.no_grad():
            for features, returns, masks in tqdm(test_loader, desc="    Testing", leave=False,
                                                   disable=not is_main_process(), mininterval=1.0):
                features = features.to(self.device, non_blocking=True)
                returns = returns.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with amp_context():
                    # Get selection
                    scores, weights, confidence = model(features, masks, hard=True)

                    # Compute portfolio return (before transaction costs)
                    base_model = model.module if hasattr(model, 'module') else model
                    portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)

                # Process each day in the batch
                batch_size = portfolio_returns.shape[0]
                for i in range(batch_size):
                    day_return = portfolio_returns[i].item()

                    # Apply transaction costs (if any stocks were selected)
                    num_selected = (weights[i] > 0.5).sum().item()
                    if num_selected > 0 and self.transaction_cost > 0:
                        day_return -= self.transaction_cost

                    daily_returns.append(day_return)
                    capital *= (1 + day_return)
                    capital_history.append(capital)

        # Compute metrics using NON-OVERLAPPING returns
        # The daily_returns are horizon-period returns (e.g., 5-day returns)
        # To avoid inflated metrics from overlapping trades, use every horizon_days-th return
        all_returns = np.array(daily_returns)
        non_overlapping_returns = all_returns[::self.horizon_days]

        # Recompute capital history with non-overlapping returns
        capital = 100000.0
        capital_history = [capital]
        for ret in non_overlapping_returns:
            capital *= (1 + ret)
            capital_history.append(capital)

        total_return = (capital / 100000.0 - 1) * 100
        avg_daily_return = np.mean(non_overlapping_returns) * 100

        # Sharpe ratio (annualized based on holding period)
        # With horizon_days holding period, there are 252/horizon_days trades per year
        trades_per_year = 252 / self.horizon_days
        if len(non_overlapping_returns) > 1 and np.std(non_overlapping_returns) > 1e-6:
            sharpe = (np.mean(non_overlapping_returns) / np.std(non_overlapping_returns)) * np.sqrt(trades_per_year)
            sharpe = np.clip(sharpe, -100, 100)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = capital_history[0]
        max_drawdown = 0.0
        for cap in capital_history:
            if cap > peak:
                peak = cap
            drawdown = (peak - cap) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Win rate (on non-overlapping returns)
        win_rate = np.mean(non_overlapping_returns > 0) * 100 if len(non_overlapping_returns) > 0 else 0

        results = {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'win_rate': win_rate,
            'avg_daily_return_pct': avg_daily_return,
            'num_trading_days': len(non_overlapping_returns),  # Non-overlapping trades
            'daily_returns': non_overlapping_returns.tolist(),  # Non-overlapping returns
            'capital_history': capital_history
        }

        # Generate backtest plot
        self._plot_backtest_results(
            fold_idx=fold_idx,
            test_start=test_dates[0],
            test_end=test_dates[-1],
            daily_returns=non_overlapping_returns,
            capital_history=capital_history,
            eval_results=results
        )

        # Clean up test dataset
        del test_loader, test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _plot_loss_curves(
        self,
        fold_idx: int,
        train_losses: List[float],
        val_losses: List[float],
        best_epoch: int
    ):
        """Generate loss curve plot."""
        # Only plot on main process
        if not is_main_process():
            return

        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)

        if best_epoch > 0:
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
                       label=f'Best Epoch ({best_epoch})')
            plt.scatter([best_epoch], [val_losses[best_epoch-1]], color='g', s=100, zorder=5)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold_idx + 1} - Portfolio Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_loss.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _save_step_loss_curves(
        self,
        fold_idx: int,
        step_train_losses: List[float],
        step_val_losses: List[float],
        step_eval_points: List[int],
        best_step: int,
        current_step: int,
        final: bool = False
    ):
        """
        Save smoothed step-level loss curves as PNG.

        Creates a detailed plot showing:
        - Raw training loss (faded)
        - EMA-smoothed training loss
        - Validation loss at evaluation checkpoints
        - Best step marker
        """
        if not is_main_process() or len(step_train_losses) == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        steps = np.arange(1, len(step_train_losses) + 1)

        # Compute smoothed losses with different windows
        def ema_smooth(data, window):
            if len(data) == 0:
                return []
            alpha = 2.0 / (window + 1)
            smoothed = [data[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[-1])
            return smoothed

        smoothed_train = ema_smooth(step_train_losses, self.loss_smoothing_window)
        smoothed_train_tight = ema_smooth(step_train_losses, max(10, self.loss_smoothing_window // 5))

        # Top plot: Training loss
        ax1 = axes[0]
        ax1.plot(steps, step_train_losses, 'b-', alpha=0.2, linewidth=0.5, label='Raw')
        ax1.plot(steps, smoothed_train_tight, 'b-', alpha=0.5, linewidth=1, label=f'EMA-{max(10, self.loss_smoothing_window // 5)}')
        ax1.plot(steps, smoothed_train, 'b-', linewidth=2, label=f'EMA-{self.loss_smoothing_window}')

        # Mark best step
        if best_step > 0 and best_step <= len(smoothed_train):
            ax1.axvline(x=best_step, color='g', linestyle='--', alpha=0.7, label=f'Best Step ({best_step})')
            ax1.scatter([best_step], [smoothed_train[best_step-1]], color='g', s=80, zorder=5)

        ax1.set_ylabel('Training Loss')
        ax1.set_title(f'Fold {fold_idx + 1} - Step-Level Training Loss (Smoothed)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Set y-axis limits to exclude outliers
        if len(smoothed_train) > 10:
            y_vals = smoothed_train[10:]  # Skip first few noisy steps
            y_min, y_max = np.percentile(y_vals, [1, 99])
            margin = (y_max - y_min) * 0.1
            ax1.set_ylim(y_min - margin, y_max + margin)

        # Bottom plot: Validation loss at checkpoints
        ax2 = axes[1]
        if len(step_val_losses) > 0 and len(step_eval_points) > 0:
            ax2.plot(step_eval_points, step_val_losses, 'r.-', linewidth=2, markersize=8, label='Val Loss')

            # Smooth val losses too
            if len(step_val_losses) > 3:
                smoothed_val = ema_smooth(step_val_losses, min(10, len(step_val_losses) // 2))
                ax2.plot(step_eval_points, smoothed_val, 'r-', linewidth=2, alpha=0.5, label='Val Smoothed')

            # Mark best
            if best_step > 0 and best_step in step_eval_points:
                idx = step_eval_points.index(best_step)
                ax2.scatter([best_step], [step_val_losses[idx]], color='g', s=100, zorder=5)
                ax2.axvline(x=best_step, color='g', linestyle='--', alpha=0.7)

            ax2.legend(loc='upper right')

            # Set y-axis limits
            if len(step_val_losses) > 1:
                y_min, y_max = min(step_val_losses), max(step_val_losses)
                margin = (y_max - y_min) * 0.1
                ax2.set_ylim(y_min - margin, y_max + margin)

        ax2.set_xlabel('Optimizer Step')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss at Checkpoints')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with step number in filename for iterative saves
        if final:
            plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_step_loss_final.png')
        else:
            plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_step_loss_step{current_step}.png')

        plt.savefig(plot_path, dpi=150)
        plt.close()

        # Also save the loss data as numpy for later analysis
        if final:
            np.savez(
                os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_loss_history.npz'),
                step_train_losses=np.array(step_train_losses),
                step_val_losses=np.array(step_val_losses),
                step_eval_points=np.array(step_eval_points),
                smoothed_train=np.array(smoothed_train),
                best_step=best_step
            )

    def _plot_backtest_results(
        self,
        fold_idx: int,
        test_start: str,
        test_end: str,
        daily_returns: np.ndarray,
        capital_history: List[float],
        eval_results: Dict
    ):
        """Generate backtest performance plots for a fold."""
        # Only plot on main process
        if not is_main_process():
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Normalize capital to start at 1.0 for easy comparison
        portfolio_values = np.array(capital_history) / capital_history[0]

        # 1. Portfolio value over time (equity curve)
        ax = axes[0, 0]
        ax.plot(portfolio_values, linewidth=1.5, color='#2ecc71')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(range(len(portfolio_values)), 1.0, portfolio_values,
                       where=(portfolio_values >= 1.0), alpha=0.3, color='green')
        ax.fill_between(range(len(portfolio_values)), 1.0, portfolio_values,
                       where=(portfolio_values < 1.0), alpha=0.3, color='red')
        ax.set_title('Portfolio Value (Equity Curve)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Portfolio Value (normalized)')
        ax.grid(alpha=0.3)

        # 2. Daily returns distribution
        ax = axes[0, 1]
        returns_pct = daily_returns * 100
        ax.hist(returns_pct, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(returns_pct.mean(), color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {returns_pct.mean():.3f}%')
        ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Drawdown over time
        ax = axes[1, 0]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        ax.fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.7, color='#e74c3c')
        ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(alpha=0.3)

        # 4. Cumulative returns with key metrics
        ax = axes[1, 1]
        cumulative_returns = (portfolio_values - 1) * 100
        ax.plot(cumulative_returns, linewidth=2, color='#9b59b6')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Cumulative Return', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return (%)')
        ax.grid(alpha=0.3)

        # Add text box with metrics
        metrics_text = (
            f"Total Return: {eval_results['total_return_pct']:+.2f}%\n"
            f"Sharpe Ratio: {eval_results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {eval_results['max_drawdown_pct']:.2f}%\n"
            f"Win Rate: {eval_results['win_rate']:.1f}%\n"
            f"Trading Days: {eval_results['num_trading_days']}"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Fold {fold_idx + 1} Backtest: {test_start} to {test_end}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_backtest.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved backtest plot: {plot_path}")

    def run(self) -> List[PortfolioFoldResult]:
        """Run walk-forward training and evaluation."""
        if is_main_process():
            print(f"\n{'='*80}")
            print("WALK-FORWARD PORTFOLIO TRAINING")
            print(f"{'='*80}")

            print(f"\nConfiguration:")
            print(f"  Mode: {self.mode}")
            print(f"  Folds: {self.num_folds}")
            print(f"  Top-k: {self.top_k}")
            print(f"  Selection: {self.selection_method}")
            print(f"  Horizon: {self.horizon_days} days")
            print(f"  Model: {self.hidden_dim}d, {self.num_layers}L, {self.num_heads}H")
            print(f"  Temperature: {self.initial_temperature} -> {self.min_temperature} ({self.annealing_schedule})")
            if self.top_m_filter > 0:
                print(f"  Top-m pre-filter: {self.top_m_filter}")
            if self.decoupled_backward_temp:
                print(f"  Decoupled backward temp: ratio={self.backward_temp_ratio}")
            if self.learnable_temperature:
                print(f"  Learnable temperature: enabled")
            if self.batch_entropy_weight > 0 and not self.ranking_only:
                print(f"  Batch entropy weight: {self.batch_entropy_weight}")
            if self.ranking_only:
                print(f"  🎯 RANKING-ONLY MODE: Using only pairwise ranking loss (data-efficient)")
            if self.transaction_cost > 0:
                print(f"  Transaction cost: {self.transaction_cost*100:.2f}%")
            if self.grad_accumulation_steps > 1:
                effective_batch = self.batch_size * self.grad_accumulation_steps
                print(f"  Gradient Accumulation: {self.grad_accumulation_steps} steps (effective batch: {effective_batch})")
            print(f"  Performance: AMP={'ON' if self.use_amp else 'OFF'}, compile={'ON' if self.use_compile else 'OFF'}, eval_batch={self.eval_batch_size}")
            if self.ddp:
                print(f"  DDP: Enabled (world_size={get_world_size()})")

        # Set seeds (different per rank for data diversity)
        seed = self.seed + get_rank()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get folds
        folds = self._get_folds()
        if is_main_process():
            print(f"\nGenerated {len(folds)} folds:")
            for i, (train_dates, test_dates) in enumerate(folds):
                print(f"  Fold {i+1}: Train [{train_dates[0]} to {train_dates[-1]}] ({len(train_dates)} days)")
                print(f"           Test  [{test_dates[0]} to {test_dates[-1]}] ({len(test_dates)} days)")

        # Run each fold, reusing checkpoint from previous fold
        self.fold_results = []
        best_overall_sharpe = -float('inf')
        current_checkpoint_path = None

        for fold_idx, (train_dates, test_dates) in enumerate(folds):
            if is_main_process():
                print(f"\n{'='*80}")
                print(f"FOLD {fold_idx + 1}/{len(folds)}")
                print(f"{'='*80}")

            # Create model - load from previous fold's checkpoint if available
            model = self._create_model(wrap_ddp=False)  # Don't wrap yet

            if current_checkpoint_path is not None and os.path.exists(current_checkpoint_path):
                if is_main_process():
                    print(f"  Loading checkpoint from previous fold: {current_checkpoint_path}")
                checkpoint = torch.load(current_checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])

            # Now wrap with DDP if enabled
            if self.ddp:
                # find_unused_parameters=True needed for ranking_only mode (confidence head unused)
                model = DDP(model, device_ids=[get_rank()], output_device=get_rank(),
                           find_unused_parameters=self.ranking_only)

            # Train
            train_loss, val_loss, epochs, best_epoch, train_losses, val_losses = \
                self._train_fold(fold_idx, train_dates, model)

            # Evaluate
            eval_results = self._evaluate_fold(fold_idx, test_dates, model)

            # Save checkpoint (only on main process)
            fold_checkpoint_path = os.path.join(self.checkpoint_dir, f'fold_{fold_idx}_best.pt')
            if is_main_process():
                # Get underlying model state dict (unwrap DDP if needed)
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_data = {
                    'model_state_dict': model_to_save.state_dict(),
                    'config': {
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers,
                        'num_heads': self.num_heads,
                        'dropout': self.dropout,
                        'top_k': self.top_k,
                        'selection_method': self.selection_method,
                        'input_dim': self.input_dim,
                        'seq_len': self.seq_len,
                        'horizon_days': self.horizon_days,
                    },
                    'fold_idx': fold_idx,
                    'train_dates': (train_dates[0], train_dates[-1]),
                    'test_dates': (test_dates[0], test_dates[-1]),
                    'loss_history': {'train': train_losses, 'val': val_losses},
                    'eval_results': {
                        'sharpe_ratio': eval_results['sharpe_ratio'],
                        'total_return_pct': eval_results['total_return_pct'],
                    }
                }
                torch.save(checkpoint_data, fold_checkpoint_path)

                # Save as best overall if this fold has the best Sharpe
                if eval_results['sharpe_ratio'] > best_overall_sharpe:
                    best_overall_sharpe = eval_results['sharpe_ratio']
                    best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_overall.pt')
                    torch.save(checkpoint_data, best_checkpoint_path)
                    print(f"  New best overall model (Sharpe: {best_overall_sharpe:.2f})")

            # Use this fold's checkpoint as starting point for next fold
            current_checkpoint_path = fold_checkpoint_path

            # Sync checkpoint path across ranks
            if self.ddp:
                dist.barrier()

            # Create result
            result = PortfolioFoldResult(
                fold_idx=fold_idx,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_days=len(train_dates),
                test_days=len(test_dates),
                final_train_loss=train_loss,
                final_val_loss=val_loss,
                epochs_trained=epochs,
                best_epoch=best_epoch,
                total_return_pct=eval_results['total_return_pct'],
                sharpe_ratio=eval_results['sharpe_ratio'],
                max_drawdown_pct=eval_results['max_drawdown_pct'],
                win_rate=eval_results['win_rate'],
                avg_daily_return_pct=eval_results['avg_daily_return_pct'],
                num_trading_days=eval_results['num_trading_days']
            )
            self.fold_results.append(result)

            # Print results
            if is_main_process():
                print(f"\n  Fold {fold_idx + 1} Results:")
                print(f"    Total Return: {eval_results['total_return_pct']:+.2f}%")
                print(f"    Sharpe Ratio: {eval_results['sharpe_ratio']:.2f}")
                print(f"    Max Drawdown: {eval_results['max_drawdown_pct']:.2f}%")
                print(f"    Win Rate: {eval_results['win_rate']:.1f}%")

            # Clean up GPU memory between folds
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate summary plot across all folds
        self._plot_summary(folds)

        return self.fold_results

    def _plot_summary(self, folds: List[Tuple[List[str], List[str]]]):
        """Generate summary plot across all folds."""
        if not is_main_process() or len(self.fold_results) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Returns by fold (bar chart)
        ax = axes[0, 0]
        fold_nums = [f"Fold {r.fold_idx + 1}" for r in self.fold_results]
        returns = [r.total_return_pct for r in self.fold_results]
        colors = ['green' if r > 0 else 'red' for r in returns]
        bars = ax.bar(fold_nums, returns, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(np.mean(returns), color='blue', linestyle='--',
                  label=f'Mean: {np.mean(returns):+.2f}%')
        ax.set_title('Total Return by Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # 2. Sharpe ratios by fold
        ax = axes[0, 1]
        sharpes = [r.sharpe_ratio for r in self.fold_results]
        colors = ['green' if s > 0 else 'red' for s in sharpes]
        ax.bar(fold_nums, sharpes, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(np.mean(sharpes), color='blue', linestyle='--',
                  label=f'Mean: {np.mean(sharpes):.2f}')
        ax.set_title('Sharpe Ratio by Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # 3. Drawdowns by fold
        ax = axes[1, 0]
        drawdowns = [r.max_drawdown_pct for r in self.fold_results]
        ax.bar(fold_nums, drawdowns, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.axhline(np.mean(drawdowns), color='blue', linestyle='--',
                  label=f'Mean: {np.mean(drawdowns):.2f}%')
        ax.set_title('Max Drawdown by Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max Drawdown (%)')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # 4. Win rates by fold
        ax = axes[1, 1]
        win_rates = [r.win_rate for r in self.fold_results]
        ax.bar(fold_nums, win_rates, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax.axhline(np.mean(win_rates), color='blue', linestyle='--',
                  label=f'Mean: {np.mean(win_rates):.1f}%')
        ax.set_title('Win Rate by Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        plt.suptitle(f'Walk-Forward Portfolio Summary ({len(self.fold_results)} Folds)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(self.checkpoint_dir, 'walk_forward_summary.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSaved summary plot: {plot_path}")


def print_summary(results: List[PortfolioFoldResult]):
    """Print summary of results."""
    print(f"\n{'='*80}")
    print("WALK-FORWARD PORTFOLIO SUMMARY")
    print(f"{'='*80}")

    if len(results) == 0:
        print("No results!")
        return

    returns = [r.total_return_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]
    win_rates = [r.win_rate for r in results]

    print(f"\nFolds: {len(results)}")

    print(f"\nReturn:")
    print(f"  Mean: {np.mean(returns):+.2f}%")
    print(f"  Std:  {np.std(returns):.2f}%")

    print(f"\nRisk-Adjusted:")
    print(f"  Sharpe Ratio: {np.mean(sharpes):.2f} (±{np.std(sharpes):.2f})")
    print(f"  Max Drawdown: {np.mean(drawdowns):.2f}%")
    print(f"  Win Rate: {np.mean(win_rates):.1f}%")

    # Statistical significance
    if len(returns) >= 2:
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        print(f"\nStatistical Test (H0: mean return = 0):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")

    print(f"\n{'─'*80}")
    print("Per-Fold Results:")
    print(f"{'─'*80}")
    print(f"{'Fold':<6} {'Test Period':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")

    for r in results:
        test_period = f"{r.test_start} to {r.test_end}"
        print(f"{r.fold_idx+1:<6} {test_period:<25} {r.total_return_pct:>+9.2f}% "
              f"{r.sharpe_ratio:>8.2f} {r.max_drawdown_pct:>7.2f}% {r.win_rate:>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Portfolio Training')

    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--prices', type=str, required=True)
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for training data (e.g., 2020-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for training data (e.g., 2025-12-31)')

    # Walk-forward config
    parser.add_argument('--num-folds', type=int, default=0,
                       help='Number of folds (0 = auto, span entire dataset)')
    parser.add_argument('--mode', type=str, default='expanding',
                       choices=['expanding', 'sliding'])
    parser.add_argument('--min-train-months', type=int, default=64)
    parser.add_argument('--test-months', type=int, default=8)
    parser.add_argument('--gap-days', type=int, default=5)

    # Model config
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seq-len', type=int, default=300)

    # Portfolio config
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--selection', type=str, default='neuralsort',
                       choices=['gumbel', 'neuralsort', 'ste', 'soft_attention'])
    parser.add_argument('--horizon-days', type=int, default=1)
    parser.add_argument('--horizon-idx', type=int, default=0)
    parser.add_argument('--initial-temp', type=float, default=1.0)
    parser.add_argument('--min-temp', type=float, default=0.2,
                       help='Minimum temperature (increased default to prevent collapse)')

    # Gumbel-softmax improvements
    parser.add_argument('--annealing-schedule', type=str, default='exponential',
                       choices=['linear', 'exponential', 'cosine'],
                       help='Temperature annealing schedule')
    parser.add_argument('--top-m-filter', type=int, default=0,
                       help='Pre-filter to top-m candidates before Gumbel selection (0=disabled)')
    parser.add_argument('--decoupled-backward-temp', action='store_true',
                       help='Use higher temperature for backward pass')
    parser.add_argument('--backward-temp-ratio', type=float, default=2.0,
                       help='Ratio of backward to forward temperature')
    parser.add_argument('--learnable-temperature', action='store_true',
                       help='Make temperature a learnable parameter')
    parser.add_argument('--batch-entropy-weight', type=float, default=0.01,
                       help='Weight for batch entropy regularization')
    parser.add_argument('--auxiliary-loss-weight', type=float, default=0.1,
                       help='Weight for auxiliary loss providing gradients to all stocks')
    parser.add_argument('--auxiliary-loss-type', type=str, default='ranking',
                       choices=['ranking', 'mse', 'contrastive'],
                       help='Type of auxiliary loss: ranking (pairwise), mse (regression), contrastive')
    parser.add_argument('--ranking-only', action='store_true',
                       help='Use ONLY ranking loss (more data-efficient, recommended for small datasets)')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Round-trip transaction cost for evaluation (default: 0.1%%)')
    parser.add_argument('--eval-batch-size', type=int, default=8,
                       help='Batch size for evaluation (larger = faster)')
    parser.add_argument('--eval-every-n-steps', type=int, default=25,
                       help='Evaluate on val set every N optimizer steps (0=disable)')
    parser.add_argument('--loss-smoothing-window', type=int, default=50,
                       help='EMA window size for smoothed loss curves')
    parser.add_argument('--save-loss-curves-every', type=int, default=25,
                       help='Save loss curve PNG every N steps (0=only at end)')

    # Performance
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for faster execution (PyTorch 2.0+)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory (slower but less VRAM)')

    # Training config
    parser.add_argument('--epochs-per-fold', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.03)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--grad-accumulation-steps', type=int, default=6,
                       help='Number of steps to accumulate gradients (effective batch = batch_size * steps)')
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--max-stocks', type=int, default=100)

    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/walk_forward_portfolio_neuralsort',)
    parser.add_argument('--output', type=str, default='walk_forward_portfolio_results.pt')
    parser.add_argument('--ddp', action='store_true',
                       help='Enable Distributed Data Parallel (use with torchrun)')

    args = parser.parse_args()

    # Setup DDP if enabled
    if args.ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        setup_ddp(local_rank, world_size)
        device = f'cuda:{local_rank}'
        args.device = device
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
            args.device = device

    trainer = WalkForwardPortfolioTrainer(
        dataset_path=args.data,
        prices_path=args.prices,
        start_date=args.start_date,
        end_date=args.end_date,
        num_folds=args.num_folds,
        mode=args.mode,
        min_train_months=args.min_train_months,
        test_months=args.test_months,
        gap_days=args.gap_days,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        seq_len=args.seq_len,
        top_k=args.top_k,
        selection_method=args.selection,
        horizon_days=args.horizon_days,
        horizon_idx=args.horizon_idx,
        initial_temperature=args.initial_temp,
        min_temperature=args.min_temp,
        annealing_schedule=args.annealing_schedule,
        top_m_filter=args.top_m_filter,
        decoupled_backward_temp=args.decoupled_backward_temp,
        backward_temp_ratio=args.backward_temp_ratio,
        learnable_temperature=args.learnable_temperature,
        batch_entropy_weight=args.batch_entropy_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        auxiliary_loss_type=args.auxiliary_loss_type,
        ranking_only=args.ranking_only,
        transaction_cost=args.transaction_cost,
        eval_batch_size=args.eval_batch_size,
        eval_every_n_steps=args.eval_every_n_steps,
        loss_smoothing_window=args.loss_smoothing_window,
        save_loss_curves_every=args.save_loss_curves_every,
        use_amp=not args.no_amp,
        use_compile=args.compile,
        use_gradient_checkpointing=args.gradient_checkpointing,
        epochs_per_fold=args.epochs_per_fold,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        grad_accumulation_steps=args.grad_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        max_stocks_per_day=args.max_stocks,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir
    )

    # Set DDP flag
    trainer.ddp = args.ddp

    try:
        results = trainer.run()

        # Print summary and save results (only on main process)
        if is_main_process():
            print_summary(results)

            # Save results
            results_dict = {
                'fold_results': [asdict(r) for r in results],
                'config': vars(args)
            }
            torch.save(results_dict, args.output)
            print(f"\nResults saved to: {args.output}")

            # Also save JSON
            json_path = args.output.replace('.pt', '.json')
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            print(f"JSON saved to: {json_path}")

    finally:
        # Clean up DDP
        if args.ddp:
            cleanup_ddp()


if __name__ == '__main__':
    main()
