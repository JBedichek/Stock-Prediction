#!/usr/bin/env python3
"""
Differentiable Portfolio Training

Trains a stock prediction model end-to-end with differentiable top-k selection.
Instead of training on individual stock predictions, this trains on actual
portfolio returns - selecting top-k stocks and backpropagating through the selection.

Key Features:
- Process all stocks for a given date in one forward pass
- Differentiable top-k selection (Gumbel-Softmax, NeuralSort, or STE)
- Portfolio-level loss (maximize returns, minimize drawdown)
- Temperature annealing for sharper selection over time

Usage:
    python -m training.train_portfolio_differentiable \
        --data all_complete_dataset.h5 \
        --prices actual_prices.h5 \
        --top-k 5 \
        --selection gumbel \
        --epochs 50
"""

import os
import sys
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import h5py

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import SimpleTransformerPredictor


# =============================================================================
# DDP Utilities
# =============================================================================

def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# =============================================================================
# Differentiable Top-K Selection Methods
# =============================================================================

class StraightThroughTopK(torch.autograd.Function):
    """
    Straight-Through Estimator for top-k selection.
    Forward: hard top-k (binary mask)
    Backward: gradient flows through softmax
    """
    @staticmethod
    def forward(ctx, scores, k):
        # Hard top-k selection
        _, indices = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, indices, 1.0)
        ctx.save_for_backward(scores)
        ctx.k = k
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, = ctx.saved_tensors
        # Gradient flows through softmax approximation
        soft_weights = F.softmax(scores, dim=-1)
        return grad_output * soft_weights, None


def gumbel_top_k(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    hard: bool = True,
    top_m_filter: int = 0,
    backward_temperature: Optional[float] = None
) -> torch.Tensor:
    """
    Differentiable top-k using Gumbel-Softmax trick with improvements.

    Args:
        logits: (batch, num_stocks) raw scores
        k: number of stocks to select
        temperature: softmax temperature for forward pass (lower = sharper)
        hard: if True, use straight-through estimator
        top_m_filter: if > 0, pre-filter to top-m candidates before Gumbel selection
                      (reduces noise, recommended: m=50 for n=300)
        backward_temperature: if set, use different temperature for backward pass
                              (decoupled temperatures for better bias-variance tradeoff)

    Returns:
        weights: (batch, num_stocks) selection weights summing to k
    """
    batch_size, num_stocks = logits.shape

    # Use same temperature for backward if not specified
    if backward_temperature is None:
        backward_temperature = temperature

    # Top-m pre-filtering: only consider top-m candidates for Gumbel selection
    # This reduces noise and speeds up training (Petersen et al. 2022)
    if top_m_filter > 0 and top_m_filter < num_stocks:
        # Get top-m indices
        _, top_m_indices = torch.topk(logits, top_m_filter, dim=-1)

        # Create mask for top-m
        top_m_mask = torch.zeros_like(logits)
        top_m_mask.scatter_(-1, top_m_indices, 1.0)

        # Mask out non-top-m with large negative (use -1e4 for float16 compatibility)
        filtered_logits = logits.masked_fill(top_m_mask == 0, -1e4)
    else:
        filtered_logits = logits

    # Add Gumbel noise for stochastic sampling
    if filtered_logits.requires_grad:
        gumbels = -torch.log(-torch.log(torch.rand_like(filtered_logits) + 1e-10) + 1e-10)
        perturbed = (filtered_logits + gumbels) / temperature
    else:
        perturbed = filtered_logits / temperature

    # Iterative soft top-k: repeatedly select and mask
    soft_mask = torch.zeros_like(filtered_logits)
    remaining = perturbed.clone()

    for _ in range(k):
        # Soft selection of next item
        soft_selection = F.softmax(remaining, dim=-1)
        soft_mask = soft_mask + soft_selection
        # "Remove" selected item by masking with large negative (use 1e4 for float16 compatibility)
        remaining = remaining - soft_selection * 1e4

    if hard:
        # Straight-through: use hard mask in forward, soft in backward
        _, indices = torch.topk(logits, k, dim=-1)
        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(-1, indices, 1.0)

        # Decoupled temperature: recompute soft_mask with backward_temperature for gradients
        if backward_temperature != temperature:
            if filtered_logits.requires_grad:
                gumbels_back = -torch.log(-torch.log(torch.rand_like(filtered_logits) + 1e-10) + 1e-10)
                perturbed_back = (filtered_logits + gumbels_back) / backward_temperature
            else:
                perturbed_back = filtered_logits / backward_temperature

            soft_mask_back = torch.zeros_like(filtered_logits)
            remaining_back = perturbed_back.clone()
            for _ in range(k):
                soft_selection = F.softmax(remaining_back, dim=-1)
                soft_mask_back = soft_mask_back + soft_selection
                remaining_back = remaining_back - soft_selection * 1e4

            # Use backward soft_mask for gradient computation
            return hard_mask - soft_mask_back.detach() + soft_mask_back

        # This trick: forward uses hard_mask, backward uses soft_mask gradient
        return hard_mask - soft_mask.detach() + soft_mask

    return soft_mask


def neuralsort_top_k(scores: torch.Tensor, k: int,
                     temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable top-k using NeuralSort (Grover et al. 2019).
    Creates a soft permutation matrix and selects top-k.

    Args:
        scores: (batch, num_stocks) raw scores
        k: number of stocks to select
        temperature: controls sharpness

    Returns:
        weights: (batch, num_stocks) soft selection weights
    """
    batch_size, n = scores.shape

    # Compute pairwise score differences
    # scores_i - scores_j for all pairs
    scores_expanded = scores.unsqueeze(-1)  # (batch, n, 1)
    pairwise_diff = scores_expanded - scores_expanded.transpose(-1, -2)  # (batch, n, n)

    # Soft permutation matrix via row-wise softmax
    # P[i,j] ≈ probability that item j has rank i
    P = F.softmax(pairwise_diff / temperature, dim=-1)  # (batch, n, n)

    # Probability of being in top-k = sum of probabilities for ranks 0 to k-1
    # P[:, :k, :].sum(dim=1) gives probability each item is in top-k positions
    top_k_probs = P[:, :k, :].sum(dim=1)  # (batch, n)

    return top_k_probs


def soft_attention_selection(
    scores: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    masks: Optional[torch.Tensor] = None,
    hard: bool = False
) -> torch.Tensor:
    """
    Full soft attention selection using temperature-scaled softmax over all stocks.

    Unlike iterative Gumbel-top-k, this provides gradients to ALL stocks through
    the softmax normalization. The weights are scaled to sum to k for compatibility
    with portfolio return computation.

    Args:
        scores: (batch, num_stocks) raw scores
        k: number of stocks (weights will sum to k)
        temperature: softmax temperature (lower = sharper, more concentrated)
        masks: (batch, num_stocks) optional validity mask
        hard: if True, use straight-through estimator for discrete selection

    Returns:
        weights: (batch, num_stocks) soft selection weights summing to k
    """
    # Apply mask if provided (mask out invalid stocks)
    if masks is not None:
        # Use -1e4 for float16 compatibility
        masked_scores = scores.masked_fill(masks == 0, -1e4)
    else:
        masked_scores = scores

    # Temperature-scaled softmax gives probability distribution
    soft_weights = F.softmax(masked_scores / temperature, dim=-1)

    # Scale weights to sum to k (instead of 1)
    # This makes portfolio returns comparable to top-k selection
    scaled_weights = soft_weights * k

    if hard:
        # Straight-through: hard top-k in forward, soft gradients in backward
        # IMPORTANT: Use masked_scores for topk, not raw scores!
        # Otherwise we might select invalid/masked stocks
        _, indices = torch.topk(masked_scores, k, dim=-1)
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(-1, indices, 1.0)

        # Zero out any invalid stocks (safety check)
        if masks is not None:
            hard_mask = hard_mask * masks

        # Forward uses hard_mask, backward uses scaled_weights gradient
        return hard_mask - scaled_weights.detach() + scaled_weights

    return scaled_weights


class LearnableTemperature(nn.Module):
    """
    Learnable temperature parameter for Gumbel-softmax.
    Uses softplus to ensure temperature stays positive.
    """
    def __init__(self, init_temp: float = 1.0, min_temp: float = 0.1):
        super().__init__()
        # Initialize log_temp such that softplus(log_temp) + min_temp = init_temp
        # softplus(x) ≈ x for large x, so init to (init_temp - min_temp)
        init_val = np.log(np.exp(init_temp - min_temp) - 1)  # Inverse softplus
        self.log_temp = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.min_temp = min_temp

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temp) + self.min_temp

    def get_temperature(self) -> float:
        """Get temperature as float for use in functions."""
        return self.temperature.item()


class DifferentiableTopKSelector(nn.Module):
    """
    Unified interface for differentiable top-k selection methods.

    Supports multiple improvements from recent research:
    - Exponential temperature annealing (vs linear)
    - Top-m pre-filtering to reduce noise
    - Decoupled forward/backward temperatures
    - Learnable temperature
    """
    def __init__(
        self,
        k: int,
        method: str = 'gumbel',
        initial_temperature: float = 1.0,
        min_temperature: float = 0.2,  # Increased default from 0.1
        annealing_schedule: str = 'exponential',  # 'linear', 'exponential', 'cosine'
        top_m_filter: int = 0,  # Pre-filter to top-m before Gumbel (0 = disabled)
        decoupled_backward_temp: bool = False,  # Use higher temp for backward pass
        backward_temp_ratio: float = 2.0,  # backward_temp = forward_temp * ratio
        learnable_temperature: bool = False,  # Make temperature a learnable parameter
    ):
        super().__init__()
        self.k = k
        self.method = method
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.annealing_schedule = annealing_schedule
        self.top_m_filter = top_m_filter
        self.decoupled_backward_temp = decoupled_backward_temp
        self.backward_temp_ratio = backward_temp_ratio
        self.learnable_temperature = learnable_temperature

        # Setup temperature
        if learnable_temperature:
            self.temp_module = LearnableTemperature(initial_temperature, min_temperature)
            self.temperature = self.temp_module.get_temperature()
        else:
            self.temp_module = None
            self.temperature = initial_temperature

    def forward(self, scores: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """
        Select top-k from scores.

        Args:
            scores: (batch, num_stocks) prediction scores
            hard: use hard selection in forward pass

        Returns:
            weights: (batch, num_stocks) selection weights
        """
        # Get current temperature
        if self.learnable_temperature and self.temp_module is not None:
            temp = self.temp_module.get_temperature()
        else:
            temp = self.temperature

        # Compute backward temperature if decoupled
        backward_temp = temp * self.backward_temp_ratio if self.decoupled_backward_temp else None

        if self.method == 'gumbel':
            return gumbel_top_k(
                scores, self.k, temp, hard,
                top_m_filter=self.top_m_filter,
                backward_temperature=backward_temp
            )
        elif self.method == 'neuralsort':
            return neuralsort_top_k(scores, self.k, temp)
        elif self.method == 'ste':
            return StraightThroughTopK.apply(scores, self.k)
        elif self.method == 'soft_attention':
            return soft_attention_selection(scores, self.k, temp, masks=None, hard=hard)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def anneal_temperature(self, progress: float):
        """
        Anneal temperature based on training progress.

        Args:
            progress: float in [0, 1] indicating training progress
        """
        if self.learnable_temperature:
            # Don't anneal if temperature is learned
            if self.temp_module is not None:
                self.temperature = self.temp_module.get_temperature()
            return

        temp_range = self.initial_temperature - self.min_temperature

        if self.annealing_schedule == 'linear':
            # Linear decay
            self.temperature = self.min_temperature + temp_range * (1 - progress)

        elif self.annealing_schedule == 'exponential':
            # Exponential decay (from original Gumbel-Softmax paper)
            # τ = τ_min + (τ_init - τ_min) * exp(-5 * progress)
            # The -5 factor gives ~99% decay at progress=1
            self.temperature = self.min_temperature + temp_range * np.exp(-5 * progress)

        elif self.annealing_schedule == 'cosine':
            # Cosine annealing (smooth decay)
            self.temperature = self.min_temperature + temp_range * 0.5 * (1 + np.cos(np.pi * progress))

        else:
            raise ValueError(f"Unknown annealing schedule: {self.annealing_schedule}")


# =============================================================================
# Portfolio-Aware Dataset
# =============================================================================

class PortfolioDataset(torch.utils.data.Dataset):
    """
    Dataset that groups all stocks by date for portfolio-level training.

    Memory-efficient: uses memory-mapped files so all DDP ranks share the same
    physical memory instead of loading separate copies.
    """

    def __init__(
        self,
        dataset_path: str,
        prices_path: str,
        start_date: str,
        end_date: str,
        seq_len: int = 60,
        horizon_days: int = 1,
        min_stocks_per_day: int = 50,
        max_stocks_per_day: int = 500,
        cache_dir: str = '/tmp/portfolio_cache',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_path = dataset_path
        self.prices_path = prices_path
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.horizon_days = horizon_days
        self.min_stocks_per_day = min_stocks_per_day
        self.max_stocks_per_day = max_stocks_per_day
        self.cache_dir = cache_dir
        self.rank = rank
        self.world_size = world_size

        self._build_index_and_load()

    def _get_cache_paths(self):
        """Get paths for cached memmap files."""
        import hashlib
        # Create unique cache key based on parameters
        key = f"{self.dataset_path}_{self.prices_path}_{self.start_date}_{self.end_date}"
        cache_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        prefix = os.path.join(self.cache_dir, f"portfolio_{cache_hash}")
        return {
            'features': f"{prefix}_features.npy",
            'offsets': f"{prefix}_offsets.npy",
            'base_indices': f"{prefix}_base_indices.npy",  # original start index per ticker
            'samples': f"{prefix}_samples.npy",      # memmap instead of pickle
            'sample_offsets': f"{prefix}_sample_offsets.npy",  # index into samples
            'meta': f"{prefix}_meta.pkl",
        }

    def _build_index_and_load(self):
        """Build index and load data. Rank 0 builds, others wait and load memmap."""
        import pickle
        import filelock

        os.makedirs(self.cache_dir, exist_ok=True)
        paths = self._get_cache_paths()
        lock_path = paths['meta'] + '.lock'

        # Use file lock to ensure only one process builds the cache
        lock = filelock.FileLock(lock_path, timeout=600)

        with lock:
            if os.path.exists(paths['meta']):
                # Cache exists, load it
                if self.rank == 0:
                    print(f"Loading cached dataset from {self.cache_dir}...")
                self._load_from_cache(paths)
            else:
                # Need to build cache - only rank 0 does this
                if self.rank == 0:
                    print(f"Building and caching dataset...")
                    self._build_and_cache(paths)
                else:
                    # This shouldn't happen with the lock, but just in case
                    raise RuntimeError("Cache not found and rank != 0")

        if self.rank == 0:
            print(f"  Dataset ready: {len(self.valid_dates)} dates, "
                  f"{self.features_mmap.shape[0]} total days across {self.num_tickers} tickers")

    def _build_and_cache(self, paths):
        """Build the dataset and save to memmap files (rank 0 only).

        Memory-efficient: only stores the exact feature slices needed, not full ticker data.
        """
        import pickle

        print(f"Building portfolio index for {self.start_date} to {self.end_date}...")

        h5_features = h5py.File(self.dataset_path, 'r')
        h5_prices = h5py.File(self.prices_path, 'r')

        feature_tickers = set(h5_features.keys())
        price_tickers = set(h5_prices.keys())
        all_tickers = sorted(feature_tickers & price_tickers)

        print(f"  Found {len(all_tickers)} tickers with both features and prices")

        sample_ticker = all_tickers[0]
        self.feature_dim = h5_features[sample_ticker]['features'].shape[1]

        # First pass: identify valid samples and compute returns
        # Also track the min/max indices used per ticker to minimize data stored
        date_to_samples = defaultdict(list)
        ticker_to_idx = {}
        ticker_index_ranges = {}  # ticker_idx -> (min_start_idx, max_end_idx)

        for ticker in tqdm(all_tickers, desc="  Indexing tickers"):
            ticker_dim = h5_features[ticker]['features'].shape[1]
            if ticker_dim != self.feature_dim:
                continue

            feature_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                           for d in h5_features[ticker]['dates'][:]]
            price_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                         for d in h5_prices[ticker]['dates'][:]]

            feature_date_to_idx = {d: i for i, d in enumerate(feature_dates)}
            price_date_to_idx = {d: i for i, d in enumerate(price_dates)}

            prices = h5_prices[ticker]['prices'][:]

            for date in feature_dates:
                if date < self.start_date or date > self.end_date:
                    continue

                feat_idx = feature_date_to_idx.get(date)
                if feat_idx is None or feat_idx < self.seq_len:
                    continue

                price_idx = price_date_to_idx.get(date)
                if price_idx is None:
                    continue

                future_idx = price_idx + self.horizon_days
                if future_idx >= len(price_dates):
                    continue

                current_price = prices[price_idx]
                future_price = prices[future_idx]
                if current_price > 0:
                    ret = np.float32(np.clip((future_price / current_price) - 1.0, -0.5, 0.5))
                else:
                    ret = np.float32(0.0)

                if ticker not in ticker_to_idx:
                    ticker_to_idx[ticker] = len(ticker_to_idx)

                ticker_idx = ticker_to_idx[ticker]
                start_idx = feat_idx - self.seq_len
                end_idx = feat_idx  # exclusive

                # Track the range of indices we need for this ticker
                if ticker_idx not in ticker_index_ranges:
                    ticker_index_ranges[ticker_idx] = (start_idx, end_idx)
                else:
                    old_min, old_max = ticker_index_ranges[ticker_idx]
                    ticker_index_ranges[ticker_idx] = (min(old_min, start_idx), max(old_max, end_idx))

                date_to_samples[date].append((ticker_idx, start_idx, ret))

        # Filter dates with enough stocks
        self.valid_dates = sorted([
            date for date, samples in date_to_samples.items()
            if len(samples) >= self.min_stocks_per_day
        ])

        if len(self.valid_dates) == 0:
            raise ValueError("No valid dates found!")

        print(f"  Valid dates: {len(self.valid_dates)}")
        print(f"  Tickers needed: {len(ticker_to_idx)}")

        # Calculate total size for memmap - only store the ranges we actually need
        ticker_stored_lengths = []
        for ticker_idx in range(len(ticker_to_idx)):
            if ticker_idx in ticker_index_ranges:
                min_idx, max_idx = ticker_index_ranges[ticker_idx]
                ticker_stored_lengths.append(max_idx - min_idx)
            else:
                ticker_stored_lengths.append(0)

        total_rows = sum(ticker_stored_lengths)
        print(f"  Total feature rows (optimized): {total_rows:,}")

        # Create memmap file for features
        print("  Creating memory-mapped feature file...")
        features_mmap = np.memmap(
            paths['features'], dtype=np.float32, mode='w+',
            shape=(total_rows, self.feature_dim)
        )

        # Create offset array and base index array
        # offset[i] = where ticker i's data starts in memmap
        # base_idx[i] = the original index that maps to offset[i]
        offsets = np.zeros(len(ticker_to_idx) + 1, dtype=np.int64)
        base_indices = np.zeros(len(ticker_to_idx), dtype=np.int64)

        # Load only the needed features into memmap
        idx_to_ticker = {v: k for k, v in ticker_to_idx.items()}
        current_offset = 0

        for i in tqdm(range(len(ticker_to_idx)), desc="  Loading features"):
            ticker = idx_to_ticker[i]
            offsets[i] = current_offset

            if i in ticker_index_ranges:
                min_idx, max_idx = ticker_index_ranges[i]
                base_indices[i] = min_idx

                # Load only the slice we need
                features = h5_features[ticker]['features'][min_idx:max_idx].astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

                features_mmap[current_offset:current_offset + len(features)] = features
                current_offset += len(features)

        offsets[-1] = current_offset
        features_mmap.flush()

        # Save offsets and base indices
        np.save(paths['offsets'], offsets)
        np.save(paths['base_indices'], base_indices)

        # Save date samples as memmap (not pickle) to share across processes
        # Flatten all samples into one array with an offset index
        total_samples = sum(len(date_to_samples[d]) for d in self.valid_dates)
        print(f"  Total samples: {total_samples:,}")

        samples_mmap = np.memmap(
            paths['samples'], dtype=np.float32, mode='w+',
            shape=(total_samples, 3)  # (ticker_idx, start_idx, return)
        )
        sample_offsets = np.zeros(len(self.valid_dates) + 1, dtype=np.int64)

        sample_idx = 0
        for date_idx, date in enumerate(self.valid_dates):
            samples = date_to_samples[date]
            sample_offsets[date_idx] = sample_idx
            for s in samples:
                samples_mmap[sample_idx] = s
                sample_idx += 1
        sample_offsets[-1] = sample_idx
        samples_mmap.flush()

        np.save(paths['sample_offsets'], sample_offsets)

        # Save metadata
        meta = {
            'feature_dim': self.feature_dim,
            'valid_dates': self.valid_dates,
            'num_tickers': len(ticker_to_idx),
            'total_days': total_rows,
            'total_samples': total_samples,
            'seq_len': self.seq_len,
        }
        with open(paths['meta'], 'wb') as f:
            pickle.dump(meta, f)

        h5_features.close()
        h5_prices.close()

        # Now load from cache (to use memmap in read mode)
        del features_mmap
        self._load_from_cache(paths)

        # Print memory info
        mmap_bytes = total_rows * self.feature_dim * 4
        print(f"  Memmap file size: {mmap_bytes / 1e9:.2f} GB (shared across all ranks)")

    def _load_from_cache(self, paths):
        """Load dataset from cached memmap files."""
        import pickle

        with open(paths['meta'], 'rb') as f:
            meta = pickle.load(f)

        self.feature_dim = meta['feature_dim']
        self.valid_dates = meta['valid_dates']
        self.num_tickers = meta['num_tickers']
        self.seq_len = meta['seq_len']

        # Load memmap in read-only mode (shared across processes)
        self.features_mmap = np.memmap(
            paths['features'], dtype=np.float32, mode='r',
            shape=(meta['total_days'], self.feature_dim)
        )

        self.offsets = np.load(paths['offsets'])
        self.base_indices = np.load(paths['base_indices'])

        # Load samples memmap (shared across processes)
        self.samples_mmap = np.memmap(
            paths['samples'], dtype=np.float32, mode='r',
            shape=(meta['total_samples'], 3)
        )
        self.sample_offsets = np.load(paths['sample_offsets'])

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        """Get all stock features and returns for a given date."""
        # Get samples for this date from memmap
        start = self.sample_offsets[idx]
        end = self.sample_offsets[idx + 1]
        samples = self.samples_mmap[start:end]  # (num_stocks, 3)

        num_stocks = samples.shape[0]

        if num_stocks > self.max_stocks_per_day:
            indices = np.random.choice(num_stocks, self.max_stocks_per_day, replace=False)
            samples = samples[indices]
            num_stocks = self.max_stocks_per_day

        features = np.empty((num_stocks, self.seq_len, self.feature_dim), dtype=np.float32)
        returns = np.empty(num_stocks, dtype=np.float32)

        for i in range(num_stocks):
            ticker_idx = int(samples[i, 0])
            start_idx = int(samples[i, 1])
            ret = samples[i, 2]
            # Get offset into the contiguous memmap
            # start_idx is the original index, subtract base_indices to get local index
            local_idx = start_idx - self.base_indices[ticker_idx]
            global_start = self.offsets[ticker_idx] + local_idx
            features[i] = self.features_mmap[global_start:global_start + self.seq_len]
            returns[i] = ret

        return (
            torch.from_numpy(features.copy()),  # copy to own the memory
            torch.from_numpy(returns),
            torch.ones(num_stocks, dtype=torch.float32)
        )


def collate_portfolio_batch(batch):
    """
    Custom collate function for variable-sized portfolio batches.
    Pads to the maximum number of stocks in the batch.
    """
    features_list, returns_list, masks_list = zip(*batch)

    # Find max stocks in batch
    max_stocks = max(f.shape[0] for f in features_list)
    seq_len = features_list[0].shape[1]
    feature_dim = features_list[0].shape[2]
    batch_size = len(batch)

    # Create padded tensors
    features = torch.zeros(batch_size, max_stocks, seq_len, feature_dim)
    returns = torch.zeros(batch_size, max_stocks)
    masks = torch.zeros(batch_size, max_stocks)

    for i, (f, r, m) in enumerate(batch):
        num_stocks = f.shape[0]
        features[i, :num_stocks] = f
        returns[i, :num_stocks] = r
        masks[i, :num_stocks] = m

    return features, returns, masks


# =============================================================================
# Portfolio Model Wrapper
# =============================================================================

class PortfolioModel(nn.Module):
    """
    Wraps a stock predictor with differentiable portfolio selection.

    Supports advanced Gumbel-softmax training techniques:
    - Exponential/cosine temperature annealing
    - Top-m pre-filtering
    - Decoupled forward/backward temperatures
    - Learnable temperature
    """

    def __init__(
        self,
        encoder: nn.Module,
        k: int = 5,
        selection_method: str = 'gumbel',
        initial_temperature: float = 1.0,
        min_temperature: float = 0.2,  # Increased default
        horizon_idx: int = 0,
        # New Gumbel-softmax improvements
        annealing_schedule: str = 'exponential',  # 'linear', 'exponential', 'cosine'
        top_m_filter: int = 0,  # Pre-filter to top-m (0 = disabled, recommended: 50)
        decoupled_backward_temp: bool = False,
        backward_temp_ratio: float = 2.0,
        learnable_temperature: bool = False,
    ):
        """
        Args:
            encoder: Stock feature encoder (e.g., SimpleTransformerPredictor)
            k: Number of stocks to select
            selection_method: 'gumbel', 'neuralsort', or 'ste'
            initial_temperature: Starting temperature for annealing
            min_temperature: Final temperature
            horizon_idx: Which prediction horizon to use (0=1d, 1=5d, etc.)
            annealing_schedule: Temperature annealing schedule
            top_m_filter: Pre-filter to top-m candidates before Gumbel selection
            decoupled_backward_temp: Use different temperature for backward pass
            backward_temp_ratio: Ratio of backward to forward temperature
            learnable_temperature: Make temperature a learnable parameter
        """
        super().__init__()
        self.encoder = encoder
        self.k = k
        self.horizon_idx = horizon_idx

        self.selector = DifferentiableTopKSelector(
            k=k,
            method=selection_method,
            initial_temperature=initial_temperature,
            min_temperature=min_temperature,
            annealing_schedule=annealing_schedule,
            top_m_filter=top_m_filter,
            decoupled_backward_temp=decoupled_backward_temp,
            backward_temp_ratio=backward_temp_ratio,
            learnable_temperature=learnable_temperature,
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode all stocks, select top-k.

        Args:
            features: (batch, num_stocks, seq_len, feature_dim)
            masks: (batch, num_stocks) valid stock mask
            hard: use hard selection in forward pass

        Returns:
            scores: (batch, num_stocks) raw prediction scores
            weights: (batch, num_stocks) selection weights (sum to k)
            confidence: (batch, num_stocks) confidence values
        """
        batch_size, num_stocks, seq_len, feature_dim = features.shape

        # Flatten for batch processing through encoder
        flat_features = features.view(batch_size * num_stocks, seq_len, feature_dim)

        # Process in chunks to avoid GPU OOM
        # Max ~2000 sequences per chunk is a safe limit for most GPUs
        chunk_size = 2048
        total_seqs = flat_features.shape[0]

        if total_seqs <= chunk_size:
            # Small enough to process at once
            predictions, confidence = self.encoder(flat_features)
        else:
            # Process in chunks
            pred_chunks = []
            conf_chunks = []
            for start in range(0, total_seqs, chunk_size):
                end = min(start + chunk_size, total_seqs)
                chunk_pred, chunk_conf = self.encoder(flat_features[start:end])
                pred_chunks.append(chunk_pred)
                conf_chunks.append(chunk_conf)
            predictions = torch.cat(pred_chunks, dim=0)
            confidence = torch.cat(conf_chunks, dim=0)

        # Reshape back
        # predictions: (batch*num_stocks, num_bins, num_pred_days) for classification
        #           or (batch*num_stocks, num_pred_days) for regression
        if self.encoder.pred_mode == 'classification':
            # Use expected value as score
            predictions = predictions.view(batch_size, num_stocks, -1, self.encoder.num_pred_days)
            # Simple approach: use mean of distribution as score
            # predictions: (batch, num_stocks, num_bins, num_pred_days)
            scores = predictions.mean(dim=2)[:, :, self.horizon_idx]  # (batch, num_stocks)
        else:
            predictions = predictions.view(batch_size, num_stocks, -1)
            scores = predictions[:, :, self.horizon_idx]  # (batch, num_stocks)

        confidence = confidence.view(batch_size, num_stocks, -1)
        confidence = confidence[:, :, self.horizon_idx]  # (batch, num_stocks)

        # Mask invalid stocks with large negative score (use -1e4 for float16 compatibility)
        scores = scores.masked_fill(masks == 0, -1e4)

        # Differentiable top-k selection
        weights = self.selector(scores, hard=hard)

        # Zero out weights for invalid stocks
        weights = weights * masks

        return scores, weights, confidence

    def compute_portfolio_return(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute portfolio return from selection weights and actual returns.

        Args:
            weights: (batch, num_stocks) selection weights
            returns: (batch, num_stocks) actual future returns
            masks: (batch, num_stocks) valid stock mask

        Returns:
            portfolio_return: (batch,) weighted portfolio returns
        """
        # Normalize weights to sum to 1 (equal weight among selected)
        weight_sum = (weights * masks).sum(dim=-1, keepdim=True) + 1e-8
        normalized_weights = weights / weight_sum

        # Portfolio return is weighted sum
        portfolio_return = (normalized_weights * returns * masks).sum(dim=-1)

        return portfolio_return

    def anneal_temperature(self, progress: float):
        """Anneal selection temperature."""
        self.selector.anneal_temperature(progress)


# =============================================================================
# Loss Functions
# =============================================================================

class PortfolioLoss(nn.Module):
    """
    Portfolio-level loss combining return maximization and risk management.

    Includes:
    - Batch entropy regularization to prevent mode collapse
    - Auxiliary ranking loss for gradient signal to ALL stocks (not just selected)

    For data-limited scenarios, use ranking_only=True for more principled training:
    - Ranking loss provides O(n²) pairwise comparisons per day instead of O(1) portfolio returns
    - All stocks receive gradients, not just the selected top-k
    - If ranking is correct, top-k selection automatically gives good returns
    """

    def __init__(
        self,
        return_weight: float = 1.0,
        sharpe_weight: float = 0.1,
        concentration_weight: float = 0.01,
        confidence_weight: float = 0.1,
        batch_entropy_weight: float = 0.01,  # Diversity regularization
        auxiliary_loss_weight: float = 0.1,  # Ranking/prediction loss on all stocks
        auxiliary_loss_type: str = 'ranking',  # 'ranking', 'mse', or 'contrastive'
        ranking_only: bool = False,  # If True, use ONLY ranking loss (more data-efficient)
    ):
        super().__init__()
        self.return_weight = return_weight
        self.sharpe_weight = sharpe_weight
        self.concentration_weight = concentration_weight
        self.confidence_weight = confidence_weight
        self.batch_entropy_weight = batch_entropy_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_loss_type = auxiliary_loss_type
        self.ranking_only = ranking_only

    def _ranking_loss(
        self,
        scores: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor,
        num_pairs: int = 1000
    ) -> torch.Tensor:
        """
        ListNet-style ranking loss: softmax(scores) should match softmax(returns).

        This is better than margin-based pairwise loss because:
        1. It encourages score SEPARATION proportional to return differences
        2. No margin saturation - always provides gradient for better ranking
        3. All stocks get gradients through softmax normalization
        """
        batch_size, num_stocks = scores.shape
        device = scores.device

        total_loss = torch.tensor(0.0, device=device)
        num_valid_samples = 0

        for b in range(batch_size):
            valid_mask = masks[b] > 0
            n_valid = valid_mask.sum().item()

            if n_valid < 2:
                continue

            valid_scores = scores[b, valid_mask]
            valid_returns = returns[b, valid_mask]

            # === ListNet Loss ===
            # Convert returns to target distribution (softmax with temperature)
            # Temperature controls sharpness: lower = more focus on top stocks
            # We use a moderate temperature to spread probability mass
            return_temperature = 20.0  # returns ~0.01-0.05, so *20 gives ~0.2-1.0 range
            target_dist = F.softmax(valid_returns * return_temperature, dim=-1)

            # Score distribution (temperature=1 for scores)
            score_log_dist = F.log_softmax(valid_scores, dim=-1)

            # Cross-entropy loss: -sum(target * log(pred))
            # This encourages scores to match return-based distribution
            listnet_loss = -(target_dist * score_log_dist).sum()

            # === Normalization ===
            # Divide by n_valid to make loss comparable across different batch sizes
            listnet_loss = listnet_loss / n_valid

            total_loss = total_loss + listnet_loss
            num_valid_samples += 1

        if num_valid_samples > 0:
            return total_loss / num_valid_samples
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _mse_loss(
        self,
        scores: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        MSE prediction loss: scores should predict actual returns.
        """
        # Scale returns to roughly match score range
        scaled_returns = returns * 10  # returns are typically -0.05 to 0.05

        mse = F.mse_loss(scores * masks, scaled_returns * masks, reduction='sum')
        num_valid = masks.sum() + 1e-8

        return mse / num_valid

    def _contrastive_loss(
        self,
        scores: torch.Tensor,
        weights: torch.Tensor,
        returns: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss: selected stocks should have higher returns than non-selected.
        """
        # Binary selection mask
        selected = (weights > 0.5).float() * masks
        unselected = (weights <= 0.5).float() * masks

        # Average returns for selected and unselected
        selected_sum = (returns * selected).sum(dim=-1)
        selected_count = selected.sum(dim=-1) + 1e-8
        selected_return = selected_sum / selected_count

        unselected_sum = (returns * unselected).sum(dim=-1)
        unselected_count = unselected.sum(dim=-1) + 1e-8
        unselected_return = unselected_sum / unselected_count

        # Margin loss: selected should beat unselected by margin
        margin = 0.001  # 0.1% margin
        contrastive_loss = F.relu(unselected_return - selected_return + margin).mean()

        return contrastive_loss

    def forward(
        self,
        portfolio_returns: torch.Tensor,
        weights: torch.Tensor,
        confidence: torch.Tensor,
        actual_returns: torch.Tensor,
        masks: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute portfolio loss.

        Args:
            portfolio_returns: (batch,) portfolio returns
            weights: (batch, num_stocks) selection weights
            confidence: (batch, num_stocks) model confidence
            actual_returns: (batch, num_stocks) actual stock returns
            masks: (batch, num_stocks) valid mask

        Returns:
            loss: scalar loss
            metrics: dict of loss components
        """
        batch_size = portfolio_returns.shape[0]
        device = portfolio_returns.device

        # =====================================================================
        # RANKING-ONLY MODE: More data-efficient for limited data scenarios
        # =====================================================================
        if self.ranking_only:
            # Only use ranking loss - provides O(n²) pairwise comparisons
            # instead of O(1) portfolio returns per day
            if scores is not None:
                ranking_loss = self._ranking_loss(scores, actual_returns, masks)
            else:
                ranking_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Still compute portfolio metrics for monitoring (no gradients)
            with torch.no_grad():
                avg_return = portfolio_returns.mean().item()
                selected_confidence = (weights * confidence * masks).sum(dim=-1)
                total_weight = (weights * masks).sum(dim=-1) + 1e-8
                per_sample_confidence = selected_confidence / total_weight

            metrics = {
                'return_loss': 0.0,  # Not used
                'sharpe_loss': 0.0,  # Not used
                'concentration_loss': 0.0,  # Not used
                'confidence_loss': 0.0,  # Not used
                'batch_entropy_loss': 0.0,  # Not used
                'batch_entropy': 0.0,  # Not used
                'auxiliary_loss': ranking_loss.item(),
                'total_loss': ranking_loss.item(),
                'avg_portfolio_return': avg_return,
                'avg_confidence': per_sample_confidence.mean().item()
            }

            return ranking_loss, metrics

        # =====================================================================
        # FULL LOSS MODE: Multiple objectives (requires more data)
        # =====================================================================

        # 1. Maximize returns (negative because we minimize)
        return_loss = -portfolio_returns.mean()

        # 2. Risk-adjusted: penalize variance of returns
        if batch_size > 1:
            return_std = portfolio_returns.std()
            mean_return = portfolio_returns.mean()
            # Pseudo-Sharpe (higher is better, so negate)
            # Use larger epsilon (0.01) to prevent explosion when std is small
            # Also clip to reasonable range to prevent gradient explosion
            raw_sharpe = -mean_return / (return_std + 0.01)
            sharpe_loss = torch.clamp(raw_sharpe, -10.0, 10.0)
        else:
            sharpe_loss = torch.tensor(0.0, device=device)

        # 3. Concentration penalty: encourage decisive selection
        # Entropy of selection weights (lower = more concentrated)
        weight_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        concentration_loss = weight_entropy  # We want low entropy

        # 4. Confidence calibration: confidence should correlate with actual returns
        # Instead of just maximizing confidence, we want confidence to predict whether
        # the selected stocks actually have positive returns
        selected_confidence = (weights * confidence * masks).sum(dim=-1)  # (batch,)
        total_weight = (weights * masks).sum(dim=-1) + 1e-8  # (batch,)
        per_sample_confidence = selected_confidence / total_weight  # (batch,)

        # Create confidence targets based on actual portfolio returns
        # High return -> should have high confidence, low return -> low confidence
        # Scale returns to [0, 1] range using sigmoid (returns ~0.01 -> ~0.62, returns ~-0.01 -> ~0.38)
        confidence_targets = torch.sigmoid(portfolio_returns * 50)  # (batch,)

        # MSE between predicted confidence and target based on actual returns
        # This encourages the model to be confident when returns are good
        confidence_loss = F.mse_loss(per_sample_confidence, confidence_targets)

        # 5. Batch entropy regularization: encourage diverse selections across the batch
        # This prevents mode collapse where the model always picks the same stocks
        # Average selection probability per stock across batch
        if batch_size > 1:
            # Normalize weights per sample, then average across batch
            normalized_weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            avg_selection_prob = (normalized_weights * masks).mean(dim=0)  # (num_stocks,)

            # Only consider stocks that are valid in at least one sample
            valid_stocks_mask = masks.sum(dim=0) > 0
            avg_selection_prob = avg_selection_prob[valid_stocks_mask]

            # Normalize to sum to 1
            avg_selection_prob = avg_selection_prob / (avg_selection_prob.sum() + 1e-8)

            # Compute entropy (higher = more diverse selection across batch)
            batch_entropy = -(avg_selection_prob * torch.log(avg_selection_prob + 1e-8)).sum()

            # We want HIGH entropy (diverse), so negate for loss
            batch_entropy_loss = -batch_entropy
        else:
            batch_entropy_loss = torch.tensor(0.0, device=device)
            batch_entropy = torch.tensor(0.0, device=device)

        # 6. Auxiliary loss: gradient signal for ALL stocks, not just selected
        # This helps the model learn about stocks it didn't select
        if self.auxiliary_loss_weight > 0 and scores is not None:
            if self.auxiliary_loss_type == 'ranking':
                auxiliary_loss = self._ranking_loss(scores, actual_returns, masks)
            elif self.auxiliary_loss_type == 'mse':
                auxiliary_loss = self._mse_loss(scores, actual_returns, masks)
            elif self.auxiliary_loss_type == 'contrastive':
                auxiliary_loss = self._contrastive_loss(scores, weights, actual_returns, masks)
            else:
                auxiliary_loss = torch.tensor(0.0, device=device)
        else:
            auxiliary_loss = torch.tensor(0.0, device=device)

        # Combined loss
        total_loss = (
            self.return_weight * return_loss +
            self.sharpe_weight * sharpe_loss +
            self.concentration_weight * concentration_loss +
            self.confidence_weight * confidence_loss +
            self.batch_entropy_weight * batch_entropy_loss +
            self.auxiliary_loss_weight * auxiliary_loss
        )

        metrics = {
            'return_loss': return_loss.item(),
            'sharpe_loss': sharpe_loss.item() if isinstance(sharpe_loss, torch.Tensor) else sharpe_loss,
            'concentration_loss': concentration_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'batch_entropy_loss': batch_entropy_loss.item() if isinstance(batch_entropy_loss, torch.Tensor) else batch_entropy_loss,
            'batch_entropy': batch_entropy.item() if isinstance(batch_entropy, torch.Tensor) else batch_entropy,
            'auxiliary_loss': auxiliary_loss.item() if isinstance(auxiliary_loss, torch.Tensor) else auxiliary_loss,
            'total_loss': total_loss.item(),
            'avg_portfolio_return': portfolio_returns.mean().item(),
            'avg_confidence': per_sample_confidence.mean().item()
        }

        return total_loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: PortfolioModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: PortfolioLoss,
    device: str,
    epoch: int,
    total_epochs: int,
    grad_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    # Get underlying model if DDP-wrapped
    base_model = model.module if hasattr(model, 'module') else model

    total_metrics = defaultdict(float)
    num_batches = 0

    # Only show progress bar on main process
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}",
                disable=not is_main_process())

    for features, returns, masks in pbar:
        features = features.to(device)
        returns = returns.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        scores, weights, confidence = model(features, masks, hard=True)

        # Compute portfolio return
        portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)

        # Compute loss
        loss, metrics = loss_fn(portfolio_returns, weights, confidence, returns, masks, scores)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] += v
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'ret': f"{metrics['avg_portfolio_return']*100:.2f}%"
        })

    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Anneal temperature
    progress = (epoch + 1) / total_epochs
    base_model.anneal_temperature(progress)
    avg_metrics['temperature'] = base_model.selector.temperature

    return avg_metrics


def validate(
    model: PortfolioModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: PortfolioLoss,
    device: str
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    # Get underlying model if DDP-wrapped
    base_model = model.module if hasattr(model, 'module') else model

    total_metrics = defaultdict(float)
    all_portfolio_returns = []
    num_batches = 0

    with torch.no_grad():
        for features, returns, masks in dataloader:
            features = features.to(device)
            returns = returns.to(device)
            masks = masks.to(device)

            # Forward pass (hard selection for evaluation)
            scores, weights, confidence = model(features, masks, hard=True)

            # Compute portfolio return
            portfolio_returns = base_model.compute_portfolio_return(weights, returns, masks)
            all_portfolio_returns.extend(portfolio_returns.cpu().tolist())

            # Compute loss
            loss, metrics = loss_fn(portfolio_returns, weights, confidence, returns, masks, scores)

            for k, v in metrics.items():
                total_metrics[k] += v
            num_batches += 1

    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Compute Sharpe ratio
    returns_array = np.array(all_portfolio_returns)
    if len(returns_array) > 1 and returns_array.std() > 1e-6:
        sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    avg_metrics['sharpe_ratio'] = sharpe
    avg_metrics['cumulative_return'] = (1 + returns_array).prod() - 1

    return avg_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Differentiable Portfolio Training')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to features HDF5')
    parser.add_argument('--prices', type=str, required=True, help='Path to prices HDF5')
    parser.add_argument('--train-start', type=str, default='2010-01-01')
    parser.add_argument('--train-end', type=str, default='2024-12-31')
    parser.add_argument('--val-start', type=str, default='2025-01-01')
    parser.add_argument('--val-end', type=str, default='2025-12-31')

    # Model
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seq-len', type=int, default=500)

    # Portfolio
    parser.add_argument('--top-k', type=int, default=20, help='Number of stocks to select')
    parser.add_argument('--selection', type=str, default='gumbel',
                       choices=['gumbel', 'neuralsort', 'ste', 'soft_attention'])
    parser.add_argument('--horizon-days', type=int, default=1)
    parser.add_argument('--horizon-idx', type=int, default=0)
    parser.add_argument('--initial-temp', type=float, default=1.0)
    parser.add_argument('--min-temp', type=float, default=0.2,
                       help='Minimum temperature (increased from 0.1 to prevent collapse)')

    # Gumbel-softmax improvements
    parser.add_argument('--annealing-schedule', type=str, default='exponential',
                       choices=['linear', 'exponential', 'cosine'],
                       help='Temperature annealing schedule')
    parser.add_argument('--top-m-filter', type=int, default=0,
                       help='Pre-filter to top-m candidates before Gumbel selection (0=disabled, recommended: 50)')
    parser.add_argument('--decoupled-backward-temp', action='store_true',
                       help='Use higher temperature for backward pass (reduces gradient variance)')
    parser.add_argument('--backward-temp-ratio', type=float, default=2.0,
                       help='Ratio of backward to forward temperature when decoupled')
    parser.add_argument('--learnable-temperature', action='store_true',
                       help='Make temperature a learnable parameter instead of annealing')

    # Training
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--max-stocks', type=int, default=300,
                       help='Max stocks per day (memory limit)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0=main process, avoids memory duplication)')

    # Loss weights
    parser.add_argument('--return-weight', type=float, default=1.0)
    parser.add_argument('--sharpe-weight', type=float, default=0.1)
    parser.add_argument('--concentration-weight', type=float, default=0.01)
    parser.add_argument('--confidence-weight', type=float, default=0.1)
    parser.add_argument('--batch-entropy-weight', type=float, default=0.01,
                       help='Weight for batch entropy regularization (prevents mode collapse)')
    parser.add_argument('--auxiliary-loss-weight', type=float, default=0.1,
                       help='Weight for auxiliary loss providing gradients to all stocks')
    parser.add_argument('--auxiliary-loss-type', type=str, default='ranking',
                       choices=['ranking', 'mse', 'contrastive'],
                       help='Type of auxiliary loss: ranking (pairwise), mse (regression), contrastive')

    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/portfolio')
    parser.add_argument('--cache-dir', type=str, default='portfolio_cache',
                       help='Directory for memory-mapped cache files (shared across ranks)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
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

    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set seeds (different per rank for data diversity)
    seed = args.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if is_main_process():
        print(f"\n{'='*80}")
        print("DIFFERENTIABLE PORTFOLIO TRAINING")
        print(f"{'='*80}")
        print(f"\nConfiguration:")
        print(f"  Top-k: {args.top_k}")
        print(f"  Selection method: {args.selection}")
        print(f"  Horizon: {args.horizon_days} days")
        print(f"  Temperature: {args.initial_temp} -> {args.min_temp} ({args.annealing_schedule})")
        if args.top_m_filter > 0:
            print(f"  Top-m pre-filter: {args.top_m_filter}")
        if args.decoupled_backward_temp:
            print(f"  Decoupled backward temp: ratio={args.backward_temp_ratio}")
        if args.learnable_temperature:
            print(f"  Learnable temperature: enabled")
        print(f"  Batch entropy weight: {args.batch_entropy_weight}")
        print(f"  DataLoader workers: {args.num_workers}")
        if args.ddp:
            print(f"  DDP: Enabled (world_size={get_world_size()})")

        # Create datasets
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)

    # Get rank info for shared memory-mapped files
    rank = get_rank()
    world_size = get_world_size()

    # Rank 0 builds cache first, then others load from it
    if rank == 0:
        train_dataset = PortfolioDataset(
            dataset_path=args.data,
            prices_path=args.prices,
            start_date=args.train_start,
            end_date=args.train_end,
            seq_len=args.seq_len,
            horizon_days=args.horizon_days,
            max_stocks_per_day=args.max_stocks,
            cache_dir=args.cache_dir,
            rank=rank,
            world_size=world_size,
        )
        val_dataset = PortfolioDataset(
            dataset_path=args.data,
            prices_path=args.prices,
            start_date=args.val_start,
            end_date=args.val_end,
            seq_len=args.seq_len,
            horizon_days=args.horizon_days,
            max_stocks_per_day=args.max_stocks,
            cache_dir=args.cache_dir,
            rank=rank,
            world_size=world_size,
        )

    # Synchronize before other ranks load
    if args.ddp:
        dist.barrier()

    # Non-zero ranks load after rank 0 has built the cache
    if rank != 0:
        train_dataset = PortfolioDataset(
            dataset_path=args.data,
            prices_path=args.prices,
            start_date=args.train_start,
            end_date=args.train_end,
            seq_len=args.seq_len,
            horizon_days=args.horizon_days,
            max_stocks_per_day=args.max_stocks,
            cache_dir=args.cache_dir,
            rank=rank,
            world_size=world_size,
        )
        val_dataset = PortfolioDataset(
            dataset_path=args.data,
            prices_path=args.prices,
            start_date=args.val_start,
            end_date=args.val_end,
            seq_len=args.seq_len,
            horizon_days=args.horizon_days,
            max_stocks_per_day=args.max_stocks,
            cache_dir=args.cache_dir,
            rank=rank,
            world_size=world_size,
        )

    # Synchronize again after all ranks have loaded
    if args.ddp:
        dist.barrier()

    # Create samplers for DDP
    train_sampler = None
    val_sampler = None
    if args.ddp:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_portfolio_batch,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_portfolio_batch,
        num_workers=args.num_workers,
        pin_memory=(args.device != 'cpu')
    )

    # Create model
    if is_main_process():
        print("\n" + "="*80)
        print("CREATING MODEL")
        print("="*80)

    encoder = SimpleTransformerPredictor(
        input_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_pred_days=4,
        pred_mode='regression'
    )

    model = PortfolioModel(
        encoder=encoder,
        k=args.top_k,
        selection_method=args.selection,
        initial_temperature=args.initial_temp,
        min_temperature=args.min_temp,
        horizon_idx=args.horizon_idx,
        annealing_schedule=args.annealing_schedule,
        top_m_filter=args.top_m_filter,
        decoupled_backward_temp=args.decoupled_backward_temp,
        backward_temp_ratio=args.backward_temp_ratio,
        learnable_temperature=args.learnable_temperature,
    )
    model = model.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        if is_main_process():
            print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap with DDP
    if args.ddp:
        model = DDP(model, device_ids=[get_rank()], output_device=get_rank())

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss function
    loss_fn = PortfolioLoss(
        return_weight=args.return_weight,
        sharpe_weight=args.sharpe_weight,
        concentration_weight=args.concentration_weight,
        confidence_weight=args.confidence_weight,
        batch_entropy_weight=args.batch_entropy_weight,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        auxiliary_loss_type=args.auxiliary_loss_type,
    )

    # Training loop
    if is_main_process():
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)

    best_val_sharpe = -float('inf')

    try:
        for epoch in range(args.epochs):
            # Set epoch for distributed sampler (ensures proper shuffling)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                epoch, args.epochs, args.grad_clip
            )

            # Validate
            val_metrics = validate(model, val_loader, loss_fn, device)

            # Update scheduler
            scheduler.step()

            # Memory cleanup after each epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Print epoch summary (only on main process)
            if is_main_process():
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train - Loss: {train_metrics['total_loss']:.4f}, "
                      f"Return: {train_metrics['avg_portfolio_return']*100:+.2f}%")
                print(f"  Val   - Loss: {val_metrics['total_loss']:.4f}, "
                      f"Return: {val_metrics['avg_portfolio_return']*100:+.2f}%, "
                      f"Sharpe: {val_metrics['sharpe_ratio']:.2f}, "
                      f"Cumul: {val_metrics['cumulative_return']*100:+.2f}%")
                print(f"  Temperature: {train_metrics['temperature']:.3f}")

            # Save best model (only on main process)
            if is_main_process() and val_metrics['sharpe_ratio'] > best_val_sharpe:
                best_val_sharpe = val_metrics['sharpe_ratio']
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_portfolio_model.pt')
                # Get underlying model state dict (unwrap DDP if needed)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_sharpe': best_val_sharpe,
                    'args': vars(args)
                }, checkpoint_path)
                print(f"  Saved best model (Sharpe: {best_val_sharpe:.2f})")

            # Save latest (only on main process)
            if is_main_process():
                latest_path = os.path.join(args.checkpoint_dir, 'latest_portfolio_model.pt')
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_sharpe': val_metrics['sharpe_ratio'],
                    'args': vars(args)
                }, latest_path)

        if is_main_process():
            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print(f"Best validation Sharpe: {best_val_sharpe:.2f}")
            print(f"{'='*80}\n")

    finally:
        # Clean up DDP
        if args.ddp:
            cleanup_ddp()


if __name__ == '__main__':
    main()
