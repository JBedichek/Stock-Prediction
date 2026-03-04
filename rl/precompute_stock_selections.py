#!/usr/bin/env python3
"""
Pre-compute randomized top-4 and bottom-4 stock selections for all dates.

This eliminates repeated computation during training by generating a pool of
diverse stock selections upfront. Training then samples from this pool.

Benefits:
- Eliminates CPU-intensive stock selection during training
- Maintains diversity (random sampling from pool)
- Can be reused across multiple training runs
- Much faster training startup
"""

import h5py
import torch
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.reduced_action_space import get_top_k_stocks_per_horizon, get_bottom_4_stocks, sample_top_4_from_top_k
from inference.backtest_simulation import DatasetLoader
from rl.rl_components import ActorCriticAgent


def precompute_stock_selections(
    data_loader: DatasetLoader,
    agent: ActorCriticAgent,
    trading_days: List[str],
    state_cache_path: str,
    output_path: str,
    num_samples_per_date: int = 100,
    top_k_per_horizon: int = 3,
    sample_fraction: float = 0.3,
    device: str = 'cuda'
):
    """
    Pre-compute diverse stock selections for all trading days.

    For each date, generates N different randomized samples of top-4 and bottom-4 stocks.
    This pool provides diversity during training without repeated computation.

    Args:
        data_loader: DatasetLoader instance
        agent: ActorCriticAgent (for state computation)
        trading_days: List of trading days to process
        state_cache_path: Path to pre-computed state cache
        output_path: Path to save stock selections
        num_samples_per_date: Number of diverse samples to generate per date
        top_k_per_horizon: K value for top-K selection
        sample_fraction: Fraction of stocks to sample (for diversity)
        device: Device for computation
    """
    print("\n" + "="*80)
    print("PRE-COMPUTING RANDOMIZED STOCK SELECTIONS")
    print("="*80)
    print(f"Trading days: {len(trading_days)}")
    print(f"Samples per date: {num_samples_per_date}")
    print(f"Top-K per horizon: {top_k_per_horizon}")
    print(f"Sample fraction: {sample_fraction}")
    print(f"Total samples to generate: {len(trading_days) * num_samples_per_date:,}")
    print(f"Output: {output_path}")

    # Load state cache
    print(f"\n1. Loading state cache from {state_cache_path}...")
    state_cache = {}
    price_cache = {}

    with h5py.File(state_cache_path, 'r') as f:
        print(f"   Loading {len(f.keys())} dates...")
        for date in tqdm(list(f.keys()), desc="   Loading cache"):
            if date not in trading_days:
                continue

            date_grp = f[date]
            tickers = [t.decode('utf-8') for t in date_grp['tickers'][:]]
            states = date_grp['states'][:]
            prices = date_grp['prices'][:]

            state_cache[date] = {}
            price_cache[date] = {}

            for i, ticker in enumerate(tickers):
                state_cache[date][ticker] = torch.from_numpy(states[i]).float()
                price_cache[date][ticker] = float(prices[i])

    print(f"   ✅ Loaded {len(state_cache)} dates")

    # Pre-compute stock selections
    print(f"\n2. Pre-computing stock selections...")

    # Create output file
    with h5py.File(output_path, 'w') as out_file:
        out_file.attrs['num_samples_per_date'] = num_samples_per_date
        out_file.attrs['top_k_per_horizon'] = top_k_per_horizon
        out_file.attrs['sample_fraction'] = sample_fraction
        out_file.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Process each date
        for date in tqdm(trading_days, desc="   Generating samples"):
            if date not in state_cache:
                print(f"   ⚠️  Skipping {date} (not in cache)")
                continue

            # Get cached states for this date
            cached_states = state_cache[date]

            # Filter to stocks with price data
            cached_states_with_prices = {
                ticker: state for ticker, state in cached_states.items()
                if ticker in price_cache.get(date, {})
            }

            if len(cached_states_with_prices) == 0:
                print(f"   ⚠️  Skipping {date} (no stocks with prices)")
                continue

            # Generate N diverse samples for this date
            top_4_samples = []
            bottom_4_samples = []

            for sample_idx in range(num_samples_per_date):
                # Get top-K stocks (with random sampling for diversity)
                top_k_stocks = get_top_k_stocks_per_horizon(
                    cached_states_with_prices,
                    k=top_k_per_horizon,
                    sample_fraction=sample_fraction
                )

                # Sample 4 from top-K (random for diversity)
                top_4_stocks = sample_top_4_from_top_k(
                    top_k_stocks,
                    sample_size=4,
                    deterministic=False
                )

                # Get bottom-4 stocks for shorting
                bottom_4_stocks = get_bottom_4_stocks(
                    cached_states_with_prices,
                    sample_fraction=sample_fraction
                )

                # Store as list of (ticker, horizon_idx) tuples
                top_4_samples.append(top_4_stocks)
                bottom_4_samples.append(bottom_4_stocks)

            # Save to HDF5
            date_grp = out_file.create_group(date)

            # Convert to structured arrays for compact storage
            # Each sample: [(ticker1, horizon1), (ticker2, horizon2), ...]

            # Top-4 samples: (num_samples, 4, 2) where 2 = (ticker, horizon_idx)
            top_4_tickers = np.array([
                [ticker for ticker, _ in sample] for sample in top_4_samples
            ], dtype='S10')
            top_4_horizons = np.array([
                [horizon for _, horizon in sample] for sample in top_4_samples
            ], dtype=np.int32)

            # Bottom-4 samples
            bottom_4_tickers = np.array([
                [ticker for ticker, _ in sample] for sample in bottom_4_samples
            ], dtype='S10')
            bottom_4_horizons = np.array([
                [horizon for _, horizon in sample] for sample in bottom_4_samples
            ], dtype=np.int32)

            date_grp.create_dataset('top_4_tickers', data=top_4_tickers, compression='gzip')
            date_grp.create_dataset('top_4_horizons', data=top_4_horizons, compression='gzip')
            date_grp.create_dataset('bottom_4_tickers', data=bottom_4_tickers, compression='gzip')
            date_grp.create_dataset('bottom_4_horizons', data=bottom_4_horizons, compression='gzip')
            date_grp.attrs['num_stocks'] = len(cached_states_with_prices)

    print(f"\n✅ Pre-computation complete!")
    print(f"   Saved to: {output_path}")

    # Print file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")

    print("="*80 + "\n")


def load_stock_selections(
    selections_path: str,
    date: str
) -> Tuple[List[List[Tuple[str, int]]], List[List[Tuple[str, int]]]]:
    """
    Load pre-computed stock selections for a date.

    Args:
        selections_path: Path to pre-computed selections file
        date: Date to load

    Returns:
        Tuple of (top_4_samples, bottom_4_samples)
        - top_4_samples: List of samples, each containing 4 (ticker, horizon) tuples
        - bottom_4_samples: List of samples, each containing 4 (ticker, horizon) tuples
    """
    with h5py.File(selections_path, 'r') as f:
        if date not in f:
            return [], []

        date_grp = f[date]

        # Load top-4 samples
        top_4_tickers = date_grp['top_4_tickers'][:]
        top_4_horizons = date_grp['top_4_horizons'][:]

        top_4_samples = []
        for i in range(len(top_4_tickers)):
            sample = [
                (top_4_tickers[i][j].decode('utf-8'), int(top_4_horizons[i][j]))
                for j in range(len(top_4_tickers[i]))
            ]
            top_4_samples.append(sample)

        # Load bottom-4 samples
        bottom_4_tickers = date_grp['bottom_4_tickers'][:]
        bottom_4_horizons = date_grp['bottom_4_horizons'][:]

        bottom_4_samples = []
        for i in range(len(bottom_4_tickers)):
            sample = [
                (bottom_4_tickers[i][j].decode('utf-8'), int(bottom_4_horizons[i][j]))
                for j in range(len(bottom_4_tickers[i]))
            ]
            bottom_4_samples.append(sample)

        return top_4_samples, bottom_4_samples


def sample_from_pool(
    top_4_samples: List[List[Tuple[str, int]]],
    bottom_4_samples: List[List[Tuple[str, int]]]
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Randomly sample one selection from the pre-computed pool.

    Args:
        top_4_samples: Pool of top-4 samples
        bottom_4_samples: Pool of bottom-4 samples

    Returns:
        Tuple of (top_4_stocks, bottom_4_stocks)
    """
    import random
    idx = random.randint(0, len(top_4_samples) - 1)
    return top_4_samples[idx], bottom_4_samples[idx]


if __name__ == '__main__':
    # Configuration
    config = {
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'state_cache_path': 'data/rl_state_cache_4yr.h5',
        'output_path': 'data/rl_stock_selections_4yr.h5',
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',
        'num_samples_per_date': 50,  # Generate 100 diverse samples per date
        'top_k_per_horizon': 3,
        'sample_fraction': 0.4,  # Match training config
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Load data
    print("\n1. Loading data...")
    data_loader = DatasetLoader(
        dataset_path=config['dataset_path'],
        prices_path=config['prices_path'],
        num_test_stocks=100
    )

    # Get trading days (last 4 years)
    cutoff_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
    sample_ticker = list(data_loader.prices_file.keys())[0]
    prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
    all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
    trading_days = [d for d in all_trading_days if d >= cutoff_date]

    print(f"   ✅ Trading days: {len(trading_days)}")
    print(f"   Date range: {trading_days[0]} to {trading_days[-1]}")

    # Initialize agent (for state computation compatibility)
    print("\n2. Initializing agent...")
    agent = ActorCriticAgent(
        predictor_checkpoint_path=config['predictor_checkpoint'],
        state_dim=11761,
        hidden_dim=1024,
        action_dim=9
    ).to(config['device'])

    # Pre-compute stock selections
    precompute_stock_selections(
        data_loader=data_loader,
        agent=agent,
        trading_days=trading_days,
        state_cache_path=config['state_cache_path'],
        output_path=config['output_path'],
        num_samples_per_date=config['num_samples_per_date'],
        top_k_per_horizon=config['top_k_per_horizon'],
        sample_fraction=config['sample_fraction'],
        device=config['device']
    )

    print("\n✅ All done! Use this file in training with:")
    print(f"   config['stock_selections_cache'] = '{config['output_path']}'")
