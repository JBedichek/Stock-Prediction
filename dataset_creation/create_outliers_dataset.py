#!/usr/bin/env python3
"""
Create Outliers Dataset - Top 1% Price Movements

This script filters the complete dataset to include only sequences with the top 1%
of price movements for a given prediction horizon. The filtered dataset maintains
the same HDF5 structure and temporal ordering for training.

Usage:
    python -m dataset_creation.create_outliers_dataset \
        --input all_complete_dataset.h5 \
        --output outliers_full_dataset.h5 \
        --horizon 0 \
        --percentile 99 \
        [--seq-len 2000] \
        [--min-future-days 20]
"""

import argparse
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
import sys


def calculate_price_movements(
    input_path: str,
    horizon: int = 0,
    seq_len: int = 2000,
    min_future_days: int = 20,
    pred_days: List[int] = [1, 5, 10, 20]
) -> Tuple[List[Tuple[str, int, float]], Dict[str, np.ndarray]]:
    """
    Calculate price movements for all valid sequences.

    Args:
        input_path: Path to input HDF5 dataset
        horizon: Prediction horizon index (0=pred_days[0], 1=pred_days[1], etc.)
        seq_len: Sequence length
        min_future_days: Minimum future days required
        pred_days: Prediction day offsets

    Returns:
        movements: List of (ticker, end_idx, price_movement) tuples
        ticker_data: Dict mapping ticker to its data for copying later
    """
    print(f"\n{'='*80}")
    print(f"CALCULATING PRICE MOVEMENTS")
    print(f"{'='*80}")
    print(f"  Input dataset: {input_path}")
    print(f"  Horizon: {horizon} (day offset: {pred_days[horizon]})")
    print(f"  Sequence length: {seq_len}")
    print(f"  Min future days: {min_future_days}")
    print()

    movements = []
    ticker_data = {}

    with h5py.File(input_path, 'r') as f:
        tickers = list(f.keys())
        print(f"  Processing {len(tickers)} tickers...")

        for ticker in tqdm(tickers, desc="  Scanning sequences"):
            group = f[ticker]
            features = group['features'][:]  # (num_dates, num_features)
            dates = group['dates'][:]
            num_dates = features.shape[0]

            # Store ticker data for later copying
            ticker_data[ticker] = {
                'features': features,
                'dates': dates,
                'num_dates': num_dates
            }

            # Find valid sequences
            pred_day_offset = pred_days[horizon]
            for end_idx in range(seq_len, num_dates - min_future_days):
                # Get current price (first feature is close price)
                current_price = features[end_idx - 1, 0]

                # Get future price at horizon
                future_idx = end_idx - 1 + pred_day_offset
                if future_idx >= num_dates:
                    continue

                future_price = features[future_idx, 0]

                # Calculate percentage change
                if current_price > 0:
                    pct_change = ((future_price - current_price) / current_price) * 100.0
                    movements.append((ticker, end_idx, pct_change))

    print(f"\n  ✅ Found {len(movements):,} valid sequences")

    return movements, ticker_data


def filter_top_movements(
    movements: List[Tuple[str, int, float]],
    percentile: float = 99.0
) -> Tuple[List[Tuple[str, int]], float]:
    """
    Filter to keep only top percentile of price movements.

    Args:
        movements: List of (ticker, end_idx, price_movement) tuples
        percentile: Percentile threshold (99.0 = top 1%)

    Returns:
        filtered_sequences: List of (ticker, end_idx) tuples to keep
        threshold: The price movement threshold used
    """
    print(f"\n{'='*80}")
    print(f"FILTERING TOP {100-percentile:.1f}% MOVEMENTS")
    print(f"{'='*80}")

    # Extract just the movements
    all_movements = np.array([m[2] for m in movements])

    # Calculate statistics
    print(f"\n  📊 Movement Statistics:")
    print(f"    Mean: {np.mean(all_movements):.4f}%")
    print(f"    Std: {np.std(all_movements):.4f}%")
    print(f"    Min: {np.min(all_movements):.4f}%")
    print(f"    Max: {np.max(all_movements):.4f}%")
    print(f"    Median: {np.median(all_movements):.4f}%")
    print(f"    25th percentile: {np.percentile(all_movements, 25):.4f}%")
    print(f"    75th percentile: {np.percentile(all_movements, 75):.4f}%")
    print(f"    95th percentile: {np.percentile(all_movements, 95):.4f}%")

    # Calculate threshold
    threshold = np.percentile(all_movements, percentile)
    print(f"\n  🎯 Threshold for top {100-percentile:.1f}%: {threshold:.4f}%")

    # Filter sequences
    filtered_sequences = [(ticker, end_idx)
                         for ticker, end_idx, movement in movements
                         if movement >= threshold]

    print(f"  ✅ Kept {len(filtered_sequences):,} sequences (top {100-percentile:.1f}%)")
    print(f"  📉 Filtered out {len(movements) - len(filtered_sequences):,} sequences")

    return filtered_sequences, threshold


def create_filtered_dataset(
    filtered_sequences: List[Tuple[str, int]],
    ticker_data: Dict[str, np.ndarray],
    output_path: str,
    seq_len: int = 2000,
    min_future_days: int = 20
):
    """
    Create new HDF5 dataset with only filtered sequences.

    Args:
        filtered_sequences: List of (ticker, end_idx) tuples to keep
        ticker_data: Dict mapping ticker to its full data
        output_path: Path for output HDF5 file
        seq_len: Sequence length
        min_future_days: Minimum future days required
    """
    print(f"\n{'='*80}")
    print(f"CREATING FILTERED DATASET")
    print(f"{'='*80}")
    print(f"  Output: {output_path}")

    # Group sequences by ticker
    sequences_by_ticker = {}
    for ticker, end_idx in filtered_sequences:
        if ticker not in sequences_by_ticker:
            sequences_by_ticker[ticker] = []
        sequences_by_ticker[ticker].append(end_idx)

    print(f"\n  📦 Creating dataset with {len(sequences_by_ticker)} tickers...")

    with h5py.File(output_path, 'w') as out_f:
        # Set dataset attributes
        out_f.attrs['num_tickers'] = len(sequences_by_ticker)
        out_f.attrs['seq_len'] = seq_len
        out_f.attrs['min_future_days'] = min_future_days
        out_f.attrs['description'] = 'Top 1% price movement outliers dataset'

        for ticker in tqdm(sequences_by_ticker.keys(), desc="  Writing tickers"):
            # Get the original ticker data
            data = ticker_data[ticker]
            features = data['features']
            dates = data['dates']

            # Get sequence indices for this ticker (sorted)
            end_indices = sorted(sequences_by_ticker[ticker])

            # Determine date range needed
            # We need: earliest sequence start to latest sequence end + min_future_days
            min_start_idx = min(end_indices) - seq_len
            max_end_idx = max(end_indices) + min_future_days

            # Extract the needed date range
            filtered_features = features[min_start_idx:max_end_idx]
            filtered_dates = dates[min_start_idx:max_end_idx]

            # Create ticker group
            ticker_group = out_f.create_group(ticker)

            # Store features and dates
            ticker_group.create_dataset(
                'features',
                data=filtered_features,
                compression='gzip',
                compression_opts=4
            )
            ticker_group.create_dataset(
                'dates',
                data=filtered_dates,
                compression='gzip',
                compression_opts=4
            )

            # Store metadata
            ticker_group.attrs['num_dates'] = len(filtered_dates)
            ticker_group.attrs['original_num_dates'] = data['num_dates']
            ticker_group.attrs['num_sequences'] = len(end_indices)
            ticker_group.attrs['date_range_start'] = filtered_dates[0].decode('utf-8')
            ticker_group.attrs['date_range_end'] = filtered_dates[-1].decode('utf-8')

    # Get file size
    file_size_gb = Path(output_path).stat().st_size / (1024**3)

    print(f"\n  ✅ Dataset created successfully!")
    print(f"    Tickers: {len(sequences_by_ticker)}")
    print(f"    Total sequences: {len(filtered_sequences):,}")
    print(f"    File size: {file_size_gb:.2f} GB")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create outliers dataset with top percentage of price movements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        default="all_complete_dataset.h5",
        help="Input HDF5 dataset path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outliers_full_dataset.h5",
        help="Output HDF5 dataset path"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Prediction horizon index (0=1 day, 1=5 days, 2=10 days, 3=20 days)"
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile threshold (99.0 = top 1%%, 95.0 = top 5%%)"
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=2000,
        help="Sequence length"
    )

    parser.add_argument(
        "--min-future-days",
        type=int,
        default=20,
        help="Minimum future days required"
    )

    parser.add_argument(
        "--pred-days",
        type=int,
        nargs='+',
        default=[1, 5, 10, 20],
        help="Prediction day offsets"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.percentile < 0 or args.percentile > 100:
        parser.error("--percentile must be between 0 and 100")

    if args.horizon < 0 or args.horizon >= len(args.pred_days):
        parser.error(f"--horizon must be between 0 and {len(args.pred_days)-1}")

    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check if output file already exists
    if Path(args.output).exists():
        response = input(f"⚠️  Output file already exists: {args.output}\n   Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        Path(args.output).unlink()

    print(f"\n{'='*80}")
    print(f"CREATE OUTLIERS DATASET")
    print(f"{'='*80}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Horizon: {args.horizon} (day offset: {args.pred_days[args.horizon]})")
    print(f"  Percentile: {args.percentile} (top {100-args.percentile:.1f}%)")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Min future days: {args.min_future_days}")
    print(f"{'='*80}\n")

    # Step 1: Calculate all price movements
    movements, ticker_data = calculate_price_movements(
        input_path=args.input,
        horizon=args.horizon,
        seq_len=args.seq_len,
        min_future_days=args.min_future_days,
        pred_days=args.pred_days
    )

    # Step 2: Filter to top percentile
    filtered_sequences, threshold = filter_top_movements(
        movements=movements,
        percentile=args.percentile
    )

    # Step 3: Create new dataset
    create_filtered_dataset(
        filtered_sequences=filtered_sequences,
        ticker_data=ticker_data,
        output_path=args.output,
        seq_len=args.seq_len,
        min_future_days=args.min_future_days
    )

    print(f"✅ DONE! Outliers dataset created at: {args.output}\n")


if __name__ == "__main__":
    main()
