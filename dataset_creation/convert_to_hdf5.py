#!/usr/bin/env python3
"""
Convert Pickle Dataset to HDF5 Format

Converts the {ticker: {date: tensor}} pickle format to HDF5 for faster loading.

HDF5 Structure:
    dataset.h5/
        AAPL/
            dates: array of date strings (YYYY-MM-DD)
            features: 2D array (num_dates, num_features)
        MSFT/
            dates: ...
            features: ...
        ...

Usage:
    python dataset_creation/convert_to_hdf5.py \
        --input all_complete_dataset.pkl \
        --output all_complete_dataset.h5
"""

import h5py
import numpy as np
import torch
from tqdm import tqdm
import argparse
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pic_load


def convert_pickle_to_hdf5(input_path: str, output_path: str, compression: str = None):
    """
    Convert pickle dataset to HDF5 format.

    Args:
        input_path: Path to input pickle file
        output_path: Path to output HDF5 file
        compression: Compression type ('gzip', 'lzf', None)
    """
    print("\n" + "="*80)
    print("CONVERTING PICKLE TO HDF5")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    if compression:
        print(f"Compression: {compression}")

    # Load pickle data
    print("\n📦 Loading pickle file...")
    data = pic_load(input_path)
    print(f"  ✅ Loaded {len(data)} tickers")

    # Get feature dimension from first sample
    sample_ticker = list(data.keys())[0]
    sample_date = list(data[sample_ticker].keys())[0]
    num_features = data[sample_ticker][sample_date].shape[0]
    print(f"  ✅ Feature dimension: {num_features}")

    # Create HDF5 file
    print(f"\n💾 Creating HDF5 file: {output_path}")
    with h5py.File(output_path, 'w') as h5f:
        # Store metadata
        h5f.attrs['num_tickers'] = len(data)
        h5f.attrs['num_features'] = num_features
        h5f.attrs['created_at'] = datetime.now().isoformat()
        h5f.attrs['source_file'] = input_path

        # Convert each ticker
        print(f"\n🔄 Converting tickers...")
        for ticker, date_dict in tqdm(data.items(), desc="  Processing"):
            # Sort dates
            sorted_dates = sorted(date_dict.keys())

            # Convert dates to strings
            date_strings = [d.strftime('%Y-%m-%d') for d in sorted_dates]

            # Stack all feature tensors into 2D array
            feature_list = []
            for date in sorted_dates:
                tensor = date_dict[date]
                # Convert to numpy
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.cpu().numpy()
                feature_list.append(tensor)

            features_2d = np.stack(feature_list)  # (num_dates, num_features)

            # Create group for this ticker
            ticker_group = h5f.create_group(ticker)

            # Store dates as fixed-length strings
            dt = h5py.string_dtype(encoding='utf-8', length=10)
            ticker_group.create_dataset(
                'dates',
                data=np.array(date_strings, dtype=dt),
                compression=compression
            )

            # Store features
            ticker_group.create_dataset(
                'features',
                data=features_2d,
                dtype='float32',
                compression=compression
            )

            # Store metadata
            ticker_group.attrs['num_dates'] = len(sorted_dates)
            ticker_group.attrs['start_date'] = date_strings[0]
            ticker_group.attrs['end_date'] = date_strings[-1]

    print(f"\n✅ Conversion complete!")
    print(f"  Output: {output_path}")

    # Show file sizes
    import os
    input_size_gb = os.path.getsize(input_path) / 1024**3
    output_size_gb = os.path.getsize(output_path) / 1024**3
    compression_ratio = (1 - output_size_gb / input_size_gb) * 100

    print(f"\n📊 File Sizes:")
    print(f"  Input (pickle):  {input_size_gb:.2f} GB")
    print(f"  Output (HDF5):   {output_size_gb:.2f} GB")
    if compression:
        print(f"  Compression:     {compression_ratio:.1f}% reduction")
    print(f"\n{'='*80}\n")


def verify_hdf5_file(hdf5_path: str, num_samples: int = 3):
    """
    Verify the HDF5 file was created correctly.

    Args:
        hdf5_path: Path to HDF5 file
        num_samples: Number of random samples to check
    """
    print("\n" + "="*80)
    print("VERIFYING HDF5 FILE")
    print("="*80)

    with h5py.File(hdf5_path, 'r') as h5f:
        # Print metadata
        print(f"\n📊 Metadata:")
        print(f"  Tickers: {h5f.attrs['num_tickers']}")
        print(f"  Features: {h5f.attrs['num_features']}")
        print(f"  Created: {h5f.attrs['created_at']}")

        # List some tickers
        tickers = list(h5f.keys())
        print(f"\n📈 Sample Tickers:")
        for ticker in tickers[:5]:
            group = h5f[ticker]
            print(f"  {ticker}: {group.attrs['num_dates']} dates "
                  f"({group.attrs['start_date']} to {group.attrs['end_date']})")

        # Sample random data
        print(f"\n🔍 Checking {num_samples} random samples:")
        import random
        sample_tickers = random.sample(tickers, min(num_samples, len(tickers)))

        for ticker in sample_tickers:
            group = h5f[ticker]
            dates = group['dates'][:]
            features = group['features'][:]

            print(f"\n  {ticker}:")
            print(f"    Dates shape: {dates.shape}")
            print(f"    Features shape: {features.shape}")
            print(f"    First date: {dates[0]}")
            print(f"    Last date: {dates[-1]}")
            print(f"    Feature range: [{features.min():.4f}, {features.max():.4f}]")
            print(f"    NaN count: {np.isnan(features).sum()}")

    print(f"\n✅ Verification complete!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Convert pickle dataset to HDF5')

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output HDF5 file')
    parser.add_argument('--compression', type=str, default=None,
                       choices=[None, 'gzip', 'lzf'],
                       help='Compression type (default: None for speed)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify output file after conversion')

    args = parser.parse_args()

    # Convert
    convert_pickle_to_hdf5(args.input, args.output, args.compression)

    # Verify if requested
    if args.verify:
        verify_hdf5_file(args.output)


if __name__ == '__main__':
    main()
