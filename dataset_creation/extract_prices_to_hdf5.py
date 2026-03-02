#!/usr/bin/env python3
"""
Extract actual prices from all_price_data.pkl to HDF5 for backtesting.

Creates a lightweight HDF5 file with just ticker -> date -> actual_price mapping.
This is used for calculating returns while keeping normalized features for inference.

Usage:
    python dataset_creation/extract_prices_to_hdf5.py \
        --input all_price_data.pkl \
        --output actual_prices.h5
"""

import os
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pic_load


def extract_prices_from_pickle(input_path: str, output_path: str, max_price: float = 10000.0):
    """
    Extract actual prices from all_price_data.pkl to HDF5.

    Assumes input format: {ticker: DataFrame with dates as index/column and prices}

    Args:
        input_path: Path to input pickle file
        output_path: Path to output HDF5 file
        max_price: Maximum reasonable price - stocks with prices above this are skipped
                   (helps filter out stocks with corrupted/unadjusted split data)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING PRICES TO HDF5")
    print(f"{'='*80}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Max price filter: ${max_price:,.0f}")

    # Load price data
    print(f"\n📦 Loading price data...")
    price_data = pic_load(input_path)
    tickers = sorted(price_data.keys())
    print(f"  ✅ Loaded {len(tickers)} tickers")

    skipped_empty = 0
    skipped_bad_price = []
    extracted = 0

    with h5py.File(output_path, 'w') as f_out:
        for ticker in tqdm(tickers, desc="Extracting"):
            df = price_data[ticker]

            # Handle empty DataFrames
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                skipped_empty += 1
                continue

            # Extract dates and prices from DataFrame
            if isinstance(df, pd.DataFrame):
                # DataFrame could have dates as index or as a column
                if 'Date' in df.columns or 'date' in df.columns:
                    date_col = 'Date' if 'Date' in df.columns else 'date'
                    dates = df[date_col].astype(str).tolist()
                    # Price could be in various columns
                    # Prioritize split-adjusted prices (Adj Close) over raw Close
                    price_col = None
                    for col in ['Adj Close', 'adj_close', 'close', 'Close', 'price', 'Price']:
                        if col in df.columns:
                            price_col = col
                            break
                    if price_col is None:
                        # Use first numeric column
                        price_col = df.select_dtypes(include=[np.number]).columns[0]
                    prices = df[price_col].values.astype(np.float32)
                else:
                    # Dates are in index
                    dates = df.index.astype(str).tolist()
                    # Get prices from first column
                    prices = df.iloc[:, 0].values.astype(np.float32)
            else:
                # Not a DataFrame, skip
                skipped_empty += 1
                continue

            # Data quality filter: skip stocks with unreasonable prices
            # This catches stocks with corrupted/unadjusted split data
            price_max = np.nanmax(prices)
            if price_max > max_price:
                skipped_bad_price.append((ticker, price_max))
                continue

            # Convert dates to bytes for HDF5
            dates_bytes = np.array([d.encode('utf-8') for d in dates])

            # Create group for this ticker
            grp = f_out.create_group(ticker)
            grp.create_dataset('prices', data=prices, compression='gzip')
            grp.create_dataset('dates', data=dates_bytes, compression='gzip')
            extracted += 1

    print(f"\n✅ Prices extracted successfully!")
    print(f"  Extracted: {extracted} tickers")
    print(f"  Skipped (empty): {skipped_empty}")
    print(f"  Skipped (bad price data): {len(skipped_bad_price)}")

    if skipped_bad_price:
        print(f"\n⚠️  Tickers skipped due to unreasonable prices (>${max_price:,.0f}):")
        for ticker, price in sorted(skipped_bad_price, key=lambda x: -x[1])[:20]:
            print(f"    {ticker}: ${price:,.2f}")
        if len(skipped_bad_price) > 20:
            print(f"    ... and {len(skipped_bad_price) - 20} more")

    print(f"\n💾 Saved to: {output_path}")


def verify_prices_file(prices_path: str):
    """Verify the prices file was created correctly."""
    print(f"\n{'='*80}")
    print("VERIFYING PRICES FILE")
    print(f"{'='*80}")

    with h5py.File(prices_path, 'r') as f:
        tickers = list(f.keys())
        print(f"✅ {len(tickers)} tickers")

        # Check first ticker
        ticker = tickers[0]
        prices = f[ticker]['prices'][:]
        dates = f[ticker]['dates'][:]

        print(f"\n📊 Sample ({ticker}):")
        print(f"  Dates: {len(dates)}")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  First date: {dates[0].decode('utf-8')}")
        print(f"  Last date: {dates[-1].decode('utf-8')}")


def main():
    parser = argparse.ArgumentParser(description='Extract actual prices from all_price_data.pkl to HDF5')

    parser.add_argument('--input', type=str, default='all_price_data.pkl',
                       help='Input price pickle file')
    parser.add_argument('--output', type=str, default='actual_prices.h5',
                       help='Output prices HDF5')
    parser.add_argument('--max-price', type=float, default=10000.0,
                       help='Maximum reasonable stock price - stocks above this are skipped (default: 10000)')

    args = parser.parse_args()

    # Extract prices
    extract_prices_from_pickle(args.input, args.output, args.max_price)

    # Verify
    verify_prices_file(args.output)

    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print(f"{'='*80}")
    print(f"\nUse this file for backtesting:")
    print(f"  python inference/backtest_simulation.py \\")
    print(f"    --data all_complete_dataset.h5 \\")
    print(f"    --prices {args.output}")
    print()


if __name__ == '__main__':
    main()
