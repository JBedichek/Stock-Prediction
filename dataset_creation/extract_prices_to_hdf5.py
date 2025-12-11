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

import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pic_load


def extract_prices_from_pickle(input_path: str, output_path: str):
    """
    Extract actual prices from all_price_data.pkl to HDF5.

    Assumes input format: {ticker: DataFrame with dates as index/column and prices}
    """
    print(f"\n{'='*80}")
    print("EXTRACTING PRICES TO HDF5")
    print(f"{'='*80}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Load price data
    print(f"\n📦 Loading price data...")
    price_data = pic_load(input_path)
    tickers = sorted(price_data.keys())
    print(f"  ✅ Loaded {len(tickers)} tickers")

    with h5py.File(output_path, 'w') as f_out:
        for ticker in tqdm(tickers, desc="Extracting"):
            df = price_data[ticker]

            # Handle empty DataFrames
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue

            # Extract dates and prices from DataFrame
            if isinstance(df, pd.DataFrame):
                # DataFrame could have dates as index or as a column
                if 'Date' in df.columns or 'date' in df.columns:
                    date_col = 'Date' if 'Date' in df.columns else 'date'
                    dates = df[date_col].astype(str).tolist()
                    # Price could be in various columns
                    price_col = None
                    for col in ['Close', 'close', 'price', 'Price', 'Adj Close']:
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
                continue

            # Convert dates to bytes for HDF5
            dates_bytes = np.array([d.encode('utf-8') for d in dates])

            # Create group for this ticker
            grp = f_out.create_group(ticker)
            grp.create_dataset('prices', data=prices, compression='gzip')
            grp.create_dataset('dates', data=dates_bytes, compression='gzip')

    print(f"\n✅ Prices extracted successfully!")
    print(f"💾 Saved to: {output_path}")


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

    args = parser.parse_args()

    # Extract prices
    extract_prices_from_pickle(args.input, args.output)

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
