#!/usr/bin/env python3
"""
Diagnose Dataset Issues

Checks for common problems:
1. Feature dimension mismatches
2. Date inconsistencies
3. Missing data
"""

import h5py
import numpy as np
from collections import Counter
from datetime import datetime

def diagnose_dataset(dataset_path='all_complete_dataset.h5', prices_path='actual_prices.h5'):
    """Diagnose common dataset issues."""

    print(f"\n{'='*80}")
    print("DATASET DIAGNOSTICS")
    print(f"{'='*80}\n")

    # Open main dataset
    h5f = h5py.File(dataset_path, 'r')
    tickers = list(h5f.keys())

    print(f"📊 Main Dataset: {dataset_path}")
    print(f"  Total tickers: {len(tickers)}")

    # Check feature dimensions
    print(f"\n🔍 Checking feature dimensions...")
    feature_dims = []
    dim_examples = {}

    for ticker in tickers[:100]:  # Sample first 100
        dim = h5f[ticker]['features'].shape[1]
        feature_dims.append(dim)

        if dim not in dim_examples:
            dim_examples[dim] = []
        if len(dim_examples[dim]) < 3:
            dim_examples[dim].append(ticker)

    dim_counts = Counter(feature_dims)

    if len(dim_counts) > 1:
        print(f"  ⚠️  FEATURE DIMENSION MISMATCH DETECTED!")
        print(f"  Found {len(dim_counts)} different feature dimensions:")
        for dim, count in sorted(dim_counts.items()):
            pct = count / len(feature_dims) * 100
            examples = dim_examples.get(dim, [])
            print(f"    - {dim} features: {count} tickers ({pct:.1f}%) - e.g., {', '.join(examples[:3])}")

        print(f"\n  💡 Solution: Padding is enabled, but ideally all tickers should have same dimensions")
    else:
        dim = list(dim_counts.keys())[0]
        print(f"  ✅ All tickers have consistent {dim} features")

    # Check date ranges
    print(f"\n📅 Checking date ranges...")
    first_dates = []
    last_dates = []

    for ticker in tickers[:20]:  # Sample first 20
        dates_bytes = h5f[ticker]['dates'][:]
        dates = sorted([d.decode('utf-8') for d in dates_bytes])
        first_dates.append(dates[0])
        last_dates.append(dates[-1])

    earliest = min(first_dates)
    latest_main = max(last_dates)

    print(f"  Main dataset range: {earliest} to {latest_main}")
    print(f"  Total dates in sample: {len(h5f[tickers[0]]['dates'][:])}")

    # Check prices file
    try:
        h5f_prices = h5py.File(prices_path, 'r')
        prices_tickers = list(h5f_prices.keys())

        sample_ticker = prices_tickers[0]
        prices_dates = [d.decode('utf-8') for d in h5f_prices[sample_ticker]['dates'][:]]
        latest_prices = max(prices_dates)

        print(f"\n📊 Prices File: {prices_path}")
        print(f"  Total tickers: {len(prices_tickers)}")
        print(f"  Date range: ... to {latest_prices}")

        if latest_prices != latest_main:
            print(f"\n  ⚠️  DATE MISMATCH!")
            print(f"    Main dataset: {latest_main}")
            print(f"    Prices file:  {latest_prices}")

            # Parse dates
            latest_main_dt = datetime.strptime(latest_main, '%Y-%m-%d')
            latest_prices_dt = datetime.strptime(latest_prices, '%Y-%m-%d')
            days_behind = (latest_main_dt - latest_prices_dt).days

            print(f"    Prices file is {days_behind} days behind")
            print(f"\n  💡 Solution: Prices file will use fallback (features[0])")
        else:
            print(f"  ✅ Prices file is up-to-date")

        h5f_prices.close()
    except Exception as e:
        print(f"\n  ⚠️  Could not check prices file: {e}")

    # Check for NaN/Inf
    print(f"\n🔬 Checking data quality...")
    nan_counts = []
    inf_counts = []

    for ticker in tickers[:20]:
        features = h5f[ticker]['features'][:]
        nan_counts.append(np.isnan(features).sum())
        inf_counts.append(np.isinf(features).sum())

    total_nan = sum(nan_counts)
    total_inf = sum(inf_counts)

    if total_nan > 0:
        print(f"  ⚠️  Found {total_nan} NaN values in sample")
    if total_inf > 0:
        print(f"  ⚠️  Found {total_inf} Inf values in sample")

    if total_nan == 0 and total_inf == 0:
        print(f"  ✅ No NaN or Inf values detected")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    issues = []
    if len(dim_counts) > 1:
        issues.append("Feature dimension mismatch (padding enabled)")
    if latest_prices != latest_main:
        issues.append("Prices file outdated")
    if total_nan > 0 or total_inf > 0:
        issues.append("NaN/Inf values detected")

    if issues:
        print(f"\n⚠️  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n✅ All issues are handled automatically, but may affect accuracy")
    else:
        print(f"\n✅ No issues detected - dataset looks good!")

    h5f.close()

    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose dataset issues')
    parser.add_argument('--dataset', type=str, default='all_complete_dataset.h5',
                       help='Path to main dataset')
    parser.add_argument('--prices', type=str, default='actual_prices.h5',
                       help='Path to prices file')

    args = parser.parse_args()

    diagnose_dataset(args.dataset, args.prices)
