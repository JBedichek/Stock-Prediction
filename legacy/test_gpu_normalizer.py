#!/usr/bin/env python3
"""
Quick test of GPU-accelerated cross-sectional normalizer
"""

import pandas as pd
import numpy as np
import torch
import sys
from datetime import datetime, timedelta

sys.path.append('/home/james/Desktop/Stock-Prediction')
from data_scraping.cross_sectional_normalizer import CrossSectionalNormalizer

def create_test_data(n_stocks=100, n_dates=500, n_features=50):
    """Create synthetic test data."""
    print(f"Creating test data: {n_stocks} stocks × {n_dates} dates × {n_features} features")

    date_range = pd.date_range(start='2020-01-01', periods=n_dates, freq='D')

    test_data = {}

    for i in range(n_stocks):
        ticker = f"TEST{i:03d}"

        # Create random data
        data = {
            f'pe_ratio': np.random.randn(n_dates) * 10 + 20,  # Cross-sectional
            f'revenue': np.random.randn(n_dates) * 1e6 + 5e6,  # Cross-sectional
            f'roe': np.random.randn(n_dates) * 5 + 15,  # Cross-sectional
            f'price': np.random.randn(n_dates) * 10 + 100,  # Temporal
            f'return': np.random.randn(n_dates) * 0.02,  # Temporal
            f'volume': np.random.randn(n_dates) * 1e6 + 1e7,  # Cross-sectional
        }

        # Add more features to reach n_features
        for j in range(6, n_features):
            if j % 2 == 0:
                # Cross-sectional feature
                data[f'fundamental_metric_{j}'] = np.random.randn(n_dates) * 100
            else:
                # Temporal feature
                data[f'tech_indicator_{j}'] = np.random.randn(n_dates)

        df = pd.DataFrame(data, index=date_range)
        test_data[ticker] = df

    return test_data

def main():
    print("="*80)
    print("GPU-ACCELERATED CROSS-SECTIONAL NORMALIZER TEST")
    print("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n⚠️  No GPU available, using CPU")

    # Create test data
    print("\n" + "="*80)
    test_data = create_test_data(n_stocks=100, n_dates=500, n_features=50)

    # Test normalization
    print("\n" + "="*80)
    print("TESTING NORMALIZATION")
    print("="*80)

    normalizer = CrossSectionalNormalizer()

    import time
    start = time.time()

    normalized_data = normalizer.normalize_dataframes(test_data, verbose=True)

    end = time.time()

    print(f"\n⏱️  Total time: {end - start:.2f} seconds")

    # Verify results
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Check a cross-sectional feature (should have mean≈0, std≈1 across stocks at each date)
    test_ticker = list(normalized_data.keys())[0]
    test_date = normalized_data[test_ticker].index[100]

    cross_sectional_feature = 'pe_ratio'
    values_at_date = [df.loc[test_date, cross_sectional_feature]
                      for df in normalized_data.values()
                      if test_date in df.index and cross_sectional_feature in df.columns]

    if values_at_date:
        mean = np.mean(values_at_date)
        std = np.std(values_at_date)
        print(f"\nCross-sectional feature '{cross_sectional_feature}' at {test_date.date()}:")
        print(f"  Mean across stocks: {mean:.6f} (should be ≈0)")
        print(f"  Std across stocks: {std:.6f} (should be ≈1)")

    # Check a temporal feature (should have mean≈0, std≈1 over time for each stock)
    temporal_feature = 'return'
    if temporal_feature in normalized_data[test_ticker].columns:
        ticker_values = normalized_data[test_ticker][temporal_feature].values
        mean = np.mean(ticker_values)
        std = np.std(ticker_values)
        print(f"\nTemporal feature '{temporal_feature}' for {test_ticker}:")
        print(f"  Mean over time: {mean:.6f} (should be ≈0)")
        print(f"  Std over time: {std:.6f} (should be ≈1)")

    # Check for NaNs and Infs
    has_issues = False
    for ticker, df in normalized_data.items():
        if df.isna().any().any():
            print(f"\n⚠️  NaN values found in {ticker}")
            has_issues = True
        if np.isinf(df.values).any():
            print(f"\n⚠️  Inf values found in {ticker}")
            has_issues = True

    if not has_issues:
        print(f"\n✅ No NaN or Inf values found!")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
