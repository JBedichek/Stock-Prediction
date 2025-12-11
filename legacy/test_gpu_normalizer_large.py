#!/usr/bin/env python3
"""
Test GPU-accelerated cross-sectional normalizer with larger dataset
Similar to real production workload (370 stocks × 6000 dates × 400 features)
"""

import pandas as pd
import numpy as np
import torch
import sys
from datetime import datetime, timedelta
import time

sys.path.append('/home/james/Desktop/Stock-Prediction')
from data_scraping.cross_sectional_normalizer import CrossSectionalNormalizer

def create_large_test_data(n_stocks=370, n_dates=1000, n_features=200):
    """Create synthetic test data matching production scale."""
    print(f"Creating test data: {n_stocks} stocks × {n_dates} dates × {n_features} features")

    date_range = pd.date_range(start='2015-01-01', periods=n_dates, freq='D')

    test_data = {}

    for i in range(n_stocks):
        ticker = f"STOCK{i:04d}"

        data = {}

        # Add many fundamental features (cross-sectional)
        for j in range(n_features // 2):
            data[f'fundamental_metric_{j}'] = np.random.randn(n_dates) * 100 + 500

        # Add many technical features (temporal)
        for j in range(n_features // 2):
            data[f'tech_indicator_{j}'] = np.random.randn(n_dates) * 10

        df = pd.DataFrame(data, index=date_range)
        test_data[ticker] = df

    total_size = n_stocks * n_dates * n_features * 4 / 1e9  # 4 bytes per float32
    print(f"Total data size: {total_size:.2f} GB")

    return test_data

def monitor_gpu():
    """Monitor GPU utilization during normalization."""
    if torch.cuda.is_available():
        print(f"\n📊 GPU Memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   GPU Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def main():
    print("="*80)
    print("LARGE-SCALE GPU NORMALIZATION TEST")
    print("="*80)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  No GPU available")
        return

    # Create large test data
    print("\n" + "="*80)
    test_data = create_large_test_data(n_stocks=370, n_dates=1000, n_features=200)

    # Test normalization
    print("\n" + "="*80)
    print("RUNNING NORMALIZATION")
    print("="*80)

    monitor_gpu()

    normalizer = CrossSectionalNormalizer()

    start = time.time()
    normalized_data = normalizer.normalize_dataframes(test_data, verbose=True)
    end = time.time()

    print(f"\n⏱️  Total time: {end - start:.2f} seconds")

    monitor_gpu()

    # Verify a sample
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    test_ticker = list(normalized_data.keys())[0]
    test_date = normalized_data[test_ticker].index[100]

    # Check cross-sectional feature
    cross_feature = 'fundamental_metric_0'
    if cross_feature in normalized_data[test_ticker].columns:
        values_at_date = [df.loc[test_date, cross_feature]
                          for df in normalized_data.values()
                          if test_date in df.index and cross_feature in df.columns]

        if values_at_date:
            mean = np.mean(values_at_date)
            std = np.std(values_at_date)
            print(f"Cross-sectional feature at {test_date.date()}:")
            print(f"  Mean: {mean:.6f} (should be ≈0)")
            print(f"  Std: {std:.6f} (should be ≈1)")

    # Check temporal feature
    temporal_feature = 'tech_indicator_0'
    if temporal_feature in normalized_data[test_ticker].columns:
        ticker_values = normalized_data[test_ticker][temporal_feature].values
        mean = np.mean(ticker_values)
        std = np.std(ticker_values)
        print(f"\nTemporal feature for {test_ticker}:")
        print(f"  Mean: {mean:.6f} (should be ≈0)")
        print(f"  Std: {std:.6f} (should be ≈1)")

    print("\n✅ TEST COMPLETE!")

if __name__ == '__main__':
    main()
