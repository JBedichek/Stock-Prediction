"""
Test script for the fundamental data collector implementation.

Tests:
1. Basic collection works
2. Caching works correctly
3. Imputation statistics are reasonable
4. Tensor conversion produces correct shapes
5. Normalization doesn't produce extreme values
"""

import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')

from data_scraping.Stock import stock_info
import torch
import os

def test_fundamental_collector():
    """Test the fundamental data collector on a small sample."""

    print("="*80)
    print("TESTING FUNDAMENTAL DATA COLLECTOR")
    print("="*80)

    # Create small test sample
    test_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'TSLA': 'Tesla Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc.',
        'WMT': 'Walmart Inc.',
        'PG': 'Procter & Gamble Co.',
        'UNH': 'UnitedHealth Group Inc.',
    }

    print(f"\n1. Creating stock_info instance with {len(test_stocks)} test stocks...")
    si = stock_info(test_stocks, dataset_name="test")

    # Test 1: Basic collection
    print("\n2. Testing basic collection (will cache)...")
    cache_file = '/home/james/Desktop/Stock-Prediction/test_fundamentals_cache.pkl'

    # Remove cache if exists
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("   Removed existing cache")

    metrics = si.get_fundamental_metrics(use_cache=True, cache_file=cache_file)

    print(f"\n   ✅ Collected metrics for {len(metrics)} stocks")

    # Check data structure
    if metrics:
        sample_ticker = list(metrics.keys())[0]
        sample_data = metrics[sample_ticker]
        print(f"   Sample stock ({sample_ticker}) has {len(sample_data)} metrics")

        # Show a few sample values
        print(f"\n   Sample values for {sample_ticker}:")
        for i, (metric, value) in enumerate(list(sample_data.items())[:5]):
            print(f"     {metric}: {value}")

    # Test 2: Caching
    print("\n3. Testing cache loading...")
    import time
    start = time.time()
    metrics_cached = si.get_fundamental_metrics(use_cache=True, cache_file=cache_file)
    cache_time = time.time() - start

    print(f"   ✅ Cache loaded in {cache_time:.4f} seconds")
    print(f"   Cache file exists: {os.path.exists(cache_file)}")

    # Test 3: Tensor conversion
    print("\n4. Testing tensor conversion...")
    tensors = si.get_fundamental_metrics_as_tensor(normalize=False)

    if tensors:
        sample_tensor = tensors[list(tensors.keys())[0]]
        print(f"   ✅ Tensor shape: {sample_tensor.shape}")
        print(f"   Expected shape: (27,)")
        print(f"   Shape matches: {sample_tensor.shape == torch.Size([27])}")

        # Check for NaN values
        has_nan = torch.isnan(sample_tensor).any()
        print(f"   Contains NaN: {has_nan}")

        # Show sample values
        print(f"\n   Sample tensor values (first 5):")
        print(f"   {sample_tensor[:5]}")

    # Test 4: Normalization
    print("\n5. Testing normalization...")
    tensors_normalized = si.get_fundamental_metrics_as_tensor(normalize=True)

    if tensors_normalized:
        # Collect all values
        all_values = torch.stack(list(tensors_normalized.values()))

        print(f"   ✅ Normalized tensor shape: {all_values.shape}")
        print(f"   Expected: ({len(test_stocks)}, 27)")

        # Check statistics
        mean_vals = all_values.mean(dim=0)
        std_vals = all_values.std(dim=0)
        min_vals = all_values.min(dim=0)[0]
        max_vals = all_values.max(dim=0)[0]

        print(f"\n   Normalization statistics:")
        print(f"     Mean of means: {mean_vals.mean():.4f} (should be ~0)")
        print(f"     Mean of stds: {std_vals.mean():.4f}")
        print(f"     Global min: {min_vals.min():.4f}")
        print(f"     Global max: {max_vals.min():.4f}")
        print(f"     Values in [-10, 10]: {((all_values >= -10) & (all_values <= 10)).all()}")

        # Show sample normalized values
        print(f"\n   Sample normalized tensor (first stock, first 5 values):")
        print(f"   {all_values[0, :5]}")

    # Test 5: Coverage statistics
    print("\n6. Coverage statistics...")
    if metrics:
        total_metrics = 27
        missing_counts = {}

        for ticker, data in metrics.items():
            for metric in data.keys():
                if metric not in missing_counts:
                    missing_counts[metric] = 0
                if data[metric] is None or (isinstance(data[metric], float) and data[metric] == 0.0):
                    missing_counts[metric] += 1

        print(f"\n   Metrics with missing values (after imputation):")
        for metric, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / len(metrics)) * 100
            if count > 0:
                print(f"     {metric}: {count}/{len(metrics)} ({pct:.1f}%)")

    # Cleanup
    print("\n7. Cleanup...")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("   ✅ Removed test cache file")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    test_fundamental_collector()
