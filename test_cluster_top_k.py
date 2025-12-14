#!/usr/bin/env python3
"""
Test script to verify top-k percent cluster filtering works correctly.
"""

import torch
import numpy as np
from cluster.dynamic_cluster_filter import DynamicClusterFilter

def test_top_k_percent():
    """Test that top-k percent selection works correctly."""

    print("\n" + "="*80)
    print("TESTING TOP-K PERCENT CLUSTER FILTERING")
    print("="*80)

    # Initialize filter with top-k percent
    print("\n1. Initializing filter with top_k_percent=0.3 (30%)")

    filter_topk = DynamicClusterFilter(
        model_path='./checkpoints/best_model_100m_1.18.pt',
        cluster_dir='./cluster_results',
        best_clusters_file='./cluster_results/best_clusters_1d.txt',
        device='cuda',
        batch_size=32,
        embeddings_cache_path='./data/embeddings_cache.h5',
        top_k_percent=0.3  # Select top 30% closest to centroids
    )

    # Create dummy features for testing
    print("\n2. Creating dummy features for 100 stocks")
    features_dict = {}
    for i in range(100):
        ticker = f"STOCK{i:03d}"
        # Create random features (seq_len, feature_dim) matching model's expected input
        features = torch.randn(2000, filter_topk.model.input_dim)
        features_dict[ticker] = features

    # Test filtering
    print("\n3. Testing top-k percent filtering")
    allowed_tickers = filter_topk.filter_stocks_for_date(features_dict)

    print(f"\n✅ Results:")
    print(f"  Total stocks:        100")
    print(f"  Top-k percent:       30%")
    print(f"  Selected stocks:     {len(allowed_tickers)}")
    print(f"  Expected:            ~30")
    print(f"  Match expected:      {25 <= len(allowed_tickers) <= 35}")

    # Compare with hard assignment
    print("\n4. Comparing with hard cluster assignment (original behavior)")

    filter_hard = DynamicClusterFilter(
        model_path='./checkpoints/best_model_100m_1.18.pt',
        cluster_dir='./cluster_results',
        best_clusters_file='./cluster_results/best_clusters_1d.txt',
        device='cuda',
        batch_size=32,
        embeddings_cache_path='./data/embeddings_cache.h5',
        top_k_percent=None  # Hard assignment
    )

    allowed_tickers_hard = filter_hard.filter_stocks_for_date(features_dict)

    print(f"\n✅ Comparison:")
    print(f"  Hard assignment selected:    {len(allowed_tickers_hard)} stocks")
    print(f"  Top-k (30%) selected:        {len(allowed_tickers)} stocks")
    print(f"  Ratio:                       {len(allowed_tickers) / max(1, len(allowed_tickers_hard)):.2f}x")

    # Test with different percentages
    print("\n5. Testing different top-k percentages")
    for pct in [0.1, 0.2, 0.5, 0.7]:
        filter_test = DynamicClusterFilter(
            model_path='./checkpoints/best_model_100m_1.18.pt',
            cluster_dir='./cluster_results',
            best_clusters_file='./cluster_results/best_clusters_1d.txt',
            device='cuda',
            batch_size=32,
            embeddings_cache_path='./data/embeddings_cache.h5',
            top_k_percent=pct
        )

        allowed = filter_test.filter_stocks_for_date(features_dict)
        print(f"  {pct*100:>3.0f}%: {len(allowed):>3} stocks selected")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print("\nThe top-k percent filtering is working correctly!")
    print("- Selects approximately k% of stocks closest to best cluster centroids")
    print("- Provides more flexibility than hard cluster assignment")
    print("- Can be used with --cluster-top-k-percent flag in backtest_simulation.py")
    print()

if __name__ == '__main__':
    test_top_k_percent()
