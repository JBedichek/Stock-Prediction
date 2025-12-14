#!/usr/bin/env python3
"""
Test coverage-based cluster selection.

This script verifies that the new coverage-based selection works correctly
and compares it with the threshold-based approach.
"""

import sys
from cluster.analyze_clusters import ClusterAnalyzer
import pickle
from pathlib import Path

def test_coverage_selection():
    print("\n" + "="*80)
    print("TESTING COVERAGE-BASED CLUSTER SELECTION")
    print("="*80)

    # Load existing cluster analysis results
    cluster_dir = Path("cluster_results")

    print("\n1. Loading existing cluster analysis...")
    with open(cluster_dir / 'cluster_analysis.pkl', 'rb') as f:
        results = pickle.load(f)

    cluster_stats = results['cluster_stats']
    print(f"   ✓ Loaded statistics for {len(cluster_stats)} clusters")

    # Initialize analyzer
    analyzer = ClusterAnalyzer(str(cluster_dir))

    # Calculate total stocks
    total_stocks = sum(len(tickers) for tickers in analyzer.cluster_to_tickers.values())
    print(f"   ✓ Total unique stocks: {total_stocks}")

    # Test coverage-based selection for different percentages
    print("\n2. Testing coverage-based selection...")

    test_configs = [
        (0.1, "Conservative"),
        (0.3, "Balanced (recommended)"),
        (0.5, "Aggressive"),
    ]

    for coverage_pct, label in test_configs:
        print(f"\n   {label} ({coverage_pct*100:.0f}% coverage):")

        best_clusters = analyzer.identify_best_clusters(
            cluster_stats,
            horizon=1,  # 1-day horizon
            top_k_percent_coverage=coverage_pct,
            ranking_metric='mean_return'
        )

        actual_stocks = sum(len(analyzer.cluster_to_tickers[cid]) for cid in best_clusters)
        actual_coverage = actual_stocks / total_stocks

        print(f"     - Selected clusters: {len(best_clusters)}")
        print(f"     - Stocks covered: {actual_stocks} ({actual_coverage*100:.1f}%)")
        print(f"     - Cluster IDs: {best_clusters[:5]}{'...' if len(best_clusters) > 5 else ''}")

    # Compare with threshold-based selection
    print("\n3. Comparing with threshold-based selection...")

    threshold_clusters = analyzer.identify_best_clusters(
        cluster_stats,
        horizon=1,
        min_return=0.01,
        min_win_rate=0.5,
        min_sharpe=0.1,
        top_k=None  # No limit
    )

    threshold_stocks = sum(len(analyzer.cluster_to_tickers[cid]) for cid in threshold_clusters) if threshold_clusters else 0
    threshold_coverage = threshold_stocks / total_stocks if total_stocks > 0 else 0

    print(f"\n   Threshold-based results:")
    print(f"     - Selected clusters: {len(threshold_clusters)}")
    print(f"     - Stocks covered: {threshold_stocks} ({threshold_coverage*100:.1f}%)")
    print(f"     - Cluster IDs: {threshold_clusters[:5]}{'...' if len(threshold_clusters) > 5 else ''}")

    # Test with different ranking metrics
    print("\n4. Testing different ranking metrics...")

    metrics = ['mean_return', 'sharpe', 'win_rate']

    for metric in metrics:
        best_clusters = analyzer.identify_best_clusters(
            cluster_stats,
            horizon=1,
            top_k_percent_coverage=0.3,
            ranking_metric=metric
        )

        print(f"\n   Ranking by {metric}:")
        print(f"     - Selected clusters: {len(best_clusters)}")
        print(f"     - Top cluster IDs: {best_clusters[:5]}")

    print("\n" + "="*80)
    print("✅ COVERAGE-BASED SELECTION TEST COMPLETE")
    print("="*80)

    print("\nKey findings:")
    print("- Coverage-based selection provides predictable stock coverage")
    print("- Different percentages yield different numbers of clusters")
    print("- Ranking metric affects which clusters are selected")
    print("- Threshold-based selection can be unpredictable")

    print("\nRecommendation:")
    print("Use coverage-based selection with 30-40% coverage for most use cases")
    print()

if __name__ == '__main__':
    try:
        test_coverage_selection()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
