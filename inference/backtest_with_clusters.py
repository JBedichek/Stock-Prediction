#!/usr/bin/env python3
"""
Example: Running Backtest with Cluster-Based Stock Filtering

This example shows how to integrate cluster filtering into your backtesting workflow.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.backtest_simulation import DatasetLoader, ModelPredictor, BacktestSimulator
from cluster.cluster_filter import ClusterFilter


def run_backtest_with_clusters(
    dataset_path: str,
    model_path: str,
    prices_path: str,
    cluster_dir: str,
    best_clusters_file: str,
    num_test_stocks: int = 1000,
    subset_size: int = 100,
    top_k: int = 5,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30"
):
    """
    Run backtest with cluster-based filtering.

    Args:
        dataset_path: Path to HDF5 dataset with features
        model_path: Path to trained model checkpoint
        prices_path: Path to actual prices HDF5
        cluster_dir: Directory with cluster results
        best_clusters_file: File with best cluster IDs
        num_test_stocks: Number of stocks from dataset to consider
        subset_size: Number of stocks to randomly sample each day
        top_k: Number of top stocks to invest in each day
        start_date: Start date for backtest
        end_date: End date for backtest
    """

    print(f"\n{'='*80}")
    print("BACKTEST WITH CLUSTER FILTERING")
    print(f"{'='*80}")

    # Step 1: Initialize cluster filter
    print(f"\n📊 Step 1: Initialize Cluster Filter")
    cluster_filter = ClusterFilter(
        cluster_dir=cluster_dir,
        best_clusters_file=best_clusters_file
    )
    cluster_filter.print_stats()

    # Step 2: Load dataset
    print(f"\n📊 Step 2: Load Dataset")
    data_loader = DatasetLoader(
        dataset_path=dataset_path,
        num_test_stocks=num_test_stocks,
        subset_size=subset_size,
        prices_path=prices_path
    )

    print(f"\n  Before filtering: {len(data_loader.test_tickers)} stocks")

    # Step 3: Apply cluster filter
    print(f"\n📊 Step 3: Apply Cluster Filter")
    original_count = len(data_loader.test_tickers)
    filtered_tickers = cluster_filter.filter_tickers(data_loader.test_tickers)

    # Update the data loader with filtered tickers
    data_loader.test_tickers = filtered_tickers
    data_loader.full_pool = filtered_tickers.copy()

    print(f"  ✓ Filtered: {original_count} → {len(filtered_tickers)} stocks")
    print(f"  ✓ Filter rate: {len(filtered_tickers)/original_count*100:.1f}%")

    # Step 4: Load model
    print(f"\n📊 Step 4: Load Model")
    predictor = ModelPredictor(model_path=model_path)

    # Step 5: Run backtest
    print(f"\n📊 Step 5: Run Backtest")
    simulator = BacktestSimulator(
        data_loader=data_loader,
        predictor=predictor,
        top_k=top_k
    )

    results = simulator.run_backtest(
        start_date=start_date,
        end_date=end_date
    )

    # Step 6: Show results
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS (WITH CLUSTER FILTERING)")
    print(f"{'='*80}")

    print(f"\n📈 Performance Metrics:")
    print(f"  Total Return:        {results['total_return']*100:>8.2f}%")
    print(f"  Win Rate:            {results['win_rate']*100:>8.2f}%")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown:        {results['max_drawdown']*100:>8.2f}%")
    print(f"  Total Trades:        {results['num_trades']:>8,}")

    print(f"\n💼 Portfolio:")
    print(f"  Stocks Used:         {len(filtered_tickers):>8,}")
    print(f"  From Clusters:       {len(cluster_filter.best_cluster_ids):>8}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run backtest with cluster filtering')

    # Data paths
    parser.add_argument('--dataset-path', type=str, default='./data/all_complete_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model_100m_1.18.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prices-path', type=str, default='./data/actual_prices.h5',
                       help='Path to actual prices HDF5')

    # Cluster filtering
    parser.add_argument('--cluster-dir', type=str, default='./cluster_results',
                       help='Directory with cluster results')
    parser.add_argument('--best-clusters-file', type=str, default='./cluster_results/best_clusters_1d.txt',
                       help='File with best cluster IDs')

    # Backtest parameters
    parser.add_argument('--num-test-stocks', type=int, default=1000,
                       help='Number of stocks to consider')
    parser.add_argument('--subset-size', type=int, default=100,
                       help='Number of stocks to sample each day')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top stocks to invest in')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date for backtest')
    parser.add_argument('--end-date', type=str, default='2024-06-30',
                       help='End date for backtest')

    args = parser.parse_args()

    # Run backtest
    results = run_backtest_with_clusters(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        prices_path=args.prices_path,
        cluster_dir=args.cluster_dir,
        best_clusters_file=args.best_clusters_file,
        num_test_stocks=args.num_test_stocks,
        subset_size=args.subset_size,
        top_k=args.top_k,
        start_date=args.start_date,
        end_date=args.end_date
    )
