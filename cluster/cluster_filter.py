"""
Cluster-based Stock Filtering

Use clusters to filter stocks during inference/trading.

Only consider stocks from profitable clusters identified in analysis.
This reduces search space and improves trading performance.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Set, Optional, Dict
import torch


class ClusterFilter:
    """
    Filter stocks based on cluster membership.

    Usage:
        # Initialize
        filter = ClusterFilter(cluster_dir, best_clusters_file)

        # Filter stocks
        allowed_tickers = filter.filter_tickers(all_tickers)

        # Or check individual stock
        if filter.is_allowed(ticker):
            trade(ticker)
    """

    def __init__(self, cluster_dir: str, best_clusters_file: Optional[str] = None,
                 best_cluster_ids: Optional[List[int]] = None):
        """
        Initialize cluster filter.

        Args:
            cluster_dir: Directory with cluster results
            best_clusters_file: Path to file with best cluster IDs (one per line)
            best_cluster_ids: List of best cluster IDs (alternative to file)
        """
        self.cluster_dir = Path(cluster_dir)

        print(f"\n{'='*80}")
        print("INITIALIZING CLUSTER FILTER")
        print(f"{'='*80}")

        # Load cluster assignments
        print(f"\n📂 Loading cluster assignments...")
        with open(self.cluster_dir / 'cluster_assignments.pkl', 'rb') as f:
            self.cluster_assignments = pickle.load(f)

        print(f"   ✓ Loaded {len(self.cluster_assignments)} stock assignments")

        # Load best clusters
        if best_clusters_file:
            self.best_cluster_ids = self._load_best_clusters_from_file(best_clusters_file)
        elif best_cluster_ids:
            self.best_cluster_ids = set(best_cluster_ids)
        else:
            raise ValueError("Must provide either best_clusters_file or best_cluster_ids")

        print(f"   ✓ Loaded {len(self.best_cluster_ids)} best clusters")

        # Create set of allowed tickers
        self.allowed_tickers = self._create_allowed_set()

        print(f"   ✓ {len(self.allowed_tickers)} stocks pass filter ({len(self.allowed_tickers)/len(self.cluster_assignments)*100:.1f}%)")

    def _load_best_clusters_from_file(self, filepath: str) -> Set[int]:
        """Load best cluster IDs from file."""
        cluster_ids = set()

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    cluster_ids.add(int(line))

        return cluster_ids

    def _create_allowed_set(self) -> Set[str]:
        """Create set of allowed tickers based on cluster membership."""
        allowed = set()

        for ticker, cluster_id in self.cluster_assignments.items():
            if cluster_id in self.best_cluster_ids:
                allowed.add(ticker)

        return allowed

    def filter_tickers(self, tickers: List[str]) -> List[str]:
        """
        Filter list of tickers to only those in good clusters.

        Args:
            tickers: List of tickers to filter

        Returns:
            Filtered list of tickers
        """
        return [t for t in tickers if t in self.allowed_tickers]

    def is_allowed(self, ticker: str) -> bool:
        """
        Check if ticker is in a good cluster.

        Args:
            ticker: Stock ticker

        Returns:
            True if ticker is in a good cluster
        """
        return ticker in self.allowed_tickers

    def get_cluster_id(self, ticker: str) -> Optional[int]:
        """
        Get cluster ID for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Cluster ID or None if not found
        """
        return self.cluster_assignments.get(ticker)

    def get_stats(self) -> Dict:
        """Get filter statistics."""
        total_stocks = len(self.cluster_assignments)
        allowed_stocks = len(self.allowed_tickers)

        return {
            'total_stocks': total_stocks,
            'allowed_stocks': allowed_stocks,
            'filter_rate': allowed_stocks / total_stocks if total_stocks > 0 else 0,
            'num_clusters_total': len(set(self.cluster_assignments.values())),
            'num_clusters_allowed': len(self.best_cluster_ids)
        }

    def print_stats(self):
        """Print filter statistics."""
        stats = self.get_stats()

        print(f"\n{'='*80}")
        print("CLUSTER FILTER STATISTICS")
        print(f"{'='*80}")
        print(f"\n  Total stocks:        {stats['total_stocks']:>6,}")
        print(f"  Allowed stocks:      {stats['allowed_stocks']:>6,} ({stats['filter_rate']*100:.1f}%)")
        print(f"  Total clusters:      {stats['num_clusters_total']:>6}")
        print(f"  Allowed clusters:    {stats['num_clusters_allowed']:>6}")
        print(f"\n  → Filtering out {stats['total_stocks'] - stats['allowed_stocks']:,} stocks ({(1-stats['filter_rate'])*100:.1f}%)")


def apply_cluster_filter_to_backtest(cluster_filter: ClusterFilter,
                                     data_loader,
                                     inplace: bool = True):
    """
    Apply cluster filter to a data loader.

    Args:
        cluster_filter: ClusterFilter instance
        data_loader: DatasetLoader instance (from backtest_simulation.py)
        inplace: If True, modify data_loader in place. If False, return filtered copy.

    Returns:
        Filtered data_loader (or None if inplace=True)
    """
    print(f"\n📊 Applying cluster filter to data loader...")

    original_count = len(data_loader.test_tickers)
    filtered_tickers = cluster_filter.filter_tickers(data_loader.test_tickers)

    if inplace:
        data_loader.test_tickers = filtered_tickers
        print(f"   ✓ Filtered: {original_count} → {len(filtered_tickers)} stocks ({len(filtered_tickers)/original_count*100:.1f}%)")
        return None
    else:
        # Create copy
        import copy
        filtered_loader = copy.deepcopy(data_loader)
        filtered_loader.test_tickers = filtered_tickers
        print(f"   ✓ Filtered (copy): {original_count} → {len(filtered_tickers)} stocks")
        return filtered_loader


def apply_cluster_filter_to_rl_env(cluster_filter: ClusterFilter,
                                   environment,
                                   inplace: bool = True):
    """
    Apply cluster filter to RL trading environment.

    Args:
        cluster_filter: ClusterFilter instance
        environment: TradingEnvironment instance
        inplace: If True, modify environment in place

    Returns:
        Filtered environment (or None if inplace=True)
    """
    print(f"\n📊 Applying cluster filter to RL environment...")

    original_count = len(environment.data_loader.test_tickers)
    filtered_tickers = cluster_filter.filter_tickers(environment.data_loader.test_tickers)

    if inplace:
        environment.data_loader.test_tickers = filtered_tickers
        print(f"   ✓ Filtered: {original_count} → {len(filtered_tickers)} stocks ({len(filtered_tickers)/original_count*100:.1f}%)")
        return None
    else:
        import copy
        filtered_env = copy.deepcopy(environment)
        filtered_env.data_loader.test_tickers = filtered_tickers
        print(f"   ✓ Filtered (copy): {original_count} → {len(filtered_tickers)} stocks")
        return filtered_env


def create_filtered_ticker_list(cluster_filter: ClusterFilter,
                                input_tickers_file: str,
                                output_tickers_file: str):
    """
    Create filtered ticker list file.

    Args:
        cluster_filter: ClusterFilter instance
        input_tickers_file: Path to input ticker list (pickle or text)
        output_tickers_file: Path to output filtered ticker list
    """
    print(f"\n📝 Creating filtered ticker list...")

    # Load input tickers
    if input_tickers_file.endswith('.pkl'):
        import pickle
        with open(input_tickers_file, 'rb') as f:
            input_tickers = pickle.load(f)
    else:
        with open(input_tickers_file, 'r') as f:
            input_tickers = [line.strip() for line in f if line.strip()]

    # Filter
    filtered_tickers = cluster_filter.filter_tickers(input_tickers)

    # Save
    if output_tickers_file.endswith('.pkl'):
        import pickle
        with open(output_tickers_file, 'wb') as f:
            pickle.dump(filtered_tickers, f)
    else:
        with open(output_tickers_file, 'w') as f:
            for ticker in filtered_tickers:
                f.write(f"{ticker}\n")

    print(f"   ✓ Saved {len(filtered_tickers)} filtered tickers to: {output_tickers_file}")
    print(f"   ✓ Filter rate: {len(filtered_tickers)/len(input_tickers)*100:.1f}%")


def main():
    """Example usage of cluster filter."""
    import argparse

    parser = argparse.ArgumentParser(description='Filter stocks using clusters')

    # Required
    parser.add_argument('--cluster-dir', type=str, required=True, help='Directory with cluster results')
    parser.add_argument('--best-clusters-file', type=str, required=True, help='File with best cluster IDs')

    # Optional: Filter ticker list
    parser.add_argument('--input-tickers', type=str, help='Input ticker list to filter')
    parser.add_argument('--output-tickers', type=str, help='Output filtered ticker list')

    args = parser.parse_args()

    # Initialize filter
    cluster_filter = ClusterFilter(args.cluster_dir, args.best_clusters_file)

    # Print stats
    cluster_filter.print_stats()

    # Filter ticker list if provided
    if args.input_tickers and args.output_tickers:
        create_filtered_ticker_list(
            cluster_filter,
            args.input_tickers,
            args.output_tickers
        )

    print(f"\n{'='*80}")
    print("FILTER READY FOR USE")
    print(f"{'='*80}")
    print(f"\nUsage in code:")
    print(f"")
    print(f"  from cluster import ClusterFilter")
    print(f"")
    print(f"  # Initialize")
    print(f"  filter = ClusterFilter('{args.cluster_dir}', '{args.best_clusters_file}')")
    print(f"")
    print(f"  # Filter tickers")
    print(f"  allowed = filter.filter_tickers(all_tickers)")
    print(f"")
    print(f"  # Or check individual")
    print(f"  if filter.is_allowed(ticker):")
    print(f"      trade(ticker)")


if __name__ == '__main__':
    main()
