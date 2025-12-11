"""
Analyze Cluster Performance

Compute performance metrics for each cluster to identify profitable ones.

Metrics computed per cluster:
- Mean return (1-day, 5-day, 10-day, 20-day)
- Win rate (probability of profit)
- Sharpe ratio
- Max drawdown
- Return volatility

Output: Ranked list of clusters by profitability
"""

import numpy as np
import h5py
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ClusterAnalyzer:
    """
    Analyze performance of stock clusters.
    """

    def __init__(self, cluster_dir: str):
        """
        Initialize analyzer.

        Args:
            cluster_dir: Directory containing cluster results
        """
        self.cluster_dir = Path(cluster_dir)

        print(f"\n{'='*80}")
        print("INITIALIZING CLUSTER ANALYZER")
        print(f"{'='*80}")

        # Load cluster assignments
        print(f"\n📂 Loading cluster assignments...")
        with open(self.cluster_dir / 'cluster_assignments.pkl', 'rb') as f:
            self.cluster_assignments = pickle.load(f)

        print(f"   ✓ Loaded {len(self.cluster_assignments)} stock assignments")

        # Group tickers by cluster
        # Note: cluster_assignments keys may be "TICKER_t123" format (temporal sampling)
        # We extract the base ticker for grouping
        self.cluster_to_tickers = defaultdict(set)
        for key, cluster_id in self.cluster_assignments.items():
            # Extract base ticker (handles both "TICKER" and "TICKER_t123" formats)
            ticker = key.split('_t')[0] if '_t' in key else key
            self.cluster_to_tickers[cluster_id].add(ticker)

        # Convert sets to lists
        self.cluster_to_tickers = {k: list(v) for k, v in self.cluster_to_tickers.items()}

        print(f"   ✓ Found {len(self.cluster_to_tickers)} unique clusters")

        # Count unique tickers
        all_unique_tickers = set()
        for tickers in self.cluster_to_tickers.values():
            all_unique_tickers.update(tickers)
        print(f"   ✓ {len(all_unique_tickers)} unique tickers across all clusters")

    def compute_returns(self, dataset_path: str, prices_path: str, horizons: List[int] = [1, 5, 10, 20]) -> Dict:
        """
        Compute returns for each stock at various horizons.

        Args:
            dataset_path: Path to features dataset
            prices_path: Path to prices HDF5
            horizons: List of horizons to compute returns for (in days)

        Returns:
            Dictionary with return statistics per cluster
        """
        print(f"\n{'='*80}")
        print("COMPUTING RETURNS")
        print(f"{'='*80}")

        print(f"\n📂 Opening prices file: {prices_path}")
        prices_file = h5py.File(prices_path, 'r')

        # Get unique tickers (handle temporal sampling format)
        unique_tickers = set()
        for key in self.cluster_assignments.keys():
            ticker = key.split('_t')[0] if '_t' in key else key
            unique_tickers.add(ticker)

        print(f"   ✓ {len(unique_tickers)} unique tickers to analyze")
        print(f"   ✓ {len(list(prices_file.keys()))} tickers in prices file")

        # Compute returns for each stock
        stock_returns = defaultdict(lambda: {h: [] for h in horizons})

        print(f"\n📊 Computing returns for each stock...")
        tickers_with_prices = 0
        for ticker in tqdm(unique_tickers):
            if ticker not in prices_file:
                continue

            tickers_with_prices += 1
            # Try 'prices' first, then 'close' for compatibility
            if 'prices' in prices_file[ticker]:
                prices = prices_file[ticker]['prices'][:]
            elif 'close' in prices_file[ticker]:
                prices = prices_file[ticker]['close'][:]
            else:
                continue

            # Compute forward returns at each horizon
            for horizon in horizons:
                for i in range(len(prices) - horizon):
                    ret = (prices[i + horizon] / prices[i]) - 1.0
                    if not np.isnan(ret) and not np.isinf(ret):
                        stock_returns[ticker][horizon].append(ret)

        prices_file.close()

        print(f"\n   ✓ Found prices for {tickers_with_prices}/{len(unique_tickers)} tickers ({tickers_with_prices/len(unique_tickers)*100:.1f}%)")

        if tickers_with_prices == 0:
            print(f"\n   ⚠️  WARNING: No tickers matched between cluster assignments and prices file!")
            print(f"   This likely means the ticker names don't match between the two files.")
            return {}

        # Aggregate by cluster
        print(f"\n📈 Aggregating returns by cluster...")
        cluster_stats = {}

        for cluster_id, tickers in self.cluster_to_tickers.items():
            cluster_stats[cluster_id] = {}

            for horizon in horizons:
                # Collect all returns for this cluster and horizon
                all_returns = []
                for ticker in tickers:
                    if ticker in stock_returns:
                        all_returns.extend(stock_returns[ticker][horizon])

                if len(all_returns) == 0:
                    continue

                all_returns = np.array(all_returns)

                # Compute statistics
                cluster_stats[cluster_id][horizon] = {
                    'mean_return': np.mean(all_returns),
                    'median_return': np.median(all_returns),
                    'std_return': np.std(all_returns),
                    'win_rate': (all_returns > 0).mean(),
                    'sharpe': np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0,
                    'skewness': self._compute_skewness(all_returns),
                    'max_return': np.max(all_returns),
                    'min_return': np.min(all_returns),
                    'num_samples': len(all_returns)
                }

        print(f"   ✓ Computed stats for {len(cluster_stats)} clusters")

        return cluster_stats

    def _compute_skewness(self, returns: np.ndarray) -> float:
        """Compute skewness of returns."""
        if len(returns) < 3:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        skew = np.mean(((returns - mean) / std) ** 3)
        return skew

    def rank_clusters(self, cluster_stats: Dict, horizon: int = 5, metric: str = 'mean_return') -> pd.DataFrame:
        """
        Rank clusters by performance metric.

        Args:
            cluster_stats: Cluster statistics
            horizon: Horizon to rank by
            metric: Metric to rank by

        Returns:
            DataFrame with ranked clusters
        """
        print(f"\n{'='*80}")
        print(f"RANKING CLUSTERS (Horizon: {horizon} days, Metric: {metric})")
        print(f"{'='*80}")

        rows = []
        for cluster_id, stats in cluster_stats.items():
            if horizon not in stats:
                continue

            horizon_stats = stats[horizon]

            rows.append({
                'cluster_id': cluster_id,
                'num_stocks': len(self.cluster_to_tickers[cluster_id]),
                'mean_return': horizon_stats['mean_return'],
                'median_return': horizon_stats['median_return'],
                'win_rate': horizon_stats['win_rate'],
                'sharpe': horizon_stats['sharpe'],
                'volatility': horizon_stats['std_return'],
                'skewness': horizon_stats['skewness'],
                'num_samples': horizon_stats['num_samples']
            })

        df = pd.DataFrame(rows)

        if df.empty:
            print(f"\n   ⚠️  No data available for {horizon}-day horizon")
            return df

        df = df.sort_values(metric, ascending=False).reset_index(drop=True)

        return df

    def identify_best_clusters(self, cluster_stats: Dict, horizon: int = 5,
                              min_return: float = 0.0,
                              min_win_rate: float = 0.5,
                              min_sharpe: float = 0.0,
                              top_k: Optional[int] = None) -> List[int]:
        """
        Identify best clusters based on criteria.

        Args:
            cluster_stats: Cluster statistics
            horizon: Horizon to evaluate
            min_return: Minimum mean return
            min_win_rate: Minimum win rate
            min_sharpe: Minimum Sharpe ratio
            top_k: If set, return top K clusters by mean return

        Returns:
            List of cluster IDs meeting criteria
        """
        print(f"\n🎯 Identifying best clusters...")
        print(f"   Criteria:")
        print(f"     - Horizon: {horizon} days")
        print(f"     - Min return: {min_return*100:.2f}%")
        print(f"     - Min win rate: {min_win_rate*100:.1f}%")
        print(f"     - Min Sharpe: {min_sharpe:.2f}")

        good_clusters = []

        for cluster_id, stats in cluster_stats.items():
            if horizon not in stats:
                continue

            horizon_stats = stats[horizon]

            if (horizon_stats['mean_return'] >= min_return and
                horizon_stats['win_rate'] >= min_win_rate and
                horizon_stats['sharpe'] >= min_sharpe):
                good_clusters.append((cluster_id, horizon_stats['mean_return']))

        # Sort by return
        good_clusters.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            good_clusters = good_clusters[:top_k]

        cluster_ids = [c[0] for c in good_clusters]

        print(f"\n   ✓ Found {len(cluster_ids)} clusters meeting criteria")

        if len(cluster_ids) > 0:
            total_stocks = sum(len(self.cluster_to_tickers[cid]) for cid in cluster_ids)
            print(f"   ✓ Total stocks in good clusters: {total_stocks}")

        return cluster_ids

    def visualize_cluster_performance(self, cluster_stats: Dict, output_dir: str, horizon: int = 5):
        """
        Create visualizations of cluster performance.

        Args:
            cluster_stats: Cluster statistics
            output_dir: Output directory
            horizon: Horizon to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📊 Creating visualizations...")

        # Prepare data
        cluster_ids = []
        mean_returns = []
        win_rates = []
        sharpes = []
        num_stocks = []

        for cluster_id, stats in cluster_stats.items():
            if horizon not in stats:
                continue

            cluster_ids.append(cluster_id)
            mean_returns.append(stats[horizon]['mean_return'] * 100)  # Convert to %
            win_rates.append(stats[horizon]['win_rate'] * 100)
            sharpes.append(stats[horizon]['sharpe'])
            num_stocks.append(len(self.cluster_to_tickers[cluster_id]))

        # 1. Return vs Win Rate scatter
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Return vs Win Rate
        ax = axes[0, 0]
        scatter = ax.scatter(win_rates, mean_returns, s=num_stocks, alpha=0.6, c=sharpes, cmap='RdYlGn')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(50, color='black', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel(f'Mean Return (%) - {horizon}d')
        ax.set_title('Cluster Performance: Return vs Win Rate\n(size = # stocks, color = Sharpe)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')

        # Plot 2: Return distribution
        ax = axes[0, 1]
        ax.hist(mean_returns, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(mean_returns), color='green', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel(f'Mean Return (%) - {horizon}d')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Distribution of Cluster Returns')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Win rate distribution
        ax = axes[1, 0]
        ax.hist(win_rates, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50%')
        ax.axvline(np.mean(win_rates), color='green', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Distribution of Cluster Win Rates')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Sharpe distribution
        ax = axes[1, 1]
        ax.hist(sharpes, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(sharpes), color='green', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Distribution of Cluster Sharpe Ratios')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'cluster_performance_{horizon}d.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()


def analyze_cluster_performance(cluster_dir: str, dataset_path: str, prices_path: str,
                                horizons: List[int] = [1, 5, 10, 20],
                                output_dir: Optional[str] = None) -> Dict:
    """
    Main function to analyze cluster performance.

    Args:
        cluster_dir: Directory with cluster results
        dataset_path: Path to dataset
        prices_path: Path to prices
        horizons: Horizons to analyze
        output_dir: Output directory for results

    Returns:
        Analysis results dictionary
    """
    if output_dir is None:
        output_dir = cluster_dir

    # Initialize analyzer
    analyzer = ClusterAnalyzer(cluster_dir)

    # Compute returns
    cluster_stats = analyzer.compute_returns(dataset_path, prices_path, horizons)

    # Rank clusters for each horizon
    results = {
        'cluster_stats': cluster_stats,
        'rankings': {},
        'best_clusters': {}
    }

    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"ANALYZING HORIZON: {horizon} days")
        print(f"{'='*80}")

        # Rank clusters
        ranking_df = analyzer.rank_clusters(cluster_stats, horizon)
        results['rankings'][horizon] = ranking_df

        if ranking_df.empty:
            print(f"\n   ⚠️  Skipping {horizon}-day horizon (no data)")
            continue

        # Print top 10
        print(f"\n🏆 Top 10 Clusters ({horizon}-day returns):")
        print(ranking_df.head(10).to_string(index=False))

        # Identify best clusters
        best_clusters = analyzer.identify_best_clusters(
            cluster_stats,
            horizon=horizon,
            min_return=0.005,  # 0.5% min return
            min_win_rate=0.52,  # 52% win rate
            min_sharpe=0.1,
            top_k=20  # Top 20 clusters
        )
        results['best_clusters'][horizon] = best_clusters

        print(f"\n✨ Best {len(best_clusters)} clusters for {horizon}-day horizon:")
        print(f"   Cluster IDs: {best_clusters}")

        # Save ranking
        ranking_path = Path(output_dir) / f'cluster_ranking_{horizon}d.csv'
        ranking_df.to_csv(ranking_path, index=False)
        print(f"   💾 Saved ranking: {ranking_path}")

        # Visualize
        analyzer.visualize_cluster_performance(cluster_stats, output_dir, horizon)

    return results


def save_analysis_results(results: Dict, output_dir: str):
    """Save analysis results."""
    output_dir = Path(output_dir)

    print(f"\n💾 Saving analysis results...")

    # Save full results
    with open(output_dir / 'cluster_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Save best clusters for each horizon
    for horizon, cluster_ids in results['best_clusters'].items():
        with open(output_dir / f'best_clusters_{horizon}d.txt', 'w') as f:
            f.write(f"# Best clusters for {horizon}-day horizon\n")
            f.write(f"# Total: {len(cluster_ids)} clusters\n")
            for cid in cluster_ids:
                f.write(f"{cid}\n")

    print(f"   ✓ Saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze cluster performance')

    # Required
    parser.add_argument('--cluster-dir', type=str, default="./cluster_results", help='Directory with cluster results')
    parser.add_argument('--dataset-path', type=str, default="./data/all_complete_dataset.h5", help='Path to dataset')
    parser.add_argument('--prices-path', type=str, default="./data/actual_prices.h5", help='Path to prices HDF5')

    # Analysis
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20], help='Horizons to analyze')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: cluster_dir)')

    # Filtering
    parser.add_argument('--min-return', type=float, default=0.005, help='Min mean return for best clusters')
    parser.add_argument('--min-win-rate', type=float, default=0.52, help='Min win rate for best clusters')
    parser.add_argument('--min-sharpe', type=float, default=0.1, help='Min Sharpe for best clusters')
    parser.add_argument('--top-k', type=int, default=5, help='Top K clusters to select')

    args = parser.parse_args()

    # Run analysis
    results = analyze_cluster_performance(
        args.cluster_dir,
        args.dataset_path,
        args.prices_path,
        args.horizons,
        args.output_dir
    )

    # Save results
    save_analysis_results(results, args.output_dir or args.cluster_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext step: Use best clusters for filtering")
    print(f"  python -m cluster.cluster_filter \\")
    print(f"    --cluster-dir {args.cluster_dir} \\")
    print(f"    --best-clusters-file {args.output_dir or args.cluster_dir}/best_clusters_5d.txt")


if __name__ == '__main__':
    main()
