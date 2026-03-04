#!/usr/bin/env python3
"""
Feature IR Analysis

Calculates the Information Ratio (IR) of each feature directly against future returns.
This shows which features have consistent predictive power on their own.

IR = mean(IC) / std(IC)

Where IC is the cross-sectional correlation between feature value and future return
computed daily.

Usage:
    python sanity_checks/feature_ir_analysis.py \
        --dataset all_complete_dataset.h5 \
        --prices actual_prices_clean.h5 \
        --output results/feature_ir_analysis
"""

import argparse
import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import json
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_feature_ics(
    h5f,
    prices_h5f,
    dates: List[str],
    horizon_days: int = 1,
    max_dates: int = 500,
    min_stocks: int = 50,
) -> Dict[int, List[float]]:
    """
    Calculate daily IC for each feature against future returns.

    Returns dict mapping feature_idx -> list of daily ICs
    """
    tickers = list(h5f.keys())
    num_features = h5f[tickers[0]]['features'].shape[1]

    # Build price data with sorted date lists for efficient lookup
    print("  Building price lookup...")
    price_data = {}
    price_date_lists = {}
    for ticker in tickers:
        if ticker in prices_h5f:
            try:
                price_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                              for d in prices_h5f[ticker]['dates'][:]]
                # Handle both 'close' and 'prices' keys
                if 'close' in prices_h5f[ticker]:
                    closes = prices_h5f[ticker]['close'][:]
                elif 'prices' in prices_h5f[ticker]:
                    closes = prices_h5f[ticker]['prices'][:]
                else:
                    continue
                price_data[ticker] = dict(zip(price_dates, closes))
                price_date_lists[ticker] = price_dates  # Keep ordered list
            except Exception as e:
                continue

    print(f"  Loaded prices for {len(price_data)} tickers")

    # Build feature data lookup
    print("  Building feature lookup...")
    feature_data = {}
    feature_date_maps = {}
    for ticker in tickers:
        if ticker in h5f:
            try:
                ticker_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                               for d in h5f[ticker]['dates'][:]]
                feature_data[ticker] = h5f[ticker]['features'][:]
                feature_date_maps[ticker] = {d: i for i, d in enumerate(ticker_dates)}
            except:
                continue

    # Find common dates across price data (use first ticker's price dates as reference)
    reference_ticker = list(price_date_lists.keys())[0]
    available_dates = price_date_lists[reference_ticker]

    # Sample dates if too many (leave room for horizon)
    if len(available_dates) > max_dates + horizon_days:
        indices = np.linspace(0, len(available_dates) - 1 - horizon_days, max_dates, dtype=int)
        sampled_dates = [available_dates[i] for i in indices]
    else:
        sampled_dates = available_dates[:-horizon_days] if len(available_dates) > horizon_days else []

    print(f"  Using {len(sampled_dates)} dates for IC calculation")

    # Feature ICs storage
    feature_ics = defaultdict(list)

    print(f"  Calculating ICs for {num_features} features across {len(sampled_dates)} dates...")

    for date_idx, date in enumerate(sampled_dates):
        if date_idx % 50 == 0:
            print(f"    Processing date {date_idx}/{len(sampled_dates)}...")

        # Collect feature values and returns for this date
        feature_values = [[] for _ in range(num_features)]  # Use list of lists for speed
        returns = []

        for ticker in tickers:
            if ticker not in feature_data or ticker not in price_data:
                continue

            try:
                # Check if date exists in feature data
                date_map = feature_date_maps.get(ticker)
                if date_map is None or date not in date_map:
                    continue

                feat_idx_in_data = date_map[date]
                features = feature_data[ticker][feat_idx_in_data]

                # Get prices
                prices = price_data[ticker]
                price_dates = price_date_lists[ticker]

                if date not in prices:
                    continue

                # Find future date (horizon_days ahead in price data)
                try:
                    date_pos = price_dates.index(date)
                    future_pos = date_pos + horizon_days
                    if future_pos >= len(price_dates):
                        continue
                    future_date = price_dates[future_pos]
                except ValueError:
                    continue

                current_price = prices[date]
                future_price = prices.get(future_date)

                if future_price is None or current_price <= 0:
                    continue

                ret = (future_price - current_price) / current_price

                # Store feature values (only if return is valid)
                valid_features = True
                for f_idx in range(num_features):
                    feat_val = features[f_idx]
                    if not np.isfinite(feat_val):
                        valid_features = False
                        break

                if valid_features:
                    for f_idx in range(num_features):
                        feature_values[f_idx].append(features[f_idx])
                    returns.append(ret)

            except Exception as e:
                continue

        # Calculate IC for each feature on this date
        if len(returns) >= min_stocks:
            returns_arr = np.array(returns)
            returns_std = np.std(returns_arr)

            if returns_std < 1e-10:
                continue

            for feat_idx in range(num_features):
                if len(feature_values[feat_idx]) == len(returns):
                    feat_vals = np.array(feature_values[feat_idx])
                    feat_std = np.std(feat_vals)

                    # Skip if no variation
                    if feat_std < 1e-10:
                        continue

                    try:
                        ic, _ = pearsonr(feat_vals, returns_arr)
                        if np.isfinite(ic):
                            feature_ics[feat_idx].append(ic)
                    except:
                        continue

    return feature_ics


def calculate_feature_stats(feature_ics: Dict[int, List[float]]) -> List[Dict]:
    """
    Calculate statistics for each feature.

    Returns list of dicts with feature stats, sorted by |IR|.
    """
    stats = []

    for feat_idx, ics in feature_ics.items():
        if len(ics) < 10:
            continue

        ics = np.array(ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        ir = mean_ic / std_ic if std_ic > 1e-10 else 0

        # T-statistic for IC != 0
        t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 1e-10 else 0

        stats.append({
            'feature_idx': feat_idx,
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            'ir': float(ir),
            't_stat': float(t_stat),
            'pct_positive': float(np.mean(ics > 0) * 100),
            'num_observations': len(ics),
            'abs_ir': float(abs(ir)),
        })

    # Sort by |IR|
    stats.sort(key=lambda x: x['abs_ir'], reverse=True)

    return stats


def print_feature_report(stats: List[Dict], top_n: int = 50):
    """Print feature IR report."""

    print("\n" + "=" * 100)
    print("FEATURE IR ANALYSIS - TOP FEATURES BY |IR|")
    print("=" * 100)

    if len(stats) == 0:
        print("\nNo features with sufficient data for analysis.")
        return

    print(f"{'Rank':<6} {'Feature':<10} {'Mean IC':>12} {'Std IC':>10} {'IR':>10} {'T-stat':>10} {'IC>0 %':>8} {'N':>6}")
    print("-" * 100)

    for rank, s in enumerate(stats[:top_n], 1):
        sig = "***" if abs(s['t_stat']) > 3 else "**" if abs(s['t_stat']) > 2 else "*" if abs(s['t_stat']) > 1.65 else ""
        print(f"{rank:<6} {s['feature_idx']:<10} {s['mean_ic']:>+12.5f} {s['std_ic']:>10.5f} "
              f"{s['ir']:>+10.3f} {s['t_stat']:>+10.2f} {s['pct_positive']:>7.1f}% {s['num_observations']:>6} {sig}")

    print("-" * 100)

    # Summary stats
    all_irs = [s['ir'] for s in stats]
    positive_ir = [s for s in stats if s['ir'] > 0]
    significant = [s for s in stats if abs(s['t_stat']) > 1.96]

    print(f"\nSummary:")
    print(f"  Total features analyzed: {len(stats)}")
    print(f"  Features with positive IR: {len(positive_ir)} ({len(positive_ir)/len(stats)*100:.1f}%)")
    print(f"  Features with |t| > 1.96: {len(significant)} ({len(significant)/len(stats)*100:.1f}%)")
    print(f"  Max IR: {max(all_irs):+.4f}")
    print(f"  Min IR: {min(all_irs):+.4f}")
    print(f"  Mean |IR|: {np.mean(np.abs(all_irs)):.4f}")


def categorize_features(stats: List[Dict], num_base_features: int = 355) -> Dict:
    """
    Categorize features and compute average IR by category.

    Based on dataset structure:
    - Features 0-354: Base features (fundamentals, technicals, cross-sectional)
    - Features 355-1122: News embeddings (768 dims)
    """
    categories = {
        'base_features': [],
        'news_embeddings': [],
    }

    for s in stats:
        if s['feature_idx'] < num_base_features:
            categories['base_features'].append(s)
        else:
            categories['news_embeddings'].append(s)

    summary = {}
    for cat, features in categories.items():
        if features:
            irs = [f['ir'] for f in features]
            summary[cat] = {
                'count': len(features),
                'mean_ir': float(np.mean(irs)),
                'std_ir': float(np.std(irs)),
                'max_ir': float(max(irs)),
                'min_ir': float(min(irs)),
                'pct_positive_ir': float(np.mean(np.array(irs) > 0) * 100),
                'top_5_features': [f['feature_idx'] for f in sorted(features, key=lambda x: x['abs_ir'], reverse=True)[:5]],
            }

    return summary


def plot_feature_ir_distribution(stats: List[Dict], output_dir: str):
    """Plot feature IR distribution."""
    import matplotlib.pyplot as plt

    irs = [s['ir'] for s in stats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # IR distribution
    ax = axes[0, 0]
    ax.hist(irs, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(irs), color='blue', linestyle='--', linewidth=2, label=f'Mean={np.mean(irs):.3f}')
    ax.set_xlabel('Information Ratio (IR)')
    ax.set_ylabel('Frequency')
    ax.set_title('Feature IR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IR by feature index
    ax = axes[0, 1]
    feature_indices = [s['feature_idx'] for s in stats]
    ax.scatter(feature_indices, irs, alpha=0.5, s=10)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.axvline(355, color='green', linestyle='--', linewidth=2, label='News embeddings start')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Information Ratio (IR)')
    ax.set_title('IR by Feature Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top features bar chart
    ax = axes[1, 0]
    top_n = 20
    top_stats = sorted(stats, key=lambda x: x['abs_ir'], reverse=True)[:top_n]
    colors = ['green' if s['ir'] > 0 else 'red' for s in top_stats]
    ax.barh(range(top_n), [s['ir'] for s in top_stats], color=colors, alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"F{s['feature_idx']}" for s in top_stats])
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Information Ratio (IR)')
    ax.set_title(f'Top {top_n} Features by |IR|')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Mean IC vs Std IC scatter
    ax = axes[1, 1]
    mean_ics = [s['mean_ic'] for s in stats]
    std_ics = [s['std_ic'] for s in stats]
    colors = ['green' if s['ir'] > 0.05 else 'red' if s['ir'] < -0.05 else 'gray' for s in stats]
    ax.scatter(std_ics, mean_ics, c=colors, alpha=0.5, s=20)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('IC Std Dev')
    ax.set_ylabel('Mean IC')
    ax.set_title('Mean IC vs IC Volatility')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_ir_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feature_ir_distribution.png")


def main():
    parser = argparse.ArgumentParser(description='Feature IR Analysis')
    parser.add_argument('--dataset', type=str, default='all_complete_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--prices', type=str, default='actual_prices_clean.h5',
                       help='Path to prices HDF5 file')
    parser.add_argument('--output', type=str, default='results/feature_ir_analysis',
                       help='Output directory')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon in days')
    parser.add_argument('--max-dates', type=int, default=500,
                       help='Max dates to sample for analysis')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("\n" + "=" * 80)
    print("FEATURE IR ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Prices: {args.prices}")
    print(f"Horizon: {args.horizon} day(s)")
    print(f"Max dates: {args.max_dates}")

    # Load data
    print("\nLoading data...")
    with h5py.File(args.dataset, 'r') as h5f:
        with h5py.File(args.prices, 'r') as prices_h5f:
            # Get all dates
            sample_ticker = list(h5f.keys())[0]
            all_dates = sorted([d.decode('utf-8') if isinstance(d, bytes) else d
                               for d in h5f[sample_ticker]['dates'][:]])

            num_features = h5f[sample_ticker]['features'].shape[1]
            print(f"Total features: {num_features}")
            print(f"Total dates: {len(all_dates)}")

            # Calculate feature ICs
            print("\nCalculating feature ICs...")
            feature_ics = calculate_feature_ics(
                h5f, prices_h5f, all_dates,
                horizon_days=args.horizon,
                max_dates=args.max_dates,
            )

    # Calculate statistics
    print("\nCalculating feature statistics...")
    stats = calculate_feature_stats(feature_ics)

    # Print report
    print_feature_report(stats, top_n=50)

    # Categorize features
    print("\n" + "=" * 80)
    print("FEATURE CATEGORY ANALYSIS")
    print("=" * 80)

    category_summary = categorize_features(stats)
    for cat, summary in category_summary.items():
        print(f"\n{cat.upper().replace('_', ' ')}:")
        print(f"  Count: {summary['count']}")
        print(f"  Mean IR: {summary['mean_ir']:+.4f}")
        print(f"  Max IR: {summary['max_ir']:+.4f}")
        print(f"  Min IR: {summary['min_ir']:+.4f}")
        print(f"  % Positive IR: {summary['pct_positive_ir']:.1f}%")
        print(f"  Top 5 features: {summary['top_5_features']}")

    # Plot distributions
    print("\nGenerating plots...")
    plot_feature_ir_distribution(stats, args.output)

    # Save results
    results = {
        'horizon_days': args.horizon,
        'num_features_analyzed': len(stats),
        'feature_stats': stats,
        'category_summary': category_summary,
    }

    results_path = os.path.join(args.output, 'feature_ir_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
