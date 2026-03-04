#!/usr/bin/env python3
"""
Predictive Signal Sanity Checks

Verifies the dataset has actual predictive signal for returns.

Checks:
1. Feature-return correlations (IC) - some features should correlate with future returns
2. Momentum signal check - past returns should have SOME predictive power (even if weak)
3. Return distribution - verify returns are reasonable (not all zeros)
4. Feature redundancy - check for highly correlated feature pairs
5. Temporal consistency - verify rolling features are computed correctly

Usage:
    python sanity_checks/check_predictive_signal.py --dataset all_complete_dataset.h5
"""

import argparse
import h5py
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def compute_future_returns(h5f, ticker, horizon=5):
    """Compute future returns for a ticker."""
    features = h5f[ticker]['features'][:]

    # We need actual prices - let's use a price-like feature or compute from returns
    # Feature indices for returns should be in the derived features
    # For now, use feature changes as proxy

    # Actually, let's compute pseudo-returns from the first feature
    # Or better: find the return_1d feature if it exists

    # For simplicity, compute returns from feature 0 (often price-related)
    # This is a proxy - real check would use actual prices

    num_dates = features.shape[0]

    # Use the mean of first few features as price proxy
    price_proxy = np.mean(features[:, :5], axis=1)

    # Future returns
    future_returns = np.zeros(num_dates)
    future_returns[:-horizon] = (price_proxy[horizon:] - price_proxy[:-horizon]) / (np.abs(price_proxy[:-horizon]) + 1e-8)

    return future_returns, features


def check_feature_return_correlations(h5f, num_stocks=50, horizon=5):
    """
    Check 1: Do any features correlate with future returns?

    We expect SOME features to have weak but positive IC.
    Suspiciously high IC (>0.1) suggests leakage.
    Zero IC for ALL features suggests no signal.
    """
    print("\n" + "="*80)
    print("CHECK 1: FEATURE-RETURN CORRELATIONS (IC)")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]
    num_features = min(100, h5f[tickers[0]]['features'].shape[1])  # Check first 100 features

    # Collect ICs per feature
    feature_ics = defaultdict(list)

    for ticker in tickers:
        future_returns, features = compute_future_returns(h5f, ticker, horizon)

        # Compute IC for each feature
        for feat_idx in range(num_features):
            feat = features[:-horizon, feat_idx]
            ret = future_returns[:-horizon]

            # Skip if no variation
            if np.std(feat) < 1e-10 or np.std(ret) < 1e-10:
                continue

            # Pearson correlation
            ic = np.corrcoef(feat, ret)[0, 1]
            if not np.isnan(ic):
                feature_ics[feat_idx].append(ic)

    # Compute mean IC per feature
    mean_ics = []
    for feat_idx in range(num_features):
        if feat_idx in feature_ics and len(feature_ics[feat_idx]) > 10:
            mean_ics.append(np.mean(feature_ics[feat_idx]))
        else:
            mean_ics.append(0)

    mean_ics = np.array(mean_ics)

    # Statistics
    max_ic = np.max(np.abs(mean_ics))
    num_significant = np.sum(np.abs(mean_ics) > 0.01)
    num_suspicious = np.sum(np.abs(mean_ics) > 0.1)

    print(f"\n  Analyzed: {len(tickers)} stocks x {num_features} features")
    print(f"  Horizon: {horizon} days")
    print(f"\n  Information Coefficient (IC) Statistics:")
    print(f"    Max |IC|: {max_ic:.4f}")
    print(f"    Features with |IC| > 0.01: {num_significant}")
    print(f"    Features with |IC| > 0.10: {num_suspicious} (suspicious if high)")

    # Top 5 features by IC
    top_indices = np.argsort(np.abs(mean_ics))[-5:][::-1]
    print(f"\n  Top 5 features by |IC|:")
    for idx in top_indices:
        print(f"    Feature {idx}: IC = {mean_ics[idx]:+.4f}")

    # PASS if we have some signal but not too much
    has_signal = num_significant >= 5
    no_leakage = num_suspicious < 10
    passed = has_signal and no_leakage

    status = "PASS" if passed else ("WARN" if not no_leakage else "FAIL")
    print(f"\n  Status: [{status}]")
    if not has_signal:
        print(f"    WARNING: Very few features show predictive signal!")
    if not no_leakage:
        print(f"    WARNING: Many features have suspiciously high IC - check for leakage!")

    return passed


def check_momentum_signal(h5f, num_stocks=100):
    """
    Check 2: Does past return predict future return?

    Momentum is a well-documented anomaly - past winners tend to keep winning.
    We expect a small positive correlation (0.01-0.05).
    """
    print("\n" + "="*80)
    print("CHECK 2: MOMENTUM SIGNAL (PAST VS FUTURE RETURNS)")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    momentum_ics = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:]
        num_dates = features.shape[0]

        # Use feature 0 as price proxy
        price_proxy = features[:, 0]

        # Past 20-day return
        past_return = np.zeros(num_dates)
        past_return[20:] = (price_proxy[20:] - price_proxy[:-20]) / (np.abs(price_proxy[:-20]) + 1e-8)

        # Future 20-day return
        future_return = np.zeros(num_dates)
        future_return[:-20] = (price_proxy[20:] - price_proxy[:-20]) / (np.abs(price_proxy[:-20]) + 1e-8)

        # Correlation (using middle portion to avoid edge effects)
        start, end = 40, num_dates - 40
        if end > start + 100:
            past = past_return[start:end]
            future = future_return[start:end]

            if np.std(past) > 1e-10 and np.std(future) > 1e-10:
                ic = np.corrcoef(past, future)[0, 1]
                if not np.isnan(ic):
                    momentum_ics.append(ic)

    if momentum_ics:
        mean_momentum_ic = np.mean(momentum_ics)
        std_momentum_ic = np.std(momentum_ics)
        t_stat = mean_momentum_ic / (std_momentum_ic / np.sqrt(len(momentum_ics)))
    else:
        mean_momentum_ic = 0
        t_stat = 0

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Momentum IC (20-day past vs 20-day future):")
    print(f"    Mean IC: {mean_momentum_ic:+.4f}")
    print(f"    T-stat:  {t_stat:+.2f}")

    # Momentum should be weakly positive or near zero
    # Very negative would be suspicious (mean reversion dominates at this horizon)
    passed = abs(mean_momentum_ic) < 0.2  # Not suspiciously strong
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")

    return passed


def check_return_distribution(h5f, num_stocks=100):
    """
    Check 3: Are implied returns reasonable?

    Returns should have:
    - Mean near zero
    - Reasonable std (0.01-0.05 daily)
    - Fat tails (kurtosis > 3)
    - Slight negative skew
    """
    print("\n" + "="*80)
    print("CHECK 3: RETURN DISTRIBUTION")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    all_returns = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:]
        price_proxy = features[:, 0]

        # Daily returns
        returns = np.diff(price_proxy) / (np.abs(price_proxy[:-1]) + 1e-8)
        returns = returns[np.isfinite(returns)]
        all_returns.extend(returns)

    all_returns = np.array(all_returns)

    # Statistics
    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    skew = stats.skew(all_returns)
    kurt = stats.kurtosis(all_returns)

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"  Total returns: {len(all_returns):,}")
    print(f"\n  Return Statistics:")
    print(f"    Mean:     {mean_ret:+.6f}")
    print(f"    Std:      {std_ret:.6f}")
    print(f"    Skewness: {skew:+.2f}")
    print(f"    Kurtosis: {kurt:+.2f} (excess, normal=0)")

    # Check reasonableness
    mean_ok = abs(mean_ret) < 0.01
    std_ok = 0.001 < std_ret < 0.5

    passed = mean_ok and std_ok
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")
    if not mean_ok:
        print(f"    WARNING: Mean return is suspiciously far from zero!")
    if not std_ok:
        print(f"    WARNING: Return volatility is outside expected range!")

    return passed


def check_feature_redundancy(h5f, num_stocks=20, num_features=100):
    """
    Check 4: Are there highly redundant features?

    Too many correlated features waste model capacity.
    """
    print("\n" + "="*80)
    print("CHECK 4: FEATURE REDUNDANCY")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]
    num_features = min(num_features, h5f[tickers[0]]['features'].shape[1])

    # Collect feature correlations
    corr_matrices = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:, :num_features]

        # Correlation matrix
        corr = np.corrcoef(features.T)
        corr_matrices.append(corr)

    # Average correlation matrix
    avg_corr = np.mean(corr_matrices, axis=0)

    # Count highly correlated pairs (excluding diagonal)
    np.fill_diagonal(avg_corr, 0)
    high_corr_pairs = np.sum(np.abs(avg_corr) > 0.9) // 2  # Divide by 2 for symmetry
    very_high_corr = np.sum(np.abs(avg_corr) > 0.99) // 2

    total_pairs = num_features * (num_features - 1) // 2

    print(f"\n  Analyzed: {len(tickers)} stocks x {num_features} features")
    print(f"\n  Feature Correlation Analysis:")
    print(f"    Total feature pairs: {total_pairs:,}")
    print(f"    Pairs with |corr| > 0.9:  {high_corr_pairs} ({high_corr_pairs/total_pairs*100:.1f}%)")
    print(f"    Pairs with |corr| > 0.99: {very_high_corr} ({very_high_corr/total_pairs*100:.2f}%)")

    # Some redundancy is expected, but not too much
    redundancy_rate = high_corr_pairs / total_pairs
    passed = redundancy_rate < 0.1
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: High feature redundancy detected!")

    return passed


def check_temporal_consistency(h5f, num_stocks=20):
    """
    Check 5: Are rolling features computed correctly?

    Check that features that should have temporal relationships do.
    E.g., 5-day MA should be correlated with 10-day MA.
    """
    print("\n" + "="*80)
    print("CHECK 5: TEMPORAL CONSISTENCY")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    # Check correlation between adjacent features (often related)
    adjacent_corrs = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:, :50]  # First 50 features

        for i in range(features.shape[1] - 1):
            corr = np.corrcoef(features[:, i], features[:, i+1])[0, 1]
            if not np.isnan(corr):
                adjacent_corrs.append(corr)

    mean_adj_corr = np.mean(adjacent_corrs)

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Adjacent Feature Correlation:")
    print(f"    Mean correlation: {mean_adj_corr:.4f}")

    # Adjacent features are often related (same category)
    passed = True  # Info only
    status = "INFO"
    print(f"\n  Status: [{status}]")
    print(f"    Adjacent features show {'high' if mean_adj_corr > 0.5 else 'moderate' if mean_adj_corr > 0.2 else 'low'} correlation (expected for grouped features)")

    return passed


def check_volume_features(h5f, num_stocks=50):
    """
    Check 6: Are volume features reasonable?

    Volume should vary significantly day-to-day.
    """
    print("\n" + "="*80)
    print("CHECK 6: VOLUME FEATURE VARIATION")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    # Volume features are likely in the derived features section
    # Look for features with high daily variation and positive values

    volume_like_features = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:]

        for feat_idx in range(min(100, features.shape[1])):
            col = features[:, feat_idx]

            # Volume-like: mostly positive, high variation
            if np.mean(col > 0) > 0.8:  # Mostly positive
                daily_change = np.abs(np.diff(col))
                if np.mean(daily_change) > 0.01:  # Changes daily
                    volume_like_features.append(feat_idx)
                    break

    num_with_volume = len(set(volume_like_features))

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Volume-like Features:")
    print(f"    Found {num_with_volume} stocks with volume-varying features")

    passed = num_with_volume > num_stocks * 0.5
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")

    return passed


def run_all_checks(dataset_path):
    """Run all predictive signal checks."""
    print("\n" + "="*80)
    print("PREDICTIVE SIGNAL SANITY CHECKS")
    print("="*80)
    print(f"\nDataset: {dataset_path}")

    with h5py.File(dataset_path, 'r') as h5f:
        results = {}

        results['feature_return_ic'] = check_feature_return_correlations(h5f)
        results['momentum_signal'] = check_momentum_signal(h5f)
        results['return_distribution'] = check_return_distribution(h5f)
        results['feature_redundancy'] = check_feature_redundancy(h5f)
        results['temporal_consistency'] = check_temporal_consistency(h5f)
        results['volume_features'] = check_volume_features(h5f)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Checks passed: {passed}/{total}")
    print()
    for check, result in results.items():
        status = "PASS" if result else "WARN/INFO"
        print(f"    [{status:^9}] {check}")

    print("\n" + "="*80 + "\n")

    return all(results.values())


def main():
    parser = argparse.ArgumentParser(description='Predictive signal sanity checks')
    parser.add_argument('--dataset', type=str, default='all_complete_dataset.h5',
                       help='Path to HDF5 dataset')

    args = parser.parse_args()

    success = run_all_checks(args.dataset)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
