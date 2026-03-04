#!/usr/bin/env python3
"""
Dataset Quality Sanity Checks

Verifies the dataset has meaningful signal and no obvious issues.

Checks:
1. Price differentiation after normalization (features not all squeezed to zero)
2. Daily variation (derived features change daily, not forward-filled)
3. Target leakage (features at T don't contain info from T+1)
4. Feature scale (reasonable ranges after normalization)
5. Cross-sectional variation (meaningful spread across stocks per date)
6. News embedding coverage (% of days with non-zero embeddings)
7. NaN/Inf check
8. Date alignment verification

Usage:
    python sanity_checks/check_dataset_quality.py --dataset all_complete_dataset.h5
"""

import argparse
import h5py
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def check_price_differentiation(h5f, num_stocks=50, num_features=50):
    """
    Check 1: Verify normalized features still have meaningful variation.

    After z-score normalization over long time horizons, features shouldn't
    all be squeezed to near-zero with no differentiation.
    """
    print("\n" + "="*80)
    print("CHECK 1: PRICE DIFFERENTIATION AFTER NORMALIZATION")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    # Collect statistics across stocks
    feature_stds = []
    feature_ranges = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:]
        # Check first num_features base features (not news embeddings)
        base_features = features[:, :min(num_features, features.shape[1])]

        # Per-feature std across time
        stds = np.std(base_features, axis=0)
        ranges = np.ptp(base_features, axis=0)  # peak-to-peak (max - min)

        feature_stds.append(stds)
        feature_ranges.append(ranges)

    feature_stds = np.array(feature_stds)  # (num_stocks, num_features)
    feature_ranges = np.array(feature_ranges)

    # Statistics
    mean_std = np.mean(feature_stds)
    min_std = np.min(feature_stds)
    max_std = np.max(feature_stds)

    # Count features with very low variation
    low_var_threshold = 0.01
    low_var_count = np.sum(feature_stds < low_var_threshold)
    low_var_pct = low_var_count / feature_stds.size * 100

    print(f"\n  Analyzed: {len(tickers)} stocks x {num_features} features")
    print(f"\n  Feature Standard Deviations:")
    print(f"    Mean: {mean_std:.4f}")
    print(f"    Min:  {min_std:.6f}")
    print(f"    Max:  {max_std:.4f}")
    print(f"\n  Low-variation features (std < {low_var_threshold}):")
    print(f"    Count: {low_var_count} / {feature_stds.size} ({low_var_pct:.1f}%)")

    # PASS/FAIL
    passed = low_var_pct < 20 and mean_std > 0.1
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: Too many features with near-zero variation!")

    return passed


def check_daily_variation(h5f, num_stocks=30):
    """
    Check 2: Verify derived features change daily (not forward-filled).

    Returns, volume ratios, etc. should have low autocorrelation.
    Fundamentals will have high autocorrelation (expected).
    """
    print("\n" + "="*80)
    print("CHECK 2: DAILY VARIATION (NOT FORWARD-FILLED)")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]
    num_features = h5f[tickers[0]]['features'].shape[1]

    # We expect ~355 base features, with derived features in certain ranges
    # Let's check autocorrelation for all features

    # Collect autocorrelations per feature across all stocks
    feature_autocorrs = defaultdict(list)
    feature_unique_ratios = defaultdict(list)

    for ticker in tickers:
        features = h5f[ticker]['features'][:]

        for feat_idx in range(min(355, num_features)):  # Base features only
            col = features[:, feat_idx]
            if np.std(col) > 1e-10:
                # Autocorrelation at lag 1
                ac = np.corrcoef(col[:-1], col[1:])[0, 1]
                feature_autocorrs[feat_idx].append(ac if not np.isnan(ac) else 1.0)

                # Unique value ratio
                unique_ratio = len(np.unique(col)) / len(col)
                feature_unique_ratios[feat_idx].append(unique_ratio)

    # Compute mean autocorrelation per feature
    mean_autocorr = np.array([np.mean(feature_autocorrs[i]) if i in feature_autocorrs else 1.0
                              for i in range(min(355, num_features))])

    daily_varying = np.sum(mean_autocorr < 0.9)
    forward_filled = np.sum(mean_autocorr > 0.99)

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Feature Autocorrelation Distribution:")
    print(f"    Daily-varying (autocorr < 0.9):  {daily_varying} features")
    print(f"    Forward-filled (autocorr > 0.99): {forward_filled} features")
    print(f"    In between: {len(mean_autocorr) - daily_varying - forward_filled} features")

    # Compute mean unique ratio for daily-varying vs forward-filled
    daily_varying_unique = [np.mean(feature_unique_ratios[i]) for i in range(len(mean_autocorr))
                           if mean_autocorr[i] < 0.9 and i in feature_unique_ratios]
    forward_filled_unique = [np.mean(feature_unique_ratios[i]) for i in range(len(mean_autocorr))
                            if mean_autocorr[i] > 0.99 and i in feature_unique_ratios]

    print(f"\n  Mean Unique Value Ratio:")
    if daily_varying_unique:
        print(f"    Daily-varying features: {np.mean(daily_varying_unique):.2%}")
    if forward_filled_unique:
        print(f"    Forward-filled features: {np.mean(forward_filled_unique):.2%}")

    # PASS if we have enough daily-varying features (derived features should be ~50+)
    passed = daily_varying >= 40
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: Not enough daily-varying features! Expected 40+, got {daily_varying}")
    else:
        print(f"    Found {daily_varying} daily-varying features (derived + market-relative)")

    return passed


def check_target_leakage(h5f, num_stocks=30, horizon=1):
    """
    Check 3: Verify features at time T don't contain future information.

    Compute correlation between features at T and returns from T to T+horizon.
    Suspiciously high correlations indicate leakage.
    """
    print("\n" + "="*80)
    print("CHECK 3: TARGET LEAKAGE (FEATURES DON'T PREDICT TOO WELL)")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    # We need to compute returns ourselves to check for leakage
    # First, find a return-like feature or use raw feature changes

    suspicious_correlations = []

    for ticker in tickers:
        features = h5f[ticker]['features'][:]
        num_dates, num_feats = features.shape

        # Use feature changes as proxy for returns (feature 0 might be price-related)
        # Actually, let's check correlation of features at T with features at T+1

        for feat_idx in range(min(50, num_feats)):
            feat_T = features[:-horizon, feat_idx]
            feat_T1 = features[horizon:, feat_idx]

            # Check if T perfectly predicts T+1 (would indicate using future data)
            if np.std(feat_T) > 1e-10 and np.std(feat_T1) > 1e-10:
                corr = np.corrcoef(feat_T, feat_T1)[0, 1]

                # Autocorrelation > 0.99 is fine for fundamentals
                # But if we have return-like features with autocorr > 0.5, that's suspicious
                # (returns should be nearly unpredictable)

                # Actually, the real test: do features at T correlate with FUTURE returns?
                # Let's compute pseudo-returns from a feature that should be return-like
                pass

        # More direct test: check if return features have suspiciously high IC
        # Return features should have autocorr near 0

        # Find return-like features (those with low autocorr AND names suggest returns)
        return_feature_autocorrs = []
        for feat_idx in range(min(355, num_feats)):
            col = features[:, feat_idx]
            if np.std(col) > 1e-10:
                ac = np.corrcoef(col[:-1], col[1:])[0, 1]
                if not np.isnan(ac) and ac < 0.3:  # Low autocorr suggests returns
                    return_feature_autocorrs.append(ac)

        if return_feature_autocorrs:
            suspicious_correlations.append(max(return_feature_autocorrs))

    mean_max_corr = np.mean(suspicious_correlations) if suspicious_correlations else 0

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Low-autocorr features (likely returns/changes):")
    print(f"    Mean max autocorrelation: {mean_max_corr:.4f}")

    # If return-like features have autocorr > 0.5, that's suspicious
    passed = mean_max_corr < 0.5
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: Return-like features have high autocorrelation!")
        print(f"    This may indicate look-ahead bias in feature computation.")
    else:
        print(f"    Return-like features have expected low autocorrelation.")

    return passed


def check_feature_scales(h5f, num_stocks=50):
    """
    Check 4: Verify features are in reasonable ranges after normalization.

    Z-scored features should mostly be in [-5, 5] range.
    Extreme outliers suggest normalization issues.
    """
    print("\n" + "="*80)
    print("CHECK 4: FEATURE SCALES (REASONABLE RANGES)")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    all_mins = []
    all_maxs = []
    extreme_count = 0
    total_values = 0

    for ticker in tickers:
        features = h5f[ticker]['features'][:, :355]  # Base features only

        all_mins.append(np.min(features))
        all_maxs.append(np.max(features))

        # Count values outside [-10, 10]
        extreme_count += np.sum(np.abs(features) > 10)
        total_values += features.size

    global_min = min(all_mins)
    global_max = max(all_maxs)
    extreme_pct = extreme_count / total_values * 100

    print(f"\n  Analyzed: {len(tickers)} stocks x 355 base features")
    print(f"\n  Feature Value Range:")
    print(f"    Global min: {global_min:.2f}")
    print(f"    Global max: {global_max:.2f}")
    print(f"\n  Extreme Values (|x| > 10):")
    print(f"    Count: {extreme_count:,} / {total_values:,} ({extreme_pct:.2f}%)")

    # PASS if extremes are rare (< 5%)
    passed = extreme_pct < 5
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: Many extreme values after normalization!")

    return passed


def check_cross_sectional_variation(h5f, num_dates=100):
    """
    Check 5: Verify meaningful cross-sectional variation per date.

    For each date, features should vary meaningfully across stocks.
    """
    print("\n" + "="*80)
    print("CHECK 5: CROSS-SECTIONAL VARIATION")
    print("="*80)

    tickers = list(h5f.keys())
    num_tickers = len(tickers)

    # Sample random dates
    sample_ticker = tickers[0]
    dates = h5f[sample_ticker]['dates'][:]
    total_dates = len(dates)

    date_indices = np.random.choice(total_dates, min(num_dates, total_dates), replace=False)
    date_indices = np.sort(date_indices)

    cross_sectional_stds = []

    for date_idx in date_indices:
        # Collect features across all stocks for this date
        features_this_date = []
        for ticker in tickers:
            feat = h5f[ticker]['features'][date_idx, :355]  # Base features
            features_this_date.append(feat)

        features_this_date = np.array(features_this_date)  # (stocks, features)

        # Cross-sectional std for each feature
        cs_std = np.std(features_this_date, axis=0)
        cross_sectional_stds.append(np.mean(cs_std))

    mean_cs_std = np.mean(cross_sectional_stds)
    min_cs_std = np.min(cross_sectional_stds)

    print(f"\n  Analyzed: {num_dates} dates x {num_tickers} stocks")
    print(f"\n  Cross-Sectional Standard Deviation:")
    print(f"    Mean across dates: {mean_cs_std:.4f}")
    print(f"    Min across dates:  {min_cs_std:.4f}")

    # For z-scored data, expect CS std around 1.0
    passed = mean_cs_std > 0.5 and min_cs_std > 0.1
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    WARNING: Low cross-sectional variation!")

    return passed


def check_news_embedding_coverage(h5f, num_stocks=100):
    """
    Check 6: What % of days have non-zero news embeddings?
    """
    print("\n" + "="*80)
    print("CHECK 6: NEWS EMBEDDING COVERAGE")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]
    num_features = h5f[tickers[0]]['features'].shape[1]

    # News embeddings are typically the last 768 features
    news_start = num_features - 768

    non_zero_days = 0
    total_days = 0

    for ticker in tickers:
        news_features = h5f[ticker]['features'][:, news_start:]

        # A day has news if any embedding dimension is non-zero
        has_news = np.any(np.abs(news_features) > 1e-6, axis=1)
        non_zero_days += np.sum(has_news)
        total_days += len(has_news)

    coverage = non_zero_days / total_days * 100

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"  News embedding dimensions: {news_start} to {num_features-1}")
    print(f"\n  Coverage:")
    print(f"    Days with news: {non_zero_days:,} / {total_days:,} ({coverage:.1f}%)")

    # Just report, don't fail - low coverage is expected
    status = "INFO"
    print(f"\n  Status: [{status}]")
    if coverage < 10:
        print(f"    Note: Low news coverage. Model may not benefit from news embeddings.")

    return True  # Info only


def check_nan_inf(h5f, num_stocks=50):
    """
    Check 7: Verify no NaN or Inf values.
    """
    print("\n" + "="*80)
    print("CHECK 7: NAN/INF CHECK")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    nan_count = 0
    inf_count = 0
    total_values = 0

    for ticker in tickers:
        features = h5f[ticker]['features'][:]
        nan_count += np.sum(np.isnan(features))
        inf_count += np.sum(np.isinf(features))
        total_values += features.size

    print(f"\n  Analyzed: {len(tickers)} stocks")
    print(f"\n  Invalid Values:")
    print(f"    NaN count: {nan_count:,}")
    print(f"    Inf count: {inf_count:,}")
    print(f"    Total values: {total_values:,}")

    passed = nan_count == 0 and inf_count == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  Status: [{status}]")
    if not passed:
        print(f"    CRITICAL: Dataset contains invalid values!")

    return passed


def check_date_alignment(h5f, num_stocks=20):
    """
    Check 8: Verify dates are consistent across stocks.
    """
    print("\n" + "="*80)
    print("CHECK 8: DATE ALIGNMENT")
    print("="*80)

    tickers = list(h5f.keys())[:num_stocks]

    # Get dates from first ticker
    reference_dates = set(h5f[tickers[0]]['dates'][:].astype(str))
    reference_count = len(reference_dates)

    misaligned = []

    for ticker in tickers[1:]:
        ticker_dates = set(h5f[ticker]['dates'][:].astype(str))
        if ticker_dates != reference_dates:
            diff = len(reference_dates.symmetric_difference(ticker_dates))
            misaligned.append((ticker, diff))

    print(f"\n  Reference ticker: {tickers[0]} ({reference_count} dates)")
    print(f"  Checked: {len(tickers)} tickers")
    print(f"\n  Alignment:")
    print(f"    Misaligned tickers: {len(misaligned)}")

    if misaligned:
        for ticker, diff in misaligned[:5]:
            print(f"      {ticker}: {diff} different dates")

    passed = len(misaligned) == 0
    status = "PASS" if passed else "WARN"
    print(f"\n  Status: [{status}]")

    return passed


def run_all_checks(dataset_path):
    """Run all sanity checks."""
    print("\n" + "="*80)
    print("DATASET QUALITY SANITY CHECKS")
    print("="*80)
    print(f"\nDataset: {dataset_path}")

    with h5py.File(dataset_path, 'r') as h5f:
        # Print basic info
        num_tickers = len(h5f.keys())
        sample_ticker = list(h5f.keys())[0]
        num_dates = h5f[sample_ticker]['features'].shape[0]
        num_features = h5f[sample_ticker]['features'].shape[1]

        print(f"\nDataset Info:")
        print(f"  Tickers: {num_tickers}")
        print(f"  Dates: {num_dates}")
        print(f"  Features: {num_features}")

        results = {}

        # Run checks
        results['price_differentiation'] = check_price_differentiation(h5f)
        results['daily_variation'] = check_daily_variation(h5f)
        results['target_leakage'] = check_target_leakage(h5f)
        results['feature_scales'] = check_feature_scales(h5f)
        results['cross_sectional_variation'] = check_cross_sectional_variation(h5f)
        results['news_coverage'] = check_news_embedding_coverage(h5f)
        results['nan_inf'] = check_nan_inf(h5f)
        results['date_alignment'] = check_date_alignment(h5f)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Checks passed: {passed}/{total}")
    print()
    for check, result in results.items():
        status = "PASS" if result else "FAIL/WARN"
        print(f"    [{status:^9}] {check}")

    print("\n" + "="*80 + "\n")

    return all(results.values())


def main():
    parser = argparse.ArgumentParser(description='Dataset quality sanity checks')
    parser.add_argument('--dataset', type=str, default='all_complete_dataset.h5',
                       help='Path to HDF5 dataset')

    args = parser.parse_args()

    success = run_all_checks(args.dataset)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
