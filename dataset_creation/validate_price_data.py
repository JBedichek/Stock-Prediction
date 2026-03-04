#!/usr/bin/env python3
"""
Validate and Clean Price Data

Detects and handles various data quality issues:
1. yfinance dividend adjustment bugs (negative prices)
2. Untracked corporate actions (mergers, spinoffs)
3. Extreme but legitimate volatility

Usage:
    python dataset_creation/validate_price_data.py --input actual_prices.h5 --output actual_prices_clean.h5

    # Or just analyze without fixing:
    python dataset_creation/validate_price_data.py --input actual_prices.h5 --analyze-only
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IssueType(Enum):
    NEGATIVE_PRICE = "negative_price"
    IMPOSSIBLE_RETURN = "impossible_return"  # < -100%
    EXTREME_SINGLE_DAY = "extreme_single_day"  # > 500% in one day
    LIKELY_UNTRACKED_SPLIT = "likely_untracked_split"  # > 200% with stable pattern before/after
    ZERO_PRICE = "zero_price"
    CLEAN = "clean"


@dataclass
class TickerAnalysis:
    ticker: str
    issue_type: IssueType
    issue_dates: List[str]
    details: str
    max_return: float
    min_return: float
    has_negative_prices: bool
    recommendation: str  # 'keep', 'exclude', 'flag'


def analyze_ticker(
    ticker: str,
    dates: np.ndarray,
    prices: np.ndarray,
    max_daily_return: float = 5.0,  # 500%
    max_5d_return: float = 5.0,  # 500%
    min_price: float = 0.01,  # $0.01 minimum
) -> TickerAnalysis:
    """
    Analyze a single ticker for data quality issues.

    Returns analysis with issue type and recommendation.

    Args:
        ticker: Stock ticker
        dates: Array of dates
        prices: Array of prices
        max_daily_return: Maximum allowed daily return (default 500%)
        max_5d_return: Maximum allowed 5-day return (default 500%)
        min_price: Minimum valid price (default $0.01)
    """
    dates_str = [d.decode('utf-8') if isinstance(d, bytes) else d for d in dates]

    # Basic checks
    has_negative = np.any(prices < 0)
    has_zero = np.any(prices == 0)
    has_sub_penny = np.any((prices > 0) & (prices < min_price))

    if has_negative:
        neg_idx = np.where(prices < 0)[0]
        return TickerAnalysis(
            ticker=ticker,
            issue_type=IssueType.NEGATIVE_PRICE,
            issue_dates=[dates_str[i] for i in neg_idx[:5]],
            details=f"{len(neg_idx)} days with negative prices (min: ${prices.min():.4f})",
            max_return=0,
            min_return=0,
            has_negative_prices=True,
            recommendation='exclude'
        )

    if has_zero:
        zero_idx = np.where(prices == 0)[0]
        return TickerAnalysis(
            ticker=ticker,
            issue_type=IssueType.ZERO_PRICE,
            issue_dates=[dates_str[i] for i in zero_idx[:5]],
            details=f"{len(zero_idx)} days with zero prices",
            max_return=0,
            min_return=0,
            has_negative_prices=False,
            recommendation='exclude'
        )

    if has_sub_penny:
        sub_idx = np.where((prices > 0) & (prices < min_price))[0]
        return TickerAnalysis(
            ticker=ticker,
            issue_type=IssueType.ZERO_PRICE,  # Treat sub-penny as zero
            issue_dates=[dates_str[i] for i in sub_idx[:5]],
            details=f"{len(sub_idx)} days with sub-penny prices (min: ${prices.min():.6f})",
            max_return=0,
            min_return=0,
            has_negative_prices=False,
            recommendation='exclude'
        )

    # Calculate daily returns
    returns = np.diff(prices) / prices[:-1]

    # Check for impossible returns (< -100%)
    impossible_idx = np.where(returns < -1.0)[0]
    if len(impossible_idx) > 0:
        return TickerAnalysis(
            ticker=ticker,
            issue_type=IssueType.IMPOSSIBLE_RETURN,
            issue_dates=[dates_str[i+1] for i in impossible_idx[:5]],
            details=f"{len(impossible_idx)} days with return < -100% (min: {returns.min()*100:.1f}%)",
            max_return=float(returns.max()),
            min_return=float(returns.min()),
            has_negative_prices=False,
            recommendation='exclude'
        )

    # Check for extreme daily returns (both directions)
    extreme_daily_idx = np.where(np.abs(returns) > max_daily_return)[0]
    if len(extreme_daily_idx) > 0:
        max_abs_ret = np.abs(returns[extreme_daily_idx]).max()
        return TickerAnalysis(
            ticker=ticker,
            issue_type=IssueType.EXTREME_SINGLE_DAY,
            issue_dates=[dates_str[i+1] for i in extreme_daily_idx[:5]],
            details=f"{len(extreme_daily_idx)} days with >{max_daily_return*100:.0f}% daily return (max: {max_abs_ret*100:.1f}%)",
            max_return=float(returns.max()),
            min_return=float(returns.min()),
            has_negative_prices=False,
            recommendation='exclude'  # Changed from 'flag' to 'exclude'
        )

    # Check for extreme 5-day returns
    if len(prices) >= 6:
        returns_5d = (prices[5:] / prices[:-5]) - 1
        extreme_5d_idx = np.where(np.abs(returns_5d) > max_5d_return)[0]
        if len(extreme_5d_idx) > 0:
            max_abs_5d = np.abs(returns_5d[extreme_5d_idx]).max()
            return TickerAnalysis(
                ticker=ticker,
                issue_type=IssueType.LIKELY_UNTRACKED_SPLIT,
                issue_dates=[dates_str[i+5] for i in extreme_5d_idx[:5]],
                details=f"{len(extreme_5d_idx)} periods with >{max_5d_return*100:.0f}% 5-day return (max: {max_abs_5d*100:.1f}%)",
                max_return=float(returns.max()),
                min_return=float(returns.min()),
                has_negative_prices=False,
                recommendation='exclude'
            )

    # Ticker is clean
    return TickerAnalysis(
        ticker=ticker,
        issue_type=IssueType.CLEAN,
        issue_dates=[],
        details="No issues detected",
        max_return=float(returns.max()) if len(returns) > 0 else 0,
        min_return=float(returns.min()) if len(returns) > 0 else 0,
        has_negative_prices=False,
        recommendation='keep'
    )


def validate_price_file(
    input_path: str,
    output_path: Optional[str] = None,
    analyze_only: bool = False,
    exclude_flagged: bool = False,
) -> Dict[str, TickerAnalysis]:
    """
    Validate all tickers in a price HDF5 file.

    Args:
        input_path: Input HDF5 file
        output_path: Output HDF5 file (cleaned)
        analyze_only: If True, only analyze without creating output
        exclude_flagged: If True, also exclude 'flag' recommendations

    Returns:
        Dict of ticker -> analysis
    """
    print(f"\n{'='*70}")
    print("PRICE DATA VALIDATION")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    if output_path and not analyze_only:
        print(f"Output: {output_path}")
    print()

    analyses = {}

    with h5py.File(input_path, 'r') as h5f:
        tickers = list(h5f.keys())
        print(f"Total tickers: {len(tickers)}")

        for ticker in tqdm(tickers, desc="Analyzing"):
            try:
                dates = h5f[ticker]['dates'][:]
                prices = h5f[ticker]['prices'][:]

                analysis = analyze_ticker(ticker, dates, prices)
                analyses[ticker] = analysis

            except Exception as e:
                analyses[ticker] = TickerAnalysis(
                    ticker=ticker,
                    issue_type=IssueType.IMPOSSIBLE_RETURN,
                    issue_dates=[],
                    details=f"Error reading data: {e}",
                    max_return=0,
                    min_return=0,
                    has_negative_prices=False,
                    recommendation='exclude'
                )

    # Summarize results
    by_type = {}
    by_recommendation = {'keep': [], 'exclude': [], 'flag': []}

    for ticker, analysis in analyses.items():
        issue_type = analysis.issue_type.value
        if issue_type not in by_type:
            by_type[issue_type] = []
        by_type[issue_type].append(ticker)
        by_recommendation[analysis.recommendation].append(ticker)

    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\nBy Issue Type:")
    for issue_type, tickers_list in sorted(by_type.items()):
        print(f"  {issue_type}: {len(tickers_list)}")
        if len(tickers_list) <= 10 and issue_type != 'clean':
            for t in tickers_list:
                print(f"    - {t}: {analyses[t].details}")

    print(f"\nBy Recommendation:")
    for rec, tickers_list in by_recommendation.items():
        print(f"  {rec}: {len(tickers_list)}")

    # Create cleaned output file
    if output_path and not analyze_only:
        keep_tickers = by_recommendation['keep']
        if not exclude_flagged:
            keep_tickers += by_recommendation['flag']

        print(f"\n{'='*70}")
        print(f"CREATING CLEANED FILE")
        print(f"{'='*70}")
        print(f"Keeping {len(keep_tickers)} tickers")
        print(f"Excluding {len(tickers) - len(keep_tickers)} tickers")

        with h5py.File(input_path, 'r') as h5f_in:
            with h5py.File(output_path, 'w') as h5f_out:
                for ticker in tqdm(keep_tickers, desc="Copying"):
                    try:
                        grp = h5f_out.create_group(ticker)
                        grp.create_dataset('prices', data=h5f_in[ticker]['prices'][:], compression='gzip')
                        grp.create_dataset('dates', data=h5f_in[ticker]['dates'][:], compression='gzip')
                    except Exception as e:
                        print(f"  Warning: Failed to copy {ticker}: {e}")

        print(f"\nSaved cleaned data to: {output_path}")

    # Print detailed report for excluded tickers
    excluded = by_recommendation['exclude']
    if excluded:
        print(f"\n{'='*70}")
        print("EXCLUDED TICKERS (Details)")
        print(f"{'='*70}")
        for ticker in excluded[:30]:
            a = analyses[ticker]
            print(f"\n{ticker}:")
            print(f"  Issue: {a.issue_type.value}")
            print(f"  Details: {a.details}")
            if a.issue_dates:
                print(f"  Example dates: {', '.join(a.issue_dates[:3])}")
        if len(excluded) > 30:
            print(f"\n  ... and {len(excluded) - 30} more")

    return analyses


def verify_against_yfinance(tickers: List[str], sample_size: int = 10):
    """
    Verify problematic tickers against fresh yfinance data.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not available for verification")
        return

    print(f"\n{'='*70}")
    print("YFINANCE VERIFICATION")
    print(f"{'='*70}")

    sample = tickers[:sample_size]

    for ticker in sample:
        print(f"\n{ticker}:")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start='2020-01-01', end='2024-01-01', auto_adjust=False)

            if df.empty:
                print(f"  No data from yfinance")
                continue

            # Check adj factor
            adj_factor = df['Adj Close'] / df['Close']

            print(f"  yfinance Adj Close range: ${df['Adj Close'].min():.4f} to ${df['Adj Close'].max():.2f}")
            print(f"  Adj factor range: {adj_factor.min():.4f} to {adj_factor.max():.4f}")

            if adj_factor.min() < 0:
                print(f"  ⚠️  CONFIRMED: yfinance bug - negative adj factor")

            # Check splits
            splits = stock.splits
            if len(splits) > 0:
                print(f"  Recorded splits: {len(splits)}")
            else:
                # Check if price had extreme jump
                daily_ret = df['Close'].pct_change()
                extreme = daily_ret[daily_ret.abs() > 2.0]
                if len(extreme) > 0:
                    print(f"  ⚠️  No splits recorded but {len(extreme)} days with >200% price change")

        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Validate and clean price data')
    parser.add_argument('--input', type=str, default='actual_prices.h5',
                       help='Input price HDF5 file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output cleaned HDF5 file')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not create output file')
    parser.add_argument('--exclude-flagged', action='store_true',
                       help='Also exclude flagged (suspicious but possible) tickers')
    parser.add_argument('--verify-yfinance', action='store_true',
                       help='Verify problematic tickers against fresh yfinance data')

    args = parser.parse_args()

    if args.output is None and not args.analyze_only:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"

    analyses = validate_price_file(
        input_path=args.input,
        output_path=args.output,
        analyze_only=args.analyze_only,
        exclude_flagged=args.exclude_flagged,
    )

    # Verify against yfinance if requested
    if args.verify_yfinance:
        excluded = [t for t, a in analyses.items() if a.recommendation == 'exclude']
        verify_against_yfinance(excluded)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
