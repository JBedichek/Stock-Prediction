"""
Test script to evaluate fundamental metrics availability from yfinance.

This script tests ~30 fundamental metrics across a sample of stocks to determine:
1. Which metrics are available
2. What percentage of stocks have each metric
3. Data quality and outliers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json

# Import stock lists from Stock.py
import sys
sys.path.append('/home/james/Desktop/Stock-Prediction')
from data_scraping.Stock import a_lot_of_stocks, s_lot_of_stocks


# Define all fundamental metrics to test
FUNDAMENTAL_METRICS = {
    # Valuation Ratios
    'trailingPE': 'Trailing P/E Ratio',
    'forwardPE': 'Forward P/E Ratio',
    'priceToBook': 'Price to Book',
    'priceToSalesTrailing12Months': 'Price to Sales (TTM)',
    'pegRatio': 'PEG Ratio',
    'enterpriseToRevenue': 'Enterprise Value / Revenue',
    'enterpriseToEbitda': 'Enterprise Value / EBITDA',

    # Profitability Metrics
    'profitMargins': 'Profit Margin',
    'operatingMargins': 'Operating Margin',
    'grossMargins': 'Gross Margin',
    'ebitdaMargins': 'EBITDA Margin',
    'returnOnEquity': 'Return on Equity (ROE)',
    'returnOnAssets': 'Return on Assets (ROA)',

    # Growth Metrics
    'revenueGrowth': 'Revenue Growth',
    'earningsGrowth': 'Earnings Growth',
    'earningsQuarterlyGrowth': 'Quarterly Earnings Growth',

    # Financial Health
    'debtToEquity': 'Debt to Equity',
    'currentRatio': 'Current Ratio',
    'quickRatio': 'Quick Ratio',
    'totalCash': 'Total Cash',
    'totalDebt': 'Total Debt',
    'freeCashflow': 'Free Cash Flow',
    'operatingCashflow': 'Operating Cash Flow',

    # Per Share Metrics
    'trailingEps': 'Trailing EPS',
    'forwardEps': 'Forward EPS',
    'bookValue': 'Book Value per Share',
    'revenuePerShare': 'Revenue per Share',

    # Market Metrics
    'beta': 'Beta (Volatility)',
    'fiftyTwoWeekHigh': '52-Week High',
    'fiftyTwoWeekLow': '52-Week Low',
    'marketCap': 'Market Capitalization',

    # Additional
    'dividendYield': 'Dividend Yield',
    'payoutRatio': 'Payout Ratio',
}


def test_ticker_fundamentals(ticker_symbol):
    """
    Test a single ticker for fundamental metric availability.

    Returns:
        dict: {metric_name: value or None}
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        results = {}
        for metric_key, metric_name in FUNDAMENTAL_METRICS.items():
            value = info.get(metric_key, None)
            results[metric_key] = value

        return results
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return {key: None for key in FUNDAMENTAL_METRICS.keys()}


def analyze_metric_availability(stock_dict, sample_size=200):
    """
    Test fundamental metrics across a sample of stocks.

    Args:
        stock_dict: Dictionary of {ticker: company_name}
        sample_size: Number of stocks to test

    Returns:
        pd.DataFrame: Metrics availability analysis
    """
    # Sample stocks
    tickers = list(stock_dict.keys())[:sample_size]

    print(f"\nTesting {len(tickers)} stocks for fundamental metric availability...")

    # Collect data
    all_results = []

    for ticker in tqdm(tickers, desc="Testing tickers"):
        results = test_ticker_fundamentals(ticker)
        results['ticker'] = ticker
        all_results.append(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Analyze availability
    availability = {}
    for metric in FUNDAMENTAL_METRICS.keys():
        total = len(df)
        non_null = df[metric].notna().sum()
        percentage = (non_null / total) * 100

        # Get some statistics
        values = df[metric].dropna()
        if len(values) > 0:
            mean_val = values.mean() if pd.api.types.is_numeric_dtype(values) else None
            median_val = values.median() if pd.api.types.is_numeric_dtype(values) else None
            std_val = values.std() if pd.api.types.is_numeric_dtype(values) else None
        else:
            mean_val = median_val = std_val = None

        availability[metric] = {
            'metric_name': FUNDAMENTAL_METRICS[metric],
            'available_count': non_null,
            'total_count': total,
            'availability_pct': percentage,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
        }

    return pd.DataFrame.from_dict(availability, orient='index'), df


def categorize_metrics(availability_df):
    """
    Categorize metrics by availability.
    """
    high_availability = availability_df[availability_df['availability_pct'] >= 70]
    medium_availability = availability_df[
        (availability_df['availability_pct'] >= 40) &
        (availability_df['availability_pct'] < 70)
    ]
    low_availability = availability_df[availability_df['availability_pct'] < 40]

    return {
        'high': high_availability,
        'medium': medium_availability,
        'low': low_availability,
    }


def detect_outliers(df, metric, threshold=3):
    """
    Detect outliers using z-score method.
    """
    values = df[metric].dropna()
    if len(values) < 10:
        return []

    z_scores = np.abs((values - values.mean()) / values.std())
    outlier_indices = values[z_scores > threshold].index

    return df.loc[outlier_indices, ['ticker', metric]].to_dict('records')


def print_report(availability_df, categorized, raw_df):
    """
    Print comprehensive analysis report.
    """
    print("\n" + "="*80)
    print("FUNDAMENTAL METRICS AVAILABILITY REPORT")
    print("="*80)

    print(f"\nTotal stocks tested: {len(raw_df)}")
    print(f"Total metrics tested: {len(FUNDAMENTAL_METRICS)}")

    # Summary statistics
    print("\n" + "-"*80)
    print("AVAILABILITY SUMMARY")
    print("-"*80)
    print(f"High availability (≥70%): {len(categorized['high'])} metrics")
    print(f"Medium availability (40-70%): {len(categorized['medium'])} metrics")
    print(f"Low availability (<40%): {len(categorized['low'])} metrics")

    # High availability metrics
    print("\n" + "-"*80)
    print("HIGH AVAILABILITY METRICS (≥70%)")
    print("-"*80)
    if len(categorized['high']) > 0:
        high_sorted = categorized['high'].sort_values('availability_pct', ascending=False)
        for idx, row in high_sorted.iterrows():
            mean_str = f"{row['mean']:.2f}" if pd.notna(row['mean']) else "N/A"
            print(f"{row['metric_name']:40} {row['availability_pct']:6.2f}% "
                  f"(mean: {mean_str:>10})")
    else:
        print("None")

    # Medium availability metrics
    print("\n" + "-"*80)
    print("MEDIUM AVAILABILITY METRICS (40-70%)")
    print("-"*80)
    if len(categorized['medium']) > 0:
        med_sorted = categorized['medium'].sort_values('availability_pct', ascending=False)
        for idx, row in med_sorted.iterrows():
            mean_str = f"{row['mean']:.2f}" if pd.notna(row['mean']) else "N/A"
            print(f"{row['metric_name']:40} {row['availability_pct']:6.2f}% "
                  f"(mean: {mean_str:>10})")
    else:
        print("None")

    # Low availability metrics
    print("\n" + "-"*80)
    print("LOW AVAILABILITY METRICS (<40%)")
    print("-"*80)
    if len(categorized['low']) > 0:
        low_sorted = categorized['low'].sort_values('availability_pct', ascending=False)
        for idx, row in low_sorted.iterrows():
            print(f"{row['metric_name']:40} {row['availability_pct']:6.2f}%")
    else:
        print("None")

    # Outlier detection for high availability metrics
    print("\n" + "-"*80)
    print("OUTLIER DETECTION (for high availability metrics)")
    print("-"*80)
    for metric in categorized['high'].head(10).index:
        outliers = detect_outliers(raw_df, metric)
        if outliers:
            print(f"\n{FUNDAMENTAL_METRICS[metric]} - {len(outliers)} outliers detected:")
            for outlier in outliers[:5]:  # Show first 5
                print(f"  {outlier['ticker']}: {outlier[metric]}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    recommended = categorized['high'][categorized['high']['availability_pct'] >= 75]
    print(f"\n✅ RECOMMENDED METRICS ({len(recommended)} metrics with ≥75% availability):")
    for idx, row in recommended.sort_values('availability_pct', ascending=False).iterrows():
        print(f"  • {row['metric_name']:40} {row['availability_pct']:6.2f}%")

    conditional = categorized['high'][
        (categorized['high']['availability_pct'] >= 70) &
        (categorized['high']['availability_pct'] < 75)
    ]
    if len(conditional) > 0:
        print(f"\n⚠️  CONDITIONAL METRICS ({len(conditional)} metrics with 70-75% availability):")
        for idx, row in conditional.sort_values('availability_pct', ascending=False).iterrows():
            print(f"  • {row['metric_name']:40} {row['availability_pct']:6.2f}%")

    not_recommended = pd.concat([categorized['medium'], categorized['low']])
    if len(not_recommended) > 0:
        print(f"\n❌ NOT RECOMMENDED ({len(not_recommended)} metrics with <70% availability):")
        print("  (Consider imputation or exclusion)")


def save_results(availability_df, raw_df, output_dir='/home/james/Desktop/Stock-Prediction'):
    """
    Save results to files.
    """
    import os

    # Save availability summary
    availability_df.to_csv(f'{output_dir}/fundamental_metrics_availability.csv')
    print(f"\n✅ Saved availability summary to: {output_dir}/fundamental_metrics_availability.csv")

    # Save raw data
    raw_df.to_csv(f'{output_dir}/fundamental_metrics_raw_data.csv', index=False)
    print(f"✅ Saved raw data to: {output_dir}/fundamental_metrics_raw_data.csv")

    # Save recommended metrics as JSON
    recommended = availability_df[availability_df['availability_pct'] >= 75].index.tolist()
    with open(f'{output_dir}/recommended_fundamental_metrics.json', 'w') as f:
        json.dump({
            'recommended_metrics': recommended,
            'metric_descriptions': {k: FUNDAMENTAL_METRICS[k] for k in recommended},
            'threshold': '75% availability',
        }, f, indent=2)
    print(f"✅ Saved recommended metrics to: {output_dir}/recommended_fundamental_metrics.json")


def main():
    """
    Main testing function.
    """
    print("="*80)
    print("FUNDAMENTAL METRICS AVAILABILITY TEST")
    print("="*80)

    # Choose stock set to test
    print("\nAvailable stock sets:")
    print("1. S-companies (~200 stocks)")
    print("2. A-companies (first 200)")
    print("3. Mixed sample (100 from each)")

    choice = input("\nSelect stock set (1-3) [default: 1]: ").strip() or "1"

    if choice == "1":
        stock_dict = s_lot_of_stocks
        sample_size = min(200, len(s_lot_of_stocks))
    elif choice == "2":
        stock_dict = a_lot_of_stocks
        sample_size = 200
    else:
        # Mix
        stock_dict = {**dict(list(s_lot_of_stocks.items())[:100]),
                      **dict(list(a_lot_of_stocks.items())[:100])}
        sample_size = 200

    # Run analysis
    availability_df, raw_df = analyze_metric_availability(stock_dict, sample_size)

    # Categorize
    categorized = categorize_metrics(availability_df)

    # Print report
    print_report(availability_df, categorized, raw_df)

    # Save results
    save_results(availability_df, raw_df)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
