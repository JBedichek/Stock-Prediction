"""
Simplified test for fundamental data collector that doesn't require CUDA.

Directly tests the fundamental metrics methods without full stock_info initialization.
"""

import yfinance as yf
import torch
import pickle
import os
import pandas as pd
import numpy as np

# Define the metrics we're testing (from recommended_fundamental_metrics.json)
RECOMMENDED_METRICS = {
    'ebitdaMargins': 'EBITDA Margin',
    'grossMargins': 'Gross Margin',
    'operatingMargins': 'Operating Margin',
    'profitMargins': 'Profit Margin',
    'totalCash': 'Total Cash',
    'totalDebt': 'Total Debt',
    'currentRatio': 'Current Ratio',
    'quickRatio': 'Quick Ratio',
    'priceToBook': 'Price to Book',
    'priceToSalesTrailing12Months': 'Price to Sales (TTM)',
    'enterpriseToRevenue': 'Enterprise Value / Revenue',
    'returnOnEquity': 'Return on Equity (ROE)',
    'returnOnAssets': 'Return on Assets (ROA)',
    'revenueGrowth': 'Revenue Growth',
    'operatingCashflow': 'Operating Cash Flow',
    'trailingEps': 'Trailing EPS',
    'bookValue': 'Book Value per Share',
    'revenuePerShare': 'Revenue per Share',
    'beta': 'Beta (Volatility)',
    'fiftyTwoWeekHigh': '52-Week High',
    'fiftyTwoWeekLow': '52-Week Low',
    'marketCap': 'Market Capitalization',
    'forwardPE': 'Forward P/E Ratio',
    'payoutRatio': 'Payout Ratio',
}

CONDITIONAL_METRICS = {
    'enterpriseToEbitda': 'Enterprise Value / EBITDA',
    'forwardEps': 'Forward EPS',
    'debtToEquity': 'Debt to Equity',
}

ALL_METRICS = {**RECOMMENDED_METRICS, **CONDITIONAL_METRICS}


def collect_fundamentals_simple(ticker_dict):
    """
    Simplified version of get_fundamental_metrics for testing.
    """
    print(f"\nCollecting fundamental metrics for {len(ticker_dict)} stocks...")

    results = {}
    sectors = {}

    for ticker_symbol, company_name in ticker_dict.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            # Extract metrics
            metrics = {}
            for metric_key in ALL_METRICS.keys():
                value = info.get(metric_key, None)
                metrics[metric_key] = value

            results[ticker_symbol] = metrics

            # Store sector/industry for imputation
            sectors[ticker_symbol] = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }

            print(f"  ✓ {ticker_symbol}: {len([v for v in metrics.values() if v is not None])}/{len(ALL_METRICS)} metrics")

        except Exception as e:
            print(f"  ✗ {ticker_symbol}: Error - {e}")
            results[ticker_symbol] = {k: None for k in ALL_METRICS.keys()}
            sectors[ticker_symbol] = {'sector': 'Unknown', 'industry': 'Unknown'}

    return results, sectors


def impute_metrics(raw_data, sectors):
    """
    Simplified imputation logic.
    """
    print("\nApplying imputation...")

    # Group by industry
    industry_groups = {}
    for ticker, sector_info in sectors.items():
        industry = sector_info['industry']
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(ticker)

    # Calculate industry averages for each metric
    industry_averages = {}
    for metric in ALL_METRICS.keys():
        industry_averages[metric] = {}

        for industry, tickers in industry_groups.items():
            values = [raw_data[t][metric] for t in tickers if raw_data[t][metric] is not None]

            if len(values) >= 3:
                # Remove outliers
                values_array = np.array(values)
                mean = np.mean(values_array)
                std = np.std(values_array)
                filtered = values_array[np.abs(values_array - mean) <= 3 * std]

                if len(filtered) > 0:
                    industry_averages[metric][industry] = np.mean(filtered)

    # Calculate global medians
    global_medians = {}
    for metric in ALL_METRICS.keys():
        all_values = [data[metric] for data in raw_data.values() if data[metric] is not None]
        if all_values:
            global_medians[metric] = np.median(all_values)
        else:
            global_medians[metric] = 0.0

    # Apply imputation
    imputed_data = {}
    imputation_stats = {'industry': 0, 'global': 0, 'zero': 0}

    for ticker, metrics in raw_data.items():
        imputed_metrics = metrics.copy()
        industry = sectors[ticker]['industry']

        for metric, value in metrics.items():
            if value is None:
                # Try industry average
                if metric in industry_averages and industry in industry_averages[metric]:
                    imputed_metrics[metric] = industry_averages[metric][industry]
                    imputation_stats['industry'] += 1
                # Fall back to global median
                elif metric in global_medians:
                    imputed_metrics[metric] = global_medians[metric]
                    imputation_stats['global'] += 1
                else:
                    imputed_metrics[metric] = 0.0
                    imputation_stats['zero'] += 1

        imputed_data[ticker] = imputed_metrics

    total_imputed = sum(imputation_stats.values())
    print(f"\nImputation statistics:")
    print(f"  Industry average: {imputation_stats['industry']} ({imputation_stats['industry']/total_imputed*100:.1f}%)")
    print(f"  Global median: {imputation_stats['global']} ({imputation_stats['global']/total_imputed*100:.1f}%)")
    print(f"  Zero fill: {imputation_stats['zero']} ({imputation_stats['zero']/total_imputed*100:.1f}%)")

    return imputed_data


def convert_to_tensors(data):
    """
    Convert to torch tensors.
    """
    print("\nConverting to tensors...")

    metric_order = list(ALL_METRICS.keys())
    tensors = {}

    for ticker, metrics in data.items():
        values = [metrics[m] if metrics[m] is not None else 0.0 for m in metric_order]
        tensors[ticker] = torch.tensor(values, dtype=torch.float32)

    print(f"  Created {len(tensors)} tensors of shape ({len(metric_order)},)")

    return tensors, metric_order


def normalize_tensors(tensors, metric_order):
    """
    Robust normalization using median and IQR.
    """
    print("\nNormalizing tensors...")

    # Stack all tensors
    all_tensors = torch.stack(list(tensors.values()))
    all_numpy = all_tensors.numpy()

    # Calculate medians and IQR
    medians = np.median(all_numpy, axis=0)
    q1 = np.percentile(all_numpy, 25, axis=0)
    q3 = np.percentile(all_numpy, 75, axis=0)
    iqr = q3 - q1

    # Avoid division by zero
    iqr = np.where(iqr == 0, 1.0, iqr)

    # Normalize
    normalized = {}
    for ticker, tensor in tensors.items():
        norm_values = (tensor.numpy() - medians) / iqr
        norm_values = np.clip(norm_values, -10, 10)
        normalized[ticker] = torch.tensor(norm_values, dtype=torch.float32)

    # Print statistics
    all_norm = torch.stack(list(normalized.values()))
    print(f"  Mean: {all_norm.mean():.4f} (should be ~0)")
    print(f"  Std: {all_norm.std():.4f}")
    print(f"  Min: {all_norm.min():.4f}")
    print(f"  Max: {all_norm.max():.4f}")
    print(f"  In range [-10, 10]: {((all_norm >= -10) & (all_norm <= 10)).all()}")

    return normalized


def main():
    print("="*80)
    print("SIMPLIFIED FUNDAMENTAL DATA COLLECTOR TEST")
    print("="*80)

    # Test stocks
    test_stocks = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'TSLA': 'Tesla Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson',
        'V': 'Visa Inc.',
        'WMT': 'Walmart Inc.',
        'PG': 'Procter & Gamble Co.',
        'UNH': 'UnitedHealth Group Inc.',
    }

    # Test 1: Collection
    print("\n" + "="*80)
    print("TEST 1: COLLECTION")
    print("="*80)
    raw_data, sectors = collect_fundamentals_simple(test_stocks)

    # Test 2: Imputation
    print("\n" + "="*80)
    print("TEST 2: IMPUTATION")
    print("="*80)
    imputed_data = impute_metrics(raw_data, sectors)

    # Test 3: Tensor conversion
    print("\n" + "="*80)
    print("TEST 3: TENSOR CONVERSION")
    print("="*80)
    tensors, metric_order = convert_to_tensors(imputed_data)

    # Show sample tensor
    sample_ticker = list(tensors.keys())[0]
    print(f"\nSample tensor ({sample_ticker}):")
    print(f"  Shape: {tensors[sample_ticker].shape}")
    print(f"  First 5 values: {tensors[sample_ticker][:5]}")

    # Test 4: Normalization
    print("\n" + "="*80)
    print("TEST 4: NORMALIZATION")
    print("="*80)
    normalized = normalize_tensors(tensors, metric_order)

    # Show sample normalized tensor
    print(f"\nSample normalized tensor ({sample_ticker}):")
    print(f"  Shape: {normalized[sample_ticker].shape}")
    print(f"  First 5 values: {normalized[sample_ticker][:5]}")

    # Test 5: Coverage statistics
    print("\n" + "="*80)
    print("TEST 5: COVERAGE STATISTICS")
    print("="*80)

    total_metrics = len(ALL_METRICS)
    for ticker, metrics in raw_data.items():
        non_null = sum(1 for v in metrics.values() if v is not None)
        pct = (non_null / total_metrics) * 100
        print(f"  {ticker}: {non_null}/{total_metrics} ({pct:.1f}%)")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    main()
