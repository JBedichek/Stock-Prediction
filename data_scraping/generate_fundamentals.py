"""
Generate and save fundamental metrics for stock datasets.

This script creates fundamental metrics pickle files that can be loaded
during training, similar to how company summaries are loaded.

Usage:
    python generate_fundamentals.py --dataset a_lot_of_stocks --output a_lot_fundamentals.pkl
"""

import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .Stock import stock_info, a_lot_of_stocks, s_lot_of_stocks
import argparse
import os
from utils.utils import save_pickle


def generate_fundamentals_for_dataset(stock_dict, dataset_name, output_path=None, use_cache=True):
    """
    Generate fundamental metrics for a stock dataset.

    Args:
        stock_dict: Dictionary of {ticker: company_name}
        dataset_name: Name of the dataset (for stock_info initialization)
        output_path: Path to save the pickle file. If None, uses default naming
        use_cache: Whether to use cached yfinance data

    Returns:
        dict: {ticker: torch.Tensor of shape (27,)} with normalized fundamental metrics
    """
    print("="*80)
    print(f"GENERATING FUNDAMENTAL METRICS FOR {dataset_name}")
    print("="*80)
    print(f"\nDataset: {len(stock_dict)} stocks")

    # Create stock_info instance (will initialize RoBERTa encoder, but we don't use it here)
    print("\nInitializing stock_info...")
    try:
        si = stock_info(stock_dict, dataset_name=dataset_name)
    except Exception as e:
        print(f"Warning: Could not initialize with CUDA. Error: {e}")
        print("Note: This is OK - we only need fundamental metrics, not text embeddings")
        # We'll work around this by creating a minimal version
        return generate_fundamentals_standalone(stock_dict, dataset_name, output_path, use_cache)

    # Generate fundamental metrics
    print("\nCollecting fundamental metrics...")
    cache_file = f'/home/james/Desktop/Stock-Prediction/{dataset_name}_fundamentals_cache.pkl'

    # Get normalized tensors
    fundamentals = si.get_fundamental_metrics_as_tensor(normalize=True)

    if fundamentals is None or len(fundamentals) == 0:
        print("ERROR: No fundamental metrics were generated!")
        return None

    print(f"\n✅ Generated fundamental metrics for {len(fundamentals)} stocks")

    # Show sample
    sample_ticker = list(fundamentals.keys())[0]
    print(f"\nSample ({sample_ticker}):")
    print(f"  Shape: {fundamentals[sample_ticker].shape}")
    print(f"  First 5 values: {fundamentals[sample_ticker][:5]}")

    # Save to pickle
    if output_path is None:
        output_path = f'/home/james/Desktop/Stock-Prediction/{dataset_name}_fundamentals.pkl'

    print(f"\nSaving to: {output_path}")
    save_pickle(fundamentals, output_path)

    print(f"✅ Saved {len(fundamentals)} fundamental metric tensors")

    return fundamentals


def generate_fundamentals_standalone(stock_dict, dataset_name, output_path=None, use_cache=True):
    """
    Generate fundamentals without needing the full stock_info class.
    This is a workaround for systems without CUDA.
    """
    import yfinance as yf
    import torch
    import numpy as np
    from tqdm import tqdm

    print("\nUsing standalone fundamental generation (no CUDA required)...")

    # Metric definitions (from Stock.py)
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
    METRIC_ORDER = list(ALL_METRICS.keys())

    # Collect raw data
    print("\nCollecting raw data from yfinance...")
    raw_data = {}
    sectors = {}

    for ticker_symbol, company_name in tqdm(stock_dict.items(), desc="Fetching"):
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            metrics = {}
            for metric_key in ALL_METRICS.keys():
                value = info.get(metric_key, None)
                metrics[metric_key] = value

            raw_data[ticker_symbol] = metrics
            sectors[ticker_symbol] = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
            }
        except Exception as e:
            raw_data[ticker_symbol] = {k: None for k in ALL_METRICS.keys()}
            sectors[ticker_symbol] = {'sector': 'Unknown', 'industry': 'Unknown'}

    # Simple imputation (use global medians)
    print("\nImputing missing values...")
    global_medians = {}
    for metric in METRIC_ORDER:
        # Filter out None and non-numeric values
        all_values = []
        for data in raw_data.values():
            val = data[metric]
            if val is not None and isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                all_values.append(val)

        if all_values:
            global_medians[metric] = np.median(all_values)
        else:
            global_medians[metric] = 0.0

    imputed_data = {}
    for ticker, metrics in raw_data.items():
        imputed_metrics = {}
        for metric, value in metrics.items():
            # Check if value is valid numeric
            if value is None or not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                imputed_metrics[metric] = global_medians[metric]
            else:
                imputed_metrics[metric] = value
        imputed_data[ticker] = imputed_metrics

    # Convert to tensors
    print("\nConverting to tensors...")
    tensors = {}
    for ticker, metrics in imputed_data.items():
        values = [metrics[m] for m in METRIC_ORDER]
        tensors[ticker] = torch.tensor(values, dtype=torch.float32)

    # Normalize
    print("\nNormalizing...")
    all_tensors = torch.stack(list(tensors.values()))
    all_numpy = all_tensors.numpy()

    medians = np.median(all_numpy, axis=0)
    q1 = np.percentile(all_numpy, 25, axis=0)
    q3 = np.percentile(all_numpy, 75, axis=0)
    iqr = q3 - q1
    iqr = np.where(iqr == 0, 1.0, iqr)

    normalized = {}
    for ticker, tensor in tensors.items():
        norm_values = (tensor.numpy() - medians) / iqr
        norm_values = np.clip(norm_values, -10, 10)
        normalized[ticker] = torch.tensor(norm_values, dtype=torch.float32)

    # Save
    if output_path is None:
        output_path = f'/home/james/Desktop/Stock-Prediction/{dataset_name}_fundamentals.pkl'

    print(f"\nSaving to: {output_path}")
    save_pickle(normalized, output_path)

    print(f"✅ Saved {len(normalized)} fundamental metric tensors")

    return normalized


def main():
    parser = argparse.ArgumentParser(description='Generate fundamental metrics for stock datasets')
    parser.add_argument('--dataset', type=str, default='a_lot',
                        help='Dataset to use: a_lot, s_lot, or custom')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for pickle file')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (re-fetch from yfinance)')

    args = parser.parse_args()

    # Select dataset
    if args.dataset == 'a_lot':
        stock_dict = a_lot_of_stocks
        dataset_name = 'a_lot_of_stocks'
    elif args.dataset == 's_lot':
        stock_dict = s_lot_of_stocks
        dataset_name = 's_lot_of_stocks'
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Available: a_lot, s_lot")
        return

    # Generate fundamentals
    fundamentals = generate_fundamentals_for_dataset(
        stock_dict,
        dataset_name,
        output_path=args.output,
        use_cache=not args.no_cache
    )

    if fundamentals is not None:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nGenerated fundamental metrics for {len(fundamentals)} stocks")
        print(f"Each stock has {fundamentals[list(fundamentals.keys())[0]].shape[0]} metrics")
    else:
        print("\n" + "="*80)
        print("FAILED")
        print("="*80)


if __name__ == "__main__":
    main()
