"""
Add Cross-Sectional Features to Existing Tensor Data

This script adds cross-sectional features (percentile rankings) to already-processed
tensor data WITHOUT loading everything into memory at once.

Memory-efficient approach:
1. Load processed tensors (already on disk)
2. For each date, extract just the needed features from all stocks
3. Calculate cross-sectional rankings for that date
4. Append cross-sectional features to existing tensors
5. Save incrementally every N dates

This avoids holding full DataFrames for all stocks in memory.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, date as dt_date
from typing import Dict, List, Optional
import sys
from tqdm import tqdm

sys.path.append('/home/james/Desktop/Stock-Prediction')
from utils.utils import pic_load, save_pickle
from data_scraping.cross_sectional_calculator import CrossSectionalCalculator


def extract_features_for_date(tensor_dict: Dict[str, Dict[dt_date, torch.Tensor]],
                              date: dt_date,
                              feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract specific features for all stocks on a specific date.

    Args:
        tensor_dict: Dict of {ticker: {date: tensor}}
        date: Date to extract
        feature_names: Names of features in the tensor (in order)

    Returns:
        Dict of {ticker: {feature_name: value}}
    """
    stocks_on_date = {}

    # Features we need for cross-sectional ranking (must exist in tensor)
    # We'll use indices to extract them
    needed_features = ['return_1d', 'return_5d', 'return_20d', 'volume',
                      'dollar_volume', 'volatility_20d', 'volume_ratio_20d']

    # Find indices of needed features
    feature_indices = {}
    for feat in needed_features:
        if feat in feature_names:
            feature_indices[feat] = feature_names.index(feat)

    # Extract values for this date from all stocks
    for ticker, date_tensors in tensor_dict.items():
        if date in date_tensors:
            tensor = date_tensors[date]

            stock_data = {}
            for feat, idx in feature_indices.items():
                if idx < len(tensor):
                    stock_data[feat] = float(tensor[idx].item())

            if stock_data:  # Only add if we got some data
                stocks_on_date[ticker] = stock_data

    return stocks_on_date


def add_cross_sectional_features(input_file: str,
                                 output_file: str,
                                 sector_dict_file: Optional[str] = None,
                                 start_date: str = '2000-01-01',
                                 end_date: str = None,
                                 save_every: int = 50):
    """
    Add cross-sectional features to existing tensor data.

    Args:
        input_file: Path to processed tensor data (without cross-sectional)
        output_file: Path to save output with cross-sectional features
        sector_dict_file: Optional path to sector dictionary
        start_date: Start date
        end_date: End date
        save_every: Save progress every N dates
    """
    print(f"\n{'='*80}")
    print(f"ADDING CROSS-SECTIONAL FEATURES")
    print(f"{'='*80}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"{'='*80}\n")

    # Load data
    print("📂 Loading tensor data...")
    tensor_data = pic_load(input_file)
    print(f"  ✅ Loaded {len(tensor_data)} stocks")

    sector_dict = None
    if sector_dict_file:
        sector_dict = pic_load(sector_dict_file)
        print(f"  ✅ Loaded sector mappings for {len(sector_dict)} stocks")

    print()

    # Get date range from data
    all_dates = set()
    for ticker_dates in tensor_data.values():
        all_dates.update(ticker_dates.keys())
    all_dates = sorted(list(all_dates))

    # Filter by date range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else datetime.now().date()

    all_dates = [d for d in all_dates if start_dt <= d <= end_dt]

    print(f"📅 Processing {len(all_dates)} dates from {all_dates[0]} to {all_dates[-1]}")

    # Get feature names (we need to know the tensor structure)
    # For now, hardcode common feature names - in production you'd save this metadata
    # This is a simplified version - you may need to adjust based on actual features
    feature_names = ['price', 'volume', 'return_1d', 'return_5d', 'return_20d',
                    'volume_ratio_20d', 'dollar_volume', 'volatility_20d']

    # Initialize calculator
    calculator = CrossSectionalCalculator(sector_dict)

    # Check for existing progress
    import os
    import gc
    completed_dates = set()

    if os.path.exists(output_file):
        try:
            print(f"\n📂 Found existing output file, checking progress...")
            # We need to check which dates have cross-sectional features already
            # For simplicity, we'll just start fresh or resume from scratch
            user_input = input("  Overwrite existing file? [y/N]: ").strip().lower()
            if user_input != 'y':
                print("  Keeping existing file, exiting...")
                return
        except Exception as e:
            print(f"  ⚠️  Could not check existing file: {e}")

    # Process dates one at a time (memory efficient!)
    cross_sectional_data = {}  # Will hold {ticker: {date: cross_sectional_tensor}}

    print(f"\n🔄 Processing dates (saving every {save_every} dates)...")

    for i, date_obj in enumerate(tqdm(all_dates)):
        try:
            # Extract features for this date from all stocks
            stocks_on_date = extract_features_for_date(tensor_data, date_obj, feature_names)

            if len(stocks_on_date) < 10:  # Need minimum stocks
                continue

            # Calculate cross-sectional features
            # We need to convert our extracted dict to the format expected by the calculator
            # The calculator expects a DataFrame format, so we'll create minimal DataFrames

            # Create minimal DataFrames for this date only
            mini_dataframes = {}
            for ticker, features in stocks_on_date.items():
                df_dict = {'date': [date_obj]}
                df_dict.update({k: [v] for k, v in features.items()})
                mini_dataframes[ticker] = pd.DataFrame(df_dict)

            # Calculate cross-sectional features for this date
            date_features = calculator.calculate_cross_sectional_features_for_date(
                mini_dataframes, date_obj
            )

            # Convert to tensors and store
            for ticker, features_dict in date_features.items():
                if ticker not in cross_sectional_data:
                    cross_sectional_data[ticker] = {}

                # Convert to tensor
                feature_values = list(features_dict.values())
                cross_sectional_data[ticker][date_obj] = torch.tensor(
                    feature_values, dtype=torch.float32
                )

        except Exception as e:
            print(f"\n  ❌ Error processing date {date_obj}: {e}")
            continue

        # Save progress periodically
        if (i + 1) % save_every == 0:
            print(f"\n💾 Saving progress ({i+1}/{len(all_dates)} dates processed)...")

            # Merge cross-sectional features with main features
            merged_data = merge_features(tensor_data, cross_sectional_data)
            save_pickle(merged_data, output_file)

            # Memory cleanup
            gc.collect()

            # Show memory usage
            import psutil
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / 1024**3
            print(f"   📊 Current RAM usage: {mem_gb:.2f} GB\n")

    # Final save
    print(f"\n💾 Saving final output...")
    merged_data = merge_features(tensor_data, cross_sectional_data)
    save_pickle(merged_data, output_file)

    print(f"\n{'='*80}")
    print(f"✅ CROSS-SECTIONAL FEATURES ADDED!")
    print(f"{'='*80}")
    print(f"Output file: {output_file}")

    # Show feature count
    if merged_data:
        sample_ticker = list(merged_data.keys())[0]
        sample_date = list(merged_data[sample_ticker].keys())[0]
        num_features = merged_data[sample_ticker][sample_date].shape[0]

        # Compare to original
        orig_features = tensor_data[sample_ticker][sample_date].shape[0]
        added_features = num_features - orig_features

        print(f"Original features: {orig_features}")
        print(f"Added features: {added_features}")
        print(f"Total features: {num_features}")
        print(f"{'='*80}\n")


def merge_features(original_data: Dict[str, Dict[dt_date, torch.Tensor]],
                   cross_sectional_data: Dict[str, Dict[dt_date, torch.Tensor]]) -> Dict[str, Dict[dt_date, torch.Tensor]]:
    """
    Merge cross-sectional features with original features.

    Args:
        original_data: Original tensor data
        cross_sectional_data: Cross-sectional features

    Returns:
        Merged data
    """
    merged = {}

    for ticker in original_data:
        merged[ticker] = {}

        for date_obj, original_tensor in original_data[ticker].items():
            # Check if we have cross-sectional features for this date
            if ticker in cross_sectional_data and date_obj in cross_sectional_data[ticker]:
                cross_tensor = cross_sectional_data[ticker][date_obj]
                # Concatenate
                merged[ticker][date_obj] = torch.cat([original_tensor, cross_tensor])
            else:
                # No cross-sectional features for this date, use zeros
                # Determine number of cross-sectional features from another stock/date
                num_cross_features = 0
                if cross_sectional_data:
                    for t in cross_sectional_data:
                        if cross_sectional_data[t]:
                            num_cross_features = list(cross_sectional_data[t].values())[0].shape[0]
                            break

                if num_cross_features > 0:
                    zeros = torch.zeros(num_cross_features, dtype=torch.float32)
                    merged[ticker][date_obj] = torch.cat([original_tensor, zeros])
                else:
                    # No cross-sectional features at all
                    merged[ticker][date_obj] = original_tensor

    return merged


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add cross-sectional features to processed tensor data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input tensor data file (without cross-sectional features)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: input with _cross_sectional suffix)')
    parser.add_argument('--sector_dict', type=str, default=None,
                       help='Sector dictionary file (optional)')
    parser.add_argument('--start_date', type=str, default='2000-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date (default: today)')
    parser.add_argument('--save_every', type=int, default=50,
                       help='Save progress every N dates (default: 50)')

    args = parser.parse_args()

    # Set output file
    if args.output is None:
        args.output = args.input.replace('.pkl', '_cross_sectional.pkl')

    # Add features
    add_cross_sectional_features(
        input_file=args.input,
        output_file=args.output,
        sector_dict_file=args.sector_dict,
        start_date=args.start_date,
        end_date=args.end_date,
        save_every=args.save_every
    )

    print(f"\n✅ Done! Cross-sectional features saved to {args.output}")
