#!/usr/bin/env python3
"""
Fix Variable Feature Dimensions in Dataset

Some stocks have different feature dimensions due to missing fundamental metrics.
This script pads all tensors to the maximum dimension found in the dataset.

Usage:
    python dataset_creation/fix_variable_dimensions.py \
        --input all_complete_dataset.pkl \
        --output all_complete_dataset_fixed.pkl
"""

import torch
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import pic_load, save_pickle


def analyze_dimensions(data):
    """
    Analyze feature dimensions across all stocks.

    Returns:
        max_features: Maximum feature dimension
        min_features: Minimum feature dimension
        dimension_counts: Dict mapping dimension -> count of stocks
    """
    print("\n" + "="*80)
    print("ANALYZING FEATURE DIMENSIONS")
    print("="*80)

    max_features = 0
    min_features = float('inf')
    dimension_counts = {}

    for ticker, date_dict in tqdm(data.items(), desc="Scanning"):
        if date_dict:
            # Get dimension from first date
            sample_date = list(date_dict.keys())[0]
            num_features = date_dict[sample_date].shape[0]

            max_features = max(max_features, num_features)
            min_features = min(min_features, num_features)

            dimension_counts[num_features] = dimension_counts.get(num_features, 0) + 1

    print(f"\n📊 Dimension Analysis:")
    print(f"  Maximum features: {max_features}")
    print(f"  Minimum features: {min_features}")
    print(f"  Difference: {max_features - min_features}")

    print(f"\n📈 Dimension Distribution:")
    for dim in sorted(dimension_counts.keys()):
        count = dimension_counts[dim]
        pct = count / len(data) * 100
        print(f"  {dim} features: {count} stocks ({pct:.1f}%)")

    return max_features, min_features, dimension_counts


def pad_dataset(data, target_dim):
    """
    Pad all tensors to target dimension.

    Args:
        data: Dataset dict {ticker: {date: tensor}}
        target_dim: Target feature dimension

    Returns:
        padded_data: Dataset with all tensors padded to target_dim
    """
    print("\n" + "="*80)
    print("PADDING TENSORS")
    print("="*80)
    print(f"Target dimension: {target_dim}")

    padded_data = {}
    num_padded = 0

    for ticker, date_dict in tqdm(data.items(), desc="Padding"):
        padded_data[ticker] = {}

        for date, tensor in date_dict.items():
            current_dim = tensor.shape[0]

            if current_dim < target_dim:
                # Pad with zeros
                padding = torch.zeros(target_dim - current_dim)
                padded_tensor = torch.cat([tensor, padding])
                num_padded += 1
            elif current_dim == target_dim:
                # Already correct size
                padded_tensor = tensor
            else:
                # Should never happen, but handle it
                print(f"\n⚠️  WARNING: {ticker} has {current_dim} features > target {target_dim}")
                padded_tensor = tensor[:target_dim]

            padded_data[ticker][date] = padded_tensor

    print(f"\n✅ Padding complete!")
    print(f"  Total tensors padded: {num_padded:,}")
    print(f"  All tensors now have {target_dim} features")

    return padded_data


def verify_dimensions(data):
    """Verify all tensors have the same dimension."""
    print("\n" + "="*80)
    print("VERIFYING DIMENSIONS")
    print("="*80)

    all_dims = set()

    for ticker, date_dict in tqdm(data.items(), desc="Verifying"):
        for date, tensor in date_dict.items():
            all_dims.add(tensor.shape[0])

    if len(all_dims) == 1:
        dim = list(all_dims)[0]
        print(f"\n✅ SUCCESS: All tensors have {dim} features")
        return True
    else:
        print(f"\n❌ FAILED: Found {len(all_dims)} different dimensions: {sorted(all_dims)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix variable feature dimensions in dataset')

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input pickle file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output pickle file')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only analyze dimensions, do not fix')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("FIX VARIABLE FEATURE DIMENSIONS")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Load data
    print("\n📦 Loading dataset...")
    data = pic_load(args.input)
    print(f"  ✅ Loaded {len(data)} tickers")

    # Analyze dimensions
    max_features, min_features, dimension_counts = analyze_dimensions(data)

    if args.verify_only:
        print("\n✅ Verification complete (no changes made)")
        return

    if max_features == min_features:
        print("\n✅ All tensors already have the same dimension - no fixes needed!")
        return

    # Pad dataset
    padded_data = pad_dataset(data, max_features)

    # Verify
    if not verify_dimensions(padded_data):
        print("\n❌ Verification failed! Not saving.")
        return

    # Save
    print(f"\n💾 Saving fixed dataset to: {args.output}")
    save_pickle(padded_data, args.output)

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Verify the fixed dataset works:")
    print(f"     python training/new_data_loader.py {args.output}")
    print(f"  2. Train with the fixed dataset:")
    print(f"     python -m training.train_new_format --data {args.output}")
    print()


if __name__ == '__main__':
    main()
