#!/usr/bin/env python3
"""
Download dataset and checkpoints from Hugging Face Hub.

This script downloads all required data for running the stock prediction models.

Prerequisites:
    pip install huggingface_hub tqdm

Usage:
    # Download everything (data + checkpoints)
    python scripts/download_data.py --repo-id username/stock-prediction-data

    # Download only data (skip checkpoints)
    python scripts/download_data.py --repo-id username/stock-prediction-data --data-only

    # Download only checkpoints
    python scripts/download_data.py --repo-id username/stock-prediction-data --checkpoints-only
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from tqdm import tqdm
except ImportError:
    print("Please install required packages:")
    print("  pip install huggingface_hub tqdm")
    sys.exit(1)


# Default repository
DEFAULT_REPO_ID = "JamesBedichek/stock-prediction-data"


def download_file(repo_id: str, filename: str, local_dir: str = "."):
    """Download a single file from Hugging Face Hub."""
    print(f"  Downloading {filename}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return path
    except Exception as e:
        print(f"    Error downloading {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download stock prediction data from Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id", type=str, default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Only download data files, not checkpoints"
    )
    parser.add_argument(
        "--checkpoints-only", action="store_true",
        help="Only download checkpoints, not data"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Output directory (default: current directory)"
    )

    args = parser.parse_args()

    if args.repo_id == DEFAULT_REPO_ID:
        print(f"Warning: Using default repo ID '{DEFAULT_REPO_ID}'")
        print("You may need to specify --repo-id with the actual repository.\n")

    print(f"Downloading from: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}\n")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # List available files
    print("Checking available files...")
    try:
        files = list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error accessing repository: {e}")
        print("\nMake sure:")
        print("  1. The repository exists")
        print("  2. You have access (run 'huggingface-cli login' for private repos)")
        sys.exit(1)

    # Categorize files
    data_files = [f for f in files if f.startswith("data/")]
    checkpoint_files = [f for f in files if f.startswith("checkpoints/")]

    print(f"  Found {len(data_files)} data files")
    print(f"  Found {len(checkpoint_files)} checkpoint files\n")

    # Determine what to download
    files_to_download = []

    if not args.checkpoints_only:
        files_to_download.extend(data_files)

    if not args.data_only:
        files_to_download.extend(checkpoint_files)

    if not files_to_download:
        print("No files to download!")
        return

    print(f"Downloading {len(files_to_download)} files...\n")

    # Download files
    successful = 0
    failed = 0

    for filename in files_to_download:
        result = download_file(args.repo_id, filename, args.output_dir)
        if result:
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    # Show file locations
    print(f"\nFiles downloaded to: {os.path.abspath(args.output_dir)}")

    # Check if main files exist and suggest next steps
    main_dataset = os.path.join(args.output_dir, "data", "all_complete_dataset.h5")
    prices_file_clean = os.path.join(args.output_dir, "data", "actual_prices_clean.h5")
    prices_file_raw = os.path.join(args.output_dir, "data", "actual_prices_raw.h5")

    if os.path.exists(main_dataset):
        print(f"\nMain dataset: {main_dataset}")

    if os.path.exists(prices_file_clean):
        print(f"Prices file (clean): {prices_file_clean}")
    if os.path.exists(prices_file_raw):
        print(f"Prices file (raw): {prices_file_raw}")

    # Create symlinks for convenience (so paths work without 'data/' prefix)
    if os.path.exists(main_dataset) and args.output_dir == ".":
        target = "all_complete_dataset.h5"
        if not os.path.exists(target):
            try:
                os.symlink("data/all_complete_dataset.h5", target)
                print(f"\nCreated symlink: {target} -> data/all_complete_dataset.h5")
            except:
                pass

    # Symlink cleaned prices as the default actual_prices.h5
    if os.path.exists(prices_file_clean) and args.output_dir == ".":
        target = "actual_prices.h5"
        if not os.path.exists(target):
            try:
                os.symlink("data/actual_prices_clean.h5", target)
                print(f"Created symlink: {target} -> data/actual_prices_clean.h5 (cleaned data)")
            except:
                pass

    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print("""
# Run evaluation on pre-trained checkpoints:
python -m inference.principled_evaluation \\
    --checkpoint-dir checkpoints/walk_forward \\
    --data data/all_complete_dataset.h5 \\
    --prices data/actual_prices_clean.h5 \\
    --sweep

# Train new model:
python -m training.walk_forward_training \\
    --data data/all_complete_dataset.h5 \\
    --prices data/actual_prices_clean.h5

Note: Use actual_prices_clean.h5 (not _raw.h5) for backtesting.
      The cleaned file has 155 problematic tickers removed.
""")


if __name__ == "__main__":
    main()
