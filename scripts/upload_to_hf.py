#!/usr/bin/env python3
"""
Upload dataset and checkpoints to Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python scripts/upload_to_hf.py --repo-id your-username/stock-prediction-data

    # Upload only checkpoints
    python scripts/upload_to_hf.py --repo-id your-username/stock-prediction-data --checkpoints-only
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def get_files_to_upload():
    """Define files to upload with their HF paths."""
    files = []

    # Main dataset (26GB)
    if os.path.exists("all_complete_dataset.h5"):
        files.append(("all_complete_dataset.h5", "data/all_complete_dataset.h5"))

    # Cleaned prices for backtesting (preferred - 155 problematic tickers removed)
    # These tickers had issues like negative prices (yfinance bugs), untracked splits, etc.
    if os.path.exists("actual_prices_clean.h5"):
        files.append(("actual_prices_clean.h5", "data/actual_prices_clean.h5"))

    # Original prices (kept for reference, but use _clean version for backtesting)
    if os.path.exists("actual_prices.h5"):
        files.append(("actual_prices.h5", "data/actual_prices_raw.h5"))

    # Price data pickle (for re-extraction if needed)
    if os.path.exists("all_price_data_adjusted.pkl"):
        files.append(("all_price_data_adjusted.pkl", "data/all_price_data_adjusted.pkl"))

    return files


def get_checkpoint_files():
    """Get checkpoint files to upload."""
    files = []
    checkpoint_dir = Path("checkpoints/walk_forward")

    if checkpoint_dir.exists():
        for pt_file in checkpoint_dir.glob("fold_*_best.pt"):
            files.append((str(pt_file), f"checkpoints/walk_forward/{pt_file.name}"))

    return files


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True,
                       help="HuggingFace repo ID (e.g., username/stock-prediction-data)")
    parser.add_argument("--checkpoints-only", action="store_true",
                       help="Only upload checkpoints, not data")
    parser.add_argument("--data-only", action="store_true",
                       help="Only upload data, not checkpoints")
    parser.add_argument("--private", action="store_true",
                       help="Create private repository")
    parser.add_argument("--yes", "-y", action="store_true",
                       help="Skip confirmation prompt")

    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/accessing repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True
        )
        print(f"  Repository ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"  Note: {e}")

    # Collect files to upload
    files_to_upload = []

    if not args.checkpoints_only:
        files_to_upload.extend(get_files_to_upload())

    if not args.data_only:
        files_to_upload.extend(get_checkpoint_files())

    if not files_to_upload:
        print("No files found to upload!")
        return

    print(f"\nFiles to upload:")
    total_size = 0
    for local_path, hf_path in files_to_upload:
        size = os.path.getsize(local_path) / (1024**3)  # GB
        total_size += size
        print(f"  {local_path} -> {hf_path} ({size:.2f} GB)")

    print(f"\nTotal: {total_size:.2f} GB")

    if not args.yes:
        confirm = input("\nProceed with upload? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    # Upload files
    print("\nUploading files...")
    for local_path, hf_path in files_to_upload:
        print(f"\n  Uploading {local_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=hf_path,
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print(f"    Done!")
        except Exception as e:
            print(f"    Error: {e}")

    # Create a README for the dataset
    readme_content = f"""---
license: mit
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - finance
  - stocks
  - time-series
  - trading
size_categories:
  - 10B<n<100B
---

# Stock Prediction Dataset

This dataset contains preprocessed stock market data for training stock return prediction models.

## Contents

- `data/all_complete_dataset.h5` - Main features dataset (HDF5)
  - ~4000 stocks
  - 20+ years of daily data
  - Technical indicators, fundamentals, news sentiment

- `data/actual_prices_clean.h5` - **Cleaned price data for backtesting (RECOMMENDED)**
  - Split-adjusted close prices
  - 3,424 tickers with validated data quality
  - 155 problematic tickers removed (negative prices, untracked splits, extreme returns)

- `data/actual_prices_raw.h5` - Original raw price data (for reference only)
  - May contain data quality issues from yfinance bugs and untracked corporate actions

- `checkpoints/walk_forward/` - Pre-trained model checkpoints
  - 6-fold walk-forward validated models
  - Transformer-based architecture

## Usage

```python
from huggingface_hub import hf_hub_download

# Download the main dataset
hf_hub_download(
    repo_id="{args.repo_id}",
    filename="data/all_complete_dataset.h5",
    repo_type="dataset",
    local_dir="."
)
```

Or use the provided download script:
```bash
python scripts/download_data.py --repo-id {args.repo_id}
```

## License

MIT
"""

    # Upload README
    readme_path = "/tmp/README_dataset.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print("\nUploading README...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"{'='*60}")
    print(f"\nDataset URL: https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nUsers can download with:")
    print(f"  python scripts/download_data.py --repo-id {args.repo_id}")


if __name__ == "__main__":
    main()
