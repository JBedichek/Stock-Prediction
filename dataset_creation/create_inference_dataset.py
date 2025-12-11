#!/usr/bin/env python3
"""
Create Inference Dataset for the Past Year

This script creates a dataset for inference that includes all stocks over the past year.
It reuses the logic from dataset_processing.py by calling QTrainingData with the
inference parameter.

Usage:
    python -m dataset_creation.create_inference_dataset \
        --bulk-prices <path_to_bulk_prices.pickle> \
        --summaries <path_to_summaries.pickle> \
        --sectors <path_to_sectors> \
        --data-dict <path_to_data_dict> \
        [--seq-len 600] \
        [--months-back 12]
"""

import argparse
import datetime
import torch
from pathlib import Path

from dataset_creation.dataset_processing import QTrainingData


def get_date_range(months_back=12):
    """
    Calculate start and end dates for the dataset.

    Args:
        months_back: Number of months to go back from today

    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    end_date = datetime.date.today()

    # Calculate start date (approximately months_back months ago)
    # Using 30-day months for simplicity
    start_date = end_date - datetime.timedelta(days=30 * months_back)

    return start_date, end_date


def format_date_short(date):
    """
    Format date as MM-DD-YY (short format).

    Args:
        date: datetime.date object

    Returns:
        str: Date formatted as MM-DD-YY
    """
    return date.strftime("%m-%d-%y")


def create_inference_dataset(
    bulk_prices_pth,
    summaries_pth,
    sectors_pth,
    data_dict_pth,
    seq_len=600,
    months_back=12,
    inf_company_keep_rate=1.0,
    output_dir="."
):
    """
    Create an inference dataset for the past year.

    Args:
        bulk_prices_pth: Path to bulk prices pickle file
        summaries_pth: Path to company summaries pickle file
        sectors_pth: Path to sectors dictionary
        data_dict_pth: Path to data dictionary (without .pt extension)
        seq_len: Sequence length for the model
        months_back: Number of months to include in the dataset
        inf_company_keep_rate: Proportion of companies to keep (1.0 = all companies)
        output_dir: Directory to save the output file
    """
    # Calculate date range
    start_date, end_date = get_date_range(months_back)

    print(f"Creating inference dataset from {start_date} to {end_date}")
    print(f"  Bulk prices: {bulk_prices_pth}")
    print(f"  Summaries: {summaries_pth}")
    print(f"  Sectors: {sectors_pth}")
    print(f"  Data dict: {data_dict_pth}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Company keep rate: {inf_company_keep_rate * 100}%")
    print()

    # Create dataset with inference mode
    print("Initializing QTrainingData with inference mode...")
    dataset = QTrainingData(
        summaries_pth=summaries_pth,
        bulk_prices_pth=bulk_prices_pth,
        seq_len=seq_len,
        n=5,  # Default trade period length
        k=1,  # How often to take a point from sequence
        c_prop=1.0,  # Use all companies
        full=False,  # Don't prepare training dataset
        load=True,  # Load existing DataDict
        inference=(start_date, end_date),  # This triggers inference mode
        sectors_pth=sectors_pth,
        data_slice_num=0,
        data_dict_pth=data_dict_pth,
        inf_company_keep_rate=inf_company_keep_rate
    )

    # Format dates for filename
    start_str = format_date_short(start_date)
    end_str = format_date_short(end_date)

    # Create output filename
    output_filename = f"inference_dataset_{start_str}-{end_str}.pt"
    output_path = Path(output_dir) / output_filename

    # Save the inference dataset
    print(f"\nSaving inference dataset to: {output_path}")
    torch.save(dataset.inference_data, str(output_path))

    # Print dataset statistics
    num_dates = len(dataset.inference_data)
    if num_dates > 0:
        sample_date = list(dataset.inference_data.keys())[0]
        num_companies = len(dataset.inference_data[sample_date])
        print(f"\n✅ Dataset created successfully!")
        print(f"  Number of dates: {num_dates}")
        print(f"  Number of companies: {num_companies}")
        print(f"  Output file: {output_path}")
        print(f"  File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    else:
        print("\n⚠️  Warning: Dataset is empty!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create inference dataset for the past year",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--bulk-prices",
        type=str,
        required=True,
        help="Path to bulk prices pickle file"
    )

    parser.add_argument(
        "--summaries",
        type=str,
        required=True,
        help="Path to company summaries pickle file"
    )

    parser.add_argument(
        "--sectors",
        type=str,
        required=True,
        help="Path to sectors dictionary"
    )

    parser.add_argument(
        "--data-dict",
        type=str,
        required=True,
        help="Path to data dictionary (without .pt extension)"
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=600,
        help="Sequence length for the model"
    )

    parser.add_argument(
        "--months-back",
        type=int,
        default=12,
        help="Number of months to go back from today"
    )

    parser.add_argument(
        "--company-keep-rate",
        type=float,
        default=1.0,
        help="Proportion of companies to keep (0.0 to 1.0, 1.0 = all companies)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save the output file"
    )

    args = parser.parse_args()

    # Validate company keep rate
    if not 0.0 < args.company_keep_rate <= 1.0:
        parser.error("--company-keep-rate must be between 0.0 and 1.0")

    # Create the dataset
    create_inference_dataset(
        bulk_prices_pth=args.bulk_prices,
        summaries_pth=args.summaries,
        sectors_pth=args.sectors,
        data_dict_pth=args.data_dict,
        seq_len=args.seq_len,
        months_back=args.months_back,
        inf_company_keep_rate=args.company_keep_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
