#!/usr/bin/env python3
"""
Example runner for statistical comparison.

Quick usage:
    python inference/run_statistical_comparison.py
    python inference/run_statistical_comparison.py --device cuda:1
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Run statistical comparison')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (e.g., cuda, cuda:0, cuda:1, cpu)')
    args = parser.parse_args()

    # Default configuration
    cmd = [
        sys.executable,
        'inference/statistical_comparison.py',
        '--data', './data/all_complete_dataset.h5',
        '--prices', './data/actual_prices.h5',
        '--model', './checkpoints/best_model_outlier.pt',
        '--bin-edges', './data/adaptive_bin_edges.pt',
        '--num-trials', '30',
        '--num-random-per-trial', '30',
        '--subset-size', '512',
        '--top-k', '5',
        '--horizon-idx', '0',  # 5-day
        '--confidence-percentile', '0.002',  # Keep top 20% by confidence
        '--test-months', '3',
        '--initial-capital', '100000',
        '--batch-size', '128',
        '--output', 'statistical_comparison_results.pt',
        '--seed', '42',
        '--device', args.device,
    ]

    print("Running statistical comparison...")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd)

if __name__ == '__main__':
    main()
