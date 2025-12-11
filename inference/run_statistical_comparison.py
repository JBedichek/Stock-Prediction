#!/usr/bin/env python3
"""
Example runner for statistical comparison.

Quick usage:
    python inference/run_statistical_comparison.py
"""

import subprocess
import sys

def main():
    # Default configuration
    cmd = [
        sys.executable,
        'inference/statistical_comparison.py',
        '--data', './data/all_complete_dataset.h5',
        '--prices', './data/actual_prices.h5',
        '--model', './checkpoints/best_model_selection_aware.pt',
        '--bin-edges', './data/adaptive_bin_edges.pt',
        '--num-trials', '20',
        '--num-random-per-trial', '25',
        '--subset-size', '256',
        '--top-k', '5',
        '--horizon-idx', '0',  # 5-day
        '--confidence-percentile', '0.002',  # Keep top 20% by confidence
        '--test-months', '3',
        '--initial-capital', '100000',
        '--batch-size', '128',
        '--output', 'statistical_comparison_results.pt',
        '--seed', '42',
    ]

    print("🚀 Running statistical comparison...")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd)

if __name__ == '__main__':
    main()
