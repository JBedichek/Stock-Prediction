#!/usr/bin/env python3
"""
Quick runner for selection-aware training.

Usage:
    python training/run_selection_aware_training.py
"""

import subprocess
import sys

def main():
    cmd = [
        sys.executable,
        'training/selection_aware_training.py',
        '--data', 'all_complete_dataset.h5',
        '--sample-size', '10000',
        '--top-k', '500',
        '--train-epochs-per-iteration', '3',
        '--num-iterations', '100',
        '--horizon-idx', '1',  # 5-day horizon
        '--lr', '4e-5',
        '--batch-size', '32',
        '--inference-batch-size', '256',  # Increased for better GPU utilization during inference
        '--hidden-dim', '1024',
        '--num-layers', '16',
        '--num-heads', '16',
        '--val-every', '5',
        '--use-amp',
        '--compile',  # Enable torch.compile for faster inference
        '--use-wandb'
    ]

    print("🚀 Running selection-aware training...")
    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd)

if __name__ == '__main__':
    main()
