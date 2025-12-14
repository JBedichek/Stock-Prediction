#!/usr/bin/env python3
"""
Example usage of the backtesting simulation.

This script shows different ways to run backtests:
1. Full test set (1000 stocks)
2. Small subset (100 stocks)
3. Different holding periods
4. Different top-k strategies
"""

import subprocess
import sys


def run_backtest(args_dict):
    """Run backtest with given arguments."""
    cmd = [sys.executable, 'inference/backtest_simulation.py']

    for key, value in args_dict.items():
        cmd.append(f'--{key}')
        if value is not None:
            cmd.append(str(value))

    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def main():
    # Example 1: Full test set (1000 stocks), 5-day horizon
    print("\n" + "="*80)
    print("EXAMPLE 1: Full test set (1000 stocks), 5-day horizon, top-10")
    print("="*80)

    run_backtest({
        'data': 'all_complete_dataset.pkl',
        'model': 'checkpoints/best_model.pt',
        'bin-edges': 'adaptive_bin_edges.pt',
        'num-test-stocks': 1000,
        'top-k': 10,
        'horizon-idx': 1,  # 5-day prediction (auto-sets horizon-days=5)
        'test-months': 2,
        'initial-capital': 100000,
        'batch-size': 128,  # Larger batch for better GPU utilization
        'output': 'backtest_full_5day.pt'
    })

    # Example 2: Small subset (100 stocks) for quick testing
    print("\n" + "="*80)
    print("EXAMPLE 2: Small subset (100 stocks), 5-day horizon, top-5")
    print("="*80)

    run_backtest({
        'data': 'all_complete_dataset.pkl',
        'model': 'checkpoints/best_model.pt',
        'bin-edges': 'adaptive_bin_edges.pt',
        'num-test-stocks': 1000,
        'subset-size': 100,  # Random 100 stocks
        'top-k': 5,
        'horizon-idx': 1,  # 5-day prediction (auto-sets horizon-days=5)
        'test-months': 2,
        'initial-capital': 50000,
        'batch-size': 64,
        'output': 'backtest_subset100_5day.pt'
    })

    # Example 3: 1-day trading (day trading)
    print("\n" + "="*80)
    print("EXAMPLE 3: Day trading strategy (1-day horizon), 100 stocks, top-3")
    print("="*80)

    run_backtest({
        'data': 'all_complete_dataset.pkl',
        'model': 'checkpoints/best_model.pt',
        'bin-edges': 'adaptive_bin_edges.pt',
        'num-test-stocks': 1000,
        'subset-size': 100,
        'top-k': 3,
        'horizon-idx': 0,  # 1-day prediction (auto-sets horizon-days=1)
        'test-months': 2,
        'initial-capital': 50000,
        'batch-size': 64,
        'output': 'backtest_daytrading.pt'
    })

    # Example 4: Long-term holding (20-day horizon)
    print("\n" + "="*80)
    print("EXAMPLE 4: Long-term strategy (20-day horizon), 100 stocks, top-5")
    print("="*80)

    run_backtest({
        'data': 'all_complete_dataset.pkl',
        'model': 'checkpoints/best_model.pt',
        'bin-edges': 'adaptive_bin_edges.pt',
        'num-test-stocks': 1000,
        'subset-size': 100,
        'top-k': 5,
        'horizon-idx': 3,  # 20-day prediction (auto-sets horizon-days=20)
        'test-months': 2,
        'initial-capital': 50000,
        'batch-size': 64,
        'output': 'backtest_longterm.pt'
    })

    print("\n" + "="*80)
    print("✅ ALL EXAMPLES COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - backtest_full_5day.pt")
    print("  - backtest_subset100_5day.pt")
    print("  - backtest_daytrading.pt")
    print("  - backtest_longterm.pt")
    print("\nYou can load these with: torch.load('backtest_*.pt')")
    print()


if __name__ == '__main__':
    main()
