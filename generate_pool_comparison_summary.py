#!/usr/bin/env python3
"""
Generate Pool Size Comparison Summary Graphs

Reads results from pool size comparison runs and generates summary visualizations.

Usage:
    python generate_pool_comparison_summary.py
    python generate_pool_comparison_summary.py --output-base output/pool_size_comparison
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


def load_pool_results(output_base, pool_sizes):
    """Load results for all pool sizes."""
    model_means = []
    model_stds = []
    random_means = []
    random_stds = []
    valid_pool_sizes = []

    for pool_size in pool_sizes:
        results_path = f"{output_base}/pool_{pool_size}/results.pt"

        if not os.path.exists(results_path):
            print(f"⚠️  Warning: Results not found for pool size {pool_size}")
            print(f"   Expected: {results_path}")
            continue

        print(f"✓ Loading results for pool size {pool_size}")
        results = torch.load(results_path, weights_only=False)
        trial_results = results['trial_results']

        # Extract final capital from all trials
        model_finals = [trial['model']['final_capital'] for trial in trial_results]
        random_finals_all = []
        for trial in trial_results:
            for random_result in trial['random']:
                random_finals_all.append(random_result['final_capital'])

        valid_pool_sizes.append(pool_size)
        model_means.append(np.mean(model_finals))
        model_stds.append(np.std(model_finals))
        random_means.append(np.mean(random_finals_all))
        random_stds.append(np.std(random_finals_all))

    return valid_pool_sizes, model_means, model_stds, random_means, random_stds


def generate_summary_graph(pool_sizes, model_means, model_stds, random_means, random_stds,
                           initial_capital, output_base):
    """Generate main summary graph with portfolio values and returns."""
    print("\n📊 Generating summary graph...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean final portfolio value
    x_pos = np.arange(len(pool_sizes))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, model_means, width, yerr=model_stds,
                   label='Model', capsize=5, color='#2ecc71', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, random_means, width, yerr=random_stds,
                   label='Random', capsize=5, color='#95a5a6', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Pool Size (Number of Stocks)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Final Portfolio Value ($)', fontsize=13, fontweight='bold')
    ax1.set_title('Final Portfolio Value vs Pool Size\n(Mean ± Std Dev)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pool_sizes, fontsize=11)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Add horizontal line for initial capital
    ax1.axhline(y=initial_capital, color='red', linestyle='--', linewidth=2,
               label=f'Initial Capital (${initial_capital:,.0f})', alpha=0.7)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Return percentage
    model_returns = [(m / initial_capital - 1) * 100 for m in model_means]
    random_returns = [(r / initial_capital - 1) * 100 for r in random_means]
    model_return_stds = [(s / initial_capital) * 100 for s in model_stds]
    random_return_stds = [(s / initial_capital) * 100 for s in random_stds]

    bars3 = ax2.bar(x_pos - width/2, model_returns, width, yerr=model_return_stds,
                   label='Model', capsize=5, color='#2ecc71', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x_pos + width/2, random_returns, width, yerr=random_return_stds,
                   label='Random', capsize=5, color='#95a5a6', alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Pool Size (Number of Stocks)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total Return (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Total Return vs Pool Size\n(Mean ± Std Dev)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pool_sizes, fontsize=11)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    summary_path = f"{output_base}/pool_size_comparison_summary.png"
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    print(f"✅ Summary graph saved to: {summary_path}")
    plt.close()

    return model_returns, random_returns


def generate_statistics_table(pool_sizes, model_means, model_stds, random_means, random_stds,
                               model_returns, random_returns, output_base):
    """Generate detailed statistics table."""
    print("📊 Generating statistics table...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Prepare table data
    table_data = [['Pool Size', 'Model Mean', 'Model Std', 'Random Mean', 'Random Std',
                   'Model Return', 'Improvement']]

    for i, pool_size in enumerate(pool_sizes):
        model_return = model_returns[i]
        random_return = random_returns[i]
        improvement = model_return - random_return

        table_data.append([
            str(pool_size),
            f'${model_means[i]:,.0f}',
            f'${model_stds[i]:,.0f}',
            f'${random_means[i]:,.0f}',
            f'${random_stds[i]:,.0f}',
            f'{model_return:+.2f}%',
            f'{improvement:+.2f}%'
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            cell.set_edgecolor('gray')
            cell.set_linewidth(0.5)

    ax.set_title('Pool Size Comparison - Detailed Statistics',
                fontsize=16, fontweight='bold', pad=20)

    table_path = f"{output_base}/pool_size_statistics_table.png"
    plt.savefig(table_path, dpi=200, bbox_inches='tight')
    print(f"✅ Statistics table saved to: {table_path}")
    plt.close()


def print_summary_statistics(pool_sizes, model_means, model_stds, random_means, random_stds,
                             model_returns, random_returns):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for i, pool_size in enumerate(pool_sizes):
        print(f"\nPool Size: {pool_size}")
        print(f"  Model:  Mean = ${model_means[i]:>10,.0f},  Std = ${model_stds[i]:>10,.0f},  Return = {model_returns[i]:>+7.2f}%")
        print(f"  Random: Mean = ${random_means[i]:>10,.0f},  Std = ${random_stds[i]:>10,.0f},  Return = {random_returns[i]:>+7.2f}%")
        print(f"  Improvement: {model_returns[i] - random_returns[i]:+.2f}%")

    # Find best pool size
    best_idx = np.argmax(model_returns)
    print("\n" + "="*80)
    print("BEST PERFORMING POOL SIZE")
    print("="*80)
    print(f"Pool Size: {pool_sizes[best_idx]}")
    print(f"  Model Return: {model_returns[best_idx]:+.2f}%")
    print(f"  Improvement over Random: {model_returns[best_idx] - random_returns[best_idx]:+.2f}%")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate pool size comparison summary graphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--output-base', type=str,
                       default='output/pool_size_comparison',
                       help='Base directory containing pool size results')
    parser.add_argument('--pool-sizes', type=int, nargs='+',
                       default=[50, 100, 256, 512, 1024, 1500],
                       help='Pool sizes to include in comparison')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital used in simulations')

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     Pool Size Comparison Summary Generator                    ║")
    print("╔════════════════════════════════════════════════════════════════╗")
    print(f"\nOutput base: {args.output_base}")
    print(f"Pool sizes: {args.pool_sizes}")
    print(f"Initial capital: ${args.initial_capital:,.0f}\n")

    # Load results
    pool_sizes, model_means, model_stds, random_means, random_stds = \
        load_pool_results(args.output_base, args.pool_sizes)

    if not pool_sizes:
        print("\n❌ No valid results found!")
        print(f"   Check that results exist in: {args.output_base}/pool_*/results.pt")
        return 1

    print(f"\n✓ Loaded results for {len(pool_sizes)} pool sizes: {pool_sizes}")

    # Generate summary graph
    model_returns, random_returns = generate_summary_graph(
        pool_sizes, model_means, model_stds, random_means, random_stds,
        args.initial_capital, args.output_base
    )

    # Generate statistics table
    generate_statistics_table(
        pool_sizes, model_means, model_stds, random_means, random_stds,
        model_returns, random_returns, args.output_base
    )

    # Print summary to console
    print_summary_statistics(
        pool_sizes, model_means, model_stds, random_means, random_stds,
        model_returns, random_returns
    )

    print("\n✅ Summary generation complete!")
    return 0


if __name__ == '__main__':
    exit(main())
