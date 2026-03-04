#!/usr/bin/env python3
"""
Analyze ablation study results.

Usage:
    python sanity_checks/analyze_ablation.py checkpoints/ablation_num_layers/
"""

import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict


def parse_stats_log(stats_path: str) -> list:
    """Extract IC and IR values from stats.log file."""
    results = []

    if not os.path.exists(stats_path):
        return results

    with open(stats_path, 'r') as f:
        content = f.read()

    # Find all fold results
    fold_pattern = r'FOLD (\d+)/(\d+) EVALUATION RESULTS.*?Mean IC:\s+([\+\-]?\d+\.\d+).*?Information Ratio:([\+\-]?\d+\.\d+)'
    matches = re.findall(fold_pattern, content, re.DOTALL)

    for match in matches:
        fold_num, total_folds, ic, ir = match
        results.append({
            'fold': int(fold_num),
            'ic': float(ic),
            'ir': float(ir)
        })

    return results


def analyze_ablation_dir(ablation_dir: str):
    """Analyze all runs in an ablation directory."""
    ablation_path = Path(ablation_dir)

    if not ablation_path.exists():
        print(f"Error: Directory not found: {ablation_dir}")
        return

    # Find all checkpoint subdirectories
    results = {}

    for subdir in sorted(ablation_path.iterdir()):
        if not subdir.is_dir():
            continue

        stats_log = subdir / 'stats.log'
        if stats_log.exists():
            fold_results = parse_stats_log(str(stats_log))
            if fold_results:
                # Extract parameter value from directory name
                param_value = subdir.name.split('__')[0]  # e.g., "num-layers_2"
                results[param_value] = fold_results

    if not results:
        print("No results found. Check if training has completed any folds.")
        return

    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\n{'Parameter':<20} {'Folds':<8} {'Mean IC':<12} {'Std IC':<12} {'Mean IR':<12}")
    print("-" * 64)

    summary = []
    for param_value, folds in sorted(results.items()):
        ics = [f['ic'] for f in folds]
        irs = [f['ir'] for f in folds]

        mean_ic = sum(ics) / len(ics)
        std_ic = (sum((x - mean_ic) ** 2 for x in ics) / len(ics)) ** 0.5
        mean_ir = sum(irs) / len(irs)

        print(f"{param_value:<20} {len(folds):<8} {mean_ic:+.4f}      {std_ic:.4f}       {mean_ir:+.3f}")

        summary.append({
            'param': param_value,
            'num_folds': len(folds),
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'mean_ir': mean_ir,
            'folds': folds
        })

    print("-" * 64)

    # Find best configuration
    if summary:
        best_by_ic = max(summary, key=lambda x: x['mean_ic'])
        best_by_ir = max(summary, key=lambda x: x['mean_ir'])

        print(f"\nBest by Mean IC: {best_by_ic['param']} (IC={best_by_ic['mean_ic']:+.4f})")
        print(f"Best by Mean IR: {best_by_ir['param']} (IR={best_by_ir['mean_ir']:+.3f})")

    # Per-fold breakdown
    print("\n" + "=" * 80)
    print("PER-FOLD BREAKDOWN")
    print("=" * 80)

    for param_value, folds in sorted(results.items()):
        print(f"\n{param_value}:")
        for f in folds:
            print(f"  Fold {f['fold']}: IC={f['ic']:+.4f}, IR={f['ir']:+.3f}")

    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python sanity_checks/analyze_ablation.py <ablation_dir>")
        print("Example: python sanity_checks/analyze_ablation.py checkpoints/ablation_num_layers/")
        sys.exit(1)

    analyze_ablation_dir(sys.argv[1])
