#!/usr/bin/env python3
"""
Hyperparameter Ablation Study Runner

Launches parallel training runs on multiple GPUs with different hyperparameters,
then compares results (loss, IC, IR) in a summary table.

Usage:
    # Run ablation over num_layers on 4 GPUs
    python sanity_checks/hyperparameter_ablation.py \
        --param num_layers --values 2 4 6 8 \
        --gpus 0 1 2 3

    # Run ablation over learning rate
    python sanity_checks/hyperparameter_ablation.py \
        --param lr --values 1e-5 5e-5 1e-4 5e-4 \
        --gpus 0 1

    # Run ablation over multiple params (grid search)
    python sanity_checks/hyperparameter_ablation.py \
        --param num_layers --values 2 4 6 \
        --param2 num_heads --values2 4 8 \
        --gpus 0 1 2 3

    # Dry run to see commands
    python sanity_checks/hyperparameter_ablation.py \
        --param num_layers --values 2 4 6 8 \
        --gpus 0 1 2 3 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


@dataclass
class AblationResult:
    """Results from a single ablation run."""
    param_name: str
    param_value: str
    param2_name: Optional[str] = None
    param2_value: Optional[str] = None
    gpu: int = 0
    checkpoint_dir: str = ""
    # Metrics
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    mean_ic: Optional[float] = None
    mean_ir: Optional[float] = None
    mean_excess_return: Optional[float] = None
    training_time_minutes: Optional[float] = None
    # Status
    success: bool = False
    error_message: str = ""


def parse_results_from_checkpoint(checkpoint_dir: str) -> Dict:
    """Parse evaluation results from checkpoint directory."""
    results = {}

    # Try principled_evaluation.json first
    eval_json = os.path.join(checkpoint_dir, 'principled_evaluation.json')
    if os.path.exists(eval_json):
        try:
            with open(eval_json, 'r') as f:
                data = json.load(f)

            fold_results = data.get('fold_results', [])
            if fold_results:
                ics = [f.get('mean_ic', 0) for f in fold_results if f.get('mean_ic') is not None]
                irs = [f.get('ir', 0) for f in fold_results if f.get('ir') is not None]
                excess = [f.get('excess_vs_random', 0) for f in fold_results if f.get('excess_vs_random') is not None]

                if ics:
                    results['mean_ic'] = sum(ics) / len(ics)
                if irs:
                    results['mean_ir'] = sum(irs) / len(irs)
                if excess:
                    results['mean_excess_return'] = sum(excess) / len(excess)
        except Exception as e:
            print(f"Warning: Could not parse {eval_json}: {e}")

    # Try stats.log for loss values
    stats_log = os.path.join(checkpoint_dir, 'stats.log')
    if os.path.exists(stats_log):
        try:
            with open(stats_log, 'r') as f:
                content = f.read()

            # Find last train/val loss
            train_losses = re.findall(r'Train Loss:\s*([\d.]+)', content)
            val_losses = re.findall(r'Val Loss:\s*([\d.]+)', content)

            if train_losses:
                results['final_train_loss'] = float(train_losses[-1])
            if val_losses:
                results['final_val_loss'] = float(val_losses[-1])
        except Exception as e:
            print(f"Warning: Could not parse {stats_log}: {e}")

    # Try metadata.txt for timing
    metadata = os.path.join(checkpoint_dir, 'metadata.txt')
    if os.path.exists(metadata):
        try:
            with open(metadata, 'r') as f:
                content = f.read()

            duration_match = re.search(r'Total Duration:\s*([\d.]+)\s*hours', content)
            if duration_match:
                results['training_time_minutes'] = float(duration_match.group(1)) * 60
        except Exception:
            pass

    return results


def run_single_ablation(
    param_name: str,
    param_value: str,
    gpu: int,
    base_args: List[str],
    output_dir: str,
    param2_name: Optional[str] = None,
    param2_value: Optional[str] = None,
) -> AblationResult:
    """Run a single training ablation on specified GPU."""

    # Create result object
    result = AblationResult(
        param_name=param_name,
        param_value=param_value,
        param2_name=param2_name,
        param2_value=param2_value,
        gpu=gpu,
    )

    # Build checkpoint directory name
    if param2_name:
        checkpoint_name = f"ablation_{param_name}_{param_value}_{param2_name}_{param2_value}"
    else:
        checkpoint_name = f"ablation_{param_name}_{param_value}"

    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    result.checkpoint_dir = checkpoint_dir

    # Build command
    cmd = [
        sys.executable, "-m", "training.walk_forward_training",
        f"--device", f"cuda:{gpu}",
        f"--checkpoint-dir", checkpoint_dir,
        f"--{param_name.replace('_', '-')}", str(param_value),
    ]

    if param2_name and param2_value:
        cmd.extend([f"--{param2_name.replace('_', '-')}", str(param2_value)])

    cmd.extend(base_args)

    print(f"\n[GPU {gpu}] Starting: {param_name}={param_value}" +
          (f", {param2_name}={param2_value}" if param2_name else ""))
    print(f"  Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        # Run training
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        elapsed = (time.time() - start_time) / 60
        result.training_time_minutes = elapsed

        if proc.returncode != 0:
            result.success = False
            result.error_message = proc.stderr[-500:] if proc.stderr else "Unknown error"
            print(f"[GPU {gpu}] FAILED: {param_name}={param_value}")
            return result

        result.success = True

        # Parse results
        parsed = parse_results_from_checkpoint(checkpoint_dir)
        result.final_train_loss = parsed.get('final_train_loss')
        result.final_val_loss = parsed.get('final_val_loss')
        result.mean_ic = parsed.get('mean_ic')
        result.mean_ir = parsed.get('mean_ir')
        result.mean_excess_return = parsed.get('mean_excess_return')

        print(f"[GPU {gpu}] DONE: {param_name}={param_value} "
              f"(IC={result.mean_ic:+.4f}, IR={result.mean_ir:+.3f})" if result.mean_ic else "")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        print(f"[GPU {gpu}] ERROR: {e}")

    return result


def print_results_table(results: List[AblationResult], param_name: str, param2_name: Optional[str] = None):
    """Print results in a formatted table."""

    print("\n" + "=" * 100)
    print("ABLATION RESULTS")
    print("=" * 100)

    # Sort by param value
    results = sorted(results, key=lambda r: (str(r.param_value), str(r.param2_value or "")))

    # Header
    if param2_name:
        header = f"| {param_name:>12} | {param2_name:>12} | {'Val Loss':>10} | {'Mean IC':>10} | {'Mean IR':>10} | {'Excess %':>10} | {'Time (min)':>10} | {'Status':>8} |"
    else:
        header = f"| {param_name:>12} | {'Val Loss':>10} | {'Mean IC':>10} | {'Mean IR':>10} | {'Excess %':>10} | {'Time (min)':>10} | {'Status':>8} |"

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        val_loss = f"{r.final_val_loss:.4f}" if r.final_val_loss else "N/A"
        mean_ic = f"{r.mean_ic:+.4f}" if r.mean_ic else "N/A"
        mean_ir = f"{r.mean_ir:+.3f}" if r.mean_ir else "N/A"
        excess = f"{r.mean_excess_return:+.3f}" if r.mean_excess_return else "N/A"
        time_min = f"{r.training_time_minutes:.1f}" if r.training_time_minutes else "N/A"
        status = "OK" if r.success else "FAIL"

        if param2_name:
            print(f"| {r.param_value:>12} | {r.param2_value:>12} | {val_loss:>10} | {mean_ic:>10} | {mean_ir:>10} | {excess:>10} | {time_min:>10} | {status:>8} |")
        else:
            print(f"| {r.param_value:>12} | {val_loss:>10} | {mean_ic:>10} | {mean_ir:>10} | {excess:>10} | {time_min:>10} | {status:>8} |")

    print("-" * len(header))

    # Find best by IC
    successful = [r for r in results if r.success and r.mean_ic is not None]
    if successful:
        best_ic = max(successful, key=lambda r: r.mean_ic)
        best_ir = max(successful, key=lambda r: abs(r.mean_ir or 0))

        print(f"\nBest by IC:  {param_name}={best_ic.param_value}" +
              (f", {param2_name}={best_ic.param2_value}" if param2_name else "") +
              f" (IC={best_ic.mean_ic:+.4f})")
        print(f"Best by |IR|: {param_name}={best_ir.param_value}" +
              (f", {param2_name}={best_ir.param2_value}" if param2_name else "") +
              f" (IR={best_ir.mean_ir:+.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter ablation study across multiple GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Ablation parameters
    parser.add_argument('--param', type=str, required=True,
                       help='Hyperparameter to ablate (e.g., num_layers, lr, hidden_dim)')
    parser.add_argument('--values', type=str, nargs='+', required=True,
                       help='Values to test for the parameter')

    # Optional second parameter for grid search
    parser.add_argument('--param2', type=str, default=None,
                       help='Second hyperparameter for grid search')
    parser.add_argument('--values2', type=str, nargs='+', default=None,
                       help='Values for second parameter')

    # GPU configuration
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                       help='GPU IDs to use (runs will be distributed across these)')

    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints/ablation',
                       help='Directory for ablation checkpoints')

    # Training arguments to pass through
    parser.add_argument('--data', type=str, default='data/all_complete_dataset.h5')
    parser.add_argument('--prices', type=str, default='data/actual_prices_clean.h5')
    parser.add_argument('--num-folds', type=int, default=3,
                       help='Number of folds (use fewer for faster ablation)')
    parser.add_argument('--epochs-per-fold', type=int, default=5,
                       help='Epochs per fold (use fewer for faster ablation)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--ranking-only', action='store_true')
    parser.add_argument('--ranking-loss-type', type=str, default='correlation')
    parser.add_argument('--max-eval-dates', type=int, default=30,
                       help='Max eval dates (use fewer for faster ablation)')

    # Other
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without running')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Build base training arguments
    base_args = [
        '--data', args.data,
        '--prices', args.prices,
        '--num-folds', str(args.num_folds),
        '--epochs-per-fold', str(args.epochs_per_fold),
        '--batch-size', str(args.batch_size),
        '--seq-len', str(args.seq_len),
        '--ranking-loss-type', args.ranking_loss_type,
        '--max-eval-dates', str(args.max_eval_dates),
        '--seed', str(args.seed),
        '--no-compile',  # Faster startup for ablation
    ]

    if args.ranking_only:
        base_args.append('--ranking-only')

    # Generate all parameter combinations
    combinations = []
    for val in args.values:
        if args.param2 and args.values2:
            for val2 in args.values2:
                combinations.append((val, val2))
        else:
            combinations.append((val, None))

    print(f"\n{'='*80}")
    print("HYPERPARAMETER ABLATION STUDY")
    print(f"{'='*80}")
    print(f"Parameter: {args.param}")
    print(f"Values: {args.values}")
    if args.param2:
        print(f"Parameter 2: {args.param2}")
        print(f"Values 2: {args.values2}")
    print(f"Total runs: {len(combinations)}")
    print(f"GPUs: {args.gpus}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("DRY RUN - Commands that would be executed:\n")
        for i, (val, val2) in enumerate(combinations):
            gpu = args.gpus[i % len(args.gpus)]
            if val2:
                checkpoint_name = f"ablation_{args.param}_{val}_{args.param2}_{val2}"
            else:
                checkpoint_name = f"ablation_{args.param}_{val}"

            cmd = [
                "python", "-m", "training.walk_forward_training",
                f"--device", f"cuda:{gpu}",
                f"--checkpoint-dir", os.path.join(args.output_dir, checkpoint_name),
                f"--{args.param.replace('_', '-')}", str(val),
            ]
            if val2:
                cmd.extend([f"--{args.param2.replace('_', '-')}", str(val2)])
            cmd.extend(base_args)

            print(f"[GPU {gpu}] {' '.join(cmd)}\n")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run ablations in parallel across GPUs
    results = []

    with ProcessPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {}

        for i, (val, val2) in enumerate(combinations):
            gpu = args.gpus[i % len(args.gpus)]

            future = executor.submit(
                run_single_ablation,
                args.param,
                val,
                gpu,
                base_args,
                args.output_dir,
                args.param2,
                val2,
            )
            futures[future] = (val, val2)

        for future in as_completed(futures):
            val, val2 = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in ablation {args.param}={val}: {e}")
                results.append(AblationResult(
                    param_name=args.param,
                    param_value=val,
                    param2_name=args.param2,
                    param2_value=val2,
                    success=False,
                    error_message=str(e),
                ))

    # Print summary table
    print_results_table(results, args.param, args.param2)

    # Save results to JSON
    results_json = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_json, 'w') as f:
        json.dump([{
            'param_name': r.param_name,
            'param_value': r.param_value,
            'param2_name': r.param2_name,
            'param2_value': r.param2_value,
            'gpu': r.gpu,
            'checkpoint_dir': r.checkpoint_dir,
            'final_train_loss': r.final_train_loss,
            'final_val_loss': r.final_val_loss,
            'mean_ic': r.mean_ic,
            'mean_ir': r.mean_ir,
            'mean_excess_return': r.mean_excess_return,
            'training_time_minutes': r.training_time_minutes,
            'success': r.success,
            'error_message': r.error_message,
        } for r in results], f, indent=2)

    print(f"\nResults saved to: {results_json}")


if __name__ == '__main__':
    main()
