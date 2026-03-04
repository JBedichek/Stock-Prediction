#!/usr/bin/env python3
"""
Ablation Study Metrics Analyzer

Parses stats.log files from walk-forward training runs and generates comparative
analysis plots for IC, IR, Rank IC, and excess returns across multiple conditions.

Usage:
    python sanity_checks/analyze_ablation_metrics.py \
        --runs checkpoints/ablation_num_layers/num-layers_1__listnet_seed4 \
               checkpoints/ablation_num_layers/num-layers_2__listnet_seed4 \
        --labels "1 Layer" "2 Layers" \
        --output results/ablation_analysis

    # Or auto-discover runs in a directory:
    python sanity_checks/analyze_ablation_metrics.py \
        --ablation-dir checkpoints/ablation_num_layers \
        --output results/ablation_analysis
"""

import argparse
import re
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats as scipy_stats


@dataclass
class FoldMetrics:
    """Metrics for a single fold."""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # IC Metrics
    mean_ic: float = 0.0
    ic_std: float = 0.0
    information_ratio: float = 0.0
    rank_ic: float = 0.0
    pct_ic_positive: float = 0.0
    ic_t_stat: float = 0.0
    ic_p_value: float = 0.0

    # Quantile Analysis
    top_decile_return: float = 0.0
    bottom_decile_return: float = 0.0
    long_short_spread: float = 0.0

    # Returns (Gross)
    model_return_gross: float = 0.0
    momentum_return_gross: float = 0.0
    random_return_gross: float = 0.0
    excess_vs_random_gross: float = 0.0
    excess_vs_momentum_gross: float = 0.0

    # Returns (Net)
    model_return_net: float = 0.0
    momentum_return_net: float = 0.0
    excess_vs_random_net: float = 0.0
    excess_vs_momentum_net: float = 0.0

    # Simulation Summary
    total_return_gross: float = 0.0
    total_return_net: float = 0.0
    sharpe_gross: float = 0.0
    sharpe_net: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0

    # Daily arrays (loaded from .npz files when available)
    daily_ics: Optional[np.ndarray] = None
    daily_rank_ics: Optional[np.ndarray] = None
    daily_model_returns: Optional[np.ndarray] = None
    daily_momentum_returns: Optional[np.ndarray] = None
    daily_random_returns: Optional[np.ndarray] = None


@dataclass
class AblationRun:
    """All metrics for one ablation condition."""
    name: str
    path: str
    folds: List[FoldMetrics] = field(default_factory=list)
    has_daily_data: bool = False

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    def get_metric_series(self, metric: str) -> np.ndarray:
        """Get a metric across all folds."""
        return np.array([getattr(f, metric) for f in self.folds])

    def get_test_periods(self) -> List[Tuple[str, str]]:
        """Get test period (start, end) for each fold."""
        return [(f.test_start, f.test_end) for f in self.folds]

    def get_all_daily_ics(self) -> np.ndarray:
        """Get all daily ICs concatenated across folds."""
        if not self.has_daily_data:
            return np.array([])
        all_ics = []
        for fold in self.folds:
            if fold.daily_ics is not None:
                all_ics.extend(fold.daily_ics)
        return np.array(all_ics)

    def get_all_daily_rank_ics(self) -> np.ndarray:
        """Get all daily Rank ICs concatenated across folds."""
        if not self.has_daily_data:
            return np.array([])
        all_rank_ics = []
        for fold in self.folds:
            if fold.daily_rank_ics is not None:
                all_rank_ics.extend(fold.daily_rank_ics)
        return np.array(all_rank_ics)

    def get_all_daily_excess_returns(self) -> np.ndarray:
        """Get all daily excess returns (model - random) concatenated across folds."""
        if not self.has_daily_data:
            return np.array([])
        all_excess = []
        for fold in self.folds:
            if fold.daily_model_returns is not None and fold.daily_random_returns is not None:
                excess = np.array(fold.daily_model_returns) - np.array(fold.daily_random_returns)
                all_excess.extend(excess)
        return np.array(all_excess)

    def get_all_daily_model_returns(self) -> np.ndarray:
        """Get all daily model returns concatenated across folds."""
        if not self.has_daily_data:
            return np.array([])
        all_returns = []
        for fold in self.folds:
            if fold.daily_model_returns is not None:
                all_returns.extend(fold.daily_model_returns)
        return np.array(all_returns)


def parse_float(s: str) -> float:
    """Parse a float from a string, handling signs and percentages."""
    s = s.strip()
    s = s.replace('%', '')
    s = s.replace('+', '')
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_daily_metrics(run_path: str, fold_num: int) -> Optional[Dict[str, np.ndarray]]:
    """Load daily metrics from .npz file for a fold."""
    npz_path = os.path.join(run_path, f'fold_{fold_num}_daily_metrics.npz')
    if not os.path.exists(npz_path):
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        return {
            'daily_ics': data['daily_ics'],
            'daily_rank_ics': data['daily_rank_ics'],
            'daily_model_returns': data['daily_model_returns'],
            'daily_momentum_returns': data['daily_momentum_returns'],
            'daily_random_returns': data['daily_random_returns'],
        }
    except Exception as e:
        print(f"Warning: Could not load {npz_path}: {e}")
        return None


def parse_stats_log(log_path: str) -> List[FoldMetrics]:
    """Parse a stats.log file and extract per-fold metrics."""
    folds = []

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by fold sections
    fold_pattern = r'FOLD (\d+)/(\d+) EVALUATION RESULTS.*?(?=FOLD \d+/\d+ EVALUATION RESULTS|Stats log saved|$)'
    fold_matches = re.findall(fold_pattern, content, re.DOTALL)

    # Actually we need the full text of each fold section
    fold_sections = re.split(r'={80}\n.*?FOLD \d+/\d+ EVALUATION RESULTS\n={80}', content)

    # More robust approach: find each FOLD header and capture until next FOLD or end
    fold_starts = list(re.finditer(r'FOLD (\d+)/(\d+) EVALUATION RESULTS', content))

    for i, match in enumerate(fold_starts):
        fold_num = int(match.group(1))
        total_folds = int(match.group(2))

        # Get text from this fold to the next (or end)
        start_pos = match.start()
        if i + 1 < len(fold_starts):
            end_pos = fold_starts[i + 1].start()
        else:
            end_pos = len(content)

        fold_text = content[start_pos:end_pos]

        # Create metrics object
        metrics = FoldMetrics(fold_num=fold_num, train_start='', train_end='',
                            test_start='', test_end='')

        # Parse training/test periods
        train_match = re.search(r'Training Period:\s+(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', fold_text)
        if train_match:
            metrics.train_start = train_match.group(1)
            metrics.train_end = train_match.group(2)

        test_match = re.search(r'Test Period:\s+(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', fold_text)
        if test_match:
            metrics.test_start = test_match.group(1)
            metrics.test_end = test_match.group(2)

        # Parse IC metrics
        ic_match = re.search(r'Mean IC:\s+([\d.+-]+)', fold_text)
        if ic_match:
            metrics.mean_ic = parse_float(ic_match.group(1))

        ic_std_match = re.search(r'IC Std Dev:\s+([\d.+-]+)', fold_text)
        if ic_std_match:
            metrics.ic_std = parse_float(ic_std_match.group(1))

        ir_match = re.search(r'Information Ratio:([\d.+-]+)', fold_text)
        if ir_match:
            metrics.information_ratio = parse_float(ir_match.group(1))

        rank_ic_match = re.search(r'Mean Rank IC:\s+([\d.+-]+)', fold_text)
        if rank_ic_match:
            metrics.rank_ic = parse_float(rank_ic_match.group(1))

        pct_ic_match = re.search(r'Pct IC > 0:\s+([\d.]+)%', fold_text)
        if pct_ic_match:
            metrics.pct_ic_positive = parse_float(pct_ic_match.group(1))

        t_stat_match = re.search(r'IC T-statistic:\s+([\d.+-]+)\s+\(p=([\d.]+)', fold_text)
        if t_stat_match:
            metrics.ic_t_stat = parse_float(t_stat_match.group(1))
            metrics.ic_p_value = parse_float(t_stat_match.group(2))

        # Parse quantile analysis
        top_decile_match = re.search(r'Top Decile Ret:\s+([\d.+-]+)%', fold_text)
        if top_decile_match:
            metrics.top_decile_return = parse_float(top_decile_match.group(1))

        bottom_decile_match = re.search(r'Bottom Decile:\s+([\d.+-]+)%', fold_text)
        if bottom_decile_match:
            metrics.bottom_decile_return = parse_float(bottom_decile_match.group(1))

        spread_match = re.search(r'Long-Short Spread:([\d.+-]+)%', fold_text)
        if spread_match:
            metrics.long_short_spread = parse_float(spread_match.group(1))

        # Parse baseline comparisons (gross) - first occurrence
        gross_section = re.search(r'BASELINE COMPARISONS \(Gross Returns\)(.*?)BASELINE COMPARISONS \(Net', fold_text, re.DOTALL)
        if gross_section:
            gross_text = gross_section.group(1)

            model_ret = re.search(r'Model Return:\s+([\d.+-]+)%', gross_text)
            if model_ret:
                metrics.model_return_gross = parse_float(model_ret.group(1))

            momentum_ret = re.search(r'Momentum Return:\s+([\d.+-]+)%', gross_text)
            if momentum_ret:
                metrics.momentum_return_gross = parse_float(momentum_ret.group(1))

            random_ret = re.search(r'Random Return:\s+([\d.+-]+)%', gross_text)
            if random_ret:
                metrics.random_return_gross = parse_float(random_ret.group(1))

            excess_random = re.search(r'Excess vs Random:\s+([\d.+-]+)%', gross_text)
            if excess_random:
                metrics.excess_vs_random_gross = parse_float(excess_random.group(1))

            excess_momentum = re.search(r'Excess vs Momentum:([\d.+-]+)%', gross_text)
            if excess_momentum:
                metrics.excess_vs_momentum_gross = parse_float(excess_momentum.group(1))

        # Parse baseline comparisons (net)
        net_section = re.search(r'BASELINE COMPARISONS \(Net of Transaction Costs\)(.*?)TURNOVER METRICS', fold_text, re.DOTALL)
        if net_section:
            net_text = net_section.group(1)

            model_ret = re.search(r'Model Return:\s+([\d.+-]+)%', net_text)
            if model_ret:
                metrics.model_return_net = parse_float(model_ret.group(1))

            momentum_ret = re.search(r'Momentum Return:\s+([\d.+-]+)%', net_text)
            if momentum_ret:
                metrics.momentum_return_net = parse_float(momentum_ret.group(1))

            excess_random = re.search(r'Excess vs Random:\s+([\d.+-]+)%', net_text)
            if excess_random:
                metrics.excess_vs_random_net = parse_float(excess_random.group(1))

            excess_momentum = re.search(r'Excess vs Momentum:([\d.+-]+)%', net_text)
            if excess_momentum:
                metrics.excess_vs_momentum_net = parse_float(excess_momentum.group(1))

        # Parse simulation summary
        total_gross = re.search(r'Total Return:\s+([\d.+-]+)% \(gross\)', fold_text)
        if total_gross:
            metrics.total_return_gross = parse_float(total_gross.group(1))

        total_net = re.search(r'Total Return:\s+([\d.+-]+)% \(net\)', fold_text)
        if total_net:
            metrics.total_return_net = parse_float(total_net.group(1))

        sharpe_gross = re.search(r'Sharpe Ratio:\s+([\d.+-]+) \(gross\)', fold_text)
        if sharpe_gross:
            metrics.sharpe_gross = parse_float(sharpe_gross.group(1))

        sharpe_net = re.search(r'Sharpe Ratio:\s+([\d.+-]+) \(net\)', fold_text)
        if sharpe_net:
            metrics.sharpe_net = parse_float(sharpe_net.group(1))

        drawdown = re.search(r'Max Drawdown:\s+([\d.+-]+)%', fold_text)
        if drawdown:
            metrics.max_drawdown = parse_float(drawdown.group(1))

        win_rate = re.search(r'Win Rate:\s+([\d.]+)%', fold_text)
        if win_rate:
            metrics.win_rate = parse_float(win_rate.group(1))

        turnover = re.search(r'Mean Turnover:\s+([\d.]+)%', fold_text)
        if turnover:
            metrics.turnover = parse_float(turnover.group(1))

        folds.append(metrics)

    return folds


def load_ablation_runs(run_paths: List[str], labels: Optional[List[str]] = None) -> List[AblationRun]:
    """Load multiple ablation runs."""
    runs = []

    for i, path in enumerate(run_paths):
        stats_log = os.path.join(path, 'stats.log')
        if not os.path.exists(stats_log):
            print(f"Warning: stats.log not found in {path}")
            continue

        name = labels[i] if labels and i < len(labels) else os.path.basename(path)
        folds = parse_stats_log(stats_log)

        if folds:
            # Try to load daily metrics for each fold
            has_daily = False
            daily_counts = []
            for fold in folds:
                daily_data = load_daily_metrics(path, fold.fold_num)
                if daily_data:
                    fold.daily_ics = daily_data['daily_ics']
                    fold.daily_rank_ics = daily_data['daily_rank_ics']
                    fold.daily_model_returns = daily_data['daily_model_returns']
                    fold.daily_momentum_returns = daily_data['daily_momentum_returns']
                    fold.daily_random_returns = daily_data['daily_random_returns']
                    has_daily = True
                    daily_counts.append(len(fold.daily_ics))

            run = AblationRun(name=name, path=path, folds=folds, has_daily_data=has_daily)
            runs.append(run)

            if has_daily:
                total_daily = sum(daily_counts)
                print(f"Loaded {len(folds)} folds from {name} ({total_daily} daily observations)")
            else:
                print(f"Loaded {len(folds)} folds from {name} (no daily data)")
        else:
            print(f"Warning: No folds found in {path}")

    return runs


def discover_runs_in_directory(ablation_dir: str) -> Tuple[List[str], List[str]]:
    """Auto-discover ablation runs in a directory."""
    run_paths = []
    labels = []

    for item in sorted(os.listdir(ablation_dir)):
        item_path = os.path.join(ablation_dir, item)
        if os.path.isdir(item_path):
            stats_log = os.path.join(item_path, 'stats.log')
            if os.path.exists(stats_log):
                run_paths.append(item_path)
                # Clean up label from directory name
                label = item.replace('_', ' ').replace('--', ': ')
                # Extract the key parts
                if 'num-layers' in item.lower():
                    match = re.search(r'num-layers[_\s]*(\d+)', item, re.IGNORECASE)
                    if match:
                        label = f"{match.group(1)} Layer{'s' if int(match.group(1)) > 1 else ''}"
                labels.append(label)

    return run_paths, labels


def plot_metrics_over_folds(runs: List[AblationRun], output_dir: str):
    """Plot IC, IR, and Rank IC across folds for all runs."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Metrics to plot
    metrics_config = [
        ('mean_ic', 'Mean IC', axes[0, 0]),
        ('information_ratio', 'Information Ratio (IC/std)', axes[0, 1]),
        ('rank_ic', 'Rank IC (Spearman)', axes[1, 0]),
        ('long_short_spread', 'Long-Short Spread (%)', axes[1, 1]),
    ]

    for metric, title, ax in metrics_config:
        for i, run in enumerate(runs):
            fold_nums = [f.fold_num for f in run.folds]
            values = run.get_metric_series(metric)

            ax.plot(fold_nums, values, 'o-', color=colors[i], label=run.name,
                   linewidth=2, markersize=6)

            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max(run.num_folds for run in runs) + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ic_metrics_by_fold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/ic_metrics_by_fold.png")


def plot_returns_over_folds(runs: List[AblationRun], output_dir: str):
    """Plot return metrics across folds."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    metrics_config = [
        ('model_return_gross', 'Model Return (Gross %)', axes[0, 0]),
        ('excess_vs_random_gross', 'Excess vs Random (Gross %)', axes[0, 1]),
        ('excess_vs_momentum_gross', 'Excess vs Momentum (Gross %)', axes[1, 0]),
        ('total_return_gross', 'Total Fold Return (Gross %)', axes[1, 1]),
    ]

    for metric, title, ax in metrics_config:
        for i, run in enumerate(runs):
            fold_nums = [f.fold_num for f in run.folds]
            values = run.get_metric_series(metric)

            ax.plot(fold_nums, values, 'o-', color=colors[i], label=run.name,
                   linewidth=2, markersize=6)

            # Add mean line
            mean_val = np.mean(values)
            ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, max(run.num_folds for run in runs) + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_by_fold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/returns_by_fold.png")


def plot_metrics_by_time(runs: List[AblationRun], output_dir: str):
    """Plot metrics with actual test period dates on x-axis."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    metrics_config = [
        ('mean_ic', 'Mean IC', axes[0, 0]),
        ('information_ratio', 'Information Ratio', axes[0, 1]),
        ('model_return_gross', 'Model Return (%)', axes[1, 0]),
        ('excess_vs_random_gross', 'Excess vs Random (%)', axes[1, 1]),
    ]

    for metric, title, ax in metrics_config:
        for i, run in enumerate(runs):
            # Use test period midpoints as x values
            dates = []
            for fold in run.folds:
                try:
                    start = datetime.strptime(fold.test_start, '%Y-%m-%d')
                    end = datetime.strptime(fold.test_end, '%Y-%m-%d')
                    mid = start + (end - start) / 2
                    dates.append(mid)
                except:
                    dates.append(datetime.now())

            values = run.get_metric_series(metric)

            ax.plot(dates, values, 'o-', color=colors[i], label=run.name,
                   linewidth=2, markersize=6)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Test Period')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Over Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/metrics_by_time.png")


def plot_metric_correlations(runs: List[AblationRun], output_dir: str):
    """Plot correlations between IC metrics and returns."""

    # Combine all folds from all runs for correlation analysis
    all_ic = []
    all_ir = []
    all_rank_ic = []
    all_excess_returns = []
    all_model_returns = []
    all_labels = []

    for run in runs:
        for fold in run.folds:
            all_ic.append(fold.mean_ic)
            all_ir.append(fold.information_ratio)
            all_rank_ic.append(fold.rank_ic)
            all_excess_returns.append(fold.excess_vs_random_gross)
            all_model_returns.append(fold.model_return_gross)
            all_labels.append(run.name)

    all_ic = np.array(all_ic)
    all_ir = np.array(all_ir)
    all_rank_ic = np.array(all_rank_ic)
    all_excess_returns = np.array(all_excess_returns)
    all_model_returns = np.array(all_model_returns)

    # Create correlation plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    color_map = {run.name: colors[i] for i, run in enumerate(runs)}
    point_colors = [color_map[label] for label in all_labels]

    # IC vs Excess Return
    ax = axes[0, 0]
    ax.scatter(all_ic, all_excess_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_ic, all_excess_returns)
    ax.set_xlabel('Mean IC')
    ax.set_ylabel('Excess Return vs Random (%)')
    ax.set_title(f'IC vs Excess Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add regression line
    if len(all_ic) > 2:
        z = np.polyfit(all_ic, all_excess_returns, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_ic.min(), all_ic.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        ax.legend()

    # IR vs Excess Return
    ax = axes[0, 1]
    ax.scatter(all_ir, all_excess_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_ir, all_excess_returns)
    ax.set_xlabel('Information Ratio')
    ax.set_ylabel('Excess Return vs Random (%)')
    ax.set_title(f'IR vs Excess Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Rank IC vs Excess Return
    ax = axes[0, 2]
    ax.scatter(all_rank_ic, all_excess_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_rank_ic, all_excess_returns)
    ax.set_xlabel('Rank IC (Spearman)')
    ax.set_ylabel('Excess Return vs Random (%)')
    ax.set_title(f'Rank IC vs Excess Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # IC vs Model Return
    ax = axes[1, 0]
    ax.scatter(all_ic, all_model_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_ic, all_model_returns)
    ax.set_xlabel('Mean IC')
    ax.set_ylabel('Model Return (%)')
    ax.set_title(f'IC vs Model Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # IR vs Model Return
    ax = axes[1, 1]
    ax.scatter(all_ir, all_model_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_ir, all_model_returns)
    ax.set_xlabel('Information Ratio')
    ax.set_ylabel('Model Return (%)')
    ax.set_title(f'IR vs Model Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Rank IC vs Model Return
    ax = axes[1, 2]
    ax.scatter(all_rank_ic, all_model_returns, c=point_colors, alpha=0.7, s=60)
    corr, p_val = scipy_stats.pearsonr(all_rank_ic, all_model_returns)
    ax.set_xlabel('Rank IC (Spearman)')
    ax.set_ylabel('Model Return (%)')
    ax.set_title(f'Rank IC vs Model Return\nr={corr:.3f}, p={p_val:.3f}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[run.name],
                          markersize=10, label=run.name) for run in runs]
    fig.legend(handles=handles, loc='upper center', ncol=len(runs), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'metric_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/metric_correlations.png")


def plot_summary_comparison(runs: List[AblationRun], output_dir: str):
    """Create summary bar charts comparing runs."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    run_names = [run.name for run in runs]
    x = np.arange(len(runs))
    width = 0.6

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Metrics to summarize (mean ± std across folds)
    metrics = [
        ('mean_ic', 'Mean IC (avg across folds)', axes[0, 0]),
        ('information_ratio', 'Information Ratio (avg)', axes[0, 1]),
        ('rank_ic', 'Rank IC (avg)', axes[0, 2]),
        ('excess_vs_random_gross', 'Excess vs Random % (avg)', axes[1, 0]),
        ('sharpe_gross', 'Sharpe Ratio (avg)', axes[1, 1]),
        ('total_return_gross', 'Total Return % (sum)', axes[1, 2]),
    ]

    for metric, title, ax in metrics:
        means = []
        stds = []

        for run in runs:
            values = run.get_metric_series(metric)
            if 'Total Return' in title:
                # Sum for total return
                means.append(np.sum(values))
                stds.append(0)  # No std for sum
            else:
                means.append(np.mean(values))
                stds.append(np.std(values))

        bars = ax.bar(x, means, width, yerr=stds if max(stds) > 0 else None,
                     capsize=5, color=colors[:len(runs)], alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/summary_comparison.png")


def generate_stats_table(runs: List[AblationRun], output_dir: str):
    """Generate a summary statistics table."""

    stats = []

    for run in runs:
        row = {
            'Run': run.name,
            'Folds': run.num_folds,
            'Mean IC': f"{np.mean(run.get_metric_series('mean_ic')):.4f}",
            'IC Std': f"{np.std(run.get_metric_series('mean_ic')):.4f}",
            'Mean IR': f"{np.mean(run.get_metric_series('information_ratio')):.3f}",
            'Mean Rank IC': f"{np.mean(run.get_metric_series('rank_ic')):.4f}",
            'Avg Excess vs Random': f"{np.mean(run.get_metric_series('excess_vs_random_gross')):.3f}%",
            'Avg Model Return': f"{np.mean(run.get_metric_series('model_return_gross')):.3f}%",
            'Total Return': f"{np.sum(run.get_metric_series('total_return_gross')):.2f}%",
            'Avg Sharpe': f"{np.mean(run.get_metric_series('sharpe_gross')):.2f}",
            'Win Rate': f"{np.mean(run.get_metric_series('win_rate')):.1f}%",
        }
        stats.append(row)

    # Save as JSON
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Print table
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    # Print header
    headers = list(stats[0].keys())
    print(" | ".join(f"{h:>15}" for h in headers))
    print("-" * (17 * len(headers)))

    # Print rows
    for row in stats:
        print(" | ".join(f"{str(v):>15}" for v in row.values()))

    print("="*100 + "\n")

    print(f"Saved: {output_dir}/summary_stats.json")

    return stats


def compute_statistical_tests(runs: List[AblationRun], output_dir: str):
    """Compute statistical tests comparing runs using daily data when available."""

    if len(runs) < 2:
        print("Need at least 2 runs for statistical comparison")
        return

    # Check if daily data is available
    has_daily = all(run.has_daily_data for run in runs)

    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS")
    if has_daily:
        print("(Using DAILY observations for higher statistical power)")
    else:
        print("(Using FOLD-LEVEL summaries - run new training to get daily data)")
    print("="*80)

    results = []

    # Compare each pair of runs
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            run1, run2 = runs[i], runs[j]

            print(f"\n{run1.name} vs {run2.name}:")
            print("-" * 50)

            comparison = {
                'run1': run1.name,
                'run2': run2.name,
                'tests': {},
                'daily_tests': {} if has_daily else None
            }

            # Fold-level tests (always available)
            print("\n  FOLD-LEVEL (n=10 per condition):")
            for metric in ['mean_ic', 'information_ratio', 'excess_vs_random_gross', 'model_return_gross']:
                vals1 = run1.get_metric_series(metric)
                vals2 = run2.get_metric_series(metric)

                if len(vals1) == len(vals2):
                    t_stat, p_val = scipy_stats.ttest_rel(vals1, vals2)
                    test_type = "paired t-test"
                else:
                    t_stat, p_val = scipy_stats.ttest_ind(vals1, vals2)
                    test_type = "independent t-test"

                mean_diff = np.mean(vals1) - np.mean(vals2)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                print(f"    {metric:28s}: diff={mean_diff:+.4f}, t={t_stat:+.2f}, p={p_val:.4f} {sig}")

                comparison['tests'][metric] = {
                    'mean_diff': float(mean_diff),
                    't_stat': float(t_stat),
                    'p_value': float(p_val),
                    'test_type': test_type,
                    'n1': int(len(vals1)),
                    'n2': int(len(vals2)),
                    'significant_0.05': bool(p_val < 0.05)
                }

            # Daily-level tests (when available)
            if has_daily:
                daily_ics1 = run1.get_all_daily_ics()
                daily_ics2 = run2.get_all_daily_ics()
                daily_rank_ics1 = run1.get_all_daily_rank_ics()
                daily_rank_ics2 = run2.get_all_daily_rank_ics()
                daily_excess1 = run1.get_all_daily_excess_returns()
                daily_excess2 = run2.get_all_daily_excess_returns()
                daily_returns1 = run1.get_all_daily_model_returns()
                daily_returns2 = run2.get_all_daily_model_returns()

                print(f"\n  DAILY-LEVEL (n={len(daily_ics1)} vs n={len(daily_ics2)}):")

                daily_metrics = [
                    ('daily_ic', daily_ics1, daily_ics2),
                    ('daily_rank_ic', daily_rank_ics1, daily_rank_ics2),
                    ('daily_excess_return', daily_excess1, daily_excess2),
                    ('daily_model_return', daily_returns1, daily_returns2),
                ]

                for metric_name, v1, v2 in daily_metrics:
                    if len(v1) > 0 and len(v2) > 0:
                        # Independent t-test for daily data (not paired across runs)
                        t_stat, p_val = scipy_stats.ttest_ind(v1, v2)
                        # Also compute Mann-Whitney U for robustness
                        u_stat, u_pval = scipy_stats.mannwhitneyu(v1, v2, alternative='two-sided')

                        mean_diff = np.mean(v1) - np.mean(v2)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                        print(f"    {metric_name:28s}: diff={mean_diff:+.5f}, t={t_stat:+.2f}, p={p_val:.4f} {sig}")

                        comparison['daily_tests'][metric_name] = {
                            'mean1': float(np.mean(v1)),
                            'mean2': float(np.mean(v2)),
                            'std1': float(np.std(v1)),
                            'std2': float(np.std(v2)),
                            'mean_diff': float(mean_diff),
                            't_stat': float(t_stat),
                            'p_value': float(p_val),
                            'mann_whitney_u': float(u_stat),
                            'mann_whitney_p': float(u_pval),
                            'n1': int(len(v1)),
                            'n2': int(len(v2)),
                            'significant_0.05': bool(p_val < 0.05)
                        }

            results.append(comparison)

    # Save results
    with open(os.path.join(output_dir, 'statistical_tests.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {output_dir}/statistical_tests.json")


def plot_daily_ic_distributions(runs: List[AblationRun], output_dir: str):
    """Plot daily IC distributions for all runs."""

    # Check if daily data is available
    if not any(run.has_daily_data for run in runs):
        print("No daily data available for distribution plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Daily IC distribution
    ax = axes[0, 0]
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_ics = run.get_all_daily_ics()
            ax.hist(daily_ics, bins=50, alpha=0.5, color=colors[i], label=f"{run.name} (n={len(daily_ics)})")
            ax.axvline(np.mean(daily_ics), color=colors[i], linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Daily IC')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily IC Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Daily Rank IC distribution
    ax = axes[0, 1]
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_rank_ics = run.get_all_daily_rank_ics()
            ax.hist(daily_rank_ics, bins=50, alpha=0.5, color=colors[i], label=f"{run.name}")
            ax.axvline(np.mean(daily_rank_ics), color=colors[i], linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Daily Rank IC')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Rank IC Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Daily excess return distribution
    ax = axes[1, 0]
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_excess = run.get_all_daily_excess_returns()
            ax.hist(daily_excess * 100, bins=50, alpha=0.5, color=colors[i], label=f"{run.name}")
            ax.axvline(np.mean(daily_excess) * 100, color=colors[i], linestyle='--', linewidth=2)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Daily Excess Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Daily Excess Return Distribution (vs Random)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IC vs Excess Return scatter (daily)
    ax = axes[1, 1]
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_ics = run.get_all_daily_ics()
            daily_excess = run.get_all_daily_excess_returns()
            if len(daily_ics) == len(daily_excess):
                ax.scatter(daily_ics, daily_excess * 100, alpha=0.3, s=10, color=colors[i], label=run.name)
                # Correlation
                corr, pval = scipy_stats.pearsonr(daily_ics, daily_excess)
                print(f"  {run.name}: Daily IC vs Excess Return correlation = {corr:.3f} (p={pval:.4f})")
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Daily IC')
    ax.set_ylabel('Daily Excess Return (%)')
    ax.set_title('Daily IC vs Excess Return')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_ic_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/daily_ic_distributions.png")


def plot_daily_ic_over_time(runs: List[AblationRun], output_dir: str):
    """Plot rolling daily IC over time."""

    if not any(run.has_daily_data for run in runs):
        print("No daily data available for time series plots")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    # Rolling mean IC
    ax = axes[0]
    window = 20  # 20-day rolling window
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_ics = run.get_all_daily_ics()
            if len(daily_ics) >= window:
                rolling_ic = np.convolve(daily_ics, np.ones(window)/window, mode='valid')
                ax.plot(rolling_ic, color=colors[i], label=f"{run.name} ({window}d rolling)", alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Day Index (across all folds)')
    ax.set_ylabel('Rolling Mean IC')
    ax.set_title(f'{window}-Day Rolling Mean IC Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative IC
    ax = axes[1]
    for i, run in enumerate(runs):
        if run.has_daily_data:
            daily_ics = run.get_all_daily_ics()
            cumsum_ic = np.cumsum(daily_ics)
            ax.plot(cumsum_ic, color=colors[i], label=f"{run.name}", alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('Day Index (across all folds)')
    ax.set_ylabel('Cumulative IC')
    ax.set_title('Cumulative IC Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'daily_ic_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/daily_ic_timeseries.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze ablation study metrics')
    parser.add_argument('--runs', nargs='+', help='Paths to ablation run directories')
    parser.add_argument('--labels', nargs='+', help='Labels for each run (optional)')
    parser.add_argument('--ablation-dir', type=str,
                       help='Auto-discover runs in this directory')
    parser.add_argument('--output', type=str, default='results/ablation_analysis',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get run paths
    if args.ablation_dir:
        run_paths, labels = discover_runs_in_directory(args.ablation_dir)
        print(f"Discovered {len(run_paths)} runs in {args.ablation_dir}")
    elif args.runs:
        run_paths = args.runs
        labels = args.labels
    else:
        parser.error("Must provide either --runs or --ablation-dir")
        return

    if not run_paths:
        print("No valid runs found!")
        return

    # Load runs
    runs = load_ablation_runs(run_paths, labels)

    if not runs:
        print("Failed to load any runs!")
        return

    print(f"\nLoaded {len(runs)} ablation runs")
    print(f"Output directory: {args.output}\n")

    # Generate all analyses
    print("Generating plots...")
    plot_metrics_over_folds(runs, args.output)
    plot_returns_over_folds(runs, args.output)
    plot_metrics_by_time(runs, args.output)
    plot_metric_correlations(runs, args.output)
    plot_summary_comparison(runs, args.output)

    # Generate daily analysis plots (if daily data available)
    if any(run.has_daily_data for run in runs):
        print("\nGenerating daily analysis plots...")
        plot_daily_ic_distributions(runs, args.output)
        plot_daily_ic_over_time(runs, args.output)

    # Generate statistics
    print("\nGenerating statistics...")
    generate_stats_table(runs, args.output)
    compute_statistical_tests(runs, args.output)

    print(f"\n✓ Analysis complete! Results saved to {args.output}/")


if __name__ == '__main__':
    main()
