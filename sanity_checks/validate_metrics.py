#!/usr/bin/env python3
"""
Sanity Checks for Walk-Forward Training Results

This script validates that evaluation metrics are internally consistent:
- Negative IC should correlate with model underperforming random
- Positive IC should correlate with model outperforming random
- Long-short spread sign should match IC sign
- Monte Carlo win rates should be consistent with IC

Usage:
    python scripts/sanity_checks.py --checkpoint checkpoints/walk_forward_seed5
    python scripts/sanity_checks.py --checkpoint checkpoints/walk_forward_seed5 --verbose
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Result of a single sanity check."""
    name: str
    status: CheckStatus
    message: str
    details: str = ""


@dataclass
class FoldMetrics:
    """Metrics extracted for a single fold."""
    fold_idx: int
    # IC metrics
    mean_ic: Optional[float] = None
    ic_std: Optional[float] = None
    rank_ic: Optional[float] = None
    ir: Optional[float] = None  # Information Ratio = IC / IC_std
    pct_ic_positive: Optional[float] = None
    # Quantile analysis
    top_decile_return: Optional[float] = None
    bottom_decile_return: Optional[float] = None
    long_short_spread: Optional[float] = None
    # Baseline comparisons
    model_return: Optional[float] = None
    momentum_return: Optional[float] = None
    random_return: Optional[float] = None
    excess_vs_random: Optional[float] = None
    excess_vs_random_pvalue: Optional[float] = None
    # Monte Carlo (if available)
    mc_prob_beats_random: Optional[float] = None
    mc_excess_mean: Optional[float] = None
    mc_mean_ic: Optional[float] = None


def parse_stats_log(stats_log_path: str) -> List[FoldMetrics]:
    """
    Parse stats.log file and extract metrics for each fold.

    The stats.log format has sections like:
    ================================================================================
    FOLD 1/10 EVALUATION RESULTS
    ================================================================================
    ...
    Mean IC:          -0.0016
    ...

    Note: stats.log may contain duplicate entries from restarts.
    We keep only the LAST occurrence of each fold to get final results.
    """
    if not os.path.exists(stats_log_path):
        print(f"Warning: stats.log not found at {stats_log_path}")
        return []

    with open(stats_log_path, 'r') as f:
        content = f.read()

    # Use a dict to deduplicate - last occurrence wins
    fold_dict = {}

    # Split by fold sections
    fold_pattern = r'FOLD (\d+)/\d+ EVALUATION RESULTS'
    fold_matches = list(re.finditer(fold_pattern, content))

    for i, match in enumerate(fold_matches):
        fold_idx = int(match.group(1))

        # Get the section for this fold (until next fold or end)
        start = match.start()
        end = fold_matches[i + 1].start() if i + 1 < len(fold_matches) else len(content)
        section = content[start:end]

        metrics = FoldMetrics(fold_idx=fold_idx)

        # Parse IC metrics
        ic_match = re.search(r'Mean IC:\s*([-+]?\d*\.?\d+)', section)
        if ic_match:
            metrics.mean_ic = float(ic_match.group(1))

        ic_std_match = re.search(r'IC Std Dev:\s*([-+]?\d*\.?\d+)', section)
        if ic_std_match:
            metrics.ic_std = float(ic_std_match.group(1))

        rank_ic_match = re.search(r'Mean Rank IC:\s*([-+]?\d*\.?\d+)', section)
        if rank_ic_match:
            metrics.rank_ic = float(rank_ic_match.group(1))

        ir_match = re.search(r'Information Ratio:\s*([-+]?\d*\.?\d+)', section)
        if ir_match:
            metrics.ir = float(ir_match.group(1))

        pct_ic_match = re.search(r'Pct IC > 0:\s*([\d.]+)%', section)
        if pct_ic_match:
            metrics.pct_ic_positive = float(pct_ic_match.group(1))

        # Parse quantile analysis
        top_decile_match = re.search(r'Top Decile Ret:\s*([-+]?\d*\.?\d+)%', section)
        if top_decile_match:
            metrics.top_decile_return = float(top_decile_match.group(1))

        bottom_decile_match = re.search(r'Bottom Decile:\s*([-+]?\d*\.?\d+)%', section)
        if bottom_decile_match:
            metrics.bottom_decile_return = float(bottom_decile_match.group(1))

        spread_match = re.search(r'Long-Short Spread:\s*([-+]?\d*\.?\d+)%', section)
        if spread_match:
            metrics.long_short_spread = float(spread_match.group(1))

        # Parse baseline comparisons (Gross Returns section)
        # Model Return:     +0.030% per period
        model_match = re.search(r'Model Return:\s*([-+]?\d*\.?\d+)% per period', section)
        if model_match:
            metrics.model_return = float(model_match.group(1))

        momentum_match = re.search(r'Momentum Return:\s*([-+]?\d*\.?\d+)% per period', section)
        if momentum_match:
            metrics.momentum_return = float(momentum_match.group(1))

        random_match = re.search(r'Random Return:\s*([-+]?\d*\.?\d+)% per period', section)
        if random_match:
            metrics.random_return = float(random_match.group(1))

        # Excess vs Random with p-value
        excess_match = re.search(r'Excess vs Random:\s*([-+]?\d*\.?\d+)%\s*\(p=([\d.]+)\)', section)
        if excess_match:
            metrics.excess_vs_random = float(excess_match.group(1))
            metrics.excess_vs_random_pvalue = float(excess_match.group(2))
        else:
            # Try without p-value
            excess_match = re.search(r'Excess vs Random:\s*([-+]?\d*\.?\d+)%', section)
            if excess_match:
                metrics.excess_vs_random = float(excess_match.group(1))

        # Use fold_idx as key - last occurrence wins (handles restarts)
        fold_dict[fold_idx] = metrics

    # Return sorted by fold index
    return [fold_dict[k] for k in sorted(fold_dict.keys())]


def parse_evaluation_json(json_path: str) -> List[FoldMetrics]:
    """
    Parse principled_evaluation.json if available.
    """
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not parse {json_path}: {e}")
        return []

    folds = []

    for fold_data in data.get('fold_results', []):
        metrics = FoldMetrics(
            fold_idx=fold_data.get('fold_idx', 0),
            mean_ic=fold_data.get('mean_ic'),
            rank_ic=fold_data.get('mean_rank_ic'),
            ir=fold_data.get('ir'),
            pct_ic_positive=fold_data.get('pct_positive_ic'),
            top_decile_return=fold_data.get('top_decile_return'),
            bottom_decile_return=fold_data.get('bottom_decile_return'),
            long_short_spread=fold_data.get('long_short_spread'),
            model_return=fold_data.get('model_mean_return'),
            random_return=fold_data.get('random_mean_return'),
            excess_vs_random=fold_data.get('excess_vs_random'),
        )

        # Monte Carlo results if embedded
        mc_data = fold_data.get('monte_carlo', {})
        if mc_data:
            by_topk = mc_data.get('by_topk', {})
            # Get first top-k results as representative
            for top_k, top_k_data in by_topk.items():
                metrics.mc_prob_beats_random = top_k_data.get('prob_model_beats_random')
                metrics.mc_excess_mean = top_k_data.get('excess_mean')
                metrics.mc_mean_ic = top_k_data.get('mean_ic')
                break

        folds.append(metrics)

    return folds


def check_ic_performance_consistency(fold: FoldMetrics) -> CheckResult:
    """
    Check that IC sign matches performance vs random.

    - If IC < 0, model should underperform random (excess < 0)
    - If IC > 0, model should outperform random (excess > 0)
    """
    name = f"Fold {fold.fold_idx}: IC-Performance Consistency"

    if fold.mean_ic is None or fold.excess_vs_random is None:
        return CheckResult(name, CheckStatus.SKIP, "Missing IC or excess return data")

    ic = fold.mean_ic
    excess = fold.excess_vs_random

    # Determine expected direction
    ic_negative = ic < 0
    excess_negative = excess < 0

    # Check consistency
    consistent = (ic_negative == excess_negative)

    if consistent:
        direction = "negative" if ic_negative else "positive"
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"IC ({ic:+.4f}) and excess return ({excess:+.3f}%) are both {direction}",
            details=f"IC={ic:+.4f}, Excess={excess:+.3f}%"
        )
    else:
        # This is the concerning case - metrics disagree
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"IC ({ic:+.4f}) and excess return ({excess:+.3f}%) have opposite signs",
            details=f"IC is {'negative' if ic_negative else 'positive'} but excess is {'negative' if excess_negative else 'positive'}. "
                    f"This may indicate noise, small sample size, or evaluation issues."
        )


def check_long_short_spread_consistency(fold: FoldMetrics) -> CheckResult:
    """
    Check that long-short spread sign matches IC sign.

    With positive IC: top decile > bottom decile (spread > 0)
    With negative IC: top decile < bottom decile (spread < 0)
    """
    name = f"Fold {fold.fold_idx}: Long-Short Spread Consistency"

    if fold.mean_ic is None or fold.long_short_spread is None:
        return CheckResult(name, CheckStatus.SKIP, "Missing IC or long-short spread data")

    ic = fold.mean_ic
    spread = fold.long_short_spread

    ic_negative = ic < 0
    spread_negative = spread < 0

    # For a well-calibrated model:
    # - Positive IC means model correctly identifies winners (top decile > bottom decile)
    # - Negative IC means model incorrectly ranks (top decile < bottom decile)

    consistent = (ic_negative == spread_negative) or abs(ic) < 0.001  # Allow tiny IC to have any spread

    if consistent or abs(spread) < 0.01:  # Spread too small to be meaningful
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"IC ({ic:+.4f}) and spread ({spread:+.3f}%) are consistent",
            details=f"IC={ic:+.4f}, Spread={spread:+.3f}%"
        )
    else:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"IC ({ic:+.4f}) and spread ({spread:+.3f}%) have unexpected relationship",
            details=f"With IC {'<' if ic_negative else '>'} 0, expected spread to be {'negative' if ic_negative else 'positive'}"
        )


def check_quantile_monotonicity(fold: FoldMetrics) -> CheckResult:
    """
    Check that top decile outperforms bottom decile when IC > 0.
    """
    name = f"Fold {fold.fold_idx}: Quantile Monotonicity"

    if fold.top_decile_return is None or fold.bottom_decile_return is None:
        return CheckResult(name, CheckStatus.SKIP, "Missing quantile return data")

    top = fold.top_decile_return
    bottom = fold.bottom_decile_return

    if fold.mean_ic is not None:
        ic = fold.mean_ic
        expected_top_better = ic > 0
        actual_top_better = top > bottom

        if expected_top_better == actual_top_better or abs(ic) < 0.002:
            return CheckResult(
                name,
                CheckStatus.PASS,
                f"Top decile ({top:+.3f}%) vs bottom ({bottom:+.3f}%) matches IC sign",
            )
        else:
            return CheckResult(
                name,
                CheckStatus.WARN,
                f"Top decile ({top:+.3f}%) vs bottom ({bottom:+.3f}%) doesn't match IC={ic:+.4f}",
                details="Expected top > bottom with positive IC, or top < bottom with negative IC"
            )
    else:
        # Just report the relationship without IC context
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"Top decile: {top:+.3f}%, Bottom decile: {bottom:+.3f}%, Spread: {top-bottom:+.3f}%"
        )


def check_monte_carlo_consistency(fold: FoldMetrics) -> CheckResult:
    """
    Check that Monte Carlo win rate matches IC sign.

    - If IC > 0, prob_model_beats_random should be > 0.5
    - If IC < 0, prob_model_beats_random should be < 0.5
    """
    name = f"Fold {fold.fold_idx}: Monte Carlo Consistency"

    if fold.mc_prob_beats_random is None:
        return CheckResult(name, CheckStatus.SKIP, "No Monte Carlo data available")

    prob = fold.mc_prob_beats_random

    if fold.mean_ic is not None:
        ic = fold.mean_ic
        ic_positive = ic > 0
        model_winning = prob > 0.5

        consistent = (ic_positive == model_winning) or abs(ic) < 0.002

        if consistent:
            return CheckResult(
                name,
                CheckStatus.PASS,
                f"Win rate ({prob*100:.1f}%) is consistent with IC ({ic:+.4f})"
            )
        else:
            return CheckResult(
                name,
                CheckStatus.WARN,
                f"Win rate ({prob*100:.1f}%) doesn't match IC sign ({ic:+.4f})",
                details=f"Expected win rate {'> 50%' if ic_positive else '< 50%'} with IC {'> 0' if ic_positive else '< 0'}"
            )
    else:
        # Just report win rate
        winning = "Model winning" if prob > 0.5 else "Random winning"
        return CheckResult(
            name,
            CheckStatus.PASS,
            f"Win rate: {prob*100:.1f}% ({winning})"
        )


def check_statistical_significance(fold: FoldMetrics) -> CheckResult:
    """
    Check if excess return is statistically significant and report p-value.
    """
    name = f"Fold {fold.fold_idx}: Statistical Significance"

    if fold.excess_vs_random is None:
        return CheckResult(name, CheckStatus.SKIP, "Missing excess return data")

    excess = fold.excess_vs_random
    p_value = fold.excess_vs_random_pvalue

    if p_value is not None:
        if p_value < 0.05:
            significance = "significant (p < 0.05)"
            status = CheckStatus.PASS
        elif p_value < 0.1:
            significance = "marginally significant (p < 0.1)"
            status = CheckStatus.PASS
        else:
            significance = f"not significant (p = {p_value:.3f})"
            status = CheckStatus.WARN

        return CheckResult(
            name,
            status,
            f"Excess return {excess:+.3f}% is {significance}",
            details=f"p-value = {p_value:.4f}"
        )
    else:
        return CheckResult(
            name,
            CheckStatus.SKIP,
            f"Excess return {excess:+.3f}% (no p-value available)"
        )


def run_aggregate_checks(folds: List[FoldMetrics]) -> List[CheckResult]:
    """
    Run aggregate checks across all folds.
    """
    results = []

    if not folds:
        return [CheckResult("Aggregate", CheckStatus.SKIP, "No fold data available")]

    # Aggregate IC check
    ics = [f.mean_ic for f in folds if f.mean_ic is not None]
    if ics:
        mean_ic = sum(ics) / len(ics)
        all_negative = all(ic < 0 for ic in ics)
        all_positive = all(ic > 0 for ic in ics)

        if all_negative:
            results.append(CheckResult(
                "Aggregate: IC Sign",
                CheckStatus.FAIL,
                f"IC is CONSISTENTLY NEGATIVE across all {len(ics)} folds (mean: {mean_ic:+.4f})",
                details="This indicates the model predictions are inversely correlated with actual returns. "
                        "Possible causes: ranking loss trained without cross-sectional batching, "
                        "data leakage, or sign error in prediction code."
            ))
        elif all_positive:
            results.append(CheckResult(
                "Aggregate: IC Sign",
                CheckStatus.PASS,
                f"IC is consistently positive across all {len(ics)} folds (mean: {mean_ic:+.4f})"
            ))
        else:
            # Mixed signs
            pos_count = sum(1 for ic in ics if ic > 0)
            neg_count = sum(1 for ic in ics if ic < 0)
            results.append(CheckResult(
                "Aggregate: IC Sign",
                CheckStatus.WARN,
                f"IC has mixed signs: {pos_count} positive, {neg_count} negative (mean: {mean_ic:+.4f})",
                details="Inconsistent IC sign across folds suggests the model has weak or no predictive power."
            ))

    # Aggregate IR check (more important for ranking strategies)
    irs = [f.ir for f in folds if f.ir is not None]
    if irs:
        mean_ir = sum(irs) / len(irs)
        # IR thresholds: |IR| > 0.5 is decent, > 1.0 is good, > 2.0 is excellent
        # For ranking strategies, we care about magnitude more than sign consistency
        abs_irs = [abs(ir) for ir in irs]
        mean_abs_ir = sum(abs_irs) / len(abs_irs)

        if mean_abs_ir < 0.1:
            results.append(CheckResult(
                "Aggregate: Information Ratio (IR)",
                CheckStatus.FAIL,
                f"IR is near zero across folds (mean |IR|: {mean_abs_ir:.3f}, mean IR: {mean_ir:+.3f})",
                details="IR = IC/std(IC). Near-zero IR means predictions have no consistent signal. "
                        "For ranking strategies, aim for |IR| > 0.5."
            ))
        elif mean_abs_ir < 0.3:
            results.append(CheckResult(
                "Aggregate: Information Ratio (IR)",
                CheckStatus.WARN,
                f"IR is weak (mean |IR|: {mean_abs_ir:.3f}, mean IR: {mean_ir:+.3f})",
                details="Weak IR suggests noisy predictions. Consider more training or architecture changes."
            ))
        elif mean_abs_ir < 0.5:
            results.append(CheckResult(
                "Aggregate: Information Ratio (IR)",
                CheckStatus.PASS,
                f"IR is moderate (mean |IR|: {mean_abs_ir:.3f}, mean IR: {mean_ir:+.3f})",
                details="Moderate IR. Model shows some predictive signal."
            ))
        else:
            results.append(CheckResult(
                "Aggregate: Information Ratio (IR)",
                CheckStatus.PASS,
                f"IR is good (mean |IR|: {mean_abs_ir:.3f}, mean IR: {mean_ir:+.3f})",
                details="Strong IR indicates consistent predictive power."
            ))

    # Aggregate excess return check
    excess_rets = [f.excess_vs_random for f in folds if f.excess_vs_random is not None]
    if excess_rets:
        mean_excess = sum(excess_rets) / len(excess_rets)
        all_negative = all(ex < 0 for ex in excess_rets)
        all_positive = all(ex > 0 for ex in excess_rets)

        if all_negative:
            results.append(CheckResult(
                "Aggregate: Excess Return",
                CheckStatus.FAIL,
                f"Model UNDERPERFORMS random in ALL {len(excess_rets)} folds (mean excess: {mean_excess:+.3f}%)",
                details="Consistent underperformance indicates systematic prediction errors."
            ))
        elif all_positive:
            results.append(CheckResult(
                "Aggregate: Excess Return",
                CheckStatus.PASS,
                f"Model outperforms random in all {len(excess_rets)} folds (mean excess: {mean_excess:+.3f}%)"
            ))
        else:
            pos_count = sum(1 for ex in excess_rets if ex > 0)
            neg_count = sum(1 for ex in excess_rets if ex < 0)
            results.append(CheckResult(
                "Aggregate: Excess Return",
                CheckStatus.WARN,
                f"Mixed performance: {pos_count} folds positive, {neg_count} negative (mean: {mean_excess:+.3f}%)"
            ))

    # Check IC-Excess consistency across folds
    consistent_folds = 0
    total_folds = 0
    for fold in folds:
        if fold.mean_ic is not None and fold.excess_vs_random is not None:
            total_folds += 1
            ic_neg = fold.mean_ic < 0
            ex_neg = fold.excess_vs_random < 0
            if ic_neg == ex_neg:
                consistent_folds += 1

    if total_folds > 0:
        consistency_rate = consistent_folds / total_folds * 100
        if consistency_rate == 100:
            results.append(CheckResult(
                "Aggregate: IC-Performance Consistency",
                CheckStatus.PASS,
                f"IC sign matches excess return sign in {consistent_folds}/{total_folds} folds ({consistency_rate:.0f}%)"
            ))
        elif consistency_rate >= 80:
            results.append(CheckResult(
                "Aggregate: IC-Performance Consistency",
                CheckStatus.PASS,
                f"IC sign matches excess return sign in {consistent_folds}/{total_folds} folds ({consistency_rate:.0f}%)"
            ))
        else:
            results.append(CheckResult(
                "Aggregate: IC-Performance Consistency",
                CheckStatus.WARN,
                f"IC and excess return signs differ in {total_folds - consistent_folds}/{total_folds} folds",
                details="This may indicate noisy evaluation or issues with the evaluation procedure."
            ))

    return results


def print_results(results: List[CheckResult], verbose: bool = False):
    """Print check results with formatting."""

    # Count by status
    status_counts = {status: 0 for status in CheckStatus}
    for result in results:
        status_counts[result.status] += 1

    # Status symbols
    symbols = {
        CheckStatus.PASS: "\033[92m[PASS]\033[0m",  # Green
        CheckStatus.FAIL: "\033[91m[FAIL]\033[0m",  # Red
        CheckStatus.WARN: "\033[93m[WARN]\033[0m",  # Yellow
        CheckStatus.SKIP: "\033[90m[SKIP]\033[0m",  # Gray
    }

    print("\n" + "=" * 80)
    print("SANITY CHECK RESULTS")
    print("=" * 80)

    for result in results:
        symbol = symbols[result.status]
        print(f"\n{symbol} {result.name}")
        print(f"    {result.message}")
        if verbose and result.details:
            print(f"    Details: {result.details}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  PASS: {status_counts[CheckStatus.PASS]}")
    print(f"  FAIL: {status_counts[CheckStatus.FAIL]}")
    print(f"  WARN: {status_counts[CheckStatus.WARN]}")
    print(f"  SKIP: {status_counts[CheckStatus.SKIP]}")

    # Overall assessment
    if status_counts[CheckStatus.FAIL] > 0:
        print("\n\033[91m[OVERALL] CHECKS FAILED - Review errors above\033[0m")
        return 1
    elif status_counts[CheckStatus.WARN] > status_counts[CheckStatus.PASS]:
        print("\n\033[93m[OVERALL] MULTIPLE WARNINGS - Results may need investigation\033[0m")
        return 0
    else:
        print("\n\033[92m[OVERALL] CHECKS PASSED\033[0m")
        return 0


def print_check_explanations():
    """Print detailed explanations of each check."""
    print("""
================================================================================
SANITY CHECK EXPLANATIONS
================================================================================

1. IC-PERFORMANCE CONSISTENCY
   Checks that Information Coefficient (IC) sign matches performance vs random.

   - If IC < 0 (predictions inversely correlated with returns):
     Model should UNDERPERFORM random (excess return < 0)

   - If IC > 0 (predictions positively correlated with returns):
     Model should OUTPERFORM random (excess return > 0)

   FAIL conditions:
   - IC consistently negative across all folds indicates systematic prediction errors

2. LONG-SHORT SPREAD CONSISTENCY
   Checks that top vs bottom decile returns match IC direction.

   - If IC > 0: Top decile return > Bottom decile return (positive spread)
   - If IC < 0: Top decile return < Bottom decile return (negative spread)

   This validates that the quantile analysis matches the correlation metrics.

3. QUANTILE MONOTONICITY
   Verifies that if IC > 0, higher predicted returns lead to higher actual returns.
   This is a sanity check on the decile sorting.

4. MONTE CARLO CONSISTENCY
   If Monte Carlo simulation data is available, checks that:
   - IC > 0 implies win rate > 50%
   - IC < 0 implies win rate < 50%

5. STATISTICAL SIGNIFICANCE
   Reports whether excess returns are statistically significant (p < 0.05).
   High p-values suggest results may be due to chance.

6. INFORMATION RATIO (IR) - KEY FOR RANKING STRATEGIES
   IR = IC / std(IC) measures signal consistency, not just magnitude.

   For pairwise ranking loss, IR is MORE important than raw IC because:
   - Ranking strategies need CONSISTENT relative predictions
   - A model with low IC but stable predictions may outperform high IC with noise
   - IR accounts for prediction variance across time

   Thresholds:
   - |IR| < 0.1: No signal (FAIL)
   - |IR| 0.1-0.3: Weak signal (WARN)
   - |IR| 0.3-0.5: Moderate signal (PASS)
   - |IR| > 0.5: Good signal (PASS)

7. AGGREGATE CHECKS
   - IC Sign: Checks if IC has consistent sign across all folds
   - Excess Return: Checks if model consistently beats/loses to random
   - IC-Performance Consistency: Reports % of folds where IC and excess agree

================================================================================
INTERPRETING RESULTS
================================================================================

PASS: Check passed - metrics are internally consistent
WARN: Warning - unexpected relationship, may indicate noise or small sample
FAIL: Failure - systematic inconsistency detected
SKIP: Check skipped - required data not available

If you see consistent FAILs with negative IC across all folds:
1. Check if ranking loss was trained with cross-sectional batch sampling
2. Verify there are no sign errors in prediction code
3. Check for data leakage or look-ahead bias

""")


def main():
    parser = argparse.ArgumentParser(
        description='Run sanity checks on walk-forward training results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sanity_checks.py --checkpoint checkpoints/walk_forward_seed5
  python scripts/sanity_checks.py --checkpoint checkpoints/walk_forward_seed5 --verbose
  python scripts/sanity_checks.py --explain
        """
    )
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information for each check')
    parser.add_argument('--explain', action='store_true',
                       help='Print detailed explanations of each check')

    args = parser.parse_args()

    if args.explain:
        print_check_explanations()
        sys.exit(0)

    if not args.checkpoint:
        parser.error("--checkpoint is required unless using --explain")

    checkpoint_dir = args.checkpoint

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print(f"\nRunning sanity checks on: {checkpoint_dir}")
    print("=" * 80)

    # Parse data from available sources
    stats_log_path = os.path.join(checkpoint_dir, 'stats.log')
    eval_json_path = os.path.join(checkpoint_dir, 'principled_evaluation.json')

    folds = []

    # Try stats.log first
    stats_folds = parse_stats_log(stats_log_path)
    if stats_folds:
        print(f"  Parsed {len(stats_folds)} folds from stats.log")
        folds = stats_folds

    # Try JSON (may have more complete data)
    json_folds = parse_evaluation_json(eval_json_path)
    if json_folds:
        print(f"  Parsed {len(json_folds)} folds from principled_evaluation.json")
        # Merge or use JSON if more complete
        if not folds:
            folds = json_folds

    if not folds:
        print("\nError: No evaluation data found in checkpoint directory")
        print(f"  Checked: {stats_log_path}")
        print(f"  Checked: {eval_json_path}")
        sys.exit(1)

    # Run checks
    results = []

    # Per-fold checks
    for fold in folds:
        results.append(check_ic_performance_consistency(fold))
        results.append(check_long_short_spread_consistency(fold))
        results.append(check_quantile_monotonicity(fold))
        results.append(check_monte_carlo_consistency(fold))
        results.append(check_statistical_significance(fold))

    # Aggregate checks
    results.extend(run_aggregate_checks(folds))

    # Print results
    exit_code = print_results(results, verbose=args.verbose)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
