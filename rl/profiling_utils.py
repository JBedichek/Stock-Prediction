#!/usr/bin/env python3
"""
Profiling utilities for RL training loop.

Provides lightweight timing and profiling tools to identify bottlenecks.
"""

import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from contextlib import contextmanager


class TrainingProfiler:
    """
    Lightweight profiler for RL training loops.

    Tracks time spent in different sections and provides statistics.
    """

    def __init__(self, enabled: bool = True):
        """
        Args:
            enabled: If False, profiler has zero overhead
        """
        self.enabled = enabled
        self.timings = defaultdict(list)  # section_name -> list of times
        self.current_section = None
        self.section_start_time = None

        # Statistics
        self.total_episodes = 0
        self.total_steps = 0

    @contextmanager
    def profile(self, section_name: str):
        """
        Context manager for profiling a code section.

        Usage:
            with profiler.profile('state_creation'):
                # code to profile
                create_states()
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[section_name].append(elapsed)

    def record_step(self):
        """Record that a step was completed."""
        self.total_steps += 1

    def record_episode(self):
        """Record that an episode was completed."""
        self.total_episodes += 1

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all profiled sections.

        Returns:
            Dict mapping section_name -> stats dict with:
            - total_time: Total time spent in this section (seconds)
            - mean_time: Mean time per call (ms)
            - std_time: Std dev of time per call (ms)
            - min_time: Min time per call (ms)
            - max_time: Max time per call (ms)
            - num_calls: Number of times this section was called
            - percent_total: Percentage of total profiled time
        """
        if not self.enabled or not self.timings:
            return {}

        # Calculate total time across all sections
        total_time = sum(sum(times) for times in self.timings.values())

        stats = {}
        for section_name, times in self.timings.items():
            times_array = np.array(times)
            section_total = times_array.sum()

            stats[section_name] = {
                'total_time': section_total,
                'mean_time': times_array.mean() * 1000,  # Convert to ms
                'std_time': times_array.std() * 1000,
                'min_time': times_array.min() * 1000,
                'max_time': times_array.max() * 1000,
                'num_calls': len(times),
                'percent_total': (section_total / total_time * 100) if total_time > 0 else 0
            }

        return stats

    def print_summary(self, top_n: int = 15):
        """
        Print a summary of profiling results.

        Args:
            top_n: Number of top sections to display
        """
        if not self.enabled:
            print("Profiler is disabled")
            return

        stats = self.get_stats()
        if not stats:
            print("No profiling data collected")
            return

        print("\n" + "="*80)
        print("TRAINING PROFILER SUMMARY")
        print("="*80)
        print(f"Total episodes: {self.total_episodes}")
        print(f"Total steps: {self.total_steps}")
        print()

        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

        print(f"{'Section':<35} {'Total (s)':<12} {'Mean (ms)':<12} {'Calls':<10} {'% Total':<10}")
        print("-" * 80)

        for section_name, section_stats in sorted_stats[:top_n]:
            print(f"{section_name:<35} "
                  f"{section_stats['total_time']:>11.3f}  "
                  f"{section_stats['mean_time']:>11.2f}  "
                  f"{section_stats['num_calls']:>9,}  "
                  f"{section_stats['percent_total']:>9.1f}%")

        if len(sorted_stats) > top_n:
            # Sum up remaining sections
            remaining_time = sum(s[1]['total_time'] for s in sorted_stats[top_n:])
            remaining_calls = sum(s[1]['num_calls'] for s in sorted_stats[top_n:])
            total_time = sum(s[1]['total_time'] for s in sorted_stats)
            remaining_pct = (remaining_time / total_time * 100) if total_time > 0 else 0

            print(f"{'... (other sections)':<35} "
                  f"{remaining_time:>11.3f}  "
                  f"{'N/A':>11}  "
                  f"{remaining_calls:>9,}  "
                  f"{remaining_pct:>9.1f}%")

        print("="*80)

        # Print per-episode and per-step averages
        total_time = sum(s[1]['total_time'] for s in sorted_stats)
        if self.total_episodes > 0:
            print(f"\nTime per episode: {total_time / self.total_episodes:.3f}s")
        if self.total_steps > 0:
            print(f"Time per step: {total_time / self.total_steps * 1000:.2f}ms")
        print()

    def reset(self):
        """Reset all profiling data."""
        self.timings.clear()
        self.total_episodes = 0
        self.total_steps = 0

    def save_to_csv(self, filepath: str):
        """Save profiling stats to CSV file."""
        import pandas as pd

        stats = self.get_stats()
        if not stats:
            print("No profiling data to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame(stats).T
        df.index.name = 'section'
        df = df.sort_values('total_time', ascending=False)

        # Add metadata
        df['total_episodes'] = self.total_episodes
        df['total_steps'] = self.total_steps

        df.to_csv(filepath)
        print(f"Profiling data saved to: {filepath}")


class SimpleTimer:
    """Simple timer for one-off measurements."""

    def __init__(self, name: str = "Operation", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time = None

    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.enabled:
            elapsed = time.perf_counter() - self.start_time
            print(f"{self.name}: {elapsed*1000:.2f}ms")


# Convenience function for quick profiling
@contextmanager
def timed(name: str):
    """Quick timer context manager."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {name}: {elapsed*1000:.2f}ms")
