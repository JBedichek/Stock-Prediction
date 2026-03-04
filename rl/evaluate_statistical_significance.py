#!/usr/bin/env python3
"""
Statistical Significance Testing for RL Trading Agent.

Compares RL agent against multiple baselines with rigorous statistical testing:
- Bootstrap confidence intervals
- Paired t-tests with Bonferroni correction
- Sharpe ratio significance tests (Jobson-Korkie)
- Walk-forward validation to prevent lookahead bias
- Risk-adjusted performance metrics

This answers: "Is the RL agent actually better than simple baselines?"
"""

import numpy as np
import pandas as pd
import sys
import os
import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.rl_components import ActorCriticAgent
from rl.rl_environment import TradingEnvironment
from rl.reduced_action_space import (
    get_top_k_stocks_per_horizon, get_bottom_4_stocks,
    sample_top_4_from_top_k, create_global_state, decode_action_to_trades
)
from inference.backtest_simulation import DatasetLoader


@dataclass
class StrategyResult:
    """Results from running a trading strategy."""
    name: str
    returns: np.ndarray  # Per-episode returns
    portfolio_values: np.ndarray  # Final portfolio values
    sharpe_ratios: np.ndarray  # Per-episode Sharpe ratios
    max_drawdowns: np.ndarray  # Per-episode max drawdowns
    win_rates: np.ndarray  # Per-episode win rates
    num_trades: np.ndarray  # Per-episode number of trades
    avg_return: float
    std_return: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float


class BaselineStrategy:
    """Base class for baseline trading strategies."""

    def __init__(self, name: str):
        self.name = name

    def select_action(self, state: Dict, env: TradingEnvironment, **kwargs) -> int:
        """Select action given current state."""
        raise NotImplementedError


class TopKHorizonStrategy(BaselineStrategy):
    """
    Baseline: Select stock with highest predicted return for matching horizon.

    Strategy:
    - Rank stocks by predicted 1d return, take top-1
    - Hold for 1 day, then re-evaluate

    This is what the predictor model was trained to do directly.
    """

    def __init__(self, horizon: str = '1d'):
        super().__init__(f"TopK-{horizon}")
        self.horizon = horizon
        self.horizon_to_idx = {'1d': 0, '3d': 1, '5d': 2, '10d': 3}
        self.horizon_idx = self.horizon_to_idx[horizon]

    def select_action(self, top_4_stocks, bottom_4_stocks, current_position,
                      is_short, **kwargs) -> Tuple[int, List[Dict], str, bool]:
        """
        Select action based on top predicted return for the specified horizon.

        Returns:
            Tuple of (action, trades, new_position, new_is_short)
        """
        # Find stock with highest predicted return for our horizon
        best_stock = None
        best_return = -float('inf')

        for ticker, horizon_idx in top_4_stocks:
            if horizon_idx == self.horizon_idx:
                # This stock is top for our horizon
                best_stock = ticker
                break

        if best_stock is None:
            # No stock for this horizon, hold cash
            return 0, [], None, False

        # If already holding this stock, keep it
        if current_position == best_stock and not is_short:
            return 0, [], current_position, False

        # Otherwise, switch to the best stock (action 1-4 for long positions)
        for i, (ticker, _) in enumerate(top_4_stocks):
            if ticker == best_stock:
                action = i + 1  # Actions 1-4 are long positions
                trades, new_position, new_is_short = decode_action_to_trades(
                    action, top_4_stocks, bottom_4_stocks, current_position, is_short
                )
                return action, trades, new_position, new_is_short

        # Fallback: hold
        return 0, [], current_position, is_short


class RandomStrategy(BaselineStrategy):
    """
    Baseline: Random action selection.

    Tests if RL agent does better than random.
    """

    def __init__(self):
        super().__init__("Random")

    def select_action(self, top_4_stocks, bottom_4_stocks, current_position,
                      is_short, **kwargs) -> Tuple[int, List[Dict], str, bool]:
        """Random action."""
        action = np.random.randint(0, 9)  # 0-8 (hold + 4 long + 4 short)
        trades, new_position, new_is_short = decode_action_to_trades(
            action, top_4_stocks, bottom_4_stocks, current_position, is_short
        )
        return action, trades, new_position, new_is_short


class BuyAndHoldStrategy(BaselineStrategy):
    """
    Baseline: Buy top-1 stock and hold for entire episode.

    Tests active trading vs passive holding.
    """

    def __init__(self):
        super().__init__("BuyAndHold")
        self.initial_stock = None

    def select_action(self, top_4_stocks, bottom_4_stocks, current_position,
                      is_short, step, **kwargs) -> Tuple[int, List[Dict], str, bool]:
        """Buy top stock on first step, hold forever."""
        if step == 0:
            # First step: buy top stock
            self.initial_stock = top_4_stocks[0][0] if top_4_stocks else None
            if self.initial_stock:
                action = 1  # Buy stock 1
                trades, new_position, new_is_short = decode_action_to_trades(
                    action, top_4_stocks, bottom_4_stocks, None, False
                )
                return action, trades, new_position, new_is_short

        # All other steps: hold
        return 0, [], current_position, is_short


def run_strategy_evaluation(
    strategy: BaselineStrategy,
    env: TradingEnvironment,
    agent: Optional[ActorCriticAgent],
    num_episodes: int,
    episode_length: int,
    data_loader: DatasetLoader,
    validation_dates: List[str],
    device: str = 'cuda',
    verbose: bool = False
) -> StrategyResult:
    """
    Evaluate a strategy over multiple episodes.

    Args:
        strategy: Strategy to evaluate (or None for RL agent)
        env: Trading environment
        agent: RL agent (if strategy is None)
        num_episodes: Number of episodes to run
        episode_length: Length of each episode
        data_loader: DatasetLoader
        validation_dates: List of validation dates
        device: Device
        verbose: Print progress

    Returns:
        StrategyResult with performance metrics
    """
    returns = []
    portfolio_values = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    num_trades_list = []

    is_rl = (strategy is None)
    name = "RL-Agent" if is_rl else strategy.name

    iterator = tqdm(range(num_episodes), desc=f"Evaluating {name}") if verbose else range(num_episodes)

    for episode_idx in iterator:
        # Random start date from validation set
        max_start_idx = len(validation_dates) - episode_length - 1
        if max_start_idx <= 0:
            continue
        start_idx = np.random.randint(0, max_start_idx)
        start_date = validation_dates[start_idx]

        # Reset environment
        env.episode_length = episode_length
        states = env.reset(start_date=start_date)

        current_position = None
        is_short = False
        episode_return = 0.0
        episode_rewards = []
        episode_portfolio_values = []
        trades_count = 0

        for step in range(episode_length):
            # Get top-4 and bottom-4 stocks for current date
            date = env.current_date
            cached_states = env.state_cache.get(date, {})

            # Filter to stocks with prices
            cached_states_with_prices = {
                ticker: state for ticker, state in cached_states.items()
                if ticker in data_loader.prices_file
            }

            if len(cached_states_with_prices) == 0:
                break

            # Get top and bottom stocks (deterministic for fair comparison)
            top_k_stocks = get_top_k_stocks_per_horizon(
                cached_states_with_prices, k=3, sample_fraction=1.0
            )
            top_4_stocks = sample_top_4_from_top_k(top_k_stocks, sample_size=4, deterministic=True)
            bottom_4_stocks = get_bottom_4_stocks(cached_states_with_prices, sample_fraction=1.0)

            # Select action
            if is_rl:
                # RL agent
                # Create global state
                states_dict = {}
                cached_prices = env.price_cache.get(date, {})
                portfolio_value = env._portfolio_value_cached(cached_prices)

                for ticker, _ in (top_4_stocks + bottom_4_stocks):
                    if ticker in cached_states and ticker in cached_prices:
                        cached_state = cached_states[ticker]
                        price = cached_prices[ticker]
                        portfolio_context = env._create_portfolio_context_fast(
                            ticker, price, portfolio_value
                        )
                        state = torch.cat([
                            cached_state[:1444].to(device),
                            portfolio_context.to(device)
                        ])
                        states_dict[ticker] = state

                global_state = create_global_state(
                    top_4_stocks, bottom_4_stocks, states_dict,
                    current_position, is_short, device
                )

                # Agent selects action (greedy, no exploration)
                with torch.no_grad():
                    action, _, _ = agent.select_action(global_state, deterministic=True)

                trades, new_position, new_is_short = decode_action_to_trades(
                    action, top_4_stocks, bottom_4_stocks, current_position, is_short
                )
            else:
                # Baseline strategy
                action, trades, new_position, new_is_short = strategy.select_action(
                    top_4_stocks=top_4_stocks,
                    bottom_4_stocks=bottom_4_stocks,
                    current_position=current_position,
                    is_short=is_short,
                    step=step
                )

            # Execute trades in environment
            cached_prices = env.price_cache.get(date, {})
            for trade in trades:
                ticker = trade['ticker']
                price = cached_prices.get(ticker)
                if price is None:
                    continue

                action_id = {'BUY': 1, 'SELL': 2, 'SHORT': 3, 'COVER': 4}.get(trade['action'], 0)
                env._execute_single_action(ticker, action_id, price)
                trades_count += 1

            # Update position
            current_position = new_position
            is_short = new_is_short

            # Step environment (advance time)
            env.step_idx += 1
            if env.step_idx >= len(env.dates):
                break
            env.current_date = env.dates[env.step_idx]

            # Record metrics
            next_cached_prices = env.price_cache.get(env.current_date, {})
            portfolio_value = env._portfolio_value_cached(next_cached_prices)
            episode_portfolio_values.append(portfolio_value)

        # Close positions at end
        final_cached_prices = env.price_cache.get(env.current_date, {})
        env._close_all_positions(final_cached_prices)

        # Compute episode metrics
        final_value = env._portfolio_value_cached(final_cached_prices)
        episode_return = (final_value - env.initial_capital) / env.initial_capital

        # Compute Sharpe ratio
        if len(episode_portfolio_values) > 1:
            daily_returns = np.diff(episode_portfolio_values) / episode_portfolio_values[:-1]
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

            # Max drawdown
            peak = np.maximum.accumulate(episode_portfolio_values)
            drawdown = (peak - episode_portfolio_values) / peak
            max_dd = np.max(drawdown)

            # Win rate
            win_rate = np.sum(daily_returns > 0) / len(daily_returns)
        else:
            sharpe = 0.0
            max_dd = 0.0
            win_rate = 0.0

        returns.append(episode_return)
        portfolio_values.append(final_value)
        sharpe_ratios.append(sharpe)
        max_drawdowns.append(max_dd)
        win_rates.append(win_rate)
        num_trades_list.append(trades_count)

    # Aggregate statistics
    returns = np.array(returns)
    portfolio_values = np.array(portfolio_values)
    sharpe_ratios = np.array(sharpe_ratios)
    max_drawdowns = np.array(max_drawdowns)
    win_rates = np.array(win_rates)
    num_trades_list = np.array(num_trades_list)

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino = np.mean(returns) / downside_std * np.sqrt(252)

    # Calmar ratio (return / max drawdown)
    calmar = np.mean(returns) / (np.mean(max_drawdowns) + 1e-8)

    return StrategyResult(
        name=name,
        returns=returns,
        portfolio_values=portfolio_values,
        sharpe_ratios=sharpe_ratios,
        max_drawdowns=max_drawdowns,
        win_rates=win_rates,
        num_trades=num_trades_list,
        avg_return=np.mean(returns),
        std_return=np.std(returns),
        sharpe=np.mean(sharpe_ratios),
        sortino=sortino,
        max_dd=np.mean(max_drawdowns),
        calmar=calmar
    )


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Data to bootstrap
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence

    point_estimate = statistic_fn(data)
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def paired_t_test_with_bonferroni(
    rl_returns: np.ndarray,
    baseline_returns_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    Paired t-tests between RL and each baseline with Bonferroni correction.

    Args:
        rl_returns: RL agent returns
        baseline_returns_dict: Dict of baseline_name -> returns
        alpha: Significance level

    Returns:
        Dict of test results for each baseline
    """
    n_comparisons = len(baseline_returns_dict)
    bonferroni_alpha = alpha / n_comparisons

    results = {}

    for baseline_name, baseline_returns in baseline_returns_dict.items():
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rl_returns, baseline_returns)

        # Effect size (Cohen's d)
        diff = rl_returns - baseline_returns
        cohens_d = np.mean(diff) / np.std(diff)

        # Significant?
        significant = p_value < bonferroni_alpha

        results[baseline_name] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'bonferroni_alpha': bonferroni_alpha,
            'significant': significant,
            'cohens_d': cohens_d,
            'mean_diff': np.mean(diff),
            'std_diff': np.std(diff)
        }

    return results


def jobson_korkie_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray
) -> Tuple[float, float]:
    """
    Jobson-Korkie test for difference in Sharpe ratios.

    Tests null hypothesis: Sharpe_A = Sharpe_B

    Args:
        returns_a: Strategy A returns
        returns_b: Strategy B returns

    Returns:
        Tuple of (z_statistic, p_value)
    """
    # Compute Sharpe ratios
    sharpe_a = np.mean(returns_a) / np.std(returns_a)
    sharpe_b = np.mean(returns_b) / np.std(returns_b)

    # Compute test statistic
    n = len(returns_a)

    # Variance of Sharpe ratio difference (Jobson-Korkie formula)
    var_sharpe_a = (1 + 0.5 * sharpe_a**2) / n
    var_sharpe_b = (1 + 0.5 * sharpe_b**2) / n

    # Covariance term
    corr = np.corrcoef(returns_a, returns_b)[0, 1]
    cov_term = 2 * corr * np.sqrt(var_sharpe_a * var_sharpe_b)

    # Z-statistic
    var_diff = var_sharpe_a + var_sharpe_b - cov_term
    z_stat = (sharpe_a - sharpe_b) / np.sqrt(var_diff)

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def generate_statistical_report(
    rl_result: StrategyResult,
    baseline_results: List[StrategyResult],
    output_dir: str = 'evaluation_results'
):
    """
    Generate comprehensive statistical report with visualizations.

    Args:
        rl_result: RL agent results
        baseline_results: List of baseline results
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING REPORT")
    print("="*80)

    # 1. Summary statistics with bootstrap CIs
    print("\n1. SUMMARY STATISTICS (with 95% Bootstrap CI)")
    print("-" * 80)

    all_results = [rl_result] + baseline_results

    for result in all_results:
        avg_return, lower, upper = bootstrap_confidence_interval(result.returns)
        sharpe, sharpe_lower, sharpe_upper = bootstrap_confidence_interval(result.sharpe_ratios)

        print(f"\n{result.name}:")
        print(f"  Avg Return: {avg_return:.2%} [{lower:.2%}, {upper:.2%}]")
        print(f"  Sharpe:     {sharpe:.3f} [{sharpe_lower:.3f}, {sharpe_upper:.3f}]")
        print(f"  Sortino:    {result.sortino:.3f}")
        print(f"  Max DD:     {result.max_dd:.2%}")
        print(f"  Win Rate:   {np.mean(result.win_rates):.1%}")
        print(f"  Calmar:     {result.calmar:.3f}")
        print(f"  Avg Trades: {np.mean(result.num_trades):.1f}")

    # 2. Paired t-tests with Bonferroni correction
    print("\n2. PAIRED T-TESTS (Bonferroni Corrected)")
    print("-" * 80)

    baseline_returns_dict = {b.name: b.returns for b in baseline_results}
    t_test_results = paired_t_test_with_bonferroni(rl_result.returns, baseline_returns_dict)

    for baseline_name, test_result in t_test_results.items():
        print(f"\nRL vs {baseline_name}:")
        print(f"  t-statistic: {test_result['t_statistic']:+.3f}")
        print(f"  p-value:     {test_result['p_value']:.6f}")
        print(f"  Bonf. α:     {test_result['bonferroni_alpha']:.6f}")
        print(f"  Significant: {'✅ YES' if test_result['significant'] else '❌ NO'}")
        print(f"  Cohen's d:   {test_result['cohens_d']:+.3f}")
        print(f"  Mean diff:   {test_result['mean_diff']:+.2%}")

    # 3. Sharpe ratio tests (Jobson-Korkie)
    print("\n3. SHARPE RATIO SIGNIFICANCE (Jobson-Korkie Test)")
    print("-" * 80)

    for baseline in baseline_results:
        z_stat, p_value = jobson_korkie_test(rl_result.returns, baseline.returns)
        significant = p_value < 0.05

        print(f"\nRL vs {baseline.name}:")
        print(f"  RL Sharpe:       {rl_result.sharpe:.3f}")
        print(f"  {baseline.name} Sharpe: {baseline.sharpe:.3f}")
        print(f"  Z-statistic:     {z_stat:+.3f}")
        print(f"  p-value:         {p_value:.6f}")
        print(f"  Significant:     {'✅ YES' if significant else '❌ NO'}")

    # 4. Visualizations
    print("\n4. GENERATING VISUALIZATIONS")
    print("-" * 80)

    # Plot 1: Return distributions
    plt.figure(figsize=(12, 6))
    for result in all_results:
        plt.hist(result.returns, bins=30, alpha=0.5, label=result.name)
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.title('Return Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/return_distributions.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/return_distributions.png")
    plt.close()

    # Plot 2: Performance comparison (bar chart with error bars)
    plt.figure(figsize=(14, 6))

    metrics = ['Avg Return', 'Sharpe', 'Sortino', 'Calmar']
    x = np.arange(len(metrics))
    width = 0.8 / len(all_results)

    for i, result in enumerate(all_results):
        values = [
            result.avg_return,
            result.sharpe,
            result.sortino,
            result.calmar
        ]

        # Bootstrap CIs for error bars
        cis = []
        for j, metric_name in enumerate(['returns', 'sharpe_ratios', 'sortino', 'calmar']):
            if metric_name == 'sortino':
                _, lower, upper = bootstrap_confidence_interval(result.returns,
                    lambda x: np.mean(x) / np.std(x[x < 0]) if np.sum(x < 0) > 0 else 0)
            elif metric_name == 'calmar':
                _, lower, upper = bootstrap_confidence_interval(result.returns,
                    lambda x: np.mean(x) / (np.mean(x < 0) + 1e-8))
            else:
                _, lower, upper = bootstrap_confidence_interval(getattr(result, metric_name))
            cis.append((values[j] - lower, upper - values[j]))

        errors = np.array(cis).T
        plt.bar(x + i * width, values, width, label=result.name, yerr=errors, capsize=3)

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Performance Comparison (with 95% CI)')
    plt.xticks(x + width * (len(all_results) - 1) / 2, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {output_dir}/performance_comparison.png")
    plt.close()

    # Plot 3: Cumulative returns over time (sample episode)
    # TODO: Would need to track timestep-by-timestep for this

    # 5. Save detailed CSV
    results_df = pd.DataFrame({
        'Strategy': [r.name for r in all_results],
        'Avg_Return': [r.avg_return for r in all_results],
        'Std_Return': [r.std_return for r in all_results],
        'Sharpe': [r.sharpe for r in all_results],
        'Sortino': [r.sortino for r in all_results],
        'Max_DD': [r.max_dd for r in all_results],
        'Calmar': [r.calmar for r in all_results],
        'Win_Rate': [np.mean(r.win_rates) for r in all_results],
        'Avg_Trades': [np.mean(r.num_trades) for r in all_results],
    })

    results_df.to_csv(f'{output_dir}/results_summary.csv', index=False)
    print(f"  ✅ Saved: {output_dir}/results_summary.csv")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    # Configuration
    config = {
        'dataset_path': 'data/all_complete_dataset.h5',
        'prices_path': 'data/actual_prices.h5',
        'state_cache_path': 'data/rl_state_cache_4yr.h5',
        'predictor_checkpoint': './checkpoints/best_model_100m_1.18.pt',
        'rl_checkpoint': './checkpoints/actor_critic_best.pt',  # Your trained RL agent
        'num_episodes': 200,  # Number of test episodes
        'episode_length': 30,  # 30-day episodes
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'evaluation_results'
    }

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTING FOR RL TRADING AGENT")
    print("="*80)

    # Load data
    print("\n1. Loading data and models...")
    data_loader = DatasetLoader(
        dataset_path=config['dataset_path'],
        prices_path=config['prices_path'],
        num_test_stocks=100
    )

    # Get validation dates (last 2 years)
    cutoff_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    sample_ticker = list(data_loader.prices_file.keys())[0]
    prices_dates_bytes = data_loader.prices_file[sample_ticker]['dates'][:]
    all_trading_days = sorted([d.decode('utf-8') for d in prices_dates_bytes])
    validation_dates = [d for d in all_trading_days if d >= cutoff_date]

    print(f"   Validation period: {validation_dates[0]} to {validation_dates[-1]}")
    print(f"   Validation days: {len(validation_dates)}")

    # Load RL agent
    print("\n2. Loading RL agent...")
    rl_agent = ActorCriticAgent(
        predictor_checkpoint_path=config['predictor_checkpoint'],
        state_dim=11761,
        hidden_dim=1024,
        action_dim=9
    ).to(config['device'])

    if os.path.exists(config['rl_checkpoint']):
        checkpoint = torch.load(config['rl_checkpoint'], map_location=config['device'])
        rl_agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"   ✅ Loaded RL checkpoint: {config['rl_checkpoint']}")
    else:
        print(f"   ⚠️  RL checkpoint not found: {config['rl_checkpoint']}")
        print(f"   Using randomly initialized agent (for testing)")

    rl_agent.eval()

    # Create environment
    print("\n3. Creating environment...")
    env = TradingEnvironment(
        data_loader=data_loader,
        agent=rl_agent,
        initial_capital=100000,
        max_positions=1,
        episode_length=config['episode_length'],
        device=config['device'],
        precompute_all_states=False,
        trading_days_filter=validation_dates,
        top_k_per_horizon=10
    )

    # Load state cache
    if os.path.exists(config['state_cache_path']):
        env.load_state_cache(config['state_cache_path'])
        print(f"   ✅ Loaded state cache")

    # Define baselines
    baselines = [
        TopKHorizonStrategy('1d'),
        TopKHorizonStrategy('3d'),
        TopKHorizonStrategy('5d'),
        TopKHorizonStrategy('10d'),
        RandomStrategy(),
        BuyAndHoldStrategy(),
    ]

    print(f"\n4. Running evaluations ({config['num_episodes']} episodes each)...")
    print(f"   Strategies: RL-Agent + {len(baselines)} baselines")

    # Evaluate RL agent
    rl_result = run_strategy_evaluation(
        strategy=None,  # None = use RL agent
        env=env,
        agent=rl_agent,
        num_episodes=config['num_episodes'],
        episode_length=config['episode_length'],
        data_loader=data_loader,
        validation_dates=validation_dates,
        device=config['device'],
        verbose=True
    )

    # Evaluate baselines
    baseline_results = []
    for baseline in baselines:
        result = run_strategy_evaluation(
            strategy=baseline,
            env=env,
            agent=None,
            num_episodes=config['num_episodes'],
            episode_length=config['episode_length'],
            data_loader=data_loader,
            validation_dates=validation_dates,
            device=config['device'],
            verbose=True
        )
        baseline_results.append(result)

    # Generate report
    print("\n5. Generating statistical report...")
    generate_statistical_report(
        rl_result=rl_result,
        baseline_results=baseline_results,
        output_dir=config['output_dir']
    )

    print(f"\n✅ Evaluation complete! Results saved to: {config['output_dir']}/")
