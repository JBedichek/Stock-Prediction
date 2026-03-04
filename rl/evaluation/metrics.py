"""
Metrics for evaluating trading agent performance.

Includes risk-adjusted returns, drawdown analysis, and trade statistics.
"""

import numpy as np
from typing import List, Dict, Tuple


def total_return(portfolio_values: List[float]) -> float:
    """
    Calculate total return over the episode.

    Args:
        portfolio_values: List of portfolio values over time

    Returns:
        Total return as decimal (e.g., 0.15 = 15%)
    """
    if len(portfolio_values) < 2:
        return 0.0

    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]

    if initial_value == 0:
        return 0.0

    return (final_value - initial_value) / initial_value


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

    Args:
        returns: List of period returns (daily)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)

    # Remove any NaN or inf values
    returns_arr = returns_arr[np.isfinite(returns_arr)]

    if len(returns_arr) < 2:
        return 0.0

    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    daily_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    sharpe = (mean_return - daily_rf) / std_return * np.sqrt(periods_per_year)

    return sharpe


def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio (only penalizes downside volatility).

    Sortino = (mean_return - risk_free_rate) / downside_std * sqrt(periods_per_year)

    Args:
        returns: List of period returns (daily)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    returns_arr = returns_arr[np.isfinite(returns_arr)]

    if len(returns_arr) < 2:
        return 0.0

    mean_return = np.mean(returns_arr)

    # Only use negative returns for downside deviation
    downside_returns = returns_arr[returns_arr < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside volatility

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    # Annualize
    daily_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    sortino = (mean_return - daily_rf) / downside_std * np.sqrt(periods_per_year)

    return sortino


def max_drawdown(portfolio_values: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.

    Drawdown = (peak - trough) / peak

    Args:
        portfolio_values: List of portfolio values over time

    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    if len(portfolio_values) < 2:
        return 0.0, 0, 0

    values = np.array(portfolio_values)

    # Calculate running maximum
    running_max = np.maximum.accumulate(values)

    # Calculate drawdowns
    drawdowns = (running_max - values) / running_max
    drawdowns[running_max == 0] = 0  # Handle division by zero

    # Find maximum drawdown
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)

    # Find start of drawdown (last peak before max drawdown)
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if values[i] == running_max[max_dd_idx]:
            peak_idx = i
            break

    return float(max_dd), peak_idx, max_dd_idx


def calmar_ratio(portfolio_values: List[float], periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        portfolio_values: List of portfolio values over time
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    # Calculate annualized return
    total_ret = total_return(portfolio_values)
    num_periods = len(portfolio_values) - 1
    annualized_ret = (1 + total_ret) ** (periods_per_year / num_periods) - 1

    # Calculate max drawdown
    max_dd, _, _ = max_drawdown(portfolio_values)

    if max_dd == 0:
        return np.inf if annualized_ret > 0 else 0.0

    return annualized_ret / max_dd


def win_rate(trades: List[Dict]) -> float:
    """
    Calculate percentage of profitable trades.

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Win rate as decimal (e.g., 0.62 = 62%)
    """
    if len(trades) == 0:
        return 0.0

    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)

    return winning_trades / len(trades)


def profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (sum of wins / sum of losses).

    Args:
        trades: List of trade dictionaries with 'pnl' field

    Returns:
        Profit factor (>1 means profitable overall)
    """
    if len(trades) == 0:
        return 0.0

    wins = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    losses = sum(abs(trade.get('pnl', 0)) for trade in trades if trade.get('pnl', 0) < 0)

    if losses == 0:
        return np.inf if wins > 0 else 0.0

    return wins / losses


def average_trade_duration(trades: List[Dict]) -> float:
    """
    Calculate average holding period in days.

    Args:
        trades: List of trade dictionaries with 'days_held' field

    Returns:
        Average days held
    """
    if len(trades) == 0:
        return 0.0

    total_days = sum(trade.get('days_held', 0) for trade in trades)

    return total_days / len(trades)


def compute_returns(portfolio_values: List[float]) -> List[float]:
    """
    Compute period-over-period returns.

    Args:
        portfolio_values: List of portfolio values

    Returns:
        List of returns (one less than portfolio_values)
    """
    if len(portfolio_values) < 2:
        return []

    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] == 0:
            returns.append(0.0)
        else:
            ret = (portfolio_values[i] - portfolio_values[i-1]) / abs(portfolio_values[i-1])
            returns.append(ret)

    return returns


def compute_all_metrics(portfolio_values: List[float], trades: List[Dict],
                       risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Compute all metrics for an episode.

    Args:
        portfolio_values: List of portfolio values over time
        trades: List of completed trades
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of all metrics
    """
    returns = compute_returns(portfolio_values)
    max_dd, dd_start, dd_end = max_drawdown(portfolio_values)

    metrics = {
        # Returns
        'total_return': total_return(portfolio_values),
        'final_portfolio_value': portfolio_values[-1] if len(portfolio_values) > 0 else 0.0,

        # Risk-adjusted returns
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calmar_ratio(portfolio_values),

        # Risk metrics
        'max_drawdown': max_dd,
        'volatility': np.std(returns) if len(returns) > 0 else 0.0,

        # Trade statistics
        'num_trades': len(trades),
        'win_rate': win_rate(trades),
        'profit_factor': profit_factor(trades),
        'avg_trade_duration': average_trade_duration(trades),

        # Episode length
        'num_periods': len(portfolio_values) - 1 if len(portfolio_values) > 0 else 0
    }

    return metrics


def aggregate_metrics(metrics_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple episodes.

    Args:
        metrics_list: List of metric dictionaries from multiple episodes

    Returns:
        Dictionary with mean, std, min, max, median for each metric
    """
    if len(metrics_list) == 0:
        return {}

    # Get all metric names
    metric_names = list(metrics_list[0].keys())

    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list if metric_name in m]

        # Skip non-numeric fields (like 'start_date', 'episode_idx')
        if len(values) > 0 and not isinstance(values[0], (int, float, np.number)):
            continue

        # Filter out inf and nan
        values = [v for v in values if np.isfinite(v)]

        if len(values) == 0:
            aggregated[metric_name] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        else:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1 if len(values) > 1 else 0)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

    return aggregated


def confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.

    Args:
        values: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats

    values = np.array(values)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return (0.0, 0.0)

    mean = np.mean(values)
    sem = stats.sem(values)
    ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)

    return ci
