#!/usr/bin/env python3
"""
Baseline Models for Stock Prediction Comparison

Implements simple baseline models that can be compared against the transformer:
- Ridge Regression (linear baseline)
- LightGBM (gradient boosting baseline)
- MLP (simple neural network baseline)

All models use the same evaluation methodology as the main transformer model.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


@dataclass
class BaselineResult:
    """Results from a baseline model evaluation."""
    model_name: str
    mean_ic: float
    std_ic: float
    ir: float
    mean_rank_ic: float
    pct_positive_ic: float
    model_mean_return: float
    excess_vs_random: float
    total_return_pct: float
    sharpe_ratio: float
    training_time_seconds: float


class BaselineModelWrapper:
    """Base class for baseline models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model on training data."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict returns for given features."""
        raise NotImplementedError

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features - handle NaN, scale if needed."""
        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X


class RidgeBaseline(BaselineModelWrapper):
    """Ridge regression baseline."""

    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._prepare_features(X)
        # Scale features for ridge
        X_scaled = self.scaler.fit_transform(X)
        # Handle NaN in y
        y = np.nan_to_num(y, nan=0.0)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LightGBMBaseline(BaselineModelWrapper):
    """LightGBM gradient boosting baseline."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, num_leaves: int = 31):
        super().__init__("LightGBM")
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'verbose': -1,
            'force_col_wise': True,
            'n_jobs': -1,
        }
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        X = self._prepare_features(X)
        y = np.nan_to_num(y, nan=0.0)
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        X = self._prepare_features(X)
        return self.model.predict(X)


class MLPBaseline(BaselineModelWrapper):
    """Simple MLP baseline."""

    def __init__(self, hidden_layers: Tuple[int, ...] = (256, 128),
                 learning_rate: float = 0.001, max_iter: int = 200):
        super().__init__("MLP")
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._prepare_features(X)
        X_scaled = self.scaler.fit_transform(X)
        y = np.nan_to_num(y, nan=0.0)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


def prepare_baseline_data(
    h5f,
    prices_h5f,
    train_dates: List[str],
    tickers: List[str],
    horizon_days: int = 1,
    max_samples: int = 50000,
    seq_len: int = 1,  # For baselines, we typically use single-day features
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for baseline models.

    Returns X (features) and y (future returns) arrays.
    Uses the last day's features for each sample (no sequence).
    """
    X_list = []
    y_list = []

    # Sample dates if too many
    if len(train_dates) > max_samples // len(tickers):
        sample_indices = np.linspace(0, len(train_dates) - 1,
                                     max_samples // len(tickers), dtype=int)
        sampled_dates = [train_dates[i] for i in sample_indices]
    else:
        sampled_dates = train_dates

    for ticker in tickers:
        if ticker not in h5f:
            continue

        try:
            features = h5f[ticker]['features'][:]
            dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                    for d in h5f[ticker]['dates'][:]]

            # Get prices for return calculation
            if prices_h5f is not None and ticker in prices_h5f:
                price_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                              for d in prices_h5f[ticker]['dates'][:]]
                # Handle both 'close' and 'prices' keys
                if 'close' in prices_h5f[ticker]:
                    closes = prices_h5f[ticker]['close'][:]
                elif 'prices' in prices_h5f[ticker]:
                    closes = prices_h5f[ticker]['prices'][:]
                else:
                    continue
                price_map = {d: c for d, c in zip(price_dates, closes)}
            else:
                continue

            date_to_idx = {d: i for i, d in enumerate(dates)}

            for date in sampled_dates:
                if date not in date_to_idx:
                    continue
                idx = date_to_idx[date]

                # Get features for this date
                feat = features[idx]

                # Calculate future return
                date_idx_in_prices = price_dates.index(date) if date in price_dates else -1
                if date_idx_in_prices < 0:
                    continue

                future_idx = date_idx_in_prices + horizon_days
                if future_idx >= len(closes):
                    continue

                current_price = closes[date_idx_in_prices]
                future_price = closes[future_idx]

                if current_price > 0:
                    future_return = (future_price - current_price) / current_price
                else:
                    continue

                X_list.append(feat)
                y_list.append(future_return)

        except Exception as e:
            continue

    if len(X_list) == 0:
        return np.array([]), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def evaluate_baseline_model(
    model: BaselineModelWrapper,
    h5f,
    prices_h5f,
    test_dates: List[str],
    tickers: List[str],
    horizon_days: int = 1,
    top_k: int = 25,
) -> Dict:
    """
    Evaluate a baseline model using the same methodology as the transformer.

    Returns dict with IC, IR, returns, etc.
    """
    from scipy.stats import pearsonr, spearmanr

    daily_ics = []
    daily_rank_ics = []
    daily_model_returns = []
    daily_random_returns = []

    # Evaluate on non-overlapping periods
    eval_dates = test_dates[::horizon_days]

    for date in eval_dates:
        predictions = []
        actual_returns = []

        for ticker in tickers:
            if ticker not in h5f:
                continue

            try:
                features = h5f[ticker]['features'][:]
                dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                        for d in h5f[ticker]['dates'][:]]

                if date not in dates:
                    continue
                idx = dates.index(date)

                # Get features
                feat = features[idx:idx+1]  # Shape (1, num_features)

                # Get prediction
                pred = model.predict(feat)[0]

                # Get actual return
                if prices_h5f is not None and ticker in prices_h5f:
                    price_dates = [d.decode('utf-8') if isinstance(d, bytes) else d
                                  for d in prices_h5f[ticker]['dates'][:]]
                    # Handle both 'close' and 'prices' keys
                    if 'close' in prices_h5f[ticker]:
                        closes = prices_h5f[ticker]['close'][:]
                    elif 'prices' in prices_h5f[ticker]:
                        closes = prices_h5f[ticker]['prices'][:]
                    else:
                        continue

                    if date in price_dates:
                        date_idx = price_dates.index(date)
                        future_idx = date_idx + horizon_days

                        if future_idx < len(closes) and closes[date_idx] > 0:
                            actual_ret = (closes[future_idx] - closes[date_idx]) / closes[date_idx]
                            predictions.append(pred)
                            actual_returns.append(actual_ret)

            except Exception:
                continue

        if len(predictions) < 10:
            continue

        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        # Calculate IC
        if np.std(predictions) > 1e-8 and np.std(actual_returns) > 1e-8:
            ic, _ = pearsonr(predictions, actual_returns)
            rank_ic, _ = spearmanr(predictions, actual_returns)
            if not np.isnan(ic):
                daily_ics.append(ic)
            if not np.isnan(rank_ic):
                daily_rank_ics.append(rank_ic)

        # Calculate returns for top-K stocks
        sorted_indices = np.argsort(predictions)[::-1]
        top_k_indices = sorted_indices[:top_k]

        model_return = np.mean(actual_returns[top_k_indices])
        random_return = np.mean(actual_returns)

        daily_model_returns.append(model_return)
        daily_random_returns.append(random_return)

    # Aggregate metrics
    daily_ics = np.array(daily_ics) if daily_ics else np.array([0.0])
    daily_rank_ics = np.array(daily_rank_ics) if daily_rank_ics else np.array([0.0])
    daily_model_returns = np.array(daily_model_returns) if daily_model_returns else np.array([0.0])
    daily_random_returns = np.array(daily_random_returns) if daily_random_returns else np.array([0.0])

    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics) if len(daily_ics) > 1 else 1.0
    ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0

    # Sharpe ratio (annualized, assuming daily returns)
    mean_return = np.mean(daily_model_returns)
    std_return = np.std(daily_model_returns) if len(daily_model_returns) > 1 else 1.0
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 1e-8 else 0.0

    return {
        'mean_ic': float(mean_ic),
        'std_ic': float(std_ic),
        'ir': float(ir),
        'mean_rank_ic': float(np.mean(daily_rank_ics)),
        'pct_positive_ic': float(np.mean(daily_ics > 0) * 100),
        'model_mean_return': float(mean_return * 100),
        'random_mean_return': float(np.mean(daily_random_returns) * 100),
        'excess_vs_random': float((mean_return - np.mean(daily_random_returns)) * 100),
        'total_return_pct': float(np.sum(daily_model_returns) * 100),
        'sharpe_ratio': float(sharpe),
        'num_eval_days': len(daily_ics),
        'daily_ics': daily_ics.tolist(),
        'daily_rank_ics': daily_rank_ics.tolist(),
        'daily_model_returns': daily_model_returns.tolist(),
        'daily_random_returns': daily_random_returns.tolist(),
    }


def train_and_evaluate_baselines(
    h5f,
    prices_h5f,
    train_dates: List[str],
    test_dates: List[str],
    tickers: List[str],
    horizon_days: int = 1,
    top_k: int = 25,
    max_train_samples: int = 50000,
    include_lgb: bool = True,
) -> Dict[str, Dict]:
    """
    Train and evaluate all baseline models.

    Returns dict mapping model name to results dict.
    """
    import time

    results = {}

    # Prepare training data (shared across models)
    print("    Preparing baseline training data...")
    X_train, y_train = prepare_baseline_data(
        h5f, prices_h5f, train_dates, tickers,
        horizon_days=horizon_days, max_samples=max_train_samples
    )

    if len(X_train) == 0:
        print("    Warning: No training data for baselines")
        return results

    print(f"    Baseline training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Ridge Regression
    print("    Training Ridge regression...")
    ridge = RidgeBaseline(alpha=1.0)
    start = time.time()
    ridge.fit(X_train, y_train)
    ridge_train_time = time.time() - start

    print("    Evaluating Ridge...")
    ridge_results = evaluate_baseline_model(
        ridge, h5f, prices_h5f, test_dates, tickers, horizon_days, top_k
    )
    ridge_results['training_time_seconds'] = ridge_train_time
    results['Ridge'] = ridge_results

    # LightGBM
    if include_lgb and HAS_LIGHTGBM:
        print("    Training LightGBM...")
        lgbm = LightGBMBaseline(n_estimators=100, max_depth=6)
        start = time.time()
        lgbm.fit(X_train, y_train)
        lgbm_train_time = time.time() - start

        print("    Evaluating LightGBM...")
        lgbm_results = evaluate_baseline_model(
            lgbm, h5f, prices_h5f, test_dates, tickers, horizon_days, top_k
        )
        lgbm_results['training_time_seconds'] = lgbm_train_time
        results['LightGBM'] = lgbm_results
    elif include_lgb:
        print("    Skipping LightGBM (not installed)")

    # MLP
    print("    Training MLP...")
    mlp = MLPBaseline(hidden_layers=(256, 128), max_iter=100)
    start = time.time()
    mlp.fit(X_train, y_train)
    mlp_train_time = time.time() - start

    print("    Evaluating MLP...")
    mlp_results = evaluate_baseline_model(
        mlp, h5f, prices_h5f, test_dates, tickers, horizon_days, top_k
    )
    mlp_results['training_time_seconds'] = mlp_train_time
    results['MLP'] = mlp_results

    return results


def print_baseline_comparison(
    transformer_results: Dict,
    baseline_results: Dict[str, Dict],
    fold_idx: int,
):
    """Print a comparison table of transformer vs baselines."""

    print("\n" + "-" * 80)
    print(f"FOLD {fold_idx} - MODEL COMPARISON")
    print("-" * 80)
    print(f"{'Model':<15} {'Mean IC':>10} {'IR':>10} {'Rank IC':>10} {'Return %':>10} {'Excess %':>10} {'Sharpe':>10}")
    print("-" * 80)

    # Transformer results
    print(f"{'Transformer':<15} "
          f"{transformer_results.get('mean_ic', 0):>+10.4f} "
          f"{transformer_results.get('ir', 0):>+10.3f} "
          f"{transformer_results.get('mean_rank_ic', 0):>+10.4f} "
          f"{transformer_results.get('model_mean_return', 0):>+10.3f} "
          f"{transformer_results.get('excess_vs_random', 0):>+10.3f} "
          f"{transformer_results.get('sharpe_ratio', 0):>+10.2f}")

    # Baseline results
    for name, results in baseline_results.items():
        print(f"{name:<15} "
              f"{results.get('mean_ic', 0):>+10.4f} "
              f"{results.get('ir', 0):>+10.3f} "
              f"{results.get('mean_rank_ic', 0):>+10.4f} "
              f"{results.get('model_mean_return', 0):>+10.3f} "
              f"{results.get('excess_vs_random', 0):>+10.3f} "
              f"{results.get('sharpe_ratio', 0):>+10.2f}")

    print("-" * 80)


def save_baseline_comparison(
    all_fold_results: List[Dict],
    output_path: str,
):
    """Save all fold comparison results to JSON."""
    import json

    # Aggregate across folds
    models = set()
    for fold in all_fold_results:
        models.update(fold.get('baselines', {}).keys())
        models.add('Transformer')

    summary = {'folds': all_fold_results, 'aggregate': {}}

    for model in models:
        ics = []
        irs = []
        returns = []
        excess = []
        sharpes = []

        for fold in all_fold_results:
            if model == 'Transformer':
                results = fold.get('transformer', {})
            else:
                results = fold.get('baselines', {}).get(model, {})

            if results:
                ics.append(results.get('mean_ic', 0))
                irs.append(results.get('ir', 0))
                returns.append(results.get('model_mean_return', 0))
                excess.append(results.get('excess_vs_random', 0))
                sharpes.append(results.get('sharpe_ratio', 0))

        if ics:
            summary['aggregate'][model] = {
                'mean_ic': float(np.mean(ics)),
                'std_ic': float(np.std(ics)),
                'mean_ir': float(np.mean(irs)),
                'mean_return': float(np.mean(returns)),
                'mean_excess': float(np.mean(excess)),
                'mean_sharpe': float(np.mean(sharpes)),
                'total_return': float(np.sum(returns)),
            }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline comparison saved to: {output_path}")

    # Print aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE COMPARISON (across all folds)")
    print("=" * 80)
    print(f"{'Model':<15} {'Mean IC':>10} {'Mean IR':>10} {'Mean Ret%':>10} {'Total Ret%':>12} {'Mean Sharpe':>12}")
    print("-" * 80)

    for model, stats in summary['aggregate'].items():
        print(f"{model:<15} "
              f"{stats['mean_ic']:>+10.4f} "
              f"{stats['mean_ir']:>+10.3f} "
              f"{stats['mean_return']:>+10.3f} "
              f"{stats['total_return']:>+12.2f} "
              f"{stats['mean_sharpe']:>+12.2f}")

    print("=" * 80)
