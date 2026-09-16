"""Automated Unit Tests for QuantAlpha Engine.

Validates stationarity testing, feature calculation sanity, purged temporal split
integrity (zero data leakage), labeling correctness, and backtester risk metrics.
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd

from src.feature_engineering import (
    adfuller_test,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    extract_alpha_factors,
)
from src.labeling import generate_labels, create_forward_returns
from src.models import PurgedWalkForwardCV, evaluate_predictions
from src.backtester import compute_drawdowns, compute_performance_metrics, run_vectorized_backtest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate deterministic synthetic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    
    # Geometric Brownian motion price series
    returns = np.random.normal(0.0005, 0.015, size=250)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.008, size=250)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.008, size=250)))
    open_p = (high + low) / 2.0
    volume = np.random.randint(1_000_000, 10_000_000, size=250).astype(float)

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def test_stationarity_test():
    """Verify ADF test correctly distinguishes trend from stationary noise."""
    # Stationary series (white noise)
    np.random.seed(42)
    stationary = pd.Series(np.random.normal(0, 1, 200))
    res_stat = adfuller_test(stationary)
    assert res_stat["is_stationary"] is True

    # Non-stationary series (random walk with drift)
    non_stat = pd.Series(np.cumsum(np.random.normal(0.5, 1, 200)))
    res_non_stat = adfuller_test(non_stat)
    assert res_non_stat["is_stationary"] is False


def test_technical_indicators(sample_ohlcv):
    """Verify RSI, MACD, and Bollinger Bands calculation bounds."""
    close = sample_ohlcv["Adj Close"]

    # RSI bounded in [0, 100]
    rsi = compute_rsi(close, 14).dropna()
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()

    # MACD produces valid shapes
    macd, signal, hist = compute_macd(close)
    assert len(macd) == len(close)
    assert len(hist) == len(close)

    # Bollinger Bands %B
    pct_b, bw, ma = compute_bollinger_bands(close, window=20)
    assert len(pct_b.dropna()) > 0
    assert (bw.dropna() >= 0).all()


def test_alpha_factor_extraction(sample_ohlcv):
    """Verify full alpha factor matrix extraction and NaN handling."""
    bench = sample_ohlcv.copy()
    feats, groups = extract_alpha_factors(sample_ohlcv, benchmark_df=bench, drop_warmup=True)

    assert isinstance(feats, pd.DataFrame)
    assert len(feats) > 0
    # No NaNs or Infs in sanitized output
    assert not feats.isna().any().any()
    assert not np.isinf(feats.values).any()
    # Check that groups exist
    assert "Momentum" in groups
    assert "Volatility" in groups
    assert "Mean_Reversion" in groups


def test_forward_labeling(sample_ohlcv):
    """Verify forward label generation does not leak past timestamps."""
    labels = generate_labels(sample_ohlcv, horizon_days=5, binary_classification=True)
    assert "label" in labels.columns
    assert set(labels["label"].dropna().unique()).issubset({0, 1})
    # Last 5 observations should have NaN forward returns
    assert labels["forward_raw_return"].iloc[-5:].isna().all()


def test_purged_walk_forward_cv_leakage():
    """CRITICAL TEST: Ensure zero overlap between train and test folds with purge buffer."""
    n_samples = 150
    X = pd.DataFrame(np.zeros((n_samples, 2)), index=pd.date_range("2020-01-01", periods=n_samples, freq="D"))
    cv = PurgedWalkForwardCV(n_splits=3, purge_window=5)
    splits = cv.split(X)

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        # Check no intersection
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        # Check that test start is strictly >= train_end + purge_window
        assert test_idx[0] >= train_idx[-1] + 5


def test_backtester_metrics():
    """Verify Sharpe ratio and drawdown calculation accuracy."""
    # Constant 1% daily return should have very high Sharpe and 0 drawdown
    constant_rets = pd.Series([0.01] * 50)
    metrics = compute_performance_metrics(constant_rets, risk_free_rate=0.0)
    assert metrics["Sharpe_Ratio"] > 10.0
    assert metrics["Max_Drawdown"] == 0.0

    # Drawdown calculation
    equity = pd.Series([100, 120, 90, 110, 80])
    dd, mdd = compute_drawdowns(equity)
    assert np.isclose(mdd, (80 - 120) / 120)  # -33.33%


def test_vectorized_backtest_with_friction(sample_ohlcv):
    """Verify that transaction costs reduce net strategy returns compared to frictionless."""
    prices = sample_ohlcv["Adj Close"]
    # Alternating 1, 0 signals force high turnover
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(len(prices))], index=prices.index)

    df_cost, summ_cost = run_vectorized_backtest(prices, signals, transaction_cost_bps=20.0, prob_threshold=0.5)
    df_free, summ_free = run_vectorized_backtest(prices, signals, transaction_cost_bps=0.0, prob_threshold=0.5)

    # Strategy with friction must yield less than frictionless strategy
    assert summ_cost["ML_Strategy"]["Total_Return"] <= summ_free["ML_Strategy"]["Total_Return"]
