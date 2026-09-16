"""Quantitative Feature Engineering and Alpha Factor Generation.

Computes 25+ institutional alpha factors across Momentum, Volatility, Mean Reversion,
Volume-Liquidity, and Cross-Asset Market Regimes. Includes Augmented Dickey-Fuller
(ADF) stationarity verification to prevent non-stationary data leakage.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def adfuller_test(series: pd.Series) -> Dict[str, float]:
    """Perform Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Returns test statistic, p-value, and whether series is stationary (p < 0.05).
    """
    clean_series = series.dropna()
    n = len(clean_series)
    if n < 20:
        return {"test_stat": np.nan, "p_value": 1.0, "is_stationary": False}

    # First-order autoregressive model check
    y = clean_series.values
    dy = np.diff(y)
    y_lag = y[:-1]

    # Regress dy on y_lag with constant
    X = np.column_stack([np.ones_like(y_lag), y_lag])
    try:
        beta, residuals, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        res = dy - X @ beta
        s2 = np.sum(res**2) / (len(dy) - 2)
        var_beta = s2 * np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(np.maximum(1e-12, var_beta[1, 1]))
        t_stat = beta[1] / se_gamma
        
        # Approximate MacKinnon p-value for ADF test with constant
        # Critical values: 1%: -3.43, 5%: -2.86, 10%: -2.57
        if t_stat < -3.43:
            p_val = 0.005
        elif t_stat < -2.86:
            p_val = 0.04
        elif t_stat < -2.57:
            p_val = 0.08
        else:
            p_val = 0.50
        
        return {
            "test_stat": float(t_stat),
            "p_value": float(p_val),
            "is_stationary": bool(p_val < 0.05)
        }
    except Exception:
        return {"test_stat": np.nan, "p_value": 1.0, "is_stationary": False}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD Line, Signal Line, and Normalized MACD Histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / close  # normalized by price
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands %B and Normalized Bandwidth."""
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)

    pct_b = (close - lower) / ((upper - lower) + 1e-10)
    bandwidth = (upper - lower) / (ma + 1e-10)
    return pct_b, bandwidth, ma


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute Normalized Average True Range (NATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr / close


def compute_parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 20
) -> pd.Series:
    """Compute Parkinson High-Low Volatility estimator (Annualized)."""
    log_hl = np.log(high / (low + 1e-10)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    pv = np.sqrt(factor * log_hl.rolling(window=window).mean() * 252.0)
    return pv


def compute_on_balance_volume_zscore(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Compute rolling Z-score of On-Balance Volume (OBV)."""
    price_change = close.diff()
    direction = np.where(price_change > 0, 1.0, np.where(price_change < 0, -1.0, 0.0))
    obv = (direction * volume).cumsum()
    obv_mean = obv.rolling(window=window).mean()
    obv_std = obv.rolling(window=window).std()
    return (obv - obv_mean) / (obv_std + 1e-10)


def compute_rolling_beta(
    asset_ret: pd.Series, benchmark_ret: pd.Series, window: int = 60
) -> pd.Series:
    """Compute rolling Market Beta against benchmark."""
    cov = asset_ret.rolling(window=window).cov(benchmark_ret)
    var_bench = benchmark_ret.rolling(window=window).var()
    return cov / (var_bench + 1e-10)


def extract_alpha_factors(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    drop_warmup: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Generate comprehensive feature matrix of institutional alpha factors.
    
    Args:
        df: Target ticker OHLCV DataFrame.
        benchmark_df: S&P 500 / SPY benchmark OHLCV DataFrame.
        vix_df: Optional VIX volatility index DataFrame.
        drop_warmup: If True, drops initial NaN warmup period.
        
    Returns:
        Tuple of (features_df, factor_groups_dict).
    """
    close = df["Adj Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    feats = pd.DataFrame(index=df.index)
    factor_groups: Dict[str, List[str]] = {
        "Momentum": [],
        "Volatility": [],
        "Mean_Reversion": [],
        "Volume_Liquidity": [],
        "Market_Regime": [],
    }

    # 1. Log Returns & Multi-Horizon Momentum
    log_ret_1d = np.log(close / close.shift(1))
    feats["log_ret_1d"] = log_ret_1d
    feats["mom_5d"] = np.log(close / close.shift(5))
    feats["mom_10d"] = np.log(close / close.shift(10))
    feats["mom_21d"] = np.log(close / close.shift(21))   # 1-month
    feats["mom_63d"] = np.log(close / close.shift(63))   # 3-month
    factor_groups["Momentum"].extend(["log_ret_1d", "mom_5d", "mom_10d", "mom_21d", "mom_63d"])

    # 2. Moving Average Crossover Ratios (Trend)
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()

    feats["ema_ratio_20_50"] = (ema_20 / ema_50) - 1.0
    feats["ema_ratio_50_200"] = (ema_50 / ema_200) - 1.0
    feats["price_to_ema200"] = (close / ema_200) - 1.0
    factor_groups["Momentum"].extend(["ema_ratio_20_50", "ema_ratio_50_200", "price_to_ema200"])

    # 3. Oscillators: RSI & MACD
    feats["rsi_14"] = compute_rsi(close, 14) / 100.0  # normalized 0-1
    macd_line, macd_signal, macd_hist = compute_macd(close)
    feats["macd_line"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_hist
    factor_groups["Momentum"].extend(["rsi_14", "macd_line", "macd_signal", "macd_hist"])

    # 4. Volatility & Mean Reversion
    pct_b, bandwidth, ma20 = compute_bollinger_bands(close, window=20)
    feats["bb_pct_b"] = pct_b
    feats["bb_bandwidth"] = bandwidth
    feats["zscore_price_20d"] = (close - ma20) / (close.rolling(20).std() + 1e-10)
    feats["natr_14"] = compute_atr(high, low, close, period=14)
    factor_groups["Mean_Reversion"].extend(["bb_pct_b", "bb_bandwidth", "zscore_price_20d", "natr_14"])

    # Realized & High-Low Volatility
    feats["vol_realized_20d"] = log_ret_1d.rolling(20).std() * np.sqrt(252)
    feats["vol_realized_60d"] = log_ret_1d.rolling(60).std() * np.sqrt(252)
    feats["vol_parkinson_20d"] = compute_parkinson_volatility(high, low, window=20)
    feats["vol_ratio_20_60"] = feats["vol_realized_20d"] / (feats["vol_realized_60d"] + 1e-10)
    factor_groups["Volatility"].extend(["vol_realized_20d", "vol_realized_60d", "vol_parkinson_20d", "vol_ratio_20_60"])

    # 5. Volume & Liquidity Factors
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    feats["volume_zscore_20d"] = (volume - vol_ma20) / (vol_std20 + 1e-10)
    feats["volume_ratio_5_20"] = volume.rolling(5).mean() / (vol_ma20 + 1e-10)
    feats["obv_zscore"] = compute_on_balance_volume_zscore(close, volume, window=20)
    factor_groups["Volume_Liquidity"].extend(["volume_zscore_20d", "volume_ratio_5_20", "obv_zscore"])

    # 6. Cross-Asset & Market Regime Factors
    if benchmark_df is not None:
        bench_close = benchmark_df["Adj Close"]
        bench_ret_1d = np.log(bench_close / bench_close.shift(1))
        
        feats["bench_rel_perf_20d"] = (close / close.shift(20)) - (bench_close / bench_close.shift(20))
        feats["market_beta_60d"] = compute_rolling_beta(log_ret_1d, bench_ret_1d, window=60)
        feats["market_corr_60d"] = log_ret_1d.rolling(60).corr(bench_ret_1d)
        factor_groups["Market_Regime"].extend(["bench_rel_perf_20d", "market_beta_60d", "market_corr_60d"])

    if vix_df is not None:
        vix_close = vix_df["Close"]
        feats["vix_level"] = vix_close / 100.0
        feats["vix_change_5d"] = vix_close.pct_change(5)
        factor_groups["Market_Regime"].extend(["vix_level", "vix_change_5d"])

    if drop_warmup:
        # Drop rows with NaN due to 200-day EMA warmup
        valid_idx = feats.dropna().index
        feats = feats.loc[valid_idx]
        logger.info(f"Generated {feats.shape[1]} alpha factors across {len(feats)} observations.")

    return feats, factor_groups
