"""Financial Machine Learning Labeling Module.

Implements institutional forward return formulations, benchmark-adjusted excess
returns (Alpha generation), and volatility-adjusted thresholding inspired by the
Triple-Barrier method (Marcos López de Prado) to eliminate noise and prevent lookahead bias.
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_forward_returns(
    prices: pd.Series, horizon_days: int = 5
) -> pd.Series:
    """Calculate forward cumulative log returns over horizon H.
    
    Formula: ln(Price_{t + H} / Price_t)
    Shifted backwards by H so that the label aligns with observation date t.
    """
    forward_ret = np.log(prices.shift(-horizon_days) / prices)
    return forward_ret


def create_excess_forward_returns(
    asset_prices: pd.Series,
    benchmark_prices: pd.Series,
    horizon_days: int = 5,
) -> pd.Series:
    """Calculate forward excess return of asset over benchmark.
    
    Formula: R_{asset, t->t+H} - R_{bench, t->t+H}
    """
    r_asset = create_forward_returns(asset_prices, horizon_days)
    r_bench = create_forward_returns(benchmark_prices, horizon_days)
    return r_asset - r_bench


def generate_labels(
    asset_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    horizon_days: int = 5,
    threshold_std_multiplier: float = 0.25,
    binary_classification: bool = True,
) -> pd.DataFrame:
    """Generate risk-adjusted targets and labels for supervised ML.
    
    Args:
        asset_df: Target ticker OHLCV DataFrame.
        benchmark_df: Benchmark ETF (SPY) DataFrame.
        horizon_days: Forward prediction horizon in trading days (default 5 = 1 week).
        threshold_std_multiplier: Multiplier on rolling daily volatility to define neutral barrier.
        binary_classification: If True, returns binary (1: Outperform, 0: Underperform/Neutral).
                              If False, returns tri-state (1: Buy, 0: Hold, -1: Sell).
                              
    Returns:
        pd.DataFrame containing:
            - forward_return: Raw forward return
            - forward_excess_return: Alpha over benchmark
            - dynamic_threshold: Volatility barrier
            - label: Classification target (0/1 or -1/0/1)
    """
    close = asset_df["Adj Close"]
    
    # 1. Forward returns
    if benchmark_df is not None:
        bench_close = benchmark_df["Adj Close"]
        forward_excess = create_excess_forward_returns(close, bench_close, horizon_days)
    else:
        forward_excess = create_forward_returns(close, horizon_days)

    forward_raw = create_forward_returns(close, horizon_days)
    
    # 2. Dynamic volatility threshold (Triple Barrier inspiration)
    daily_vol = np.log(close / close.shift(1)).rolling(20).std()
    vol_barrier = threshold_std_multiplier * daily_vol * np.sqrt(horizon_days)

    labels_df = pd.DataFrame(index=asset_df.index)
    labels_df["forward_raw_return"] = forward_raw
    labels_df["forward_excess_return"] = forward_excess
    labels_df["vol_barrier"] = vol_barrier

    if binary_classification:
        # 1 if excess return exceeds positive barrier, 0 otherwise
        # If threshold_std_multiplier is 0, this is simply sign(excess_return > 0)
        if threshold_std_multiplier > 0:
            labels_df["label"] = (forward_excess > vol_barrier).astype(int)
        else:
            labels_df["label"] = (forward_excess > 0.0).astype(int)
    else:
        # Tri-state: 1 (outperform), -1 (underperform), 0 (neutral)
        cond_buy = forward_excess > vol_barrier
        cond_sell = forward_excess < -vol_barrier
        labels_df["label"] = np.where(cond_buy, 1, np.where(cond_sell, -1, 0))

    # Log class distribution
    valid_labels = labels_df["label"].dropna()
    class_dist = valid_labels.value_counts(normalize=True).to_dict()
    formatted_dist = {k: f"{v*100:.1f}%" for k, v in class_dist.items()}
    logger.info(f"Target labeling created for {len(valid_labels)} samples. Class distribution: {formatted_dist}")

    return labels_df
