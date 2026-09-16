"""Institutional Quantitative Backtesting & Risk Analytics Engine.

Simulates strategy execution with realistic transaction costs (slippage + commissions),
and calculates standard institutional metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown,
Calmar Ratio, Win Rate, and CAPM Alpha/Beta against benchmark.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_drawdowns(equity_curve: pd.Series) -> Tuple[pd.Series, float]:
    """Compute running drawdown series and maximum drawdown (MDD)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = float(drawdown.min())
    return drawdown, max_drawdown


def compute_performance_metrics(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """Calculate institutional risk and return metrics.
    
    Args:
        daily_returns: Daily strategy return series.
        benchmark_returns: Daily benchmark (SPY) return series.
        risk_free_rate: Annualized risk-free rate (e.g. 4.0% = 0.04).
        
    Returns:
        Dictionary of key quantitative performance indicators.
    """
    clean_rets = daily_returns.dropna()
    n_days = len(clean_rets)
    if n_days < 10:
        return {}

    # Equity curve (compounded)
    equity = (1.0 + clean_rets).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    
    # Compound Annual Growth Rate (CAGR)
    years = max(n_days / 252.0, 0.05)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    # Volatility
    ann_vol = float(clean_rets.std() * np.sqrt(252))

    # Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_rets = clean_rets - daily_rf
    sharpe = float(np.sqrt(252) * excess_rets.mean() / (clean_rets.std() + 1e-10))

    # Sortino Ratio (downside deviation only)
    downside = clean_rets[clean_rets < daily_rf] - daily_rf
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 1e-5
    sortino = float((cagr - risk_free_rate) / (downside_vol + 1e-10))

    # Maximum Drawdown & Calmar
    _, mdd = compute_drawdowns(equity)
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-4 else np.nan

    # Trade statistics
    positive_days = clean_rets[clean_rets > 0]
    negative_days = clean_rets[clean_rets < 0]
    win_rate = float(len(positive_days) / (len(positive_days) + len(negative_days) + 1e-10))
    profit_factor = float(positive_days.sum() / (abs(negative_days.sum()) + 1e-10))

    metrics = {
        "Total_Return": total_return,
        "CAGR": cagr,
        "Annualized_Volatility": ann_vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown": mdd,
        "Calmar_Ratio": calmar,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
    }

    # Benchmark Alpha and Beta (CAPM)
    if benchmark_returns is not None:
        aligned = pd.concat([clean_rets, benchmark_returns], axis=1).dropna()
        if len(aligned) > 20:
            strat_r = aligned.iloc[:, 0]
            bench_r = aligned.iloc[:, 1]
            cov = float(strat_r.cov(bench_r))
            var_b = float(bench_r.var())
            beta = cov / (var_b + 1e-10)
            alpha_ann = float((cagr - risk_free_rate) - beta * ((1.0 + bench_r).prod() ** (252 / len(bench_r)) - 1.0 - risk_free_rate))
            metrics["Beta_to_Benchmark"] = beta
            metrics["Alpha_Annualized"] = alpha_ann

    return metrics


def run_vectorized_backtest(
    prices: pd.Series,
    signals: pd.Series,
    benchmark_prices: Optional[pd.Series] = None,
    transaction_cost_bps: float = 10.0,
    prob_threshold: float = 0.55,
    allow_short: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute realistic friction-adjusted backtest on out-of-sample signals.
    
    Args:
        prices: Asset Close / Adj Close prices.
        signals: Predicted ML probabilities or discrete directional signals.
        benchmark_prices: SPY benchmark prices.
        transaction_cost_bps: Basis points friction per trade (10 bps = 0.0010 = 0.10%).
        prob_threshold: Probability hurdle to take a long position.
        allow_short: If True, takes short positions when prob < (1 - threshold).
        
    Returns:
        Tuple of (results_timeseries_df, performance_summary_dict).
    """
    aligned = pd.DataFrame(index=prices.index)
    aligned["price"] = prices
    aligned["signal"] = signals

    # Clean raw daily returns: R_t = (P_t / P_{t-1}) - 1
    aligned["asset_return"] = aligned["price"].pct_change().fillna(0.0)

    # Position sizing from signal: shifted by 1 day to prevent lookahead execution
    if allow_short:
        raw_position = np.where(aligned["signal"] > prob_threshold, 1.0, 
                                np.where(aligned["signal"] < (1.0 - prob_threshold), -1.0, 0.0))
    else:
        # Long-only or Cash (Flat)
        raw_position = np.where(aligned["signal"] > prob_threshold, 1.0, 0.0)

    # Shift position by 1 day: trade executes at next bar's close/open
    aligned["position"] = pd.Series(raw_position, index=aligned.index).shift(1).fillna(0.0)

    # Transaction costs: applied on position changes (turnover)
    turnover = (aligned["position"] - aligned["position"].shift(1)).abs().fillna(0.0)
    cost_penalty = turnover * (transaction_cost_bps / 10000.0)

    # Strategy daily returns after costs
    aligned["gross_strategy_return"] = aligned["position"] * aligned["asset_return"]
    aligned["net_strategy_return"] = aligned["gross_strategy_return"] - cost_penalty

    # Cumulative equity curves (starting at 1.0)
    aligned["equity_strategy"] = (1.0 + aligned["net_strategy_return"]).cumprod()
    aligned["equity_buy_and_hold"] = (1.0 + aligned["asset_return"]).cumprod()

    # Drawdowns
    dd_strat, _ = compute_drawdowns(aligned["equity_strategy"])
    dd_bh, _ = compute_drawdowns(aligned["equity_buy_and_hold"])
    aligned["drawdown_strategy"] = dd_strat
    aligned["drawdown_buy_and_hold"] = dd_bh

    # Benchmark comparison if provided
    bench_returns = None
    if benchmark_prices is not None:
        bench_ret = benchmark_prices.reindex(aligned.index).pct_change().fillna(0.0)
        aligned["benchmark_return"] = bench_ret
        aligned["equity_benchmark"] = (1.0 + bench_ret).cumprod()
        bench_returns = bench_ret

    # Compute institutional metrics
    strat_metrics = compute_performance_metrics(aligned["net_strategy_return"], bench_returns)
    bh_metrics = compute_performance_metrics(aligned["asset_return"], bench_returns)
    
    summary = {
        "ML_Strategy": strat_metrics,
        "Buy_and_Hold": bh_metrics,
    }
    if bench_returns is not None:
        summary["Benchmark_SPY"] = compute_performance_metrics(bench_returns)

    return aligned, summary
