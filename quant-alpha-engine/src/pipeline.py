"""End-to-End Quantitative ML Pipeline Orchestrator.

Integrates Ingestion -> Stationarity Testing -> Alpha Factor Engineering ->
Purged Walk-Forward Training -> Explainable AI -> Realistic Backtesting -> Report Generation.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import fetch_multi_asset_universe
from src.feature_engineering import extract_alpha_factors, adfuller_test
from src.labeling import generate_labels
from src.models import train_and_evaluate_walk_forward
from src.explainability import compute_feature_importance
from src.backtester import run_vectorized_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    ticker: str = "AAPL",
    benchmark: str = "SPY",
    start_date: str = "2019-01-01",
    end_date: Optional[str] = None,
    horizon_days: int = 5,
    threshold_multiplier: float = 0.20,
    prob_threshold: float = 0.52,
    transaction_cost_bps: float = 10.0,
    export_plots: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete end-to-end quantitative ML workflow.
    
    Returns:
        Dictionary containing datasets, metrics, models, backtest results, and factor ranks.
    """
    out_dir = output_dir or (Path(__file__).resolve().parent.parent / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"   QUANTALPHA: SYSTEMATIC ML ALPHA ENGINE & WALK-FORWARD BACKTESTER")
    print(f"{'='*70}")
    print(f"Target Ticker    : {ticker.upper()}")
    print(f"Benchmark        : {benchmark.upper()}")
    print(f"Historical Range : {start_date} to {end_date or 'latest'}")
    print(f"Prediction Horizon: {horizon_days} Trading Days (1-Week Alpha)")
    print(f"Execution Friction: {transaction_cost_bps} bps per trade\n")

    # 1. Ingestion
    logger.info("Step 1: Multi-Asset Ingestion & Caching...")
    target_df, bench_df, vix_df = fetch_multi_asset_universe(
        target_ticker=ticker,
        benchmark_ticker=benchmark,
        start_date=start_date,
        end_date=end_date,
    )

    # 2. Stationarity Verification (ML Best Practices)
    logger.info("Step 2: Checking stationarity with Augmented Dickey-Fuller test...")
    raw_price_adf = adfuller_test(target_df["Adj Close"])
    log_ret_adf = adfuller_test(np.log(target_df["Adj Close"] / target_df["Adj Close"].shift(1)))
    print(f"\n[Stationarity Diagnostics - ADF Test]")
    print(f"  - Raw Price Level: t-stat={raw_price_adf['test_stat']:.3f}, p-val={raw_price_adf['p_value']:.4f}, Stationary={raw_price_adf['is_stationary']}")
    print(f"  - Log Daily Return: t-stat={log_ret_adf['test_stat']:.3f}, p-val={log_ret_adf['p_value']:.4f}, Stationary={log_ret_adf['is_stationary']}")
    if not raw_price_adf["is_stationary"] and log_ret_adf["is_stationary"]:
        print("  -> CONFIRMED: Raw prices are non-stationary I(1). Quantitative features strictly built on stationary returns/ratios.\n")

    # 3. Factor Engineering
    logger.info("Step 3: Calculating 25+ Quantitative Alpha Factors...")
    features_df, factor_groups = extract_alpha_factors(
        df=target_df,
        benchmark_df=bench_df,
        vix_df=vix_df,
        drop_warmup=True,
    )

    # 4. Target Labeling (Triple Barrier / Forward Alpha)
    logger.info("Step 4: Formulating Risk-Adjusted Forward Labels...")
    labels_df = generate_labels(
        asset_df=target_df.loc[features_df.index],
        benchmark_df=bench_df.loc[features_df.index] if bench_df is not None else None,
        horizon_days=horizon_days,
        threshold_std_multiplier=threshold_multiplier,
        binary_classification=True,
    )

    # Align features and targets
    aligned_idx = features_df.index.intersection(labels_df.dropna().index)
    X = features_df.loc[aligned_idx]
    y = labels_df.loc[aligned_idx, "label"]

    # 5. Purged Walk-Forward Cross-Validation
    logger.info("Step 5: Executing Purged Walk-Forward Time-Series Cross Validation...")
    cv_metrics, fitted_models, oos_preds = train_and_evaluate_walk_forward(
        X=X,
        y=y,
        n_splits=5,
        purge_window=horizon_days,
    )
    print("\n[Cross-Validation Model Performance]")
    print(cv_metrics.to_string())

    # 6. Explainability & Factor Attribution
    logger.info("Step 6: Computing Global Factor Importance and Group Attributions...")
    best_model_name = "Random Forest"
    best_model = fitted_models[best_model_name]
    
    # Validation slice for permutation importance
    split_pt = int(len(X) * 0.8)
    X_val_scaled = fitted_models["scaler"].transform(X.iloc[split_pt:])
    y_val = y.iloc[split_pt:].values

    feat_importance_df, group_importance_df = compute_feature_importance(
        model=best_model,
        feature_names=list(X.columns),
        factor_groups=factor_groups,
        X_val=X_val_scaled,
        y_val=y_val,
    )
    print("\n[Top 10 Most Predictive Alpha Factors]")
    print(feat_importance_df.head(10)[["Relative_Importance_Pct"]].to_string())

    print("\n[Alpha Domain Category Importance]")
    print(group_importance_df.to_string())

    # 7. Institutional Backtesting
    logger.info("Step 7: Simulating Realistic Friction-Adjusted Strategy Execution...")
    selected_signal_col = f"{best_model_name}_prob"
    backtest_signals = oos_preds[selected_signal_col].dropna()
    aligned_prices = target_df.loc[backtest_signals.index, "Adj Close"]
    aligned_bench = bench_df.loc[backtest_signals.index, "Adj Close"] if bench_df is not None else None

    bt_timeseries, bt_summary = run_vectorized_backtest(
        prices=aligned_prices,
        signals=backtest_signals,
        benchmark_prices=aligned_bench,
        transaction_cost_bps=transaction_cost_bps,
        prob_threshold=prob_threshold,
    )

    print("\n[Out-of-Sample Backtesting Performance Summary]")
    summary_df = pd.DataFrame(bt_summary)
    print(summary_df.to_string())

    # 8. Export Publication Plots
    if export_plots:
        logger.info(f"Step 8: Exporting visual plots to {out_dir}...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

        # Plot 1: Cumulative Equity Curves
        axes[0].plot(bt_timeseries.index, bt_timeseries["equity_strategy"], label=f"QuantAlpha ML Strategy ({best_model_name})", color="#2ecc71", lw=2)
        axes[0].plot(bt_timeseries.index, bt_timeseries["equity_buy_and_hold"], label=f"Buy & Hold ({ticker.upper()})", color="#3498db", lw=1.5, ls="--")
        if "equity_benchmark" in bt_timeseries.columns:
            axes[0].plot(bt_timeseries.index, bt_timeseries["equity_benchmark"], label=f"Benchmark ({benchmark.upper()})", color="#95a5a6", lw=1.2, ls=":")
        axes[0].set_title(f"Out-of-Sample Equity Growth (Compounded, Net of {transaction_cost_bps} bps Friction)")
        axes[0].set_ylabel("Portfolio Value (Base = 1.0)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2: Drawdowns
        axes[1].fill_between(bt_timeseries.index, bt_timeseries["drawdown_strategy"] * 100, 0, color="#e74c3c", alpha=0.3, label="ML Strategy Drawdown %")
        axes[1].plot(bt_timeseries.index, bt_timeseries["drawdown_buy_and_hold"] * 100, color="#7f8c8d", alpha=0.7, ls="--", label="Asset Drawdown %")
        axes[1].set_title("Drawdown Profile (Downside Risk Preservation)")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Plot 3: Top Feature Importances
        top_feats = feat_importance_df.head(10).iloc[::-1]
        axes[2].barh(top_feats.index, top_feats["Relative_Importance_Pct"], color="#34495e")
        axes[2].set_title("Top 10 Predictive Alpha Factors (Relative Contribution %)")
        axes[2].set_xlabel("Contribution (%)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = out_dir / f"{ticker.upper()}_quant_summary.png"
        plt.savefig(chart_path, dpi=200)
        plt.close()
        print(f"\n-> Visual report exported to: {chart_path}\n")

    return {
        "target_df": target_df,
        "features_df": features_df,
        "labels_df": labels_df,
        "cv_metrics": cv_metrics,
        "fitted_models": fitted_models,
        "oos_preds": oos_preds,
        "feat_importance_df": feat_importance_df,
        "group_importance_df": group_importance_df,
        "bt_timeseries": bt_timeseries,
        "bt_summary": bt_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantAlpha: Machine Learning Alpha Factor Backtesting Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Target equity ticker symbol")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark index ticker symbol")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in trading days")
    parser.add_argument("--friction", type=float, default=10.0, help="Transaction cost in basis points")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.ticker,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
        horizon_days=args.horizon,
        transaction_cost_bps=args.friction,
    )
