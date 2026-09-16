"""Interactive Gradio Web Application for QuantAlpha Engine.

Provides an interactive GUI for quantitative researchers and recruiters to test any ticker,
visualize walk-forward ML diagnostics, explore alpha factor attribution, and evaluate
friction-adjusted backtesting results.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from src.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def build_equity_plotly_chart(bt_timeseries: pd.DataFrame, ticker: str, benchmark: str) -> go.Figure:
    """Create interactive Plotly chart for Cumulative Returns and Drawdowns."""
    fig = go.Figure()

    # Strategy Equity Curve
    fig.add_trace(
        go.Scatter(
            x=bt_timeseries.index,
            y=bt_timeseries["equity_strategy"],
            mode="lines",
            name="QuantAlpha ML Strategy",
            line=dict(color="#10b981", width=2.5),
        )
    )

    # Buy & Hold Asset
    fig.add_trace(
        go.Scatter(
            x=bt_timeseries.index,
            y=bt_timeseries["equity_buy_and_hold"],
            mode="lines",
            name=f"Buy & Hold ({ticker})",
            line=dict(color="#3b82f6", width=1.8, dash="dash"),
        )
    )

    # Benchmark SPY
    if "equity_benchmark" in bt_timeseries.columns:
        fig.add_trace(
            go.Scatter(
                x=bt_timeseries.index,
                y=bt_timeseries["equity_benchmark"],
                mode="lines",
                name=f"Benchmark ({benchmark})",
                line=dict(color="#9ca3af", width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title=f"Out-of-Sample Cumulative Growth ({ticker} vs Benchmark)",
        xaxis_title="Date",
        yaxis_title="Portfolio Equity (Base = $1.00)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def build_drawdown_plotly_chart(bt_timeseries: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive Plotly chart for Strategy Drawdowns."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bt_timeseries.index,
            y=bt_timeseries["drawdown_strategy"] * 100,
            mode="lines",
            name="ML Strategy Drawdown %",
            fill="tozeroy",
            line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239, 68, 68, 0.2)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bt_timeseries.index,
            y=bt_timeseries["drawdown_buy_and_hold"] * 100,
            mode="lines",
            name=f"{ticker} Buy & Hold Drawdown %",
            line=dict(color="#94a3b8", width=1.2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Drawdown Profile (Preserving Capital in Market Crashes)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def build_factor_importance_plotly(feat_importance_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of top alpha factor contributions."""
    top_feats = feat_importance_df.head(10).iloc[::-1]
    
    fig = go.Figure(
        go.Bar(
            x=top_feats["Relative_Importance_Pct"],
            y=top_feats.index,
            orientation="h",
            marker=dict(
                color=top_feats["Relative_Importance_Pct"],
                colorscale="Viridis",
            ),
        )
    )
    fig.update_layout(
        title="Top 10 Most Predictive Alpha Factors (% Contribution)",
        xaxis_title="Relative Contribution (%)",
        yaxis_title="Alpha Factor",
        template="plotly_dark",
        margin=dict(l=120, r=40, t=50, b=40),
    )
    return fig


def analyze_ticker(
    ticker: str,
    benchmark: str,
    start_date: str,
    end_date: str,
    horizon_days: int,
    friction_bps: float,
    prob_threshold: float,
):
    """Main Gradio callback running pipeline and returning formatted components."""
    try:
        clean_end = end_date.strip() if end_date and len(end_date.strip()) > 0 else None
        results = run_pipeline(
            ticker=ticker.strip().upper(),
            benchmark=benchmark.strip().upper(),
            start_date=start_date.strip(),
            end_date=clean_end,
            horizon_days=int(horizon_days),
            transaction_cost_bps=float(friction_bps),
            prob_threshold=float(prob_threshold),
            export_plots=True,
        )

        bt_timeseries = results["bt_timeseries"]
        summary_dict = results["bt_summary"]
        cv_metrics = results["cv_metrics"]
        feat_imp = results["feat_importance_df"]
        group_imp = results["group_importance_df"]

        # 1. Plotly charts
        equity_fig = build_equity_plotly_chart(bt_timeseries, ticker.upper(), benchmark.upper())
        dd_fig = build_drawdown_plotly_chart(bt_timeseries, ticker.upper())
        feat_fig = build_factor_importance_plotly(feat_imp)

        # 2. Performance Summary DataFrame
        summary_df = pd.DataFrame(summary_dict).T.reset_index()
        summary_df.columns = ["Strategy", "Total Return", "CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar", "Win Rate", "Profit Factor", "Beta", "Alpha"]
        # Format metrics nicely
        for col in ["Total Return", "CAGR", "Volatility", "Max DD", "Win Rate", "Alpha"]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
        for col in ["Sharpe", "Sortino", "Calmar", "Profit Factor", "Beta"]:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

        # 3. CV Metrics DataFrame
        cv_formatted = cv_metrics.reset_index()
        for col in ["ROC_AUC", "Accuracy", "Precision", "Recall", "F1_Score", "Precision_Top_Decile"]:
            if col in cv_formatted.columns:
                cv_formatted[col] = cv_formatted[col].apply(lambda x: f"{x*100:.1f}%")
        if "Brier_Score" in cv_formatted.columns:
            cv_formatted["Brier_Score"] = cv_formatted["Brier_Score"].apply(lambda x: f"{x:.4f}")

        # 4. Factor tables
        feat_table = feat_imp.head(15).reset_index().rename(columns={"index": "Factor Name"})
        feat_table["Relative_Importance_Pct"] = feat_table["Relative_Importance_Pct"].apply(lambda x: f"{x:.2f}%")
        
        group_table = group_imp.copy()
        group_table["Share_Pct"] = group_table["Share_Pct"].apply(lambda x: f"{x:.2f}%")

        status_msg = f"Analysis completed successfully for {ticker.upper()}! Walk-forward backtest evaluated from {bt_timeseries.index[0].strftime('%Y-%m-%d')} to {bt_timeseries.index[-1].strftime('%Y-%m-%d')}."
        return equity_fig, dd_fig, summary_df, cv_formatted, feat_fig, feat_table, group_table, status_msg

    except Exception as e:
        logger.exception("Error in pipeline execution")
        return None, None, pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame(), f"Error: {str(e)}"


def create_app() -> gr.Blocks:
    """Build Gradio UI interface."""
    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="QuantAlpha: Systematic ML Trading & Alpha Factor Engine") as demo:
        gr.Markdown(
            """
            # QuantAlpha: Systematic ML Alpha Factor & Walk-Forward Backtester
            ### Institutional Quantitative Machine Learning Pipeline for Alpha Factor Generation & Risk-Managed Trading
            *Designed following Marcos López de Prado's 'Advances in Financial Machine Learning' (Purged Walk-Forward Time-Series Validation, Stationary Alpha Factors, Friction-Adjusted Execution).*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Parameter Configuration")
                ticker_input = gr.Dropdown(
                    choices=["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "SPY", "QQQ"],
                    value="AAPL",
                    label="Asset Ticker Symbol",
                    allow_custom_value=True,
                )
                benchmark_input = gr.Textbox(value="SPY", label="Benchmark Symbol (Market Index)")
                start_date_input = gr.Textbox(value="2019-01-01", label="Start Date (YYYY-MM-DD)")
                end_date_input = gr.Textbox(value="", label="End Date (Blank for latest)")
                
                with gr.Accordion("Quantitative Model & Risk Settings", open=False):
                    horizon_input = gr.Slider(minimum=1, maximum=21, value=5, step=1, label="Alpha Prediction Horizon (Trading Days)")
                    friction_input = gr.Slider(minimum=0.0, maximum=30.0, value=10.0, step=1.0, label="Transaction Friction (Basis Points)")
                    prob_threshold_input = gr.Slider(minimum=0.50, maximum=0.70, value=0.52, step=0.01, label="Signal Conviction Hurdle")

                run_btn = gr.Button("Run Quantitative Pipeline", variant="primary")
                status_box = gr.Textbox(label="Execution Status", interactive=False)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Strategy Backtest & Performance"):
                        equity_plot = gr.Plot(label="Equity Curve")
                        drawdown_plot = gr.Plot(label="Drawdown Profile")
                        summary_table = gr.Dataframe(label="Institutional Performance Metrics (Out-of-Sample)")

                    with gr.TabItem("ML Diagnostics & Cross-Validation"):
                        gr.Markdown("#### Purged Walk-Forward Time-Series Cross Validation")
                        gr.Markdown(
                            "Evaluates models chronologically without lookahead bias. "
                            "Features are scaled strictly on past training windows with an embargo window separating train/test."
                        )
                        cv_table = gr.Dataframe(label="Cross-Validation Metrics by Model")

                    with gr.TabItem("Explainable AI & Factor Attribution"):
                        gr.Markdown("#### Global Alpha Factor Attribution")
                        feat_plot = gr.Plot(label="Top Predictive Alpha Signals")
                        with gr.Row():
                            feat_table_out = gr.Dataframe(label="Individual Factor Rankings")
                            group_table_out = gr.Dataframe(label="Factor Category Share")

        run_btn.click(
            fn=analyze_ticker,
            inputs=[
                ticker_input,
                benchmark_input,
                start_date_input,
                end_date_input,
                horizon_input,
                friction_input,
                prob_threshold_input,
            ],
            outputs=[
                equity_plot,
                drawdown_plot,
                summary_table,
                cv_table,
                feat_plot,
                feat_table_out,
                group_table_out,
                status_box,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
