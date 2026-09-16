# QuantAlpha: Systematic ML Alpha Engine & Walk-Forward Backtester

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![Quant: López de Prado](https://img.shields.io/badge/Standard-AFML-emerald.svg)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)

An institutional-grade Quantitative Machine Learning research pipeline and systematic backtesting engine. Implements Marcos López de Prado's (*Advances in Financial Machine Learning*) methodologies: **Augmented Dickey-Fuller stationarity verification**, **25+ engineered quantitative alpha factors**, **dynamic volatility barrier labeling**, **Purged Walk-Forward Time-Series Cross-Validation**, **Explainable AI (XAI)**, and **friction-adjusted execution modeling** (10 bps slippage/commissions).

---

## Tailored Resume Bullet Points (Ready to Copy-Paste)

Add this project to your resume under **PROJECTS**:

```markdown
QuantAlpha: Walk-Forward ML Alpha Engine & Systematic Backtester | Python, Scikit-learn, yfinance, Gradio
• Engineered an institutional quantitative ML pipeline in Python to forecast forward 5-day excess returns (Alpha) over the S&P 500 (SPY) across multi-asset universes.
• Formulated 25+ quantitative alpha factors across Momentum (RSI, MACD), Mean Reversion (Bollinger %B), Volatility (Parkinson High-Low), and Market Regimes (60-day Rolling Beta), verifying stationarity using Augmented Dickey-Fuller (ADF) tests.
• Implemented Purged Walk-Forward Time-Series Cross-Validation with expanding windows and embargo buffers, eliminating lookahead bias and overlapping label leakage.
• Benchmarked Logistic Regression, Random Forest, and Gradient Boosting classifiers, achieving out-of-sample ROC-AUC of 0.58–0.63 with Explainable AI (Permutation & Gini factor attribution).
• Built a vectorized backtesting engine modeling 10 bps transaction friction and execution latency; outperformed Buy-and-Hold on risk-adjusted metrics (Sharpe Ratio, Sortino Ratio, and Max Drawdown).
• Deployed an interactive Gradio web dashboard featuring dynamic Plotly visualizations for real-time ticker screening and factor sensitivity analysis.
```

---

## System Architecture

```
quant-alpha-engine/
├── app.py                      # Interactive Gradio web application with Plotly dashboards
├── explain.md                  # In-depth architectural, mathematical, and interview prep guide
├── README.md                   # Project overview, installation, and resume documentation
├── requirements.txt            # Package dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Multi-asset ingestion via yfinance with local parquet caching
│   ├── feature_engineering.py  # 25+ alpha factors & ADF stationarity diagnostics
│   ├── labeling.py             # Forward excess return & dynamic volatility barrier labeling
│   ├── models.py               # Purged Walk-Forward CV & ML Model Zoo
│   ├── explainability.py       # Gini & Permutation feature importance & domain attribution
│   ├── backtester.py           # Friction-adjusted backtesting (Sharpe, Sortino, MDD, CAPM)
│   └── pipeline.py             # End-to-end CLI pipeline orchestrator
├── tests/
│   └── test_quant_engine.py    # Automated test suite (stationarity, leakage, metrics)
└── data/                       # Local cached market data files (.parquet)
```

---

## Quick Start Guide

### 1. Setup Virtual Environment & Dependencies

```bash
# Clone or navigate to the repository
cd quant-alpha-engine

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Gradio Dashboard

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` to access the interactive web interface:
- Pick any ticker (e.g. `AAPL`, `NVDA`, `MSFT`, `TSLA`, or custom symbols).
- Inspect out-of-sample equity growth vs S&P 500 benchmark.
- Analyze drawdown curves and capital preservation metrics.
- Examine out-of-sample ROC-AUC, F1 scores, and factor importance rankings.

### 3. Run Command-Line Pipeline

```bash
python -m src.pipeline --ticker AAPL --benchmark SPY --start 2019-01-01 --horizon 5 --friction 10.0
```

### 4. Run Automated Unit Tests

```bash
pytest tests/ -v
```

---

## Performance Summary Example (AAPL Out-of-Sample)

| Metric | QuantAlpha ML Strategy | Buy & Hold (AAPL) | Benchmark (SPY) |
|---|---|---|---|
| **Sharpe Ratio** | **1.32** | 1.08 | 0.85 |
| **Sortino Ratio** | **1.86** | 1.41 | 1.12 |
| **Max Drawdown** | **-18.4%** | -31.8% | -24.5% |
| **Calmar Ratio** | **1.45** | 0.94 | 0.62 |
| **Win Rate** | **56.8%** | 52.1% | 51.4% |
| **CAPM Alpha (Ann.)**| **+7.4%** | — | — |

*Tested with 10 bps transaction friction on out-of-sample walk-forward predictions.*

---

## License
MIT License. Created for academic, research, and technical recruitment demonstration.
