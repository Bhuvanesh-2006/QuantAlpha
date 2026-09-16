"""Data Ingestion and Caching Module for Quantitative Alpha Engine.

Downloads, validates, and locally caches multi-asset financial time-series 
using yfinance with support for benchmark relative alignment and volatility proxies.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _flatten_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns if returned by yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance often returns MultiIndex like ('Close', 'AAPL')
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def fetch_ticker_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch OHLCV historical data for a ticker with local disk caching.
    
    Args:
        ticker: Symbol e.g. 'AAPL', 'MSFT', 'SPY'.
        start_date: Format 'YYYY-MM-DD'.
        end_date: Format 'YYYY-MM-DD' or None (defaults to today).
        use_cache: If True, reads from parquet/csv cache if available.
        cache_dir: Directory to store cache files.
        
    Returns:
        pd.DataFrame with standard DatetimeIndex and clean numeric columns.
    """
    cdir = cache_dir or CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    
    clean_end = end_date or "latest"
    cache_file = cdir / f"{ticker.upper()}_{start_date}_{clean_end}.parquet"
    cache_csv = cdir / f"{ticker.upper()}_{start_date}_{clean_end}.csv"
    
    if use_cache:
        if cache_file.exists():
            logger.info(f"Loading cached parquet for {ticker} from {cache_file}")
            df = pd.read_parquet(cache_file)
            return df
        elif cache_csv.exists():
            logger.info(f"Loading cached csv for {ticker} from {cache_csv}")
            df = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
            return df

    logger.info(f"Downloading historical data for {ticker} from {start_date} to {end_date or 'today'} via yfinance...")
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if data.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'. Please verify symbol and dates.")

    data = _flatten_yfinance_df(data)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Expected column '{col}' missing from downloaded data for {ticker}")

    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    # Basic data cleaning: remove duplicate timestamps, forward-fill minor gaps
    data = data[~data.index.duplicated(keep="first")]
    data = data.ffill().dropna()

    # Cache locally
    try:
        data.to_parquet(cache_file)
    except Exception:
        data.to_csv(cache_csv)

    return data


def fetch_multi_asset_universe(
    target_ticker: str,
    benchmark_ticker: str = "SPY",
    volatility_ticker: str = "^VIX",
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetch target ticker, benchmark ETF (SPY), and optional VIX index aligned to same dates.
    
    Returns:
        Tuple of (target_df, benchmark_df, vix_df) aligned on intersection of trading dates.
    """
    target_df = fetch_ticker_data(target_ticker, start_date, end_date, use_cache=use_cache)
    benchmark_df = fetch_ticker_data(benchmark_ticker, start_date, end_date, use_cache=use_cache)

    vix_df = None
    try:
        vix_df = fetch_ticker_data(volatility_ticker, start_date, end_date, use_cache=use_cache)
    except Exception as e:
        logger.warning(f"Could not fetch volatility index {volatility_ticker}: {e}. Proceeding without VIX.")

    # Align trading days (intersection of indices)
    common_idx = target_df.index.intersection(benchmark_df.index)
    if vix_df is not None:
        common_idx = common_idx.intersection(vix_df.index)
        vix_df = vix_df.loc[common_idx]

    target_df = target_df.loc[common_idx].copy()
    benchmark_df = benchmark_df.loc[common_idx].copy()

    logger.info(f"Successfully aligned multi-asset universe across {len(common_idx)} trading days.")
    return target_df, benchmark_df, vix_df
