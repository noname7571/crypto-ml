"""Fetch historical OHLCV data from Binance REST API."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

try:
    from binance.client import Client
except ImportError:  # pragma: no cover
    Client = None  # type: ignore[assignment,misc]

# Binance kline column names
_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]

# Columns to keep after fetching
_KEEP_COLS = ["open_time", "open", "high", "low", "close", "volume", "num_trades"]


def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start: str = "1 Jan, 2022",
    end: Optional[str] = None,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """Download OHLCV candles from Binance and return as a tidy DataFrame.

    Parameters
    ----------
    symbol:
        Trading pair symbol, e.g. ``"BTCUSDT"``.
    interval:
        Kline interval string accepted by the Binance API (``"1m"``, ``"1h"``,
        ``"1d"``, …).
    start:
        Human-readable start date, e.g. ``"1 Jan, 2022"``.
    end:
        Human-readable end date.  Defaults to *now*.
    api_key / api_secret:
        Binance API credentials.  Public endpoints work without credentials;
        providing them raises the rate-limit ceiling.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and float-typed OHLCV columns.
    """
    if Client is None:
        raise ImportError("python-binance is required: pip install python-binance")

    client = Client(api_key, api_secret)
    logger.info(f"Fetching {symbol} {interval} candles from {start} …")

    klines = client.get_historical_klines(symbol, interval, start, end)
    df = pd.DataFrame(klines, columns=_KLINE_COLS)
    return _post_process(df, symbol, interval)


def _post_process(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    df = df[_KEEP_COLS].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df = df.set_index("open_time").sort_index()
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = interval
    logger.info(f"Fetched {len(df)} rows for {symbol} [{interval}]")
    return df


def save_ohlcv(df: pd.DataFrame, path: Path) -> None:
    """Persist an OHLCV DataFrame to Parquet format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Saved {len(df)} rows → {path}")


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load an OHLCV DataFrame from Parquet."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows ← {path}")
    return df


def make_sample_ohlcv(n: int = 500, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing / demo purposes."""
    import numpy as np

    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 30_000 + rng.standard_normal(n).cumsum() * 200
    open_ = close + rng.standard_normal(n) * 50
    high = np.maximum(close, open_) + rng.uniform(0, 100, n)
    low = np.minimum(close, open_) - rng.uniform(0, 100, n)
    volume = rng.uniform(100, 5000, n)
    num_trades = rng.integers(500, 5000, n)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "num_trades": num_trades,
        },
        index=pd.DatetimeIndex(dates, name="open_time"),
    )
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = "1h"
    return df
