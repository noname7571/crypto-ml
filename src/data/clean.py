"""Clean and validate raw OHLCV DataFrames."""

from __future__ import annotations

import pandas as pd
from loguru import logger


def clean_ohlcv(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Apply a sequence of cleaning steps to a raw OHLCV DataFrame.

    Steps
    -----
    1. Require the expected columns (``open``, ``high``, ``low``, ``close``,
       ``volume``).
    2. Drop rows where any OHLCV column is NaN.
    3. Optionally drop duplicate index entries, keeping the last.
    4. Enforce OHLC sanity: ``low <= open, close <= high``.
    5. Remove rows with non-positive ``close`` or ``volume``.
    6. Sort by index.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame with a DatetimeIndex.
    drop_duplicates:
        Whether to deduplicate on the index.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df*.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    original_len = len(df)
    df = df.copy()

    # 1. Drop NaN in core columns
    df.dropna(subset=list(required), inplace=True)

    # 2. Deduplicate
    if drop_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # 3. OHLC sanity
    bad_ohlc = (df["low"] > df["open"]) | (df["low"] > df["close"]) | \
               (df["high"] < df["open"]) | (df["high"] < df["close"])
    if bad_ohlc.any():
        logger.warning(f"Dropping {bad_ohlc.sum()} rows with invalid OHLC relationships")
        df = df[~bad_ohlc]

    # 4. Positive price & volume
    df = df[(df["close"] > 0) & (df["volume"] > 0)]

    # 5. Sort
    df.sort_index(inplace=True)

    dropped = original_len - len(df)
    if dropped:
        logger.info(f"Cleaned {dropped} rows ({original_len} → {len(df)})")
    else:
        logger.info(f"No rows dropped during cleaning ({len(df)} rows)")

    return df


def forward_fill_gaps(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Reindex to a regular frequency and forward-fill gaps.

    This is useful when Binance occasionally misses candles during low-liquidity
    periods.

    Parameters
    ----------
    df:
        Cleaned OHLCV DataFrame.
    freq:
        Target frequency string (e.g. ``"1h"``, ``"1d"``).

    Returns
    -------
    pd.DataFrame
        Gap-filled DataFrame.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    df_reindexed = df.reindex(full_idx)
    filled = df_reindexed.ffill()
    gaps = df_reindexed.isna().any(axis=1).sum()
    if gaps:
        logger.info(f"Forward-filled {gaps} missing candles at freq={freq}")
    return filled
