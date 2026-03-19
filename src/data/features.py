"""Feature engineering for cryptocurrency OHLCV data.

Adds technical indicators and lag/return features that are commonly used as
inputs to price-prediction models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_features(
    df: pd.DataFrame,
    lag_periods: int = 5,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Add a comprehensive set of features to a cleaned OHLCV DataFrame.

    Features added
    --------------
    * Log returns and percentage returns
    * Rolling statistics (mean, std, z-score) over multiple windows
    * Technical indicators via the ``ta`` library:
      RSI-14, MACD, Bollinger Bands, ATR, OBV, EMA-20/50
    * Lag features for close price and volume
    * Temporal features (hour-of-day, day-of-week, month)
    * Target column: ``target`` = next-period close (for regression) and
      ``target_dir`` = 1 if next close > current close else 0 (for
      classification)

    Parameters
    ----------
    df:
        Cleaned OHLCV DataFrame with DatetimeIndex.
    lag_periods:
        How many lag candles to include.
    drop_na:
        Drop rows that contain NaN after feature generation.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()

    # ------------------------------------------------------------------
    # Rolling statistics
    # ------------------------------------------------------------------
    for window in [5, 10, 20, 50]:
        df[f"close_ma_{window}"] = df["close"].rolling(window).mean()
        df[f"close_std_{window}"] = df["close"].rolling(window).std()
        if window <= 20:
            df[f"close_zscore_{window}"] = (
                (df["close"] - df[f"close_ma_{window}"]) / df[f"close_std_{window}"]
            )

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # ------------------------------------------------------------------
    # Technical indicators (ta library)
    # ------------------------------------------------------------------
    if _TA_AVAILABLE:
        df = _add_ta_indicators(df)
    else:
        logger.warning("ta library not installed; skipping technical indicators")
        df = _add_manual_indicators(df)

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    for lag in range(1, lag_periods + 1):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"log_return_lag_{lag}"] = df["log_return"].shift(lag)

    # ------------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------------
    if hasattr(df.index, "hour"):
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ------------------------------------------------------------------
    # Target columns
    # ------------------------------------------------------------------
    df["target"] = df["close"].shift(-1)
    df["target_dir"] = (df["target"] > df["close"]).astype(int)

    # ------------------------------------------------------------------
    # Drop NaN rows introduced by rolling/lag/target
    # ------------------------------------------------------------------
    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        logger.info(f"Dropped {before - len(df)} NaN rows after feature engineering ({len(df)} remaining)")

    logger.info(f"Feature engineering complete: {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes target columns)."""
    return [c for c in df.columns if c not in ("target", "target_dir")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_ta_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators using the ``ta`` library."""
    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pband"] = bb.bollinger_pband()
    df["bb_wband"] = bb.bollinger_wband()

    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # EMA
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()

    return df


def _add_manual_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal fallback indicators when ``ta`` is unavailable."""
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # EMA
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    return df
