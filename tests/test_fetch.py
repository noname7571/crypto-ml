"""Tests for data fetching utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.fetch import make_sample_ohlcv


def test_make_sample_ohlcv_shape():
    df = make_sample_ohlcv(n=100)
    assert len(df) == 100
    assert set(df.columns) >= {"open", "high", "low", "close", "volume", "num_trades"}


def test_make_sample_ohlcv_index():
    df = make_sample_ohlcv(n=50)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "open_time"
    assert str(df.index.tz) == "UTC"


def test_make_sample_ohlcv_types():
    df = make_sample_ohlcv(n=50)
    for col in ["open", "high", "low", "close", "volume"]:
        assert df[col].dtype == float, f"{col} should be float"
    assert df["num_trades"].dtype in (int, "int64", "int32")


def test_make_sample_ohlcv_attrs():
    df = make_sample_ohlcv(n=50, symbol="ETHUSDT")
    assert df.attrs["symbol"] == "ETHUSDT"
    assert df.attrs["interval"] == "1h"


def test_make_sample_ohlcv_positive_prices():
    df = make_sample_ohlcv(n=200)
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()
