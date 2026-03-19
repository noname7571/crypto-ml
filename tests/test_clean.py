"""Tests for data cleaning utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_ohlcv, forward_fill_gaps
from src.data.fetch import make_sample_ohlcv


@pytest.fixture
def sample_df():
    return make_sample_ohlcv(n=200)


def test_clean_passes_clean_data(sample_df):
    cleaned = clean_ohlcv(sample_df)
    assert len(cleaned) == len(sample_df)


def test_clean_drops_nan_rows(sample_df):
    df = sample_df.copy()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    cleaned = clean_ohlcv(df)
    assert len(cleaned) == len(sample_df) - 1


def test_clean_drops_duplicate_index(sample_df):
    # Duplicate the first row
    dup = pd.concat([sample_df.iloc[[0]], sample_df])
    cleaned = clean_ohlcv(dup, drop_duplicates=True)
    assert not cleaned.index.duplicated().any()


def test_clean_drops_non_positive_close(sample_df):
    df = sample_df.copy()
    df.iloc[0, df.columns.get_loc("close")] = -100
    df.iloc[0, df.columns.get_loc("low")] = -200
    cleaned = clean_ohlcv(df)
    assert (cleaned["close"] > 0).all()


def test_clean_requires_ohlcv_columns():
    bad_df = pd.DataFrame({"open": [1], "close": [2]})
    with pytest.raises(ValueError, match="missing columns"):
        clean_ohlcv(bad_df)


def test_clean_sorted_index(sample_df):
    shuffled = sample_df.sample(frac=1, random_state=42)
    cleaned = clean_ohlcv(shuffled)
    assert cleaned.index.is_monotonic_increasing


def test_forward_fill_gaps():
    df = make_sample_ohlcv(n=100)
    # Remove 5 rows to create gaps
    df_gaps = df.drop(df.index[10:15])
    filled = forward_fill_gaps(df_gaps, freq="1h")
    assert len(filled) == 100
