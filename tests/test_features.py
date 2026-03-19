"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import clean_ohlcv
from src.data.features import add_features, get_feature_columns
from src.data.fetch import make_sample_ohlcv


@pytest.fixture
def clean_df():
    df = make_sample_ohlcv(n=300)
    return clean_ohlcv(df)


def test_add_features_returns_dataframe(clean_df):
    result = add_features(clean_df)
    assert isinstance(result, pd.DataFrame)


def test_add_features_has_target(clean_df):
    result = add_features(clean_df)
    assert "target" in result.columns
    assert "target_dir" in result.columns


def test_add_features_no_nan(clean_df):
    result = add_features(clean_df, drop_na=True)
    assert not result.isnull().any().any()


def test_add_features_with_nan_kept(clean_df):
    # With drop_na=False NaN rows are retained (from rolling / lag)
    result = add_features(clean_df, drop_na=False)
    assert result.isnull().any().any()


def test_add_features_lag_columns(clean_df):
    result = add_features(clean_df, lag_periods=3)
    for lag in range(1, 4):
        assert f"close_lag_{lag}" in result.columns


def test_add_features_rolling_columns(clean_df):
    result = add_features(clean_df)
    assert "close_ma_20" in result.columns
    assert "close_std_20" in result.columns


def test_add_features_temporal(clean_df):
    result = add_features(clean_df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns


def test_add_features_rsi(clean_df):
    result = add_features(clean_df)
    assert "rsi_14" in result.columns


def test_get_feature_columns_excludes_target(clean_df):
    result = add_features(clean_df)
    cols = get_feature_columns(result)
    assert "target" not in cols
    assert "target_dir" not in cols


def test_target_dir_binary(clean_df):
    result = add_features(clean_df)
    assert set(result["target_dir"].unique()).issubset({0, 1})
