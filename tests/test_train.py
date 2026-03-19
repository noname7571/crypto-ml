"""Tests for the end-to-end training pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import clean_ohlcv
from src.data.features import add_features, get_feature_columns
from src.data.fetch import make_sample_ohlcv
from src.training.train import (
    load_data,
    prepare_features,
    scale_features,
    temporal_split,
)


@pytest.fixture
def feature_df():
    raw = make_sample_ohlcv(n=400)
    return prepare_features(raw, lag_periods=3)


def test_load_data_synthetic():
    df = load_data(None, "BTCUSDT", "1h")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_prepare_features_no_nan(feature_df):
    assert not feature_df.isnull().any().any()


def test_temporal_split_sizes(feature_df):
    train, val, test = temporal_split(feature_df, val_frac=0.1, test_frac=0.1)
    total = len(train) + len(val) + len(test)
    assert total == len(feature_df)
    # Train should be largest
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_temporal_split_no_overlap(feature_df):
    train, val, test = temporal_split(feature_df)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_scale_features_range(feature_df):
    import numpy as np
    train, val, test = temporal_split(feature_df)
    cols = get_feature_columns(feature_df)
    result = scale_features(train, val, test, cols)
    (X_train, _), (X_val, _), (X_test, _), _, _ = result
    # Training data should be in [0, 1] after MinMaxScaler
    assert X_train.min() >= -1e-6
    assert X_train.max() <= 1 + 1e-6
