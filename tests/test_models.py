"""Tests for ML models."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import directional_accuracy, evaluate_regression
from src.models.lstm import LSTMModel, make_sequences
from src.models.xgboost_model import XGBoostPricePredictor


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_regression_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate_regression(y, y)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["r2"] == pytest.approx(1.0)


def test_evaluate_regression_keys():
    y_true = np.random.rand(50)
    y_pred = np.random.rand(50)
    metrics = evaluate_regression(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "mape"}


def test_directional_accuracy_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Predictions that always go in the right direction
    y_pred = y_true.copy()
    da = directional_accuracy(y_true, y_pred)
    assert da == pytest.approx(1.0)


def test_directional_accuracy_range():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100).cumsum() + 100
    y_pred = rng.standard_normal(100).cumsum() + 100
    da = directional_accuracy(y_true, y_pred)
    assert 0.0 <= da <= 1.0


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@pytest.fixture
def xgb_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = rng.standard_normal(200)
    return X, y


def test_xgboost_fit_predict(xgb_data):
    X, y = xgb_data
    model = XGBoostPricePredictor(n_estimators=20, max_depth=3)
    model.fit(X[:160], y[:160], eval_set=[(X[160:], y[160:])])
    preds = model.predict(X[160:])
    assert preds.shape == (40,)


def test_xgboost_predict_before_fit():
    model = XGBoostPricePredictor()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((5, 3)))


def test_xgboost_feature_importances(xgb_data):
    X, y = xgb_data
    model = XGBoostPricePredictor(n_estimators=10, max_depth=3)
    model.fit(X[:160], y[:160], eval_set=[(X[160:], y[160:])])
    names = [f"feat_{i}" for i in range(X.shape[1])]
    importances = model.get_feature_importances(feature_names=names)
    assert len(importances) == X.shape[1]
    assert all(v >= 0 for v in importances.values())


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def test_lstm_forward_pass():
    model = LSTMModel(input_size=10, hidden_size=32, num_layers=1)
    import torch
    x = torch.randn(8, 24, 10)  # batch=8, seq=24, features=10
    out = model(x)
    assert out.shape == (8, 1)


def test_make_sequences_shape():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    X_seq, y_seq = make_sequences(X, y, seq_len=10)
    assert X_seq.shape == (90, 10, 5)
    assert y_seq.shape == (90,)


def test_make_sequences_min_length():
    X = np.random.rand(5, 3)
    y = np.random.rand(5)
    X_seq, y_seq = make_sequences(X, y, seq_len=10)
    assert len(X_seq) == 0
