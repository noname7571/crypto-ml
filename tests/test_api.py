"""Tests for the FastAPI application."""

from __future__ import annotations

import pickle

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, _state, lifespan


@pytest.fixture
def client(monkeypatch):
    # Keep default API tests deterministic by preventing implicit startup autoload.
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL_PATH", "__missing_model_bundle__.pkl")
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure model state is cleared between tests."""
    _state["model"] = None
    _state["model_type"] = None
    _state["feature_names"] = []
    _state["feature_scaler"] = None
    _state["target_scaler"] = None
    _state["seq_len"] = 24
    _state["metadata"] = {}
    yield


def test_health_no_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_health_with_model(client):
    _state["model"] = object()  # dummy non-None object
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_info_endpoint(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data
    assert "feature_names" in data
    assert "seq_len" in data


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "crypto-ml API" in resp.text
    assert "/docs" in resp.text


def test_status_endpoint(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "crypto-ml API"
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"
    assert data["status"] == "/status"
    assert data["model_loaded"] is False


def test_predict_no_model_returns_503(client):
    resp = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 503


def test_predict_batch_no_model_returns_503(client):
    resp = client.post(
        "/predict/batch",
        json={"instances": [[1.0, 2.0], [3.0, 4.0]]},
    )
    assert resp.status_code == 503


def test_predict_with_mock_model(client):
    """Test /predict with a mock XGBoost-like model and no scalers."""
    import numpy as np

    class MockModel:
        def predict(self, X):
            return np.ones(len(X)) * 42.0

    _state["model"] = MockModel()
    _state["model_type"] = "xgboost"
    _state["feature_scaler"] = None
    _state["target_scaler"] = None

    resp = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 200
    assert resp.json()["prediction"] == pytest.approx(42.0)
    assert resp.json()["model_type"] == "xgboost"


def test_predict_batch_with_mock_model(client):
    """Test /predict/batch with a mock model."""
    import numpy as np

    class MockModel:
        def predict(self, X):
            return np.full(len(X), 99.0)

    _state["model"] = MockModel()
    _state["model_type"] = "xgboost"

    resp = client.post(
        "/predict/batch",
        json={"instances": [[1.0], [2.0], [3.0]]},
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 3
    assert all(p == pytest.approx(99.0) for p in preds)


def test_predict_wrong_feature_count_returns_422(client):
    class MockModel:
        def predict(self, X):
            return [1.0 for _ in range(len(X))]

    _state["model"] = MockModel()
    _state["model_type"] = "xgboost"
    _state["feature_names"] = ["f1", "f2", "f3"]

    resp = client.post("/predict", json={"features": [1.0, 2.0]})
    assert resp.status_code == 422
    assert "Expected 3 features, got 2." in resp.json()["detail"]


def test_predict_batch_wrong_feature_count_returns_422(client):
    class MockModel:
        def predict(self, X):
            return [1.0 for _ in range(len(X))]

    _state["model"] = MockModel()
    _state["model_type"] = "xgboost"
    _state["feature_names"] = ["f1", "f2"]

    resp = client.post(
        "/predict/batch",
        json={"instances": [[1.0], [2.0]]},
    )
    assert resp.status_code == 422
    assert "Expected 2 features, got 1." in resp.json()["detail"]


def test_predict_batch_empty_instances_returns_422(client):
    class MockModel:
        def predict(self, X):
            return [1.0 for _ in range(len(X))]

    _state["model"] = MockModel()
    _state["model_type"] = "xgboost"

    resp = client.post("/predict/batch", json={"instances": []})
    assert resp.status_code == 422
    assert "at least one feature vector" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_lifespan_autoloads_default_model_path(tmp_path, monkeypatch):
    bundle_path = tmp_path / "model_bundle.pkl"
    payload = {
        "model": "placeholder-model",
        "feature_names": ["f1", "f2"],
        "feature_scaler": None,
        "target_scaler": None,
        "seq_len": 1,
        "metadata": {"model_type": "xgboost"},
    }
    with bundle_path.open("wb") as fh:
        pickle.dump(payload, fh)

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL_PATH", str(bundle_path))
    monkeypatch.delenv("MODEL_TYPE", raising=False)

    async with lifespan(app):
        assert _state["model"] is not None
        assert _state["model_type"] == "xgboost"
        assert _state["feature_names"] == ["f1", "f2"]
