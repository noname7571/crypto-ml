"""FastAPI application for serving cryptocurrency price predictions.

Endpoints
---------
GET  /health          — liveness probe
GET  /info            — model metadata
POST /predict         — single-step price prediction
POST /predict/batch   — batch prediction

Start locally::

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# In-memory model state (populated on startup via load_model())
# ---------------------------------------------------------------------------

_state: dict = {
    "model": None,
    "model_type": None,   # "xgboost" | "lstm"
    "feature_names": [],
    "feature_scaler": None,
    "target_scaler": None,
    "seq_len": 24,
    "metadata": {},
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """A single prediction request carrying a flat feature vector."""
    features: List[float] = Field(
        ...,
        description="Feature values in the order defined by feature_names.",
    )


class BatchPredictRequest(BaseModel):
    """A batch of prediction requests."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors.",
    )


class PredictResponse(BaseModel):
    prediction: float
    model_type: Optional[str] = None


class BatchPredictResponse(BaseModel):
    predictions: List[float]
    model_type: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_type: Optional[str]
    feature_names: List[str]
    seq_len: int
    metadata: dict


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    """Attempt to load a model from MODEL_PATH env var on startup."""
    model_path = os.getenv("MODEL_PATH", "")
    model_type = os.getenv("MODEL_TYPE", "xgboost")
    if model_path:
        try:
            load_model(model_path, model_type)
        except Exception as exc:
            logger.warning(f"Could not load model at startup: {exc}")
    else:
        logger.info("MODEL_PATH not set — starting without a loaded model")
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="crypto-ml API",
    description="Serve cryptocurrency price predictions from trained ML models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(path: str, model_type: str = "xgboost") -> None:
    """Load a persisted model into the application state.

    Parameters
    ----------
    path:
        File-system path to a pickled model (``.pkl``) or an MLflow run
        directory.
    model_type:
        ``"xgboost"`` or ``"lstm"``.
    """
    import pickle

    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    _state["model"] = payload.get("model")
    _state["model_type"] = model_type
    _state["feature_names"] = payload.get("feature_names", [])
    _state["feature_scaler"] = payload.get("feature_scaler")
    _state["target_scaler"] = payload.get("target_scaler")
    _state["seq_len"] = payload.get("seq_len", 24)
    _state["metadata"] = payload.get("metadata", {})
    logger.info(f"Model loaded from {path} (type={model_type})")


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _run_prediction(features_2d: np.ndarray) -> np.ndarray:
    """Apply scaler → model → inverse-scale and return raw prices."""
    model = _state["model"]
    feature_scaler = _state["feature_scaler"]
    target_scaler = _state["target_scaler"]
    model_type = _state["model_type"]

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is loaded. Set MODEL_PATH and restart the server.",
        )

    X = feature_scaler.transform(features_2d) if feature_scaler else features_2d

    if model_type == "lstm":
        import torch
        seq_len = _state["seq_len"]
        if X.shape[0] < seq_len:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"LSTM requires at least {seq_len} feature rows; got {X.shape[0]}.",
            )
        # Use the last seq_len rows
        X_seq = X[-seq_len:][np.newaxis, ...]  # (1, seq_len, F)
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_seq, dtype=torch.float32)
            pred_scaled = model(tensor).squeeze(-1).numpy()
    else:
        pred_scaled = model.predict(X)

    if target_scaler:
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    else:
        pred = pred_scaled

    return pred


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    """Return model metadata."""
    return InfoResponse(
        model_type=_state["model_type"],
        feature_names=_state["feature_names"],
        seq_len=_state["seq_len"],
        metadata=_state["metadata"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the next-period close price from a single feature vector."""
    features_2d = np.array(request.features, dtype=float).reshape(1, -1)
    pred = _run_prediction(features_2d)
    return PredictResponse(
        prediction=float(pred[0]),
        model_type=_state["model_type"],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict next-period close prices for a batch of feature vectors."""
    features_2d = np.array(request.instances, dtype=float)
    pred = _run_prediction(features_2d)
    return BatchPredictResponse(
        predictions=pred.tolist(),
        model_type=_state["model_type"],
    )
