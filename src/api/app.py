"""FastAPI application for serving cryptocurrency price predictions.

Endpoints
---------
GET  /                — simple HTML landing page
GET  /status          — JSON service status
GET  /feature-spec    — expected feature schema for clients
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
from typing import Any, AsyncGenerator, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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


class RootResponse(BaseModel):
    service: str
    docs: str
    health: str
    status: str
    model_loaded: bool


class FeatureSpecResponse(BaseModel):
    model_loaded: bool
    model_type: Optional[str]
    expected_feature_count: int
    feature_names: List[str]


class ApiError(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ApiError


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    """Attempt to load a model from MODEL_PATH env var on startup."""
    model_path = os.getenv("MODEL_PATH", "")
    default_model_path = os.getenv("DEFAULT_MODEL_PATH", "artifacts/model_bundle.pkl")
    model_type = os.getenv("MODEL_TYPE", "xgboost")
    if model_path:
        try:
            load_model(model_path, model_type)
        except Exception as exc:
            logger.warning(f"Could not load model at startup: {exc}")
    elif os.path.exists(default_model_path):
        try:
            load_model(default_model_path, model_type)
            logger.info(f"Auto-loaded default model bundle from {default_model_path}")
        except Exception as exc:
            logger.warning(f"Could not auto-load default model bundle: {exc}")
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
# Exception handling (stable error schema)
# ---------------------------------------------------------------------------

def _http_error_code(exc: HTTPException) -> str:
    detail = str(exc.detail)
    if exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE and detail.startswith("No model is loaded"):
        return "MODEL_NOT_LOADED"
    if detail.startswith("Expected") and "features" in detail:
        return "INVALID_FEATURE_COUNT"
    if detail.startswith("instances must contain at least one"):
        return "EMPTY_BATCH"
    if detail.startswith("instances must be a 2D array"):
        return "INVALID_BATCH_SHAPE"
    if detail.startswith("LSTM requires at least"):
        return "INSUFFICIENT_SEQUENCE_LENGTH"
    return "HTTP_ERROR"


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail)
    payload = ErrorResponse(
        error=ApiError(
            code=_http_error_code(exc),
            message=message,
            details=exc.detail if not isinstance(exc.detail, str) else None,
        )
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        error=ApiError(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=exc.errors(),
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=payload.model_dump(),
    )


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model(path: str, model_type: str | None = None) -> None:
    """Load a persisted model into the application state.

    Parameters
    ----------
    path:
        File-system path to a pickled model (``.pkl``) or an MLflow run
        directory.
    model_type:
        Optional override for model type (``"xgboost"`` or ``"lstm"``).
    """
    import pickle

    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    resolved_model_type = model_type or payload.get("metadata", {}).get("model_type") or "xgboost"

    _state["model"] = payload.get("model")
    _state["model_type"] = resolved_model_type
    _state["feature_names"] = payload.get("feature_names", [])
    _state["feature_scaler"] = payload.get("feature_scaler")
    _state["target_scaler"] = payload.get("target_scaler")
    _state["seq_len"] = payload.get("seq_len", 24)
    _state["metadata"] = payload.get("metadata", {})
    logger.info(f"Model loaded from {path} (type={resolved_model_type})")


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
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
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


def _validate_feature_vector_length(features_2d: np.ndarray) -> None:
    """Validate feature width against model metadata when available."""
    expected = len(_state["feature_names"])
    if expected <= 0:
        return

    got = int(features_2d.shape[1])
    if got != expected:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Expected {expected} features, got {got}.",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Simple browser-friendly landing page."""
    model_loaded = "yes" if _state["model"] is not None else "no"
    html = """
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>crypto-ml API</title>
        <style>
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(140deg, #f2f8ff 0%, #e8f4ef 100%);
                color: #17202a;
            }
            .wrap {
                max-width: 760px;
                margin: 48px auto;
                padding: 24px;
            }
            .card {
                background: #ffffff;
                border: 1px solid #d6e2ee;
                border-radius: 14px;
                padding: 24px;
                box-shadow: 0 10px 24px rgba(8, 38, 66, 0.08);
            }
            h1 { margin: 0 0 6px; }
            p { margin: 8px 0; line-height: 1.5; }
            code {
                background: #f6f8fa;
                border: 1px solid #e5ebf1;
                border-radius: 6px;
                padding: 2px 6px;
            }
            ul { padding-left: 20px; }
            .panel {
                margin-top: 18px;
                padding: 16px;
                border: 1px solid #d6e2ee;
                border-radius: 10px;
                background: #f9fcff;
            }
            textarea {
                width: 100%;
                min-height: 90px;
                border: 1px solid #c4d4e4;
                border-radius: 8px;
                padding: 10px;
                box-sizing: border-box;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 13px;
            }
            button {
                margin-top: 10px;
                border: 0;
                border-radius: 8px;
                padding: 9px 12px;
                background: #0b5cab;
                color: #fff;
                cursor: pointer;
            }
            button:hover { background: #0a4f95; }
            .hint { color: #4c5f73; font-size: 14px; }
            pre {
                background: #0d1620;
                color: #d7e8f8;
                padding: 12px;
                border-radius: 8px;
                overflow: auto;
                font-size: 12px;
            }
            a { color: #0b5cab; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <h1>crypto-ml API</h1>
                <p>Service is running. Model loaded: <strong>__MODEL_LOADED__</strong></p>
                <ul>
                    <li><a href="/docs">Open API Docs</a></li>
                    <li><a href="/health">Health JSON</a></li>
                    <li><a href="/status">Status JSON</a></li>
                    <li><a href="/feature-spec">Feature Spec JSON</a></li>
                    <li><a href="/info">Model Info JSON</a></li>
                </ul>
                <p>Use <code>POST /predict</code> and <code>POST /predict/batch</code> for inference.</p>
                <div class="panel">
                    <h3>Try a prediction</h3>
                    <p class="hint">Enter comma-separated numeric features (in model feature order), then click Predict.</p>
                    <p class="hint" id="featureMeta">Loading expected feature count...</p>
                    <textarea id="featureInput" placeholder="0.1, 0.2, 0.3"></textarea>
                    <button id="predictBtn">Predict</button>
                    <pre id="resultBox">No request sent yet.</pre>
                </div>
            </div>
        </div>
        <script>
            const featureMeta = document.getElementById("featureMeta");
            const featureInput = document.getElementById("featureInput");
            const resultBox = document.getElementById("resultBox");
            const predictBtn = document.getElementById("predictBtn");

            async function loadFeatureSpec() {
                try {
                    const res = await fetch('/feature-spec');
                    const data = await res.json();
                    if (res.ok) {
                        featureMeta.textContent = `Expected features: ${data.expected_feature_count} (model loaded: ${data.model_loaded})`;
                        if (data.expected_feature_count > 0 && !featureInput.value.trim()) {
                            featureInput.value = Array(data.expected_feature_count).fill(0).join(', ');
                        }
                    } else {
                        featureMeta.textContent = 'Could not load feature spec.';
                    }
                } catch (_) {
                    featureMeta.textContent = 'Could not load feature spec.';
                }
            }

            predictBtn.addEventListener('click', async () => {
                try {
                    const features = featureInput.value
                        .split(',')
                        .map((x) => x.trim())
                        .filter((x) => x.length > 0)
                        .map((x) => Number(x));

                    if (features.some((x) => Number.isNaN(x))) {
                        resultBox.textContent = 'All feature values must be numeric.';
                        return;
                    }

                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features })
                    });
                    const data = await res.json();
                    resultBox.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    resultBox.textContent = String(err);
                }
            });

            loadFeatureSpec();
        </script>
    </body>
</html>
""".strip()
    return html.replace("__MODEL_LOADED__", model_loaded)


@app.get("/status", response_model=RootResponse)
def service_status() -> RootResponse:
    """JSON status endpoint for quick service checks."""
    return RootResponse(
        service="crypto-ml API",
        docs="/docs",
        health="/health",
        status="/status",
        model_loaded=_state["model"] is not None,
    )


@app.get("/feature-spec", response_model=FeatureSpecResponse)
def feature_spec() -> FeatureSpecResponse:
    """Return expected feature schema for clients before prediction calls."""
    return FeatureSpecResponse(
        model_loaded=_state["model"] is not None,
        model_type=_state["model_type"],
        expected_feature_count=len(_state["feature_names"]),
        feature_names=_state["feature_names"],
    )

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
    _validate_feature_vector_length(features_2d)
    pred = _run_prediction(features_2d)
    return PredictResponse(
        prediction=float(pred[0]),
        model_type=_state["model_type"],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict next-period close prices for a batch of feature vectors."""
    if not request.instances:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="instances must contain at least one feature vector.",
        )
    features_2d = np.array(request.instances, dtype=float)
    if features_2d.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="instances must be a 2D array of shape (N, F).",
        )
    _validate_feature_vector_length(features_2d)
    pred = _run_prediction(features_2d)
    return BatchPredictResponse(
        predictions=pred.tolist(),
        model_type=_state["model_type"],
    )
