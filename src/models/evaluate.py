"""Evaluation metrics for regression and directional accuracy."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "model",
) -> dict[str, float]:
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: ``rmse``, ``mae``, ``r2``, ``mape``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Mean Absolute Percentage Error (guard against zero prices)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
    logger.info(
        f"[{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%"
    )
    return metrics


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that correctly call the next-period direction.

    The direction is *up* when the predicted value exceeds the previous
    actual value, and *down* otherwise.

    Parameters
    ----------
    y_true:
        Actual prices/values at time *t+1*, shape ``(N,)``.
    y_pred:
        Predicted prices/values at time *t+1*, shape ``(N,)``.

    Returns
    -------
    float in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Actual and predicted directions (+1 or -1) relative to previous value
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])

    accuracy = float(np.mean(actual_dir == pred_dir))
    logger.info(f"Directional accuracy: {accuracy:.4f}")
    return accuracy
