"""End-to-end training pipeline with MLflow experiment tracking.

Usage
-----
From the project root::

    python -m src.training.train \
        --model xgboost \
        --symbol BTCUSDT \
        --interval 1h \
        --data-path data/raw/BTCUSDT_1h.parquet \
        --seq-len 24

If ``--data-path`` does not exist the pipeline generates synthetic data for
demonstration purposes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.data.clean import clean_ohlcv
from src.data.features import add_features, get_feature_columns
from src.data.fetch import load_ohlcv, make_sample_ohlcv
from src.models.evaluate import directional_accuracy, evaluate_regression
from src.models.lstm import (
    LSTMModel,
    build_dataloaders,
    make_sequences,
    predict_lstm,
    train_lstm,
)
from src.models.xgboost_model import XGBoostPricePredictor


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_data(data_path: Path | None, symbol: str, interval: str) -> pd.DataFrame:
    """Load or generate OHLCV data."""
    if data_path and Path(data_path).exists():
        df = load_ohlcv(data_path)
    else:
        logger.warning("Data file not found — using synthetic sample data")
        df = make_sample_ohlcv(n=1000, symbol=symbol)
    return df


def prepare_features(df: pd.DataFrame, lag_periods: int = 5) -> pd.DataFrame:
    """Clean + feature-engineer a raw OHLCV DataFrame."""
    df = clean_ohlcv(df)
    df = add_features(df, lag_periods=lag_periods)
    return df


def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train / val / test."""
    n = len(df)
    val_start = int(n * (1 - val_frac - test_frac))
    test_start = int(n * (1 - test_frac))
    return df.iloc[:val_start], df.iloc[val_start:test_start], df.iloc[test_start:]


def scale_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Fit scalers on train and transform all splits."""
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train = feature_scaler.fit_transform(train[feature_cols])
    X_val = feature_scaler.transform(val[feature_cols])
    X_test = feature_scaler.transform(test[feature_cols])

    y_train = target_scaler.fit_transform(train[["target"]]).ravel()
    y_val = target_scaler.transform(val[["target"]]).ravel()
    y_test = target_scaler.transform(test[["target"]]).ravel()

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        feature_scaler,
        target_scaler,
    )


# ---------------------------------------------------------------------------
# Model-specific training functions
# ---------------------------------------------------------------------------

def run_xgboost(
    train_val_test: tuple,
    feature_cols: list[str],
    target_scaler: MinMaxScaler,
    params: dict,
    experiment_name: str,
) -> None:
    """Train XGBoost and log to MLflow."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test

    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))

        model = XGBoostPricePredictor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Evaluate on test set
        y_pred_scaled = model.predict(X_test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

        metrics = evaluate_regression(y_true, y_pred, label="XGBoost-test")
        da = directional_accuracy(y_true, y_pred)
        mlflow.log_metrics({**metrics, "directional_accuracy": da})

        # Feature importances
        importances = model.get_feature_importances(feature_names=feature_cols)
        top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top-5 features: {top5}")

        mlflow.sklearn.log_model(model, artifact_path="xgboost_model")
        logger.info(f"XGBoost run logged — run_id={run.info.run_id}")


def run_lstm(
    train_val_test: tuple,
    feature_cols: list[str],
    target_scaler: MinMaxScaler,
    params: dict,
    seq_len: int,
    device: str = "cpu",
) -> None:
    """Train LSTM and log to MLflow."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test

    # Build sequences
    X_train_s, y_train_s = make_sequences(X_train, y_train, seq_len)
    X_val_s, y_val_s = make_sequences(X_val, y_val, seq_len)
    X_test_s, y_test_s = make_sequences(X_test, y_test, seq_len)

    if len(X_train_s) == 0:
        logger.error("Not enough data to create sequences. Aborting LSTM training.")
        return

    train_loader, val_loader = build_dataloaders(
        X_train_s, y_train_s, X_val_s, y_val_s, batch_size=params.get("batch_size", 64)
    )

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=params.get("hidden_size", 128),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.2),
    )

    with mlflow.start_run(run_name="lstm") as run:
        mlflow.log_params(params)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("n_features", len(feature_cols))

        history = train_lstm(
            model,
            train_loader,
            val_loader,
            epochs=params.get("epochs", 30),
            lr=params.get("lr", 1e-3),
            device=device,
        )

        # Log loss curves
        for epoch, (tl, vl) in enumerate(
            zip(history["train_loss"], history["val_loss"]), 1
        ):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

        # Evaluate on test set
        y_pred_scaled = predict_lstm(model, X_test_s, device=device)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

        metrics = evaluate_regression(y_true, y_pred, label="LSTM-test")
        da = directional_accuracy(y_true, y_pred)
        mlflow.log_metrics({**metrics, "directional_accuracy": da})

        mlflow.pytorch.log_model(model, artifact_path="lstm_model")
        logger.info(f"LSTM run logged — run_id={run.info.run_id}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train(
    model_type: Literal["xgboost", "lstm"] = "xgboost",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    data_path: Path | None = None,
    seq_len: int = 24,
    lag_periods: int = 5,
    experiment_name: str = "crypto-ml",
    mlflow_tracking_uri: str = "mlruns",
    device: str = "cpu",
) -> None:
    """Run the full training pipeline."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # --- Data ---
    raw = load_data(data_path, symbol, interval)
    df = prepare_features(raw, lag_periods=lag_periods)
    logger.info(f"Dataset shape after feature engineering: {df.shape}")

    feature_cols = get_feature_columns(df)
    train_df, val_df, test_df = temporal_split(df)

    result = scale_features(train_df, val_df, test_df, feature_cols)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feat_scaler, tgt_scaler = result
    train_val_test = (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )

    if model_type == "xgboost":
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        run_xgboost(train_val_test, feature_cols, tgt_scaler, params, experiment_name)

    elif model_type == "lstm":
        params = {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "epochs": 30,
            "lr": 1e-3,
            "batch_size": 64,
        }
        run_lstm(train_val_test, feature_cols, tgt_scaler, params, seq_len, device)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'xgboost' or 'lstm'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a crypto price prediction model")
    p.add_argument("--model", default="xgboost", choices=["xgboost", "lstm"])
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument("--seq-len", default=24, type=int)
    p.add_argument("--lag-periods", default=5, type=int)
    p.add_argument("--experiment", default="crypto-ml")
    p.add_argument("--mlflow-uri", default="mlruns")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_type=args.model,
        symbol=args.symbol,
        interval=args.interval,
        data_path=args.data_path,
        seq_len=args.seq_len,
        lag_periods=args.lag_periods,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.mlflow_uri,
        device=args.device,
    )
