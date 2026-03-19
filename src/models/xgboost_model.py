"""XGBoost wrapper for cryptocurrency price prediction."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin


class XGBoostPricePredictor(BaseEstimator, RegressorMixin):
    """Thin scikit-learn–compatible wrapper around :class:`xgb.XGBRegressor`.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth.
    learning_rate:
        Boosting learning rate (eta).
    subsample:
        Fraction of training samples to use per tree.
    colsample_bytree:
        Fraction of features to use per tree.
    early_stopping_rounds:
        Stop if validation metric does not improve for this many rounds.
    random_state:
        Random seed for reproducibility.
    **kwargs:
        Additional keyword arguments forwarded to :class:`xgb.XGBRegressor`.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 30,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.kwargs = kwargs
        self._model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = False,
    ) -> "XGBoostPricePredictor":
        """Train the model.

        Parameters
        ----------
        X:
            Training feature matrix, shape ``(N, F)``.
        y:
            Target array, shape ``(N,)``.
        eval_set:
            List of ``(X_val, y_val)`` tuples for early stopping.
        verbose:
            Whether to print XGBoost training logs.
        """
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            tree_method="hist",
            verbosity=0,
            **self.kwargs,
        )
        self._model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=verbose,
        )
        logger.info(f"XGBoost trained — best iteration: {self._model.best_iteration}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        if self._model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self._model.predict(X)

    def get_feature_importances(self, feature_names: Optional[list[str]] = None) -> dict[str, float]:
        """Return a dict mapping feature name → importance score."""
        if self._model is None:
            raise RuntimeError("Model is not fitted yet.")
        scores = self._model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(scores))]
        return dict(zip(feature_names, scores.tolist()))
