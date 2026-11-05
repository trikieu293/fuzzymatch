"""Wrapper around any sklearn classifier for pairwise matching."""

from __future__ import annotations

import pandas as pd
from sklearn.base import ClassifierMixin


class SklearnClassifier:
    """Wraps any sklearn classifier (LogisticRegression, RandomForest, etc.).

    Expects a binary classifier with `predict_proba` returning 2 columns.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        threshold: tuple[float, float, float] = (0.8, 0.5, 0.2),
    ):
        self.estimator = estimator
        self.match_thresh = threshold[0]
        self.possible_lower = threshold[1]
        self.possible_upper = threshold[2]

    def fit(self, features: pd.DataFrame, labels: pd.Series | None = None) -> SklearnClassifier:
        """Train the wrapped estimator."""
        self.estimator.fit(features, labels)
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        proba = self.estimator.predict_proba(features)
        # Take probability of positive class (column index 1)
        return pd.Series(proba[:, 1], index=features.index)

    def predict(self, features: pd.DataFrame, threshold: str = "match") -> pd.Series:
        proba = self.predict_proba(features)
        if threshold == "match":
            return proba >= self.match_thresh
        elif threshold == "possible":
            return (proba >= self.possible_lower) & (proba < self.match_thresh)
        else:
            return proba < self.possible_lower
