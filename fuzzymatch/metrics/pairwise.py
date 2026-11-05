"""Pairwise binary classification metrics for match prediction."""

from __future__ import annotations

from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pairwise_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute precision, recall, F1, MCC, PR-AUC, ROC-AUC.

    PR-AUC is preferred over ROC-AUC for fuzzy matching under the extreme
    class imbalance of ER (matches are rare vs non-matches).
    """
    m: dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None:
        try:
            m["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            m["pr_auc"] = float("nan")
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            m["roc_auc"] = float("nan")
    return m
