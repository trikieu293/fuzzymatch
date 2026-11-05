"""Tests for pairwise metrics error handlers (lines 37-38, 41-42).

Trigger ValueError from sklearn average_precision_score and roc_auc_score
to cover the except-ValueError branches.
"""

from __future__ import annotations

import numpy as np

from fuzzymatch.metrics import pairwise_metrics


class TestPairwiseMetricsErrorHandlers:
    """Cover the except-ValueError branches in pairwise_metrics."""

    def test_2d_y_score_triggers_both_value_errors(self) -> None:
        """A 2D y_score triggers ValueError in both AP and ROC-AUC.

        sklearn's average_precision_score and roc_auc_score both expect 1D
        y_score for binary classification. A 2D array triggers ValueError
        in both, exercising the except branches at lines 37-38 and 41-42.
        """
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        # 2D score array — triggers "y should be a 1d array" ValueError
        y_score = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

        m = pairwise_metrics(y_true, y_pred, y_score=y_score)

        # Both should fall back to NaN via the except ValueError branches
        assert np.isnan(m["pr_auc"])
        assert np.isnan(m["roc_auc"])
        # Non-AUC metrics should still compute fine (they ignore y_score)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["mcc"] == 1.0

    def test_2d_y_score_with_imperfect_predictions(self) -> None:
        """2D y_score with imperfect predictions still triggers both NaN fallbacks."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])  # 2 wrong
        y_score = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
            ]
        )

        m = pairwise_metrics(y_true, y_pred, y_score=y_score)

        assert np.isnan(m["pr_auc"])
        assert np.isnan(m["roc_auc"])
        assert m["precision"] > 0
        assert m["recall"] > 0
