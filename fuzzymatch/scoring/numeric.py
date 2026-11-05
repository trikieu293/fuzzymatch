"""Continuous numeric proximity scoring for financial transaction amounts.

Computes a linear decay score:
    s_num(a1, a2; epsilon) = max(0.0, 1.0 - |a1 - a2| / epsilon)

Mathematical properties:
1. Boundedness: 0.0 <= s_num(a1, a2; epsilon) <= 1.0
2. Identity: s_num(a, a; epsilon) = 1.0
3. Symmetry: s_num(a1, a2; epsilon) = s_num(a2, a1; epsilon)
4. Monotonicity: |a1 - a2| < |a1 - a3| <= epsilon ==> s_num(a1, a2) > s_num(a1, a3)
5. Missing-value neutrality: If either value is None, NaN, or non-numeric, score is 0.0.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseScorer(ABC):
    """Abstract base class for attribute scorers."""

    @abstractmethod
    def score_pair(self, left_vals: pd.Series, right_vals: pd.Series) -> pd.Series:
        """Compute pairwise scores for two equal-length Series."""
        ...


class NumericProximityScorer(BaseScorer):
    """Continuous numeric proximity scorer with linear decay within +/- epsilon.

    Score formula:
        max(0.0, 1.0 - |a1 - a2| / epsilon)
    """

    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}")
        self.epsilon = float(epsilon)

    def score(self, a1: Any, a2: Any) -> float:
        """Compute proximity score between two numeric scalars.

        Returns:
            float in [0.0, 1.0]. Missing / NaN values yield 0.0.
        """
        if a1 is None or a2 is None or pd.isna(a1) or pd.isna(a2):
            return 0.0

        try:
            v1 = float(a1)
            v2 = float(a2)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(v1) or not math.isfinite(v2):
            return 0.0

        diff = abs(v1 - v2)
        if diff >= self.epsilon:
            return 0.0

        return max(0.0, min(1.0, 1.0 - diff / self.epsilon))

    def score_pair(self, left_vals: pd.Series, right_vals: pd.Series) -> pd.Series:
        """Compute pairwise continuous numeric proximity scores across two Series.

        Returns:
            pd.Series of floats in [0.0, 1.0], indexed by left_vals.index.
        """
        if len(left_vals) != len(right_vals):
            raise ValueError(f"Series length mismatch: {len(left_vals)} vs {len(right_vals)}")

        l_num = pd.to_numeric(left_vals, errors="coerce")
        r_num = pd.to_numeric(right_vals, errors="coerce")

        valid = np.isfinite(l_num) & np.isfinite(r_num)
        diff = (l_num - r_num).abs()

        scores = (1.0 - diff / self.epsilon).clip(lower=0.0, upper=1.0)
        result = scores.where(valid, 0.0)
        result.index = left_vals.index
        return result.astype(float)


def amount_proximity_score(a1: Any, a2: Any, epsilon: float) -> float:
    """Convenience helper computing continuous numeric proximity score."""
    scorer = NumericProximityScorer(epsilon=epsilon)
    return scorer.score(a1, a2)
