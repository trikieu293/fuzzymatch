"""Continuous date window scoring for financial invoices and transactions.

Computes a linear decay score within a day window:
    s_date(t1, t2; delta) = max(0.0, 1.0 - |t1 - t2| / delta)

Mathematical properties:
1. Boundedness: 0.0 <= s_date(t1, t2; delta) <= 1.0
2. Identity: s_date(t, t; delta) = 1.0
3. Symmetry: s_date(t1, t2; delta) = s_date(t2, t1; delta)
4. Monotonicity: |t1 - t2| < |t1 - t3| <= delta ==> s_date(t1, t2) > s_date(t1, t3)
5. Missing-value neutrality: If either timestamp is None, NaT, or invalid date, score is 0.0.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .numeric import BaseScorer


class DateWindowScorer(BaseScorer):
    """Continuous date window proximity scorer with linear decay within +/- delta_days.

    Score formula:
        max(0.0, 1.0 - |t1 - t2| / delta_days)
    """

    def __init__(self, delta_days: float) -> None:
        if delta_days <= 0.0:
            raise ValueError(f"delta_days must be strictly positive, got {delta_days}")
        self.delta_days = float(delta_days)

    def score(self, t1: Any, t2: Any) -> float:
        """Compute date proximity score between two date/timestamp scalars.

        Accepts datetime objects, pd.Timestamp, and ISO-8601 date strings.
        Returns:
            float in [0.0, 1.0]. Missing / NaT values yield 0.0.
        """
        if t1 is None or t2 is None or pd.isna(t1) or pd.isna(t2):
            return 0.0

        if (isinstance(t1, (float, int)) and not math.isfinite(t1)) or (
            isinstance(t2, (float, int)) and not math.isfinite(t2)
        ):
            return 0.0

        try:
            ts1 = pd.to_datetime(t1)
            ts2 = pd.to_datetime(t2)
        except Exception:
            return 0.0

        if pd.isna(ts1) or pd.isna(ts2):
            return 0.0

        if ts1.tzinfo is not None:
            ts1 = ts1.tz_convert("UTC").tz_localize(None)
        if ts2.tzinfo is not None:
            ts2 = ts2.tz_convert("UTC").tz_localize(None)

        diff_seconds = abs((ts1 - ts2).total_seconds())
        diff_days = diff_seconds / 86400.0

        # Check for potential day/month transposition (e.g. DD-MM-YYYY parsed as MM-DD-YYYY)
        candidates = [diff_days]
        if ts1.day <= 12:
            alt_ts1 = ts1.replace(month=ts1.day, day=ts1.month)
            candidates.append(abs((alt_ts1 - ts2).total_seconds()) / 86400.0)
        if ts2.day <= 12:
            alt_ts2 = ts2.replace(month=ts2.day, day=ts2.month)
            candidates.append(abs((ts1 - alt_ts2).total_seconds()) / 86400.0)
        if ts1.day <= 12 and ts2.day <= 12:
            alt_ts1 = ts1.replace(month=ts1.day, day=ts1.month)
            alt_ts2 = ts2.replace(month=ts2.day, day=ts2.month)
            candidates.append(abs((alt_ts1 - alt_ts2).total_seconds()) / 86400.0)
        diff_days = min(candidates)

        if diff_days >= self.delta_days:
            return 0.0

        return float(max(0.0, min(1.0, 1.0 - diff_days / self.delta_days)))

    def score_pair(self, left_vals: pd.Series, right_vals: pd.Series) -> pd.Series:
        """Compute pairwise continuous date window scores across two Series.

        Returns:
            pd.Series of floats in [0.0, 1.0], indexed by left_vals.index.
        """
        if len(left_vals) != len(right_vals):
            raise ValueError(f"Series length mismatch: {len(left_vals)} vs {len(right_vals)}")

        scores = [self.score(lv, rv) for lv, rv in zip(left_vals, right_vals, strict=True)]
        return pd.Series(scores, index=left_vals.index, dtype=float)


def date_window_score(t1: Any, t2: Any, delta_days: float) -> float:
    """Convenience helper computing continuous date window score."""
    scorer = DateWindowScorer(delta_days=delta_days)
    return scorer.score(t1, t2)
