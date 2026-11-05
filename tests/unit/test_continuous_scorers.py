"""Unit tests for continuous numeric proximity and date window scorers.

Tests:
1. NumericProximityScorer:
   - Mathematical invariants: bounded in [0, 1], symmetry s(x, y) == s(y, x), reflexivity s(x, x) == 1.0.
   - Absolute tolerance mode: linear decay, exact cutoff at epsilon, 0.0 beyond.
   - Negative amounts, floating point precision, extreme differences.
   - Missing/NaN/None values and unparseable types return 0.0.
   - Input validation: zero or negative epsilon rejection.
   - amount_proximity_score helper function.
   - score_series vectorized operations across pandas Series.

2. DateWindowScorer:
   - Mathematical invariants: bounded in [0, 1], symmetry s(d1, d2) == s(d2, d1), reflexivity s(d, d) == 1.0.
   - Absolute day window decay: linear decay, exact boundary at delta_days, 0.0 beyond.
   - Multi-format support: date, datetime, Timestamp, string, np.datetime64.
   - Day-month transposition tolerance (e.g. DD-MM-YYYY vs MM-DD-YYYY parsing).
   - Leap year boundaries (Feb 28/29, leap vs non-leap).
   - Missing/NaT/None values and unparseable strings return 0.0.
   - Input validation: zero or negative delta_days rejection.
   - date_window_score helper function.
   - score_series vectorized operations across pandas Series.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.scoring.date import (
    DateWindowScorer,
    date_window_score,
)
from fuzzymatch.scoring.numeric import (
    NumericProximityScorer,
    amount_proximity_score,
)

# ============================================================================
# NumericProximityScorer Tests
# ============================================================================


def test_numeric_scorer_initialization_validation() -> None:
    """Verify input validation during NumericProximityScorer initialization."""
    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        NumericProximityScorer(epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        NumericProximityScorer(epsilon=-1.0)


def test_numeric_scorer_absolute_mode_invariants() -> None:
    """Verify mathematical invariants in absolute proximity scoring."""
    scorer = NumericProximityScorer(epsilon=10.0)

    # Reflexivity: s(x, x) == 1.0
    assert scorer.score(100.0, 100.0) == 1.0
    assert scorer.score(0.0, 0.0) == 1.0
    assert scorer.score(-50.0, -50.0) == 1.0

    # Symmetry: s(x, y) == s(y, x)
    assert scorer.score(100.0, 105.0) == scorer.score(105.0, 100.0)
    assert scorer.score(-20.0, -25.0) == scorer.score(-25.0, -20.0)

    # Linear decay: score = 1.0 - abs(x - y) / epsilon
    assert scorer.score(100.0, 105.0) == pytest.approx(0.50)
    assert scorer.score(100.0, 102.0) == pytest.approx(0.80)
    assert scorer.score(100.0, 108.0) == pytest.approx(0.20)

    # Cutoff boundary
    assert scorer.score(100.0, 110.0) == 0.0
    assert scorer.score(100.0, 90.0) == 0.0

    # Beyond cutoff
    assert scorer.score(100.0, 115.0) == 0.0
    assert scorer.score(100.0, 50.0) == 0.0


def test_numeric_scorer_missing_and_invalid_values() -> None:
    """Verify handling of NaN, None, and unparseable types."""
    scorer = NumericProximityScorer(epsilon=5.0)

    assert scorer.score(None, 100.0) == 0.0
    assert scorer.score(100.0, None) == 0.0
    assert scorer.score(np.nan, 100.0) == 0.0
    assert scorer.score(float("nan"), 100.0) == 0.0
    assert scorer.score(None, None) == 0.0
    assert scorer.score("nan", 100.0) == 0.0
    assert scorer.score("invalid", 100.0) == 0.0
    assert scorer.score("100.0", "102.0") == pytest.approx(0.60)


def test_amount_proximity_score_helper() -> None:
    """Verify amount_proximity_score convenience function."""
    assert amount_proximity_score(1250.50, 1250.50, epsilon=0.50) == 1.0
    assert amount_proximity_score(1250.50, 1250.75, epsilon=0.50) == pytest.approx(0.50)
    assert amount_proximity_score(100.0, 101.0, epsilon=0.50) == 0.0
    assert amount_proximity_score(None, 100.0, epsilon=0.50) == 0.0


def test_numeric_scorer_score_pair() -> None:
    """Verify score_pair vectorized pandas Series operations."""
    scorer = NumericProximityScorer(epsilon=10.0)

    s_left = pd.Series([100.0, 200.0, 300.0, np.nan])
    s_right = pd.Series([102.0, 250.0, 300.0, 100.0])

    scores = scorer.score_pair(s_left, s_right)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 4
    assert scores.iloc[0] == pytest.approx(0.80)
    assert scores.iloc[1] == 0.0
    assert scores.iloc[2] == 1.0
    assert scores.iloc[3] == 0.0

    # Length mismatch validation
    with pytest.raises(ValueError, match="Series length mismatch"):
        scorer.score_pair(pd.Series([1.0]), pd.Series([1.0, 2.0]))


# ============================================================================
# DateWindowScorer Tests
# ============================================================================


def test_date_scorer_initialization_validation() -> None:
    """Verify input validation during DateWindowScorer initialization."""
    with pytest.raises(ValueError, match="delta_days must be strictly positive"):
        DateWindowScorer(delta_days=0.0)

    with pytest.raises(ValueError, match="delta_days must be strictly positive"):
        DateWindowScorer(delta_days=-5.0)


def test_date_scorer_invariants_and_decay() -> None:
    """Verify mathematical invariants and day decay in DateWindowScorer."""
    scorer = DateWindowScorer(delta_days=10.0)

    d1 = datetime.date(2025, 4, 1)
    d2 = datetime.date(2025, 4, 6)  # 5 days diff
    d3 = datetime.date(2025, 4, 11)  # 10 days diff (exact boundary)
    d4 = datetime.date(2025, 4, 15)  # 14 days diff (beyond cutoff)

    # Reflexivity: s(d, d) == 1.0
    assert scorer.score(d1, d1) == 1.0

    # Symmetry: s(d1, d2) == s(d2, d1)
    assert scorer.score(d1, d2) == scorer.score(d2, d1)
    assert scorer.score(d1, d4) == scorer.score(d4, d1)

    # Linear decay: 1.0 - 5 / 10 = 0.50
    assert scorer.score(d1, d2) == pytest.approx(0.50)

    # Exact boundary: 10 days diff -> 0.0
    assert scorer.score(d1, d3) == 0.0

    # Beyond boundary: 14 days diff -> 0.0
    assert scorer.score(d1, d4) == 0.0


def test_date_scorer_type_conversions() -> None:
    """Verify DateWindowScorer across diverse date representations."""
    scorer = DateWindowScorer(delta_days=5.0)

    # datetime.datetime vs datetime.date (diff exactly 1 day)
    dt1 = datetime.datetime(2025, 6, 1, 0, 0, 0, tzinfo=datetime.UTC)
    d2 = datetime.date(2025, 6, 2)
    assert scorer.score(dt1, d2) == pytest.approx(0.80)
    assert scorer.score("2025-06-01", pd.Timestamp("2025-06-02", tz="UTC")) == pytest.approx(0.80)

    # pd.Timestamp vs string
    ts = pd.Timestamp("2025-06-01")
    s_date = "2025-06-03"
    assert scorer.score(ts, s_date) == pytest.approx(0.60)

    # np.datetime64 vs string
    np_dt = np.datetime64("2025-06-01")
    assert scorer.score(np_dt, "2025-06-01") == 1.0


def test_date_scorer_day_month_transposition() -> None:
    """Verify transposition tolerance when dates have swapped day and month."""
    scorer = DateWindowScorer(delta_days=7.0)

    # "2025-04-09" (April 9) vs pd.to_datetime("09-04-2025") parsed as Sept 4 (diff 148 days)
    # Transposing 09-04 to 04-09 yields diff of 0 days -> score 1.0
    d_standard = "2025-04-09"
    d_european_raw = "09-04-2025"  # pd.to_datetime default yields 2025-09-04
    ts_parsed = pd.to_datetime(d_european_raw)

    score = scorer.score(d_standard, ts_parsed)
    assert score == 1.0

    # Reversed argument order (exercises ts2.day <= 12 branch)
    assert scorer.score(ts_parsed, d_standard) == 1.0

    # Both dates have day <= 12 and benefit from mutual transposition
    d_both1 = "2025-09-04"
    d_both2 = "2025-04-09"
    assert scorer.score(d_both1, d_both2) == 1.0

    # Date with day 31 where replace(month=31) or replace(day=31) raises ValueError in invalid month
    d_nov31 = pd.Timestamp(2025, 5, 31)  # May 31 (day 31 cannot be a month, but day > 12 so no try)
    d_other = pd.Timestamp(2025, 2, 10)  # Feb 10 (month 2, day 10). Transpose yields Oct 2.
    assert scorer.score(d_nov31, d_other) == 0.0

    # April 7 vs parsed Sept 9 (transposed April 9) -> 2 days diff -> 1 - 2/7 = 5/7 ~ 0.714
    score2 = scorer.score("2025-04-07", ts_parsed)
    assert score2 == pytest.approx(5.0 / 7.0)


def test_date_scorer_leap_years() -> None:
    """Verify behavior across leap year boundaries."""
    scorer = DateWindowScorer(delta_days=5.0)

    # 2024 is a leap year (Feb 29 exists)
    d_feb28 = datetime.date(2024, 2, 28)
    d_mar1 = datetime.date(2024, 3, 1)
    # Diff is 2 days (Feb 28 -> Feb 29 -> Mar 1)
    assert scorer.score(d_feb28, d_mar1) == pytest.approx(1.0 - 2.0 / 5.0)

    # 2025 is NOT a leap year
    d25_feb28 = datetime.date(2025, 2, 28)
    d25_mar1 = datetime.date(2025, 3, 1)
    # Diff is 1 day (Feb 28 -> Mar 1)
    assert scorer.score(d25_feb28, d25_mar1) == pytest.approx(1.0 - 1.0 / 5.0)


def test_date_scorer_missing_and_invalid() -> None:
    """Verify handling of NaT, None, NaN, and unparseable date strings."""
    scorer = DateWindowScorer(delta_days=7.0)

    assert scorer.score(None, "2025-05-01") == 0.0
    assert scorer.score("2025-05-01", pd.NaT) == 0.0
    assert scorer.score("NaT", "2025-05-01") == 0.0
    assert scorer.score(pd.NaT, pd.NaT) == 0.0
    assert scorer.score("not-a-date", "2025-05-01") == 0.0
    assert scorer.score(float("nan"), "2025-05-01") == 0.0


def test_date_window_score_helper() -> None:
    """Verify date_window_score convenience function."""
    assert date_window_score("2025-01-10", "2025-01-10", delta_days=6.0) == 1.0
    assert date_window_score("2025-01-10", "2025-01-13", delta_days=6.0) == pytest.approx(0.50)
    assert date_window_score("2025-01-10", "2025-01-20", delta_days=6.0) == 0.0
    assert date_window_score(None, "2025-01-10", delta_days=6.0) == 0.0


def test_date_scorer_score_pair() -> None:
    """Verify score_pair vectorized pandas Series operations."""
    scorer = DateWindowScorer(delta_days=10.0)

    s_left = pd.Series(["2025-05-01", "2025-05-10", "2025-05-20", None])
    s_right = pd.Series(["2025-05-03", "2025-05-25", "2025-05-20", "2025-05-01"])

    scores = scorer.score_pair(s_left, s_right)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 4
    assert scores.iloc[0] == pytest.approx(0.80)
    assert scores.iloc[1] == 0.0
    assert scores.iloc[2] == 1.0
    assert scores.iloc[3] == 0.0

    # Length mismatch validation
    with pytest.raises(ValueError, match="Series length mismatch"):
        scorer.score_pair(pd.Series(["2025-01-01"]), pd.Series(["2025-01-01", "2025-01-02"]))
