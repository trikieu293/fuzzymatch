"""Adversarial stress-test suite for continuous proximity and composite scorers.

Empirically challenges:
1. Boundedness & Invariants: All scores in [0.0, 1.0], symmetry s(x, y) == s(y, x),
   and reflexivity s(x, x) == 1.0 for valid non-missing data.
2. Boundary behavior: Exact cutoff at epsilon / delta yields 0.0 without overshoot.
3. Pathological values: IEEE 754 extremes (NaN, inf, -inf, subnormals), zero amounts
   (0.0 vs -0.0), extreme differences (10^15 vs -10^15, 10^308 vs -10^308).
4. Date edge cases: Leap day comparisons (Feb 29 on leap vs non-leap years, century leap),
   cross-year boundaries (Dec 31 vs Jan 1), epoch boundaries, and timezone mismatches.
5. Transposition attacks: Symmetrical evaluation of day/month swaps when day <= 12,
   rejection when day > 12, and tolerance monotonicity under window widening.
6. Simplex weights: Simplex normalization, rejection of negative/zero weights,
   and non-finite weight safety.
7. Candidate column parity: Consistency between score_record_pair and score_pairs.
"""

from __future__ import annotations

import datetime
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("fuzzymatch_finance")

from fuzzymatch_finance.scoring import (
    FinancialCompositeScorer,
)

from fuzzymatch.scoring.date import (
    DateWindowScorer,
    date_window_score,
)
from fuzzymatch.scoring.numeric import (
    NumericProximityScorer,
    amount_proximity_score,
)

# =============================================================================
# 1. NumericProximityScorer Adversarial Tests
# =============================================================================


class TestNumericProximityAdversarial:
    """Adversarial stress tests for NumericProximityScorer."""

    @settings(max_examples=100)
    @given(
        x=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        eps=st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
    )
    def test_numeric_boundedness_and_invariants_hypothesis(
        self,
        x: float,
        y: float,
        eps: float,
    ) -> None:
        """Hypothesis verification: scores bounded in [0.0, 1.0], symmetric, and reflexive."""
        scorer = NumericProximityScorer(epsilon=eps)

        score_xy = scorer.score(x, y)
        score_yx = scorer.score(y, x)
        score_xx = scorer.score(x, x)

        # Boundedness
        assert 0.0 <= score_xy <= 1.0
        # Symmetry
        assert score_xy == pytest.approx(score_yx, abs=1e-12)
        # Reflexivity
        assert score_xx == pytest.approx(1.0, abs=1e-12)

    def test_exact_cutoff_and_precision_boundary(self) -> None:
        """Verify behavior exactly at epsilon and epsilon +/- machine epsilon."""
        eps = 10.0
        scorer = NumericProximityScorer(epsilon=eps)

        base = 100.0
        # Exact cutoff: |100.0 - 110.0| == 10.0 -> must be exactly 0.0
        assert scorer.score(base, base + eps) == 0.0
        assert scorer.score(base, base - eps) == 0.0

        # Just inside cutoff: diff = 9.999999999999 -> score > 0.0
        inside_val = base + eps - 1e-12
        assert scorer.score(base, inside_val) > 0.0

        # Just outside cutoff: diff = 10.000000000001 -> score == 0.0
        outside_val = base + eps + 1e-12
        assert scorer.score(base, outside_val) == 0.0

    def test_pathological_zero_and_signs(self) -> None:
        """Verify IEEE 754 positive zero and negative zero behavior."""
        scorer = NumericProximityScorer(epsilon=1.0)

        assert scorer.score(0.0, 0.0) == 1.0
        assert scorer.score(+0.0, -0.0) == 1.0
        assert scorer.score(-0.0, -0.0) == 1.0
        assert scorer.score(-0.0, 0.5) == pytest.approx(0.5)

        # Series vectorized test
        s1 = pd.Series([0.0, -0.0])
        s2 = pd.Series([-0.0, 0.0])
        res = scorer.score_pair(s1, s2)
        assert (res == 1.0).all()

    def test_subnormal_floats(self) -> None:
        """Verify subnormal floating point proximity without underflow crash."""
        scorer = NumericProximityScorer(epsilon=1.0)
        subnormal = sys.float_info.min * 1e-10

        assert subnormal > 0.0
        score = scorer.score(subnormal, 0.0)
        assert score == pytest.approx(1.0)

    def test_extreme_floating_point_differences(self) -> None:
        """Verify massive differences up to 10^308 do not cause unhandled overflow."""
        scorer = NumericProximityScorer(epsilon=100.0)

        assert scorer.score(1e15, -1e15) == 0.0
        assert scorer.score(1e308, -1e308) == 0.0
        assert scorer.score(-1e308, 1e308) == 0.0

        # Vectorized check
        s1 = pd.Series([1e15, 1e308])
        s2 = pd.Series([-1e15, -1e308])
        res = scorer.score_pair(s1, s2)
        assert (res == 0.0).all()

    def test_missing_and_corrupt_numeric_inputs(self) -> None:
        """Verify non-numeric and corrupt inputs return 0.0 without exception."""
        scorer = NumericProximityScorer(epsilon=5.0)

        corrupt_inputs: list[Any] = [
            None,
            np.nan,
            float("nan"),
            "nan",
            "None",
            "NULL",
            "",
            "   ",
            "twelve",
            [100.0],
            {"val": 100.0},
        ]

        for val in corrupt_inputs:
            assert scorer.score(val, 100.0) == 0.0
            assert scorer.score(100.0, val) == 0.0
            assert scorer.score(val, val) == 0.0

    def test_invalid_epsilon_rejection(self) -> None:
        """Verify epsilon <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be strictly positive"):
            NumericProximityScorer(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be strictly positive"):
            NumericProximityScorer(epsilon=-0.001)

    def test_amount_proximity_score_convenience_helper(self) -> None:
        """Verify amount_proximity_score helper."""
        assert amount_proximity_score(50.0, 52.0, epsilon=4.0) == pytest.approx(0.5)
        assert amount_proximity_score(50.0, 60.0, epsilon=4.0) == 0.0

    def test_scalar_and_series_pairwise_parity(self) -> None:
        """Verify score_pair produces identical results to scalar score across Series."""
        scorer = NumericProximityScorer(epsilon=20.0)
        vals_left = [0.0, 10.0, 25.0, -15.0, 100.0, np.nan, 50.0]
        vals_right = [0.0, 15.0, 5.0, -20.0, 200.0, 100.0, np.nan]

        s_l = pd.Series(vals_left)
        s_r = pd.Series(vals_right)

        pairwise = scorer.score_pair(s_l, s_r)
        for i in range(len(vals_left)):
            expected = scorer.score(vals_left[i], vals_right[i])
            assert pairwise.iloc[i] == pytest.approx(expected)

    def test_ieee754_infinity_neutrality_scalar(self) -> None:
        """Non-finite numbers (NaN, inf, -inf) must return 0.0 without exception."""
        scorer = NumericProximityScorer(epsilon=10.0)

        # Infinity amounts are non-finite financial values; proximity must be 0.0
        assert scorer.score(float("inf"), float("inf")) == 0.0
        assert scorer.score(float("-inf"), float("-inf")) == 0.0

    def test_ieee754_infinity_series_no_nan(self) -> None:
        """Vectorized score_pair must never produce NaN values on infinite inputs."""
        scorer = NumericProximityScorer(epsilon=10.0)

        s1 = pd.Series([float("inf"), float("-inf")])
        s2 = pd.Series([float("inf"), float("-inf")])
        res = scorer.score_pair(s1, s2)

        # Must be floats in [0.0, 1.0], never NaN
        assert not res.isna().any(), f"score_pair produced NaN on infinite inputs: {res.tolist()}"
        assert ((res >= 0.0) & (res <= 1.0)).all()


# =============================================================================
# 2. DateWindowScorer Adversarial Tests
# =============================================================================


class TestDateWindowAdversarial:
    """Adversarial stress tests for DateWindowScorer."""

    @settings(max_examples=100)
    @given(
        days_offset1=st.integers(min_value=0, max_value=3650),
        days_offset2=st.integers(min_value=0, max_value=3650),
        delta=st.floats(min_value=0.5, max_value=60.0, allow_nan=False, allow_infinity=False),
    )
    def test_date_boundedness_and_invariants_hypothesis(
        self,
        days_offset1: int,
        days_offset2: int,
        delta: float,
    ) -> None:
        """Hypothesis verification: scores bounded in [0.0, 1.0], symmetric, and reflexive."""
        base_date = datetime.date(2020, 1, 1)
        d1 = base_date + datetime.timedelta(days=days_offset1)
        d2 = base_date + datetime.timedelta(days=days_offset2)

        scorer = DateWindowScorer(delta_days=delta)

        score_12 = scorer.score(d1, d2)
        score_21 = scorer.score(d2, d1)
        score_11 = scorer.score(d1, d1)

        assert 0.0 <= score_12 <= 1.0
        assert score_12 == pytest.approx(score_21, abs=1e-9)
        assert score_11 == pytest.approx(1.0, abs=1e-9)

    def test_exact_boundary_cutoff(self) -> None:
        """Verify behavior exactly at delta_days and boundary edges."""
        delta = 10.0
        scorer = DateWindowScorer(delta_days=delta)

        t_base = datetime.date(2025, 3, 1)
        t_exact = datetime.date(2025, 3, 11)  # exactly 10 days diff
        assert scorer.score(t_base, t_exact) == 0.0

        # 9 days diff -> 1.0 - 9/10 = 0.1
        t_inside = datetime.date(2025, 3, 10)
        assert scorer.score(t_base, t_inside) == pytest.approx(0.10)

        # 11 days diff -> 0.0
        t_outside = datetime.date(2025, 3, 12)
        assert scorer.score(t_base, t_outside) == 0.0

    def test_leap_year_boundaries(self) -> None:
        """Verify leap year leap days (Feb 29) across leap and non-leap years."""
        scorer = DateWindowScorer(delta_days=5.0)

        # Leap year 2024: Feb 28 to Mar 1 is 2 days (Feb 29 exists)
        d_feb28_24 = "2024-02-28"
        d_mar01_24 = "2024-03-01"
        assert scorer.score(d_feb28_24, d_mar01_24) == pytest.approx(1.0 - 2.0 / 5.0)

        # Non-leap year 2023: Feb 28 to Mar 1 is 1 day
        d_feb28_23 = "2023-02-28"
        d_mar01_23 = "2023-03-01"
        assert scorer.score(d_feb28_23, d_mar01_23) == pytest.approx(1.0 - 1.0 / 5.0)

        # Feb 29 itself
        d_feb29_24 = "2024-02-29"
        assert scorer.score(d_feb29_24, "2024-02-28") == pytest.approx(0.80)
        assert scorer.score(d_feb29_24, "2024-03-01") == pytest.approx(0.80)

        # Century leap year: 2000 is leap, 1900 is not
        d_feb29_2000 = "2000-02-29"
        assert scorer.score(d_feb29_2000, "2000-03-01") == pytest.approx(0.80)
        # 1900-02-29 does not exist; returns 0.0
        assert scorer.score("1900-02-29", "1900-03-01") == 0.0

    def test_cross_year_and_epoch_boundaries(self) -> None:
        """Verify year roll-overs and historical Unix epoch boundaries."""
        scorer = DateWindowScorer(delta_days=3.0)

        # Cross-year: Dec 31 to Jan 1 is 1 day diff
        assert scorer.score("2024-12-31", "2025-01-01") == pytest.approx(1.0 - 1.0 / 3.0)
        assert scorer.score("2024-12-30", "2025-01-02") == 0.0  # 3 days diff -> exact boundary

        # Unix Epoch: 1970-01-01
        assert scorer.score("1970-01-01", "1969-12-31") == pytest.approx(1.0 - 1.0 / 3.0)

        # Year 2038 32-bit boundary
        assert scorer.score("2038-01-19 03:14:07", "2038-01-20 03:14:07") == pytest.approx(
            1.0 - 1.0 / 3.0
        )

    def test_transposition_day_le_12_symmetry(self) -> None:
        """Verify day/month transposition when day <= 12 is symmetric."""
        scorer = DateWindowScorer(delta_days=5.0)

        # April 9 vs Sept 4: raw diff 148 days, swapped diff 0 days
        d1 = "2024-04-09"
        d2 = "2024-09-04"
        assert scorer.score(d1, d2) == 1.0
        assert scorer.score(d2, d1) == 1.0

    def test_transposition_day_gt_12_rejected(self) -> None:
        """Verify no invalid transposition is attempted when day > 12."""
        scorer = DateWindowScorer(delta_days=5.0)

        # Day 15 cannot be a month; raw diff 30 days -> score 0.0
        d1 = "2024-04-15"
        d2 = "2024-05-15"
        assert scorer.score(d1, d2) == 0.0

    def test_missing_and_unparseable_dates(self) -> None:
        """Verify missing/NaT/corrupt date strings return 0.0."""
        scorer = DateWindowScorer(delta_days=5.0)

        assert scorer.score(None, "2025-01-01") == 0.0
        assert scorer.score("2025-01-01", pd.NaT) == 0.0
        assert scorer.score("NaT", "2025-01-01") == 0.0
        assert scorer.score(float("nan"), "2025-01-01") == 0.0
        assert scorer.score("not_a_valid_date", "2025-01-01") == 0.0
        assert scorer.score("", "2025-01-01") == 0.0

    def test_invalid_delta_days_rejection(self) -> None:
        """Verify delta_days <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="delta_days must be strictly positive"):
            DateWindowScorer(delta_days=0.0)

        with pytest.raises(ValueError, match="delta_days must be strictly positive"):
            DateWindowScorer(delta_days=-1.0)

    def test_date_window_score_helper(self) -> None:
        """Verify date_window_score helper function."""
        assert date_window_score("2025-05-01", "2025-05-03", delta_days=4.0) == pytest.approx(0.5)

    def test_transposition_tolerance_monotonicity(self) -> None:
        """Invariant: Widening delta_days must NEVER decrease similarity score."""
        t1 = "2024-05-06"
        t2 = "2024-06-05"  # 30 days direct, 0 days transposed

        s_narrow = DateWindowScorer(delta_days=7.0)
        s_wide = DateWindowScorer(delta_days=40.0)

        score_narrow = s_narrow.score(t1, t2)
        score_wide = s_wide.score(t1, t2)

        # Under delta=7, transposition triggers -> score = 1.0.
        assert score_narrow == 1.0

        # Monotonicity requires: delta1 <= delta2 ==> score(t1, t2; delta1) <= score(t1, t2; delta2)
        assert score_wide >= score_narrow, (
            f"Monotonicity violated: narrow delta=7 gave {score_narrow}, "
            f"but wider delta=40 gave {score_wide}!"
        )

    def test_timezone_aware_instant_identity(self) -> None:
        """Timestamps representing the exact same physical instant must score 1.0."""
        scorer = DateWindowScorer(delta_days=1.0)

        # Exact same physical moment in time: 12:00 UTC == 14:00 UTC+2
        t_utc = pd.Timestamp("2025-06-01 12:00:00+00:00")
        t_paris = pd.Timestamp("2025-06-01 14:00:00+02:00")

        score = scorer.score(t_utc, t_paris)
        assert score == pytest.approx(1.0), (
            f"Expected score 1.0 for identical instant in time, got {score}"
        )


# =============================================================================
# 3. FinancialCompositeScorer Adversarial Tests
# =============================================================================


class TestFinancialCompositeAdversarial:
    """Adversarial stress tests for FinancialCompositeScorer."""

    def test_simplex_weight_normalization_and_validation(self) -> None:
        """Verify weights on simplex sum to 1.0 and negative weights are rejected."""
        scorer = FinancialCompositeScorer(weights={"text": 4.0, "amount": 2.0, "date": 2.0})
        assert scorer.weights["text"] == pytest.approx(0.5)
        assert scorer.weights["amount"] == pytest.approx(0.25)
        assert scorer.weights["date"] == pytest.approx(0.25)
        assert sum(scorer.weights.values()) == pytest.approx(1.0)

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            FinancialCompositeScorer(weights={"text": -0.1, "amount": 0.5, "date": 0.5})

        with pytest.raises(ValueError, match="Sum of weights must be strictly positive"):
            FinancialCompositeScorer(weights={"text": 0.0, "amount": 0.0, "date": 0.0})

    def test_record_pair_invariants(self) -> None:
        """Verify boundedness, symmetry, and reflexivity on record pairs."""
        scorer = FinancialCompositeScorer(amount_epsilon=1.0, date_delta_days=3.0)

        r1 = {"name": "Industrial Supplies Co.", "amount": 100.0, "date": "2024-05-01"}
        r2 = {"name": "Industrial Supplies Inc.", "amount": 100.2, "date": "2024-05-02"}

        score_12 = scorer.score_record_pair(r1, r2)
        score_21 = scorer.score_record_pair(r2, r1)
        score_11 = scorer.score_record_pair(r1, r1)

        assert 0.0 <= score_12 <= 1.0
        assert score_12 == pytest.approx(score_21)
        assert score_11 == pytest.approx(1.0)

    def test_missing_attributes_graceful_neutrality(self) -> None:
        """Verify missing attributes do not crash and only contribute 0 to their component."""
        scorer = FinancialCompositeScorer()

        r_empty1: dict[str, Any] = {}
        r_empty2: dict[str, Any] = {}
        assert scorer.score_record_pair(r_empty1, r_empty2) == 0.0

        r_name_only = {"name": "Acme Corp"}
        r_other = {"name": "Acme Inc", "amount": 100.0, "date": "2024-01-01"}
        # Text matches (weight 0.5 * 1.0 = 0.5), amount & date missing on one side -> 0.0
        assert scorer.score_record_pair(r_name_only, r_other) == pytest.approx(0.50)

    def test_gated_mode_boundary_behavior(self) -> None:
        """Verify strict gating returns 0.0 if amount or date fails tolerance."""
        scorer = FinancialCompositeScorer(amount_epsilon=1.0, date_delta_days=2.0, gated=True)

        r_base = {"name": "Acme Corp", "amount": 100.0, "date": "2024-01-01"}
        # Amount out of tolerance: diff 1.5 > 1.0
        r_bad_amt = {"name": "Acme Corp", "amount": 101.5, "date": "2024-01-01"}
        # Date out of tolerance: diff 5 > 2
        r_bad_date = {"name": "Acme Corp", "amount": 100.0, "date": "2024-01-06"}
        # Good record
        r_good = {"name": "Acme Corp", "amount": 100.2, "date": "2024-01-02"}

        assert scorer.score_record_pair(r_base, r_bad_amt) == 0.0
        assert scorer.score_record_pair(r_base, r_bad_date) == 0.0
        assert scorer.score_record_pair(r_base, r_good) > 0.0

    def test_legal_normalization_toggle(self) -> None:
        """Verify normalize_legal toggles suffix stripping behavior."""
        s_norm = FinancialCompositeScorer(normalize_legal=True)
        s_raw = FinancialCompositeScorer(normalize_legal=False)

        r1 = {"name": "Beta Technologies LLC", "amount": 50.0, "date": "2024-01-01"}
        r2 = {"name": "Beta Technologies Inc", "amount": 50.0, "date": "2024-01-01"}

        score_norm = s_norm.score_record_pair(r1, r2)
        score_raw = s_raw.score_record_pair(r1, r2)

        # With legal normalization: 'Beta Technologies' == 'Beta Technologies' -> 1.0 text
        # Without legal normalization: LLC vs Inc yields lower text similarity
        assert score_norm > score_raw
        assert score_norm == pytest.approx(1.0)

    def test_simplex_non_finite_weight_rejection(self) -> None:
        """Weights must be strictly finite numbers; NaN or Inf must raise ValueError."""
        with pytest.raises(ValueError, match="Weights must be finite"):
            FinancialCompositeScorer(weights={"text": float("nan"), "amount": 0.3, "date": 0.2})

        with pytest.raises(ValueError, match="Weights must be finite"):
            FinancialCompositeScorer(weights={"text": float("inf"), "amount": 0.3, "date": 0.2})

    def test_score_pairs_candidate_column_fallback_parity(self) -> None:
        """score_pairs must resolve candidate column names consistently with score_record_pair."""
        scorer = FinancialCompositeScorer()

        df_l = pd.DataFrame(
            {
                "name": ["Precision Machining Corp."],
                "cleared_amount": [250.00],
                "cleared_date": ["2025-05-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Precision Machining LLC"],
                "invoice_amount": [250.00],
                "invoice_date": ["2025-05-01"],
            }
        )

        scalar_score = scorer.score_record_pair(df_l.iloc[0], df_r.iloc[0])
        vector_scores = scorer.score_pairs(df_l, df_r)

        assert scalar_score == pytest.approx(1.0)
        assert vector_scores.iloc[0] == pytest.approx(scalar_score), (
            f"Parity mismatch: score_record_pair gave {scalar_score}, "
            f"but score_pairs gave {vector_scores.iloc[0]}"
        )
