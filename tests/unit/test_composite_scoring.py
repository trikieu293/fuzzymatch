"""Tests for composite noisy-text scorers and missing-value safety."""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.scoring import RapidFuzzScorer
from fuzzymatch.scoring.rapidfuzz_wrapper import SUPPORTED_METHODS


class TestCompositeMethodsRegistered:
    def test_new_methods_listed(self) -> None:
        for method in ("token_overlap", "digit_run_similarity"):
            assert method in SUPPORTED_METHODS

    def test_new_methods_callable(self) -> None:
        scorer = RapidFuzzScorer()
        assert callable(scorer.token_overlap)
        assert callable(scorer.digit_run_similarity)


class TestTokenOverlap:
    def setup_method(self) -> None:
        self.scorer = RapidFuzzScorer()

    def test_identical_token_sets_score_one(self) -> None:
        assert self.scorer.token_overlap("sony tv black", "black sony tv") == 1.0

    def test_disjoint_tokens_score_zero(self) -> None:
        assert self.scorer.token_overlap("apple phone", "sony camera") == 0.0

    def test_containment_scores_between_zero_and_one(self) -> None:
        # Shared: {sony}; union: {sony, tv, bravia} -> 1/3
        score = self.scorer.token_overlap("sony tv", "sony bravia")
        assert abs(score - 1 / 3) < 1e-9

    def test_duplicate_tokens_counted_once(self) -> None:
        assert self.scorer.token_overlap("a a b", "a b") == 1.0


class TestDigitRunSimilarity:
    def setup_method(self) -> None:
        self.scorer = RapidFuzzScorer()

    def test_same_digit_runs_reordered_score_one(self) -> None:
        # Model numbers survive word reordering.
        score = self.scorer.digit_run_similarity("pslx350h model 45 rpm", "45 rpm pslx350h model")
        assert score == 1.0

    def test_different_digit_runs_score_below_one(self) -> None:
        score = self.scorer.digit_run_similarity("model 350", "model 450")
        assert score < 1.0

    def test_no_digits_on_both_sides_is_neutral_one(self) -> None:
        # Absence of digits carries no signal either way.
        assert self.scorer.digit_run_similarity("hello world", "world hello") == 1.0

    def test_digits_on_one_side_only_scores_zero(self) -> None:
        assert self.scorer.digit_run_similarity("abc 123", "no digits here") == 0.0


class TestMissingValueSafety:
    """NaN/empty text must never look like agreement."""

    def setup_method(self) -> None:
        self.scorer = RapidFuzzScorer()

    @pytest.mark.parametrize(
        "method",
        [
            "ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "token_overlap",
            "digit_run_similarity",
            "jaro_winkler",
        ],
    )
    @pytest.mark.parametrize("missing", ["", "nan", "none"])
    def test_missing_either_side_scores_zero(self, method: str, missing: str) -> None:
        scorer = getattr(self.scorer, method)
        assert scorer(missing, "sony tv") == 0.0
        assert scorer("sony tv", missing) == 0.0

    def test_missing_series_pairwise(self) -> None:
        left = pd.Series(["nan", "sony tv", "nan"])
        right = pd.Series(["nan", "nan", "sony tv"])
        result = self.scorer.score_pair("token_overlap", left, right)
        assert result.tolist() == [0.0, 0.0, 0.0]

    def test_right_side_missing_scores_zero(self) -> None:
        for method in (
            "partial_ratio",
            "token_sort_ratio",
            "token_set_ratio",
            "token_overlap",
            "jaro",
        ):
            scorer = getattr(self.scorer, method)
            assert scorer("sony tv", "") == 0.0, method

    def test_whitespace_only_tokens_score_zero(self) -> None:
        # Non-missing string that tokenizes to nothing.
        assert self.scorer.token_overlap("   ", "sony tv") == 0.0

    def test_direct_method_calls_remain_public_api(self) -> None:
        """Single-pair methods stay callable and consistent with batch path."""
        scorer = RapidFuzzScorer()
        assert scorer.ratio("hello world", "hello world") == 1.0
        assert scorer.token_set_ratio("a b", "b a") == 1.0
        assert scorer.jaro_winkler("martha", "martha") == 1.0
        left = pd.Series(["hello world"])
        right = pd.Series(["hello world"])
        batch = scorer.score_pair("ratio", left, right)
        assert abs(float(batch.iloc[0]) - scorer.ratio("hello world", "hello world")) < 1e-9

    def test_score_pair_rejects_unknown_method(self) -> None:
        left = pd.Series(["a"])
        right = pd.Series(["b"])
        with pytest.raises(ValueError, match="Unknown scoring method"):
            self.scorer.score_pair("nope", left, right)
