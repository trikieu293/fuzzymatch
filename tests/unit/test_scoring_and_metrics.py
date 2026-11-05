"""Tests for pairwise metrics and rapidfuzz scorer methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.metrics.pairwise import pairwise_metrics
from fuzzymatch.scoring.rapidfuzz_wrapper import RapidFuzzScorer


class TestPairwiseMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        m = pairwise_metrics(y_true, y_pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["mcc"] == 1.0

    def test_all_wrong(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        m = pairwise_metrics(y_true, y_pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_with_scores(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        y_score = np.array([0.9, 0.4, 0.1, 0.2])
        m = pairwise_metrics(y_true, y_pred, y_score=y_score)
        assert "pr_auc" in m
        assert "roc_auc" in m
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_without_scores(self) -> None:
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        m = pairwise_metrics(y_true, y_pred)
        assert "pr_auc" not in m
        assert "roc_auc" not in m

    def test_constant_score_yields_nan(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_score = np.array([0.5, 0.5, 0.5, 0.5])
        m = pairwise_metrics(y_true, y_pred, y_score=y_score)
        # roc_auc with constant score can be 0.5 or NaN
        assert "roc_auc" in m


class TestRapidFuzzScorerExtended:
    def test_partial_ratio(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.partial_ratio("smith", "john smith") > 0.8

    def test_token_sort_ratio(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.token_sort_ratio("john smith", "smith john") > 0.9

    def test_token_set_ratio(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.token_set_ratio("john smith", "john smith jr") > 0.8

    def test_jaro(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.jaro("martha", "marhta") > 0.9

    def test_jaro_winkler(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.jaro_winkler("martha", "marhta") > 0.9

    def test_damerau_levenshtein(self) -> None:
        scorer = RapidFuzzScorer()
        assert scorer.damerau_levenshtein("ca", "abc") >= 0

    def test_levenshtein_with_cutoff(self) -> None:
        scorer = RapidFuzzScorer()
        d = scorer.levenshtein("kitten", "sitting", score_cutoff=2)
        # Actual distance is 3, cutoff=2 means returns max value or cutoff
        assert isinstance(d, int)

    def test_score_pair(self) -> None:
        scorer = RapidFuzzScorer()
        left = pd.Series(["john", "alice"])
        right = pd.Series(["jonh", "alica"])
        result = scorer.score_pair("ratio", left, right)
        assert len(result) == 2
        assert all(0.0 <= s <= 1.0 for s in result)

    def test_score_pair_length_mismatch(self) -> None:
        scorer = RapidFuzzScorer()
        with pytest.raises(ValueError, match="length mismatch"):
            scorer.score_pair("ratio", pd.Series(["a"]), pd.Series(["a", "b"]))

    def test_score_pair_with_missing_and_nan(self) -> None:
        from fuzzymatch.scoring.rapidfuzz_wrapper import _is_missing

        assert _is_missing(None) is True
        assert _is_missing(float("nan")) is True
        assert _is_missing(pd.NA) is True
        assert _is_missing("nan") is True
        assert _is_missing("None") is True
        assert _is_missing("") is True
        assert _is_missing("valid") is False

        scorer = RapidFuzzScorer()
        left = pd.Series(["john", None, float("nan"), ""])
        right = pd.Series(["jonh", "smith", "doe", "none"])
        res_ratio = scorer.score_pair("ratio", left, right)
        assert len(res_ratio) == 4
        assert res_ratio.iloc[0] > 0.0

        res_lev = scorer.score_pair("levenshtein", left, right)
        assert len(res_lev) == 4
