"""iter_decisions(): memory-bounded decision streaming (B3)."""

from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm


def _fit() -> fm.Matcher:
    matcher = fm.Matcher(
        link_type="link",
        blocking=fm.ExactBlock(["name"]),
        normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
        classifier=fm.FSClassifier(),
        scoring_fields=["name"],
        scoring_methods={"name": "token_sort_ratio"},
    )
    left = pd.DataFrame({"name": ["acme", "beta", "gamma"]})
    right = pd.DataFrame({"name": ["acme", "beta inc", "delta"]})
    matcher.fit(left, right)
    return matcher


class TestIterDecisions:
    def test_yields_same_decisions_as_decisions(self) -> None:
        matcher = _fit()
        materialized = matcher.decisions()
        streamed = list(matcher.iter_decisions(chunk_size=2))
        assert len(streamed) == len(materialized)
        for a, b in zip(streamed, materialized, strict=True):
            assert a.left_id == b.left_id
            assert a.right_id == b.right_id
            assert a.status == b.status
            assert a.score == pytest.approx(b.score)

    def test_chunk_size_bounds_yield_batches(self) -> None:
        matcher = _fit()
        batches = list(matcher.iter_decision_chunks(chunk_size=2))
        total = len(matcher.decisions())
        assert total > 0
        assert all(0 < len(b) <= 2 for b in batches)
        assert sum(len(b) for b in batches) == total

    def test_invalid_chunk_size_raises(self) -> None:
        matcher = _fit()
        with pytest.raises(ValueError, match="chunk_size"):
            list(matcher.iter_decision_chunks(chunk_size=0))

    def test_before_fit_raises(self) -> None:
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["name"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        with pytest.raises(RuntimeError, match="fit"):
            list(matcher.iter_decisions())
