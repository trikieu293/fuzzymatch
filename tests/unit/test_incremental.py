"""Incremental matching: score new right batches against a fitted matcher."""

from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm


def _matcher() -> fm.Matcher:
    return fm.Matcher(
        link_type="link",
        blocking=fm.ExactBlock(["name"]),
        normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
        classifier=fm.FSClassifier(),
        scoring_fields=["name"],
    )


LEFT = pd.DataFrame({"id": [0, 1], "name": ["acme corp", "beta llc"]})
RIGHT = pd.DataFrame({"id": [0, 1], "name": ["ACME CORP", "gamma inc"]})


class TestMatchRightBatch:
    def test_reproduces_full_fit_decisions_for_seen_rows(self) -> None:
        matcher = _matcher()
        matcher.fit(LEFT, RIGHT)
        baseline = {
            int(d.right_id): round(float(d.score), 9)
            for d in matcher.decisions()
            if int(d.right_id) == 0  # the ACME CORP row
        }
        fresh = _matcher()
        fresh.fit(LEFT, RIGHT.iloc[[0]])
        batch = fresh.match_right_batch(RIGHT.iloc[[0]])
        assert len(batch) >= 1
        assert {round(float(d.score), 9) for d in batch} == set(baseline.values())

    def test_unseen_rows_scored_against_stored_left(self) -> None:
        matcher = _matcher()
        matcher.fit(LEFT, RIGHT)
        unseen = pd.DataFrame({"id": [9], "name": ["BETA LLC"]})
        decisions = matcher.match_right_batch(unseen)
        assert len(decisions) >= 1
        # best candidate must be the beta llc left row (position 1)
        best = max(decisions, key=lambda d: d.score)
        assert int(best.left_id) == 1

    def test_missing_scoring_field_skipped_gracefully(self) -> None:
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["name"]),
            normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name", "ghost_column"],
        )
        matcher.fit(LEFT, RIGHT)
        decisions = matcher.match_right_batch(pd.DataFrame({"id": [7], "name": ["ACME CORP"]}))
        assert len(decisions) == 1

    def test_before_fit_raises(self) -> None:
        matcher = _matcher()
        with pytest.raises(RuntimeError, match="fit"):
            matcher.match_right_batch(RIGHT)

    def test_requires_cross_file_fit(self) -> None:
        matcher = _matcher()
        matcher.fit(LEFT)
        with pytest.raises(RuntimeError, match="right"):
            matcher.match_right_batch(RIGHT)

    def test_decisions_explainable(self) -> None:
        matcher = _matcher()
        matcher.fit(LEFT, RIGHT)
        decisions = matcher.match_right_batch(pd.DataFrame({"id": [7], "name": ["ACME CORP"]}))
        assert all(d.explanation.field_scores for d in decisions)
