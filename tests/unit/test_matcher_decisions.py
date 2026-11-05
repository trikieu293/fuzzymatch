from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm


def test_decisions_requires_fit_first():
    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["group"]),
        classifier=fm.FSClassifier(),
        scoring_fields=["name"],
    )
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before decisions\\(\\)"):
        matcher.decisions()


def test_decisions_returns_generic_artifacts_with_field_scores():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "name": ["Office Supplies", "Office Supply", "Travel Expense"],
        },
        index=["src-1", "src-2", "src-3"],
    )
    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["group"]),
        classifier=fm.FSClassifier(),
        scoring_fields=["name"],
        scoring_method="ratio",
    )
    matcher.fit(df)
    decisions = matcher.decisions()

    assert len(decisions) == 2  # self-join produces both (l,r) and (r,l)
    decision = decisions[0]
    assert decision.left_id == "src-1"
    assert decision.right_id == "src-2"
    assert decision.status in {"match", "possible", "non_match"}
    assert 0.0 <= decision.score <= 1.0
    assert decision.explanation.threshold_band == decision.status
    assert decision.explanation.field_scores[0].field == "name"
    assert decision.explanation.field_scores[0].left_value == "Office Supplies"
    assert decision.explanation.field_scores[0].right_value == "Office Supply"
    assert 0.0 <= decision.explanation.field_scores[0].score <= 1.0
    assert decision.explanation.blocking_rules == ("ExactBlock",)


def test_decisions_without_classifier_uses_feature_mean_score():
    left = pd.DataFrame({"group": ["A"], "name": ["Acme Limited"]}, index=["l1"])
    right = pd.DataFrame({"group": ["A"], "name": ["ACME Ltd"]}, index=["r1"])
    matcher = fm.Matcher(
        link_type="fuzzy_join",
        blocking=fm.ExactBlock(["group"]),
        scoring_fields=["name"],
        scoring_method="ratio",
    )
    matcher.fit(left, right)
    decisions = matcher.decisions()

    assert len(decisions) == 1
    assert decisions[0].left_id == "l1"
    assert decisions[0].right_id == "r1"
    assert decisions[0].status == "possible"
    assert 0.0 <= decisions[0].score <= 1.0
