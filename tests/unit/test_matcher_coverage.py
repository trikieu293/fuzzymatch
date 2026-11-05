"""Comprehensive tests for matcher.py to achieve 100% coverage.

Covers constructor validation, normalizer integration, error handlers,
fallback paths, edge cases, and untested methods (clusters, join).
"""

from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm

# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """Tests for __init__ validation guards (lines 52-59)."""

    def test_invalid_link_type_raises(self):
        """Line 53: unknown link_type."""
        with pytest.raises(ValueError, match="Unknown link_type: bogus"):
            fm.Matcher(
                link_type="bogus",
                blocking=fm.ExactBlock(["group"]),
                classifier=fm.FSClassifier(),
            )

    def test_missing_blocking_raises(self):
        """Line 55: blocking=None."""
        with pytest.raises(ValueError, match="Matcher requires a blocking strategy"):
            fm.Matcher(blocking=None)

    def test_missing_classifier_for_dedupe_raises(self):
        """Line 57: classifier=None with dedupe mode."""
        with pytest.raises(ValueError, match="Matcher requires a classifier for dedupe/link modes"):
            fm.Matcher(
                link_type="dedupe",
                blocking=fm.ExactBlock(["group"]),
                classifier=None,
            )

    def test_missing_classifier_for_link_raises(self):
        """Line 57: classifier=None with link mode."""
        with pytest.raises(ValueError, match="Matcher requires a classifier for dedupe/link modes"):
            fm.Matcher(
                link_type="link",
                blocking=fm.ExactBlock(["group"]),
                classifier=None,
            )

    def test_fuzzy_join_without_classifier_is_allowed(self):
        """fuzzy_join mode does NOT require a classifier."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        assert matcher.classifier is None


# ---------------------------------------------------------------------------
# Normalizer integration
# ---------------------------------------------------------------------------


class TestNormalizerIntegration:
    """Tests for normalizer paths in pipeline_config and fit (lines 82, 113-116)."""

    def test_pipeline_config_with_normalizer(self):
        """Line 82: normalizer_steps populated from normalizer.steps."""
        norm = fm.Normalizer([fm.casefold(), fm.strip_whitespace()])
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            normalizer=norm,
        )
        config = matcher.pipeline_config()
        # Lambda-based steps report "<lambda>" as __name__
        assert len(config.normalizer_steps) == 2
        assert all(step == "<lambda>" for step in config.normalizer_steps)

    def test_pipeline_config_without_normalizer(self):
        """Normalizer is None → empty normalizer_steps tuple."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        config = matcher.pipeline_config()
        assert config.normalizer_steps == ()

    def test_fit_with_normalizer_dedupe(self):
        """Lines 113-114: normalizer applied in dedupe fit."""
        norm = fm.Normalizer([fm.casefold()])
        df = pd.DataFrame(
            {"group": ["A", "A"], "name": ["HELLO", "hello"]},
            index=[0, 1],
        )
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            normalizer=norm,
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        matcher.fit(df)
        # After normalization, both names should be lowercase
        assert matcher._df_left is not None
        assert matcher._df_left.loc[0, "name"] == "hello"
        assert matcher._df_left.loc[1, "name"] == "hello"

    def test_fit_with_normalizer_link_mode(self):
        """Lines 115-116: normalizer applied to df_right in link/fuzzy_join fit."""
        norm = fm.Normalizer([fm.casefold()])
        left = pd.DataFrame(
            {"group": ["A"], "name": ["HELLO"]},
            index=[0],
        )
        right = pd.DataFrame(
            {"group": ["A"], "name": ["WORLD"]},
            index=[0],
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            normalizer=norm,
            scoring_fields=["name"],
        )
        matcher.fit(left, right)
        assert matcher._df_left is not None
        assert matcher._df_right is not None
        assert matcher._df_left.loc[0, "name"] == "hello"
        assert matcher._df_right.loc[0, "name"] == "world"


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Tests for error-raising guard clauses (lines 134, 141, 187, 237)."""

    def test_build_feature_matrix_before_fit_raises(self):
        """Line 134: _build_feature_matrix without fit."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            scoring_fields=["name"],
        )
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before _build_feature_matrix"):
            matcher._build_feature_matrix()

    def test_nonexistent_scoring_field_is_skipped(self):
        """Line 141: scoring field not in DataFrame columns → continue (skip)."""
        df = pd.DataFrame(
            {"group": ["A", "A"], "name": ["hello", "world"]},
            index=[0, 1],
        )
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["nonexistent_field", "name"],
        )
        matcher.fit(df)
        # _features should only contain "name", not "nonexistent_field"
        assert matcher._features is not None
        assert "nonexistent_field" not in matcher._features.columns
        assert "name" in matcher._features.columns

    def test_decisions_without_scoring_fields_raises(self):
        """Line 187: decisions() requires scoring_fields for explanations."""
        df = pd.DataFrame(
            {"group": ["A", "A"], "name": ["hello", "world"]},
            index=[0, 1],
        )
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=[],  # no scoring fields
        )
        matcher.fit(df)
        with pytest.raises(
            RuntimeError,
            match=r"Matcher\.decisions\(\) requires scoring_fields",
        ):
            matcher.decisions()

    def test_predict_before_fit_raises(self):
        """Line 237: predict() without fit."""
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict\\(\\)"):
            matcher.predict()

    def test_predict_raises_when_features_missing_despite_proba(self):
        """Invariant guard: classifier present and _proba set but _features never built."""
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
        )
        matcher._candidates = fm.CandidateIndex([], pd.Index([]))
        matcher._proba = pd.Series(dtype=float)
        matcher._features = None
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict\\(\\)"):
            matcher.predict()


# ---------------------------------------------------------------------------
# Fallback paths (_status_for_score)
# ---------------------------------------------------------------------------


class TestStatusForScore:
    """Tests for _status_for_score with and without classifier (lines 159-167)."""

    def test_without_classifier_match_band(self):
        """Line 159: score >= 0.8 → 'match' without classifier."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        assert matcher._status_for_score(0.9) == "match"
        assert matcher._status_for_score(0.8) == "match"

    def test_without_classifier_possible_band(self):
        """Line 162: 0.5 <= score < 0.8 → 'possible' without classifier."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        assert matcher._status_for_score(0.5) == "possible"
        assert matcher._status_for_score(0.79) == "possible"

    def test_without_classifier_non_match_band(self):
        """Line 162: score < 0.5 → 'non_match' without classifier."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        assert matcher._status_for_score(0.0) == "non_match"
        assert matcher._status_for_score(0.49) == "non_match"

    def test_with_classifier_match_band(self):
        """Line 164: score >= match_thresh → 'match' with classifier."""
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(threshold=(0.8, 0.5, 0.2)),
        )
        assert matcher._status_for_score(0.85) == "match"
        assert matcher._status_for_score(0.8) == "match"

    def test_with_classifier_possible_band(self):
        """Line 166: possible_lower <= score < match_thresh → 'possible'."""
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(threshold=(0.8, 0.5, 0.2)),
        )
        assert matcher._status_for_score(0.5) == "possible"
        assert matcher._status_for_score(0.79) == "possible"

    def test_with_classifier_non_match_band(self):
        """score < possible_lower → 'non_match' with classifier."""
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(threshold=(0.8, 0.5, 0.2)),
        )
        assert matcher._status_for_score(0.49) == "non_match"
        assert matcher._status_for_score(0.0) == "non_match"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for _blocking_rule_names with combinators and predict without classifier."""

    def test_blocking_rule_names_with_union_block(self):
        """Line 174: UnionBlock → returns tuple of nested blocker type names."""
        blocker = fm.UnionBlock(
            fm.ExactBlock(["group"]),
            fm.ExactBlock(["name"]),
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=blocker,
        )
        names = matcher._blocking_rule_names()
        assert names == ("ExactBlock", "ExactBlock")

    def test_blocking_rule_names_with_intersect_block(self):
        """Line 174: IntersectBlock → returns tuple of nested blocker type names."""
        blocker = fm.IntersectBlock(
            fm.ExactBlock(["group"]),
            fm.ExactBlock(["name"]),
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=blocker,
        )
        names = matcher._blocking_rule_names()
        assert names == ("ExactBlock", "ExactBlock")

    def test_decisions_with_union_block(self):
        """Line 174 in context: decisions() exercises _blocking_rule_names via UnionBlock."""
        df = pd.DataFrame(
            {"group": ["A", "A"], "name": ["hello", "hello"]},
            index=["r1", "r2"],
        )
        blocker = fm.UnionBlock(
            fm.ExactBlock(["group"]),
            fm.ExactBlock(["name"]),
        )
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=blocker,
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        matcher.fit(df)
        decisions = matcher.decisions()
        assert len(decisions) > 0
        assert decisions[0].explanation.blocking_rules == ("ExactBlock", "ExactBlock")

    def test_predict_without_classifier_returns_all_candidates(self):
        """Line 239: predict() without classifier returns all candidate pairs."""
        left = pd.DataFrame(
            {"group": ["A"], "name": ["hello"]},
            index=[0],
        )
        right = pd.DataFrame(
            {"group": ["A"], "name": ["hello"]},
            index=[0],
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            scoring_fields=["name"],
        )
        matcher.fit(left, right)
        idx = matcher.predict()
        # Without classifier, all candidate pairs are returned
        assert len(idx) >= 1


# ---------------------------------------------------------------------------
# clusters() method (lines 252-255)
# ---------------------------------------------------------------------------


class TestClustersMethod:
    """Tests for the clusters() method."""

    def test_clusters_on_non_dedupe_raises(self):
        """Lines 252-253: clusters() only valid for dedupe mode."""
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        with pytest.raises(ValueError, match="clusters\\(\\) only valid for link_type='dedupe'"):
            matcher.clusters()

    def test_clusters_on_link_mode_raises(self):
        """Lines 252-253: clusters() raises for link mode too."""
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
        )
        with pytest.raises(ValueError, match="clusters\\(\\) only valid for link_type='dedupe'"):
            matcher.clusters()

    def test_clusters_dedupe_returns_entity_sets(self):
        """Lines 254-255: dedupe clusters() returns list of sets."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "name": ["Office Supplies", "Office Supply", "Travel"],
            },
            index=[0, 1, 2],
        )
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
            scoring_method="ratio",
        )
        matcher.fit(df)
        result = matcher.clusters(threshold="match")
        # Result is a list of sets (clusters)
        assert isinstance(result, list)
        for cluster in result:
            assert isinstance(cluster, set)


# ---------------------------------------------------------------------------
# join() method (lines 268-289)
# ---------------------------------------------------------------------------


class TestJoinMethod:
    """Tests for the join() fuzzy_join method."""

    def test_join_on_non_fuzzy_join_raises(self):
        """Lines 268-269: join() requires fuzzy_join mode."""
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
        )
        left = pd.DataFrame({"name": ["hello"]})
        right = pd.DataFrame({"name": ["hello"]})
        with pytest.raises(ValueError, match="join\\(\\) requires link_type='fuzzy_join'"):
            matcher.join(left, right, on="name")

    def test_join_raises_when_fit_produces_no_candidates(self, monkeypatch):
        """Invariant guard: fit() must produce a candidate index before pair filtering."""
        left = pd.DataFrame({"group": ["A"], "name": ["hello"]}, index=[0])
        right = pd.DataFrame({"group": ["A"], "name": ["hello"]}, index=[0])
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        monkeypatch.setattr(matcher, "fit", lambda *args, **kwargs: None)
        matcher._candidates = None
        with pytest.raises(RuntimeError, match="Call fit\\(\\) before join\\(\\)"):
            matcher.join(left, right, on="name")

    def test_join_unknown_key_raises(self):
        """join() validates that the on= key exists in both frames."""
        left = pd.DataFrame({"name": ["hello"]})
        right = pd.DataFrame({"name": ["hello"]})
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["name"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        with pytest.raises(ValueError, match="must exist in both frames"):
            matcher.join(left, right, on="missing")

    def test_join_without_classifier_raises(self):
        """join() without a classifier raises instead of silently returning df_left."""
        left = pd.DataFrame(
            {"group": ["A", "B"], "name": ["hello", "world"]},
            index=[0, 1],
        )
        right = pd.DataFrame(
            {"group": ["A", "B"], "name": ["helo", "word"]},
            index=[0, 1],
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
        )
        with pytest.raises(ValueError, match="requires a classifier"):
            matcher.join(left, right, on="name")

    def test_join_forwards_labels_to_supervised_fit(self, monkeypatch):
        """join() passes an optional labeled-pair signal down to fit()."""
        captured: dict = {}

        left = pd.DataFrame({"name": ["hello", "heap"]}, index=[0, 1])
        right = pd.DataFrame({"name": ["hello", "helo"]}, index=[0, 1])
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["name"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
        )
        original_fit = matcher.fit

        def spy_fit(*args, **kwargs):
            captured["labels"] = kwargs.get("labels")
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(matcher, "fit", spy_fit)
        labels = pd.Series([1, 0])
        matcher.join(left, right, on="name", labels=labels)
        assert captured["labels"] is labels

    def test_join_without_scoring_fields_raises(self):
        """join() must never silently fall back to returning df_left."""
        left = pd.DataFrame({"name": ["hello"]})
        right = pd.DataFrame({"name": ["hello"]})
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["name"]),
            classifier=fm.FSClassifier(),
        )
        assert matcher.scoring_fields == []
        with pytest.raises(ValueError, match="requires scoring_fields"):
            matcher.join(left, right, on="name")

    def test_per_field_scoring_methods(self):
        """scoring_methods overrides the global method for named fields."""
        left = pd.DataFrame({"name": ["same key"], "desc": ["alpha beta"]}, index=[0])
        right = pd.DataFrame({"name": ["same key"], "desc": ["beta alpha"]}, index=[0])
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["name"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name", "desc"],
            scoring_method="ratio",  # word-order sensitive
            scoring_methods={"desc": "token_sort_ratio"},  # order blind
        )
        matcher.fit(left, right)
        decision = matcher.decisions()[0]
        by_field = {fs.field: fs.score for fs in decision.explanation.field_scores}
        # token_sort ignores reordering -> 1.0; plain ratio penalises it.
        assert by_field["desc"] == 1.0

    def test_unknown_scoring_method_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            fm.Matcher(
                link_type="link",
                blocking=fm.ExactBlock(["k"]),
                classifier=fm.FSClassifier(),
                scoring_methods={"k": "bogus"},
            )

    def test_join_with_classifier_produces_merged_df(self):
        """Lines 274-289: join() with classifier and scoring_fields merges DataFrames."""
        left = pd.DataFrame(
            {"group": ["A"], "name": ["Office Supplies"]},
            index=[0],
        )
        right = pd.DataFrame(
            {"group": ["A"], "name": ["Office Supplies"]},
            index=[0],
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
            scoring_method="ratio",
        )
        result = matcher.join(left, right, on="name", threshold=0.01)
        # Should produce a merged DataFrame (at least 1 row for the exact match)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_join_with_high_threshold_returns_empty(self):
        """Lines 280-281: no pairs above threshold → empty DataFrame."""
        left = pd.DataFrame(
            {"group": ["A"], "name": ["hello"]},
            index=[0],
        )
        right = pd.DataFrame(
            {"group": ["A"], "name": ["xyz"]},
            index=[0],
        )
        matcher = fm.Matcher(
            link_type="fuzzy_join",
            blocking=fm.ExactBlock(["group"]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
            scoring_method="ratio",
        )
        result = matcher.join(left, right, on="name", threshold=0.999)
        # No pairs pass threshold -> empty DataFrame with proper suffixed columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "name_left" in result.columns
        assert "name_right" in result.columns
        assert "group_left" in result.columns
