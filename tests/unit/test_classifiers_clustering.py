"""Tests for classifiers and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fuzzymatch.classification.fs_classifier import FSClassifier
from fuzzymatch.classification.sklearn_classifier import SklearnClassifier
from fuzzymatch.clustering.connected_components import ConnectedComponents, PluggableResolver


class TestFSClassifier:
    def test_fit_supervised(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.8, 0.2, 0.1], "f2": [0.95, 0.85, 0.15, 0.1]})
        labels = pd.Series([1, 1, 0, 0])
        clf = FSClassifier()
        clf.fit(features, labels)
        assert "f1" in clf.weights_
        assert "f2" in clf.weights_
        assert clf.weights_["f1"] > 0

    def test_fit_unsupervised(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.8, 0.2, 0.1]})
        clf = FSClassifier()
        clf.fit(features)
        assert "f1" in clf.weights_

    def test_predict_proba_before_fit_raises(self) -> None:
        clf = FSClassifier()
        features = pd.DataFrame({"f1": [0.5]})
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(features)

    def test_predict_match(self) -> None:
        features = pd.DataFrame({"f1": [0.95, 0.90, 0.05, 0.02], "f2": [0.98, 0.92, 0.08, 0.03]})
        labels = pd.Series([1, 1, 0, 0])
        clf = FSClassifier(threshold=(0.5, 0.3, 0.2))
        clf.fit(features, labels)
        proba = clf.predict_proba(features)
        # High-scoring pairs should have higher probability than low-scoring
        assert proba.iloc[0] > proba.iloc[3]

    def test_predict_possible(self) -> None:
        features = pd.DataFrame({"f1": [0.95, 0.5, 0.05]})
        labels = pd.Series([1, 1, 0])
        clf = FSClassifier(threshold=(0.8, 0.4, 0.2))
        clf.fit(features, labels)
        preds = clf.predict(features, threshold="possible")
        assert isinstance(preds, pd.Series)

    def test_predict_non_match(self) -> None:
        features = pd.DataFrame({"f1": [0.95, 0.05]})
        labels = pd.Series([1, 0])
        clf = FSClassifier()
        clf.fit(features, labels)
        preds = clf.predict(features, threshold="non_match")
        assert bool(preds.iloc[1]) is True

    def test_predict_unknown_threshold_raises(self) -> None:
        features = pd.DataFrame({"f1": [0.5]})
        clf = FSClassifier()
        clf.fit(features)
        with pytest.raises(ValueError, match="Unknown threshold"):
            clf.predict(features, threshold="bogus")

    def test_fit_no_matches(self) -> None:
        features = pd.DataFrame({"f1": [0.3, 0.2]})
        labels = pd.Series([0, 0])
        clf = FSClassifier()
        clf.fit(features, labels)
        assert clf.m_["f1"] == 0.8  # default fallback

    def test_fit_no_non_matches(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.8]})
        labels = pd.Series([1, 1])
        clf = FSClassifier()
        clf.fit(features, labels)
        assert clf.u_["f1"] == 0.05  # default fallback


class TestFSClassifierEM:
    """Unsupervised Fellegi-Sunter EM over binned agreement levels."""

    def _separable_features(self) -> pd.DataFrame:
        """Two well-separated clusters across two fields (n=12, above EM floor)."""
        return pd.DataFrame(
            {
                "f1": [0.95, 0.92, 0.88, 0.94, 0.90, 0.96, 0.10, 0.12, 0.08, 0.11, 0.09, 0.13],
                "f2": [0.90, 0.94, 0.85, 0.92, 0.88, 0.95, 0.15, 0.09, 0.11, 0.14, 0.08, 0.12],
            }
        )

    def test_match_band_reachable_out_of_box(self) -> None:
        """The headline regression: default match band must be reachable."""
        features = self._separable_features()
        clf = FSClassifier()
        clf.fit(features)
        proba = clf.predict_proba(features)
        assert (proba[:6] > 0.8).all(), "high-agreement rows must reach match band"
        assert (proba[6:] < 0.2).all(), "low-agreement rows must fall to non-match"

    def test_predict_separates_clusters(self) -> None:
        features = self._separable_features()
        clf = FSClassifier()
        clf.fit(features)
        preds = clf.predict(features, threshold="match")
        assert preds[:6].all()
        assert not preds[6:].any()

    def test_uninformative_field_downweighted(self) -> None:
        """A field with no signal gets ~zero weight relative to a real signal."""
        features = pd.DataFrame(
            {
                "signal": [0.95, 0.92, 0.88, 0.94, 0.90, 0.10, 0.12, 0.08, 0.11, 0.09],
                "noise": [0.5] * 10,  # constant: carries no information
            }
        )
        clf = FSClassifier()
        clf.fit(features)
        assert abs(clf.weights_["noise"]) < abs(clf.weights_["signal"])

    def test_small_candidate_set_falls_back_to_legacy(self) -> None:
        """Below the EM floor the legacy estimator keeps old small-data behaviour."""
        features = pd.DataFrame({"f1": [0.97], "f2": [0.96]})
        clf = FSClassifier()
        clf.fit(features)
        assert clf._em_mode is False
        proba = clf.predict_proba(features)
        assert proba.iloc[0] > 0.8

    def test_deterministic_across_refits(self) -> None:
        features = self._separable_features()
        probas = []
        for _ in range(2):
            clf = FSClassifier()
            clf.fit(features)
            probas.append(clf.predict_proba(features).to_numpy())
        assert (probas[0] == probas[1]).all()

    def test_lambda_updated_and_bounded(self) -> None:
        features = self._separable_features()
        clf = FSClassifier()
        clf.fit(features)
        assert 0.0 < clf.lambda_ < 1.0
        assert clf.lambda_ != 0.1  # EM moved off the initial prior

    def test_proba_in_range_no_nan(self) -> None:
        features = self._separable_features()
        clf = FSClassifier()
        clf.fit(features)
        proba = clf.predict_proba(features)
        assert proba.notna().all()
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_single_row_candidate(self) -> None:
        """One agreeing pair keeps legacy behaviour: confident match."""
        features = pd.DataFrame({"f1": [0.97], "f2": [0.96]})
        clf = FSClassifier()
        clf.fit(features)
        proba = clf.predict_proba(features)
        assert proba.iloc[0] > 0.8

    def test_constant_column_only(self) -> None:
        """Degenerate all-constant input must not crash or produce NaN."""
        features = pd.DataFrame({"f1": [0.4] * 5 + [0.6] * 5})
        clf = FSClassifier()
        clf.fit(features)
        proba = clf.predict_proba(features)
        assert proba.notna().all()

    def test_em_predict_proba_preserves_custom_index(self) -> None:
        """EM predict_proba must preserve the input DataFrame's index."""
        features = self._separable_features()
        custom_index = [f"pair_{i}" for i in range(len(features))]
        features.index = pd.Index(custom_index)
        clf = FSClassifier()
        clf.fit(features)
        proba = clf.predict_proba(features)
        assert proba.index.equals(features.index)
        assert proba.index[0] == "pair_0"

    def test_em_observed_only_quantile_conditioning(self) -> None:
        """Missing values flagged via __missing must not distort quantile bin edges."""
        base_features = self._separable_features()
        clf_base = FSClassifier()
        clf_base.fit(base_features)

        # Add rows with f1 missing (f1=0.0, f1__missing=1.0)
        n_missing = 20
        missing_rows = pd.DataFrame(
            {
                "f1": [0.0] * n_missing,
                "f1__missing": [1.0] * n_missing,
                "f2": [0.9] * n_missing,
            }
        )
        base_with_missing_flag = base_features.copy()
        base_with_missing_flag["f1__missing"] = 0.0

        augmented = pd.concat([base_with_missing_flag, missing_rows], ignore_index=True)
        clf_aug = FSClassifier()
        clf_aug.fit(augmented)

        # The quantile bin edges for f1 should match because they are conditioned only on observed values
        np.testing.assert_allclose(clf_aug._edges["f1"], clf_base._edges["f1"])


class TestSklearnClassifier:
    def test_fit_and_predict(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.8, 0.2, 0.1], "f2": [0.95, 0.85, 0.15, 0.05]})
        labels = pd.Series([1, 1, 0, 0])
        clf = SklearnClassifier(LogisticRegression())
        clf.fit(features, labels)
        preds = clf.predict(features, threshold="match")
        assert len(preds) == 4

    def test_predict_proba(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.1], "f2": [0.9, 0.1]})
        labels = pd.Series([1, 0])
        clf = SklearnClassifier(LogisticRegression())
        clf.fit(features, labels)
        proba = clf.predict_proba(features)
        assert len(proba) == 2
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_predict_possible(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.5, 0.1], "f2": [0.9, 0.5, 0.1]})
        labels = pd.Series([1, 1, 0])
        clf = SklearnClassifier(LogisticRegression(), threshold=(0.8, 0.4, 0.2))
        clf.fit(features, labels)
        preds = clf.predict(features, threshold="possible")
        assert len(preds) == 3

    def test_predict_nonmatch(self) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.1], "f2": [0.9, 0.1]})
        labels = pd.Series([1, 0])
        clf = SklearnClassifier(LogisticRegression())
        clf.fit(features, labels)
        preds = clf.predict(features, threshold="other")
        assert len(preds) == 2


class TestConnectedComponents:
    def test_basic_clustering(self) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
        cc = ConnectedComponents()
        clusters = cc.resolve(pairs)
        assert len(clusters) == 2
        # {0,1,2} and {3,4}
        cluster_sizes = sorted(len(c) for c in clusters)
        assert cluster_sizes == [2, 3]

    def test_with_cutoff(self) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 1), (1, 2), (3, 4)])
        weights = pd.Series([0.9, 0.3, 0.8])
        cc = ConnectedComponents(edge_weight_cutoff=0.5)
        clusters = cc.resolve(pairs, edge_weights=weights)
        # (0,1) passes cutoff -> {0,1}, (1,2) dropped, (3,4) passes -> {3,4}, {2} isolated
        # connected_components only returns nodes that are in the graph
        assert len(clusters) == 2  # {0,1} and {3,4}

    def test_no_cutoff_keeps_all(self) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 1), (1, 2)])
        weights = pd.Series([0.9, 0.3])
        cc = ConnectedComponents()
        clusters = cc.resolve(pairs, edge_weights=weights)
        assert len(clusters) == 1

    def test_single_pair(self) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 1)])
        cc = ConnectedComponents()
        clusters = cc.resolve(pairs)
        assert len(clusters) == 1
        assert clusters[0] == {0, 1}


class TestPluggableResolver:
    def test_custom_resolver(self) -> None:
        def my_resolver(pairs, weights):
            return [{0, 1, 2}]

        pr = PluggableResolver(my_resolver)
        pairs = pd.MultiIndex.from_tuples([(0, 1), (1, 2)])
        result = pr.resolve(pairs)
        assert result == [{0, 1, 2}]
