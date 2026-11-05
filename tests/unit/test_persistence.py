"""Persistence contracts for fitted classifiers (operational requirement)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch.classification.fs_classifier import FSClassifier


def _separable() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.95, 0.92, 0.88, 0.94, 0.90, 0.10, 0.12, 0.08, 0.11, 0.09],
            "f2": [0.90, 0.94, 0.85, 0.92, 0.88, 0.15, 0.09, 0.11, 0.14, 0.08],
        }
    )


class TestFSClassifierPersistence:
    def test_em_roundtrip_preserves_probabilities(self, tmp_path: Path) -> None:
        features = _separable()
        original = FSClassifier().fit(features)
        path = tmp_path / "clf.json"
        original.save(path)

        restored = FSClassifier.load(path)
        a = original.predict_proba(features)
        b = restored.predict_proba(features)
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_supervised_roundtrip_preserves_probabilities(self, tmp_path: Path) -> None:
        features = pd.DataFrame({"f1": [0.9, 0.2], "f2": [0.8, 0.1]})
        labels = pd.Series([1, 0])
        original = FSClassifier(threshold=(0.6, 0.4, 0.2)).fit(features, labels)
        path = tmp_path / "sup.json"
        original.save(path)
        restored = FSClassifier.load(path)
        a = original.predict_proba(features).reset_index(drop=True)
        b = restored.predict_proba(features).reset_index(drop=True)
        assert a.tolist() == pytest.approx(b.tolist())

    def test_schema_markers_written(self, tmp_path: Path) -> None:
        clf = FSClassifier().fit(_separable())
        path = tmp_path / "clf.json"
        clf.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "fuzzymatch.fs_classifier"
        assert payload["schema_version"] == 1
        assert payload["em_mode"] is True
        assert payload["lambda_"] == clf.lambda_

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"schema": "fuzzymatch.fs_classifier", "schema_version": 99}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version"):
            FSClassifier.load(path)

    def test_wrong_schema_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "other.json"
        path.write_text(
            json.dumps({"schema": "something.else", "schema_version": 1}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema"):
            FSClassifier.load(path)
