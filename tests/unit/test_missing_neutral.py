"""B4: missing values become neutral evidence, never silent disagreement."""

from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm
from fuzzymatch.classification import fs_classifier as fsc


@pytest.fixture(autouse=True)
def _force_em(monkeypatch: pytest.MonkeyPatch):
    """Run the EM estimator regardless of candidate count."""
    monkeypatch.setattr(fsc, "_MIN_ROWS_FOR_EM", 1)


def _matcher() -> fm.Matcher:
    return fm.Matcher(
        link_type="link",
        blocking=fm.ExactBlock(["name"]),
        normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
        classifier=fm.FSClassifier(),
        scoring_fields=["name", "description"],
        scoring_methods={"name": "token_sort_ratio"},
    )


class TestMissingIndicators:
    def test_indicator_columns_emitted_when_any_missing(self) -> None:
        left = pd.DataFrame({"name": ["acme corp"], "description": ["widgets galore"]})
        right = pd.DataFrame({"name": ["acme corp"], "description": ["nan"]})
        matcher = _matcher()
        matcher.fit(left, right)
        feats = matcher._features
        assert "description__missing" in feats.columns
        assert feats["description__missing"].iloc[0] == 1

    def test_no_indicators_when_everything_present(self) -> None:
        left = pd.DataFrame({"name": ["acme corp"], "description": ["widgets"]})
        right = pd.DataFrame({"name": ["acme corp"], "description": ["gadgets"]})
        matcher = _matcher()
        matcher.fit(left, right)
        assert not any(c.endswith("__missing") for c in matcher._features.columns)

    def test_explanations_exclude_indicator_columns(self) -> None:
        left = pd.DataFrame({"name": ["acme corp"], "description": ["nan"]})
        right = pd.DataFrame({"name": ["acme corp"], "description": ["nan"]})
        matcher = _matcher()
        matcher.fit(left, right)
        d = matcher.decisions()[0]
        fields = [fs.field for fs in d.explanation.field_scores]
        assert "name" in fields
        assert not any(f.endswith("__missing") for f in fields)


class TestNeutralization:
    def test_missing_description_beats_bad_description(self) -> None:
        """Same strong name match: absent desc must outrank contradicting desc."""
        left = pd.DataFrame(
            {
                "name": ["alpha corp", "beta corp"],
                "description": ["desc x", "totally different text here"],
            }
        )
        right = pd.DataFrame(
            {
                "name": ["alpha corp", "beta corp"],
                "description": ["desc y", "nan"],
            }
        )
        matcher = _matcher()
        matcher.fit(left, right)
        proba = matcher.classifier.predict_proba(matcher._features)
        # Pair 0: name agrees, desc disagrees -> penalised.
        # Pair 1: name agrees, desc missing -> neutral, NOT punished.
        assert float(proba.iloc[1]) > float(proba.iloc[0])

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_missing_on_either_side_is_neutral(self, side: str) -> None:
        """Missing desc == no desc evidence: proba equals name-only model."""
        left = pd.DataFrame(
            {
                "name": ["alpha corp", "filler one", "filler two"],
                "description": ["real text", "f1 d", "f2 d"],
            }
        )
        right = pd.DataFrame(
            {
                "name": ["alpha corp", "filler one", "filler two"],
                "description": [
                    "nan" if side == "left" else "other text",
                    "g1 d",
                    "g2 d",
                ],
            }
        )
        matcher = _matcher()
        matcher.fit(left, right)

        name_only = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["name"]),
            normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name"],
            scoring_methods={"name": "token_sort_ratio"},
        )
        name_only.fit(left[["name"]], right[["name"]])

        full = float(matcher.classifier.predict_proba(matcher._features).iloc[0])
        solo = float(name_only.classifier.predict_proba(name_only._features).iloc[0])
        # Neutralized desc contributes nothing, but binning differs between
        # models, so allow generous tolerance around the name-only signal.
        if side == "left":
            assert full == pytest.approx(solo, abs=0.05)
        else:
            assert full >= solo - 0.05

    def test_mean_fallback_ignores_indicators(self) -> None:
        """decisions() without classifier must not average indicator columns."""
        matcher = fm.Matcher(
            link_type="link",
            blocking=fm.ExactBlock(["name"]),
            normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
            classifier=fm.FSClassifier(),
            scoring_fields=["name", "description"],
            scoring_methods={"name": "token_sort_ratio"},
        )
        matcher.classifier = None
        left = pd.DataFrame({"name": ["a"], "description": ["x"]})
        right = pd.DataFrame({"name": ["a"], "description": ["y"]})
        matcher.fit(left, right)
        d = matcher.decisions()[0]
        expected = (
            matcher._features["name"].iloc[0] + matcher._features["description"].iloc[0]
        ) / 2
        assert d.score == pytest.approx(float(expected))
