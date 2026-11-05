"""Tests for serializable pipeline configuration summary."""

from __future__ import annotations

import fuzzymatch as fm


def test_pipeline_config_serializes_to_plain_dict():
    """PipelineConfig.to_dict() produces a JSON-serializable dict."""
    config = fm.PipelineConfig(
        link_type="dedupe",
        blocker="ExactBlock",
        normalizer_steps=("casefold",),
        scorer="RapidFuzzScorer",
        scoring_method="ratio",
        scoring_fields=("name",),
        classifier="FSClassifier",
        clusterer="ConnectedComponents",
        library_version="0.1.0",
    )

    assert config.to_dict() == {
        "link_type": "dedupe",
        "blocker": "ExactBlock",
        "normalizer_steps": ["casefold"],
        "scorer": "RapidFuzzScorer",
        "scoring_method": "ratio",
        "scoring_fields": ["name"],
        "classifier": "FSClassifier",
        "clusterer": "ConnectedComponents",
        "library_version": "0.1.0",
    }


def test_matcher_pipeline_config_reflects_components():
    """Matcher.pipeline_config() returns config matching the configured components."""
    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["group"]),
        classifier=fm.FSClassifier(),
        clusterer=fm.ConnectedComponents(),
        scoring_method="ratio",
        scoring_fields=["name"],
    )

    config = matcher.pipeline_config()

    assert config.link_type == "dedupe"
    assert config.blocker == "ExactBlock"
    assert config.normalizer_steps == ()
    assert config.scorer == "RapidFuzzScorer"
    assert config.scoring_method == "ratio"
    assert config.scoring_fields == ("name",)
    assert config.classifier == "FSClassifier"
    assert config.clusterer == "ConnectedComponents"
