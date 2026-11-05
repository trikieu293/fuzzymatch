"""Tests for blocking quality metrics."""

from __future__ import annotations

import pytest

from fuzzymatch.metrics import block_quality_metrics


def test_block_quality_metrics_for_partial_candidate_coverage():
    candidate_pairs = [(0, 0), (0, 1), (1, 1)]
    true_pairs = [(0, 0), (1, 1), (2, 2)]

    result = block_quality_metrics(
        candidate_pairs=candidate_pairs,
        true_pairs=true_pairs,
        total_possible_pairs=9,
    )

    assert result == {
        "reduction_ratio": pytest.approx(1.0 - 3 / 9),
        "pairs_completeness": pytest.approx(2 / 3),
        "pairs_quality": pytest.approx(2 / 3),
    }


def test_block_quality_metrics_empty_candidates_with_truth():
    result = block_quality_metrics(
        candidate_pairs=[],
        true_pairs=[(0, 0)],
        total_possible_pairs=4,
    )

    assert result == {
        "reduction_ratio": 1.0,
        "pairs_completeness": 0.0,
        "pairs_quality": 0.0,
    }


def test_block_quality_metrics_no_true_pairs_is_vacuously_complete():
    result = block_quality_metrics(
        candidate_pairs=[(0, 1)],
        true_pairs=[],
        total_possible_pairs=3,
    )

    assert result == {
        "reduction_ratio": pytest.approx(1.0 - 1 / 3),
        "pairs_completeness": 1.0,
        "pairs_quality": 0.0,
    }


def test_block_quality_metrics_rejects_negative_total_possible_pairs():
    with pytest.raises(ValueError, match="total_possible_pairs must be positive"):
        block_quality_metrics(
            candidate_pairs=[],
            true_pairs=[],
            total_possible_pairs=0,
        )
