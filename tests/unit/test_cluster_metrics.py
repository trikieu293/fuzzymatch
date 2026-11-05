"""Unit tests for cluster_metrics."""

from __future__ import annotations

import pytest

from fuzzymatch.metrics.cluster import cluster_metrics


def test_perfect_agreement():
    """Identical clusters produce perfect scores."""
    clusters = [{0, 1, 2}, {3, 4}]
    result = cluster_metrics(clusters, clusters)
    assert result["b_cubed_precision"] == pytest.approx(1.0)
    assert result["b_cubed_recall"] == pytest.approx(1.0)
    assert result["b_cubed_f1"] == pytest.approx(1.0)
    assert result["ari"] == pytest.approx(1.0)
    assert result["v_measure"] == pytest.approx(1.0)


def test_empty_both():
    """Both empty returns 1.0 for all metrics (vacuously perfect)."""
    result = cluster_metrics([], [])
    assert result["b_cubed_precision"] == pytest.approx(1.0)
    assert result["b_cubed_recall"] == pytest.approx(1.0)
    assert result["b_cubed_f1"] == pytest.approx(1.0)
    assert result["ari"] == pytest.approx(1.0)
    assert result["v_measure"] == pytest.approx(1.0)


def test_complete_disagreement():
    """All singletons vs one big cluster -- low B-cubed scores."""
    clusters_true = [{0}, {1}, {2}, {3}]
    clusters_pred = [{0, 1, 2, 3}]
    result = cluster_metrics(clusters_true, clusters_pred)
    # Each singleton item: pred_cluster has size 4, intersect with true has size 1
    # precision_i = 1/4 = 0.25, recall_i = 1/1 = 1.0
    assert result["b_cubed_precision"] == pytest.approx(0.25)
    assert result["b_cubed_recall"] == pytest.approx(1.0)
    assert 0 <= result["b_cubed_f1"] <= 1


def test_empty_pred_nonempty_true():
    """Empty predictions vs non-empty ground truth."""
    clusters_true = [{0, 1}, {2, 3}]
    result = cluster_metrics(clusters_true, [])
    # Each true item forms its own predicted singleton
    # precision_i = 1/1 = 1.0, recall_i = 1/2 = 0.5
    assert result["b_cubed_precision"] == pytest.approx(1.0)
    assert result["b_cubed_recall"] == pytest.approx(0.5)
    assert 0 <= result["b_cubed_f1"] <= 1


def test_one_cluster_each_identical():
    """Single cluster each, identical membership."""
    clusters = [{0, 1, 2}]
    result = cluster_metrics(clusters, clusters)
    assert result["b_cubed_f1"] == pytest.approx(1.0)
    assert result["ari"] == pytest.approx(1.0)
    assert result["v_measure"] == pytest.approx(1.0)


def test_split_cluster():
    """One true cluster split into two predicted clusters."""
    clusters_true = [{0, 1, 2, 3}]
    clusters_pred = [{0, 1}, {2, 3}]
    result = cluster_metrics(clusters_true, clusters_pred)
    # Each item: entire predicted cluster is inside the true cluster
    # prec_i = 2/2 = 1.0, rec_i = 2/4 = 0.5
    assert result["b_cubed_precision"] == pytest.approx(1.0)
    assert result["b_cubed_recall"] == pytest.approx(0.5)
    assert result["ari"] < 1.0  # not perfect
    assert 0 <= result["b_cubed_f1"] <= 1


def test_merged_cluster():
    """Two true clusters merged into one predicted cluster."""
    clusters_true = [{0, 1}, {2, 3}]
    clusters_pred = [{0, 1, 2, 3}]
    result = cluster_metrics(clusters_true, clusters_pred)
    # Each item: pred_size=4, intersect=2 (its true cluster)
    # prec_i = 2/4 = 0.5, rec_i = 2/2 = 1.0
    assert result["b_cubed_precision"] == pytest.approx(0.5)
    assert result["b_cubed_recall"] == pytest.approx(1.0)
    assert result["ari"] < 1.0
    assert 0 <= result["b_cubed_f1"] <= 1


def test_duplicate_record_raises():
    """Item appearing in two clusters raises ValueError."""
    clusters_bad = [{0, 1}, {0, 2}]  # 0 appears twice
    clusters_ok = [{0, 1}, {2, 3}]
    with pytest.raises(ValueError, match="clusters must be a partition"):
        cluster_metrics(clusters_ok, clusters_bad)
