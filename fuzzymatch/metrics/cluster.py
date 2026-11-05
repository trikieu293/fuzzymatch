"""Cluster-level evaluation metrics for entity resolution deduplication.

Provides B-cubed precision/recall/F1, Adjusted Rand Index (ARI),
and V-measure (homogeneity + completeness).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score


def cluster_metrics(
    clusters_true: list[set[int]],
    clusters_pred: list[set[int]],
) -> dict[str, float]:
    """Compute cluster-level evaluation metrics for deduplication output.

    Parameters
    ----------
    clusters_true : list[set[int]]
        Ground-truth entity clusters. Each set contains the record indices
        belonging to one entity.
    clusters_pred : list[set[int]]
        Predicted entity clusters, same format.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - ``b_cubed_precision`` (0--1)
        - ``b_cubed_recall`` (0--1)
        - ``b_cubed_f1`` (0--1)
        - ``ari`` (-1--1) --- Adjusted Rand Index
        - ``v_measure`` (0--1)

    Raises
    ------
    ValueError
        If any record index appears in more than one cluster on either side
        (clusters must form a partition).
    """
    _validate_partition(clusters_true, "clusters_true")
    _validate_partition(clusters_pred, "clusters_pred")

    bc_prec, bc_rec, bc_f1 = _b_cubed(clusters_true, clusters_pred)

    labels_true = _clusters_to_labels(clusters_true)
    labels_pred = _clusters_to_labels(clusters_pred)
    all_items = sorted(set(labels_true.keys()) | set(labels_pred.keys()))
    true_arr = np.array([labels_true.get(i, -1) for i in all_items], dtype=int)
    pred_arr = np.array([labels_pred.get(i, -1) for i in all_items], dtype=int)

    ari = float(adjusted_rand_score(true_arr, pred_arr))
    vm = float(v_measure_score(true_arr, pred_arr))

    return {
        "b_cubed_precision": bc_prec,
        "b_cubed_recall": bc_rec,
        "b_cubed_f1": bc_f1,
        "ari": ari,
        "v_measure": vm,
    }


def _validate_partition(clusters: list[set[int]], name: str) -> None:
    """Raise ValueError if any item appears in more than one cluster."""
    seen: dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        for item in cluster:
            if item in seen:
                raise ValueError(
                    f"clusters must be a partition: record {item} appears "
                    f"in clusters {seen[item]} and {cid} of {name}"
                )
            seen[item] = cid


def _b_cubed(
    clusters_true: list[set[int]],
    clusters_pred: list[set[int]],
) -> tuple[float, float, float]:
    """Compute B-cubed precision, recall, and F1.

    B-cubed evaluates per-item: for each record, precision is the fraction
    of its predicted cluster that shares its true cluster; recall is the
    fraction of its true cluster that appears in its predicted cluster.
    Scores are macro-averaged across all items.
    """
    # Build item -> cluster_id maps
    true_map: dict[int, int] = {}
    for cid, cluster in enumerate(clusters_true):
        for item in cluster:
            true_map[item] = cid

    pred_map: dict[int, int] = {}
    for cid, cluster in enumerate(clusters_pred):
        for item in cluster:
            pred_map[item] = cid

    # Collect all items across both sides
    all_items = set(true_map.keys()) | set(pred_map.keys())

    if not all_items:
        return 1.0, 1.0, 1.0

    # Materialize cluster sets for O(1) intersection lookups
    true_sets = [set(c) for c in clusters_true]
    pred_sets = [set(c) for c in clusters_pred]

    precisions: list[float] = []
    recalls: list[float] = []

    for item in all_items:
        true_cid = true_map.get(item)
        pred_cid = pred_map.get(item)

        # Item missing from one side -> treated as singleton
        true_cluster = true_sets[true_cid] if true_cid is not None else {item}
        pred_cluster = pred_sets[pred_cid] if pred_cid is not None else {item}

        overlap = len(true_cluster & pred_cluster)
        prec = overlap / len(pred_cluster) if pred_cluster else 0.0
        rec = overlap / len(true_cluster) if true_cluster else 0.0

        precisions.append(prec)
        recalls.append(rec)

    avg_prec = float(np.mean(precisions))
    avg_rec = float(np.mean(recalls))
    f1 = (2 * avg_prec * avg_rec / (avg_prec + avg_rec)) if (avg_prec + avg_rec) > 0 else 0.0
    return avg_prec, avg_rec, f1


def _clusters_to_labels(clusters: list[set[int]]) -> dict[int, int]:
    """Convert list[set[int]] to {item: cluster_id} mapping."""
    labels: dict[int, int] = {}
    for cid, cluster in enumerate(clusters):
        for item in cluster:
            labels[item] = cid
    return labels
