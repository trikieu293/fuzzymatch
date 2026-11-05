"""Integration tests against ER benchmark datasets.

These tests load an actual ER-Magellan benchmark and run the full
fuzzymatch pipeline end-to-end. They are marked `slow` so the default
`pytest` run skips them. To run:

    pytest tests/integration -m slow --network

Requires network access to download the dataset on first run.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import fuzzymatch as fm

NETWORK_REQUIRED = pytest.mark.skipif(
    "FMM_SKIP_NETWORK" in os.environ,
    reason="set FMM_SKIP_NETWORK=1 to skip network-dependent tests",
)


@NETWORK_REQUIRED
@pytest.mark.slow
def test_abt_buy_smoke():
    """Smoke test: load Abt-Buy and run dedupe pipeline end-to-end."""
    from fuzzymatch.datasets import load

    try:
        labeled, table_a, table_b = load("Abt-Buy")
    except (OSError, TimeoutError) as exc:
        pytest.skip(f"Network unavailable: {exc}")
    assert len(labeled) > 0
    assert len(table_a) > 0
    assert len(table_b) > 0

    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["name"]),
        scorer=fm.RapidFuzzScorer(),
        classifier=fm.FSClassifier(),
        clusterer=fm.ConnectedComponents(),
        scoring_fields=["name"],
        scoring_method="ratio",
    )
    matcher.fit(table_a)
    pairs = matcher.predict(threshold="possible")
    assert isinstance(pairs, pd.MultiIndex)


@NETWORK_REQUIRED
@pytest.mark.slow
def test_cluster_metrics_on_abt_buy():
    """Load Abt-Buy, transform to dedup, run pipeline, verify cluster metrics."""
    import networkx as nx
    import pandas as pd

    from fuzzymatch.datasets import load
    from fuzzymatch.metrics import cluster_metrics

    try:
        labeled, table_a, table_b = load("Abt-Buy")
    except (OSError, TimeoutError) as exc:
        pytest.skip(f"Network unavailable: {exc}")
    n_a = len(table_a)

    # 1. Concatenate tables into one dedup DataFrame
    combined = pd.concat(
        [table_a.assign(_source="a"), table_b.assign(_source="b")],
        ignore_index=True,
    )

    # 2. Derive ground-truth clusters from labeled pairs
    graph = nx.Graph()
    for _, row in labeled[labeled["label"] == 1].iterrows():
        graph.add_edge(int(row["ltable_id"]), int(row["rtable_id"]) + n_a)
    clusters_true = [set(c) for c in nx.connected_components(graph)]

    # 3. Run full dedupe pipeline
    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["name"]),
        scorer=fm.RapidFuzzScorer(),
        classifier=fm.FSClassifier(),
        clusterer=fm.ConnectedComponents(),
        scoring_fields=["name"],
        scoring_method="ratio",
    )
    matcher.fit(combined)
    clusters_pred = matcher.clusters(threshold="possible")

    # 4. Evaluate
    metrics = cluster_metrics(clusters_true, clusters_pred)
    assert 0.0 <= metrics["b_cubed_f1"] <= 1.0
    assert -1.0 <= metrics["ari"] <= 1.0
    assert 0.0 <= metrics["v_measure"] <= 1.0
