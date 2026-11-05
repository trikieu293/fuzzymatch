"""Adversarial stress harness for StarClusterer.

Verifies:
1. Intransitive triad splitting under threshold boundaries.
2. Global intra-cluster clique verification oracle.
3. Disjointness and node conservation invariants.
4. Degenerate graph topologies (star, complete bipartite, chain, cycle, empty, isolated).
5. Heterogeneous and UUID node identifiers.
6. Numerical boundary precision (tau, tau - eps, duplicate weights).
"""

from __future__ import annotations

import heapq
import random
import uuid
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzymatch.clustering.star_cluster import StarClusterer


class TestIntransitiveTriadSplitting:
    """Stress tests specifically targeting intransitive triads:
    w(A, B) >= tau AND w(B, C) >= tau BUT w(A, C) < tau (or edge absent).

    Mathematical Guarantee:
    Under verify_clique=True, A and C must NEVER be co-clustered.
    """

    @pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7, 0.85, 0.99])
    def test_systematic_intransitive_triads_varying_cutoff(self, tau: float) -> None:
        """Test with varying cutoff thresholds and exact boundary values."""
        eps = 1e-4
        pairs = [("A", "B"), ("B", "C"), ("A", "C")]
        # A-B matches, B-C matches, A-C fails by eps
        weights = [
            tau + 0.001 if tau <= 0.998 else 1.0,
            tau + 0.001 if tau <= 0.998 else 1.0,
            max(0.0, tau - eps),
        ]

        sc = StarClusterer(cutoff=tau, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        # Assert A and C are NEVER in the same cluster
        for c in clusters:
            assert not ({"A", "C"}.issubset(c)), f"A and C co-clustered at tau={tau}: {c}"

        # Assert node conservation: all nodes {A, B, C} appear in the partition
        all_clustered_nodes = set.union(*clusters) if clusters else set()
        assert all_clustered_nodes == {"A", "B", "C"}

    def test_randomized_intransitive_triad_chains(self) -> None:
        """Generate a random network of 100 intransitive triads.

        For each triad i:
          w(A_i, B_i) >= 0.8
          w(B_i, C_i) >= 0.8
          w(A_i, C_i) <= 0.2 (below cutoff 0.6)
        """
        rng = random.Random(42)
        pairs: list[tuple[str, str]] = []
        weights: list[float] = []
        triad_records: list[tuple[str, str, str]] = []

        for i in range(100):
            a, b, c = f"A_{i}", f"B_{i}", f"C_{i}"
            triad_records.append((a, b, c))
            w_ab = rng.uniform(0.7, 1.0)
            w_bc = rng.uniform(0.7, 1.0)
            w_ac = rng.uniform(0.0, 0.4)
            pairs.extend([(a, b), (b, c), (a, c)])
            weights.extend([w_ab, w_bc, w_ac])

        sc = StarClusterer(cutoff=0.6, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        # Verify that for every triad i, {A_i, C_i} is NEVER a subset of any cluster
        for a, _b, c in triad_records:
            for cluster in clusters:
                assert not ({a, c}.issubset(cluster)), (
                    f"Triad violation: {a} and {c} co-clustered in {cluster}"
                )

    def test_interconnected_triads_dense_bridge(self) -> None:
        """Two triads sharing a common node B:
        Triad 1: A1 - B (0.9), B - C1 (0.9), A1 - C1 (0.1)
        Triad 2: A2 - B (0.9), B - C2 (0.9), A2 - C2 (0.1)
        Cross edges: A1 - A2 (0.1), C1 - C2 (0.1)
        """
        pairs = [
            ("A1", "B"),
            ("B", "C1"),
            ("A1", "C1"),
            ("A2", "B"),
            ("B", "C2"),
            ("A2", "C2"),
            ("A1", "A2"),
            ("C1", "C2"),
        ]
        weights = [0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        for c in clusters:
            assert not ({"A1", "C1"}.issubset(c))
            assert not ({"A2", "C2"}.issubset(c))
            assert not ({"A1", "A2"}.issubset(c))
            assert not ({"C1", "C2"}.issubset(c))


class TestGlobalCliqueOracle:
    """Hypothesis property-based tests verifying the formal mathematical invariant:
    FOR EVERY cluster C produced by StarClusterer(verify_clique=True),
    every pair of distinct nodes u, v in C MUST have an edge with w(u, v) >= cutoff.
    """

    @given(
        num_nodes=st.integers(min_value=3, max_value=25),
        edge_prob=st.floats(min_value=0.1, max_value=0.9),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_all_clusters_are_cliques_in_thresholded_graph(
        self, num_nodes: int, edge_prob: float, seed: int
    ) -> None:
        rng = random.Random(seed)
        nodes = [f"node_{i}" for i in range(num_nodes)]
        cutoff = 0.5

        # Construct random weighted graph
        pairs: list[tuple[str, str]] = []
        weights: list[float] = []
        adj: dict[str, dict[str, float]] = {u: {} for u in nodes}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < edge_prob:
                    u, v = nodes[i], nodes[j]
                    w = round(rng.uniform(0.0, 1.0), 4)
                    pairs.append((u, v))
                    weights.append(w)
                    if w >= cutoff:
                        adj[u][v] = w
                        adj[v][u] = w

        sc = StarClusterer(cutoff=cutoff, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        # 1. Disjointness invariant: pairwise intersection must be empty
        all_seen: set[str] = set()
        for c in clusters:
            assert len(c & all_seen) == 0, f"Clusters overlap on {c & all_seen}"
            all_seen.update(c)

        # 2. Strict Clique Invariant:
        # For every cluster of size >= 2, every pair u, v in C must satisfy adj[u][v] >= cutoff
        for c in clusters:
            node_list = list(c)
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    u, v = node_list[i], node_list[j]
                    w = adj[u].get(v, 0.0)
                    assert w >= cutoff, (
                        f"Clique violation in cluster {c}: edge ({u}, {v}) has weight {w} < {cutoff}"
                    )

        # 3. Active Node Conservation:
        # Every node that has at least one edge with w >= cutoff must be clustered
        active_nodes = {u for u in nodes if len(adj[u]) > 0}
        assert active_nodes.issubset(all_seen)


class TestDegenerateTopologies:
    """Stress test degenerate and boundary graph topologies."""

    def test_star_graph_topology(self) -> None:
        """Star graph K_{1, n}: center node P connected to n satellites with NO satellite-satellite edges.

        Under verify_clique=True:
        Center P can join with at most ONE satellite (since satellites have no mutual edges).
        All other satellites must become singletons!
        Total clusters = n, all clusters have size <= 2.
        """
        n = 50
        center = "CENTER"
        satellites = [f"SAT_{i}" for i in range(n)]
        pairs = [(center, sat) for sat in satellites]
        weights = [0.8] * n

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        # Total clusters must be exactly n (one cluster of {CENTER, SAT_0}, and n-1 singletons)
        assert len(clusters) == n
        for c in clusters:
            assert len(c) <= 2
        all_nodes = set.union(*clusters)
        assert all_nodes == {center, *satellites}

    def test_complete_bipartite_topology(self) -> None:
        """Complete bipartite graph K_{m, n}:
        All edges between set X and set Y. Zero edges within X, zero edges within Y.

        Under verify_clique=True:
        No cluster can have >1 node from X (since no intra-X edges).
        No cluster can have >1 node from Y (since no intra-Y edges).
        Therefore, every cluster must have size <= 2!
        """
        X = [f"X_{i}" for i in range(10)]
        Y = [f"Y_{j}" for j in range(10)]
        pairs = [(x, y) for x in X for y in Y]
        weights = [0.85] * len(pairs)

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        assert len(clusters) == 10  # 10 pairs of size 2
        for c in clusters:
            assert len(c) == 2
            x_count = sum(1 for node in c if node in X)
            y_count = sum(1 for node in c if node in Y)
            assert x_count == 1 and y_count == 1

    def test_linear_chain_topology(self) -> None:
        """Long linear chain: 0 - 1 - 2 - 3 - ... - 99.
        No non-adjacent edges exist.
        Clusters must have size <= 2, and no non-adjacent nodes clustered.
        """
        n = 100
        pairs = [(f"N_{i}", f"N_{i + 1}") for i in range(n - 1)]
        weights = [0.9] * len(pairs)

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        for c in clusters:
            assert len(c) <= 2
            if len(c) == 2:
                u, v = list(c)
                idx_u = int(u.split("_")[1])
                idx_v = int(v.split("_")[1])
                assert abs(idx_u - idx_v) == 1

        all_nodes = set.union(*clusters)
        assert len(all_nodes) == n

    def test_ring_cycle_topology(self) -> None:
        """Cycle graph C_{20}: 0-1-2-...-19-0."""
        n = 20
        pairs = [(f"V_{i}", f"V_{(i + 1) % n}") for i in range(n)]
        weights = [0.88] * n

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        for c in clusters:
            assert len(c) <= 2
        all_nodes = set.union(*clusters)
        assert len(all_nodes) == n

    def test_disconnected_cliques(self) -> None:
        """Three separate, disconnected dense cliques: K_4, K_5, K_6."""
        pairs: list[tuple[str, str]] = []
        weights: list[float] = []

        for prefix, k in [("A", 4), ("B", 5), ("C", 6)]:
            nodes = [f"{prefix}_{i}" for i in range(k)]
            for i in range(k):
                for j in range(i + 1, k):
                    pairs.append((nodes[i], nodes[j]))
                    weights.append(0.95)

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        assert len(clusters) == 3
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [4, 5, 6]


class TestHeterogeneousAndUUIDIdentifiers:
    """Verify arbitrary heterogeneous node types without crash or TypeError."""

    def test_uuid_and_mixed_type_identifiers(self) -> None:
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        s1 = "entity_string_1"
        i1 = 42
        tup1 = ("compound", "key")

        # Mix UUID, str, int, tuple
        pairs: list[tuple[Any, Any]] = [
            (u1, u2),
            (u2, s1),
            (s1, i1),
            (i1, tup1),
        ]
        weights = [0.9, 0.9, 0.9, 0.9]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        assert len(clusters) >= 1
        all_nodes = set.union(*clusters)
        assert all_nodes == {u1, u2, s1, i1, tup1}

    def test_special_characters_and_empty_strings(self) -> None:
        """Strings with unicode, emojis, commas, quotes, and empty string."""
        s_empty = ""
        s_emoji = "Vendor 🏢"
        s_comma = "Acme, Inc. & Co."
        s_quote = 'Quote "Test" \n Tab\t'

        pairs = [
            (s_empty, s_emoji),
            (s_emoji, s_comma),
            (s_comma, s_quote),
        ]
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster(pairs, [0.8, 0.8, 0.8])
        assert len(clusters) >= 1
        all_nodes = set.union(*clusters)
        assert s_empty in all_nodes
        assert s_emoji in all_nodes


class TestNumericalBoundariesAndDuplicates:
    """Test boundary floating point comparisons and duplicate edges."""

    def test_exact_cutoff_boundary(self) -> None:
        """Edge with weight EXACTLY equal to cutoff must be included (>= cutoff)."""
        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster([("A", "B")], [0.5])
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_infinitesimal_below_cutoff(self) -> None:
        """Edge with weight cutoff - 1e-15 must be excluded (< cutoff)."""
        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster([("A", "B")], [0.5 - 1e-15])
        assert len(clusters) == 0

    def test_conflicting_duplicate_edges(self) -> None:
        """Multiple edges between (A, B) with weights [0.2, 0.9, 0.4] takes max weight 0.9."""
        pairs = [("A", "B"), ("A", "B"), ("B", "A")]
        weights = [0.2, 0.9, 0.4]
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster(pairs, weights)
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_zero_cutoff(self) -> None:
        """Cutoff = 0.0 includes all non-negative edges."""
        sc = StarClusterer(cutoff=0.0, verify_clique=True)
        clusters = sc.cluster([("A", "B")], [0.0])
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}


class TestConnectedComponentsAdversarial:
    """Verify ConnectedComponents and PluggableResolver with adversarial inputs."""

    def test_connected_components_uuid_and_mixed(self) -> None:
        from fuzzymatch.clustering.connected_components import ConnectedComponents

        u1, u2, u3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        s1 = "company_xyz"
        pairs = [(u1, u2), (u2, s1), (s1, u3)]
        cc = ConnectedComponents()
        clusters = cc.cluster(pairs)
        assert len(clusters) == 1
        assert clusters[0] == {u1, u2, u3, s1}

    def test_connected_components_with_cutoff(self) -> None:
        from fuzzymatch.clustering.connected_components import ConnectedComponents

        pairs = [("A", "B"), ("B", "C")]
        weights = [0.9, 0.3]
        cc = ConnectedComponents(edge_weight_cutoff=0.5)
        clusters = cc.resolve(pairs, weights)
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}


class TestDensePathologicalGraphs:
    """Stress tests on dense and pathological graphs."""

    def test_dense_uniform_weights_clique(self) -> None:
        """K_30 complete graph with all identical weights 0.85."""
        k = 30
        nodes = [f"N_{i}" for i in range(k)]
        pairs = [(nodes[i], nodes[j]) for i in range(k) for j in range(i + 1, k)]
        weights = [0.85] * len(pairs)

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)
        assert len(clusters) == 1
        assert len(clusters[0]) == k


class TestDefensiveFallback:
    """Verify defensive fallback when heap validation fails on unassigned vertices."""

    def test_defensive_pivot_fallback_on_corrupted_heap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When heap entries fail validation due to simulated drift, fallback selects next(iter(U))."""
        real_heappop = heapq.heappop
        corrupted = False

        def mock_heappop(h: list[Any]) -> Any:
            nonlocal corrupted
            item = real_heappop(h)
            if not corrupted:
                corrupted = True
                # Corrupt the negative weighted degree in the heap tuple so validation fails
                return (item[0], item[1] + 999.0, item[2], item[3], item[4])
            return item

        monkeypatch.setattr(heapq, "heappop", mock_heappop)
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster([("A", "B")], [0.8])
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}
