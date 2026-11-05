"""Unit tests for StarClusterer (Pivot clustering) and ConnectedComponents string ID fix."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.clustering import (
    BaseClusterer,
    ConnectedComponents,
    PluggableResolver,
    StarClusterer,
)


class TestStarClustererProtocolAndValidation:
    """Verify protocol compliance and parameter validation."""

    def test_implements_base_clusterer_protocol(self) -> None:
        clusterer = StarClusterer(cutoff=0.5, verify_clique=True)
        assert isinstance(clusterer, BaseClusterer)
        assert hasattr(clusterer, "cluster")
        assert hasattr(clusterer, "resolve")
        assert clusterer.cutoff == 0.5
        assert clusterer.verify_clique is True

    def test_connected_components_implements_base_clusterer_protocol(self) -> None:
        cc = ConnectedComponents()
        assert isinstance(cc, BaseClusterer)
        assert hasattr(cc, "cluster")
        assert hasattr(cc, "resolve")

    def test_invalid_cutoff_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StarClusterer(cutoff=-0.1)

    def test_mismatched_weights_length_raises_value_error(self) -> None:
        clusterer = StarClusterer()
        pairs = [("A", "B"), ("B", "C")]
        weights = [0.9]  # length 1 vs 2 pairs
        with pytest.raises(ValueError, match="Length of edge_weights"):
            clusterer.cluster(pairs, edge_weights=weights)


class TestStarClustererAntiChaining:
    """Core mathematical test: eliminate false transitive chaining on intransitive triads."""

    def test_intransitive_triad_elimination(self) -> None:
        """
        Benchmark triad:
          d(A, B) = 0.90 >= 0.5 (Match)
          d(B, C) = 0.85 >= 0.5 (Match)
          d(A, C) = 0.20 < 0.5 (Non-match, or edge absent)

        Under ConnectedComponents (transitive closure):
          {A, B, C} are collapsed into a single cluster (false transitive chaining).

        Under StarClusterer(verify_clique=True):
          Pivot B forms cluster with A (weight 0.90).
          C cannot join {B, A} because w(A, C) < 0.5.
          C forms singleton cluster {C}.
          Returns [{A, B}, {C}] (or [{B, C}, {A}]).
          Crucially, {A, C} are NEVER in the same cluster!
        """
        pairs = [("A", "B"), ("B", "C"), ("A", "C")]
        weights = [0.90, 0.85, 0.20]

        # 1. Confirm ConnectedComponents fails (collapses A and C)
        cc = ConnectedComponents(edge_weight_cutoff=0.5)
        cc_clusters = cc.resolve(pd.MultiIndex.from_tuples(pairs), pd.Series(weights))
        assert len(cc_clusters) == 1
        assert cc_clusters[0] == {"A", "B", "C"}  # False transitive chaining!

        # 2. Confirm StarClusterer eliminates false chaining
        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        sc_clusters = sc.cluster(pairs, edge_weights=weights)

        assert len(sc_clusters) == 2
        # Verify that no cluster contains both A and C
        for cluster in sc_clusters:
            assert not ({"A", "C"}.issubset(cluster))

        # Expected partition is either {B, A} and {C}, or {B, C} and {A}
        cluster_sizes = sorted(len(c) for c in sc_clusters)
        assert cluster_sizes == [1, 2]

    def test_intransitive_triad_with_missing_edge(self) -> None:
        """Intransitive triad where edge (A, C) is not generated at all."""
        pairs = [("A", "B"), ("B", "C")]
        weights = [0.90, 0.85]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        assert len(clusters) == 2
        for cluster in clusters:
            assert not ({"A", "C"}.issubset(cluster))

    def test_verify_clique_false_allows_star(self) -> None:
        """When verify_clique is False, all valid satellites of the pivot are joined."""
        pairs = [("A", "B"), ("B", "C")]
        weights = [0.90, 0.85]

        sc_pure_star = StarClusterer(cutoff=0.5, verify_clique=False)
        clusters = sc_pure_star.cluster(pairs, edge_weights=weights)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C"}


class TestStarClustererTopologies:
    """Test full cliques, disconnected components, chains, and edge cases."""

    def test_full_clique(self) -> None:
        """In a true clique, all pairwise similarities >= cutoff; forms a single cluster."""
        pairs = [("A", "B"), ("B", "C"), ("A", "C")]
        weights = [0.90, 0.85, 0.95]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C"}

    def test_disconnected_graphs(self) -> None:
        """Disconnected components resolve into independent clusters."""
        pairs = [("A", "B"), ("C", "D"), ("E", "F")]
        weights = [0.9, 0.8, 0.85]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        assert len(clusters) == 3
        expected = [{"A", "B"}, {"C", "D"}, {"E", "F"}]
        for exp in expected:
            assert exp in clusters

    def test_4_node_chain_decomposition(self) -> None:
        """A-B-C-D chain where (A,C), (A,D), (B,D) are absent."""
        pairs = [("A", "B"), ("B", "C"), ("C", "D")]
        weights = [0.9, 0.9, 0.9]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, edge_weights=weights)

        # Must not form a 4-node cluster
        for c in clusters:
            assert len(c) <= 2
        # All 4 nodes covered
        all_nodes = set.union(*clusters)
        assert all_nodes == {"A", "B", "C", "D"}

    def test_empty_pairs(self) -> None:
        """Empty input returns empty cluster list."""
        sc = StarClusterer()
        assert sc.cluster([]) == []

    def test_single_pair(self) -> None:
        """Single pair above cutoff produces 1 cluster with both nodes."""
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster([("X", "Y")], [0.8])
        assert len(clusters) == 1
        assert clusters[0] == {"X", "Y"}

    def test_single_node_self_pair(self) -> None:
        """Single node self-pair produces singleton cluster."""
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster([("Node1", "Node1")], [1.0])
        assert len(clusters) == 1
        assert clusters[0] == {"Node1"}

    def test_all_edges_below_cutoff(self) -> None:
        """All edges below cutoff produces empty cluster list."""
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster([("A", "B"), ("C", "D")], [0.2, 0.3])
        assert clusters == []

    def test_duplicate_pairs_takes_max_weight(self) -> None:
        """Duplicate edges take maximum similarity weight."""
        pairs = [("A", "B"), ("A", "B")]
        weights = [0.3, 0.9]
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster(pairs, weights)
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}


class TestNodeIdentifierTypes:
    """Verify arbitrary string, integer, UUID, and mixed hashable node IDs work seamlessly."""

    def test_string_node_identifiers(self) -> None:
        pairs = [("VEND_1001", "VEND_1002"), ("VEND_1003", "VEND_1004")]
        sc = StarClusterer(cutoff=0.5)
        clusters = sc.cluster(pairs)
        assert len(clusters) == 2
        assert {"VEND_1001", "VEND_1002"} in clusters

    def test_uuid_node_identifiers(self) -> None:
        u1, u2, u3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        pairs = [(u1, u2), (u2, u3)]
        weights = [0.9, 0.85]

        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)
        assert len(clusters) == 2
        all_nodes = set.union(*clusters)
        assert all_nodes == {u1, u2, u3}

    def test_integer_node_identifiers(self) -> None:
        pairs = [(10, 20), (20, 30)]
        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, [0.9, 0.85])
        assert len(clusters) == 2

    def test_input_formats(self) -> None:
        """Test pd.MultiIndex, pd.Series, list, and None edge weights."""
        pairs_mi = pd.MultiIndex.from_tuples([("A", "B"), ("B", "C")])
        weights_s = pd.Series([0.9, 0.8])
        weights_arr = np.array([0.9, 0.8])

        sc = StarClusterer(cutoff=0.5)

        # pd.MultiIndex and pd.Series
        c1 = sc.resolve(pairs_mi, weights_s)
        assert len(c1) >= 1

        # numpy array weights
        c2 = sc.cluster(pairs_mi, weights_arr)
        assert len(c2) >= 1

        # Full triangle with default weights (1.0) forms 1 cluster
        c3 = sc.cluster([("A", "B"), ("B", "C"), ("A", "C")])
        assert len(c3) == 1
        assert c3[0] == {"A", "B", "C"}


class TestConnectedComponentsStringFix:
    """Verify ConnectedComponents string identifier fix."""

    def test_string_ids_in_connected_components(self) -> None:
        """Prior to fix, int(u) raised ValueError on string IDs."""
        pairs = pd.MultiIndex.from_tuples([("VEND_A", "VEND_B"), ("VEND_B", "VEND_C")])
        cc = ConnectedComponents()
        clusters = cc.resolve(pairs)
        assert len(clusters) == 1
        assert clusters[0] == {"VEND_A", "VEND_B", "VEND_C"}

    def test_connected_components_cluster_method(self) -> None:
        pairs = [("A", "B"), ("B", "C")]
        cc = ConnectedComponents()
        clusters = cc.cluster(pairs)
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B", "C"}

    def test_connected_components_with_list_weights(self) -> None:
        pairs = [("A", "B"), ("B", "C")]
        cc = ConnectedComponents(edge_weight_cutoff=0.5)
        clusters = cc.resolve(pairs, [0.9, 0.2])
        assert len(clusters) == 1
        assert clusters[0] == {"A", "B"}

    def test_pluggable_resolver_cluster_method(self) -> None:
        def my_resolver(pairs: pd.MultiIndex, weights: pd.Series | None) -> list[set[str]]:
            return [{"custom_cluster"}]

        resolver = PluggableResolver(my_resolver)
        # Test with list pairs and list weights
        c1 = resolver.cluster([("A", "B")], edge_weights=[0.8])
        assert c1 == [{"custom_cluster"}]

        # Test with pd.MultiIndex and pd.Series weights
        mi = pd.MultiIndex.from_tuples([("A", "B")])
        s = pd.Series([0.8])
        c2 = resolver.cluster(mi, edge_weights=s)
        assert c2 == [{"custom_cluster"}]

        # Test with empty list
        c3 = resolver.cluster([])
        assert c3 == [{"custom_cluster"}]
