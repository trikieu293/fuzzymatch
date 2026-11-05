"""Star (Pivot) clustering algorithm for non-transitive entity resolution."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaseClusterer(Protocol):
    """Protocol for entity clustering implementations."""

    def cluster(
        self,
        pairs: Sequence[tuple[Any, Any]],
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]: ...

    def resolve(
        self,
        pairs: pd.MultiIndex | Sequence[tuple[Any, Any]],
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]: ...


class StarClusterer(BaseClusterer):
    """Star / Pivot clustering with intra-cluster clique verification.

    Groups nodes around high-degree / max-weight pivots.
    When verify_clique is True, a candidate satellite v is admitted to pivot p's
    cluster only if w(u, v) >= cutoff for all existing members u in the cluster.
    This eliminates false transitive chaining on intransitive triads
    (where w(A, B) >= tau and w(B, C) >= tau, but w(A, C) < tau).

    Time complexity is sub-quadratic: O(N log N + M), where N is the number
    of entities and M is the number of candidate pairs.
    """

    def __init__(
        self,
        cutoff: float = 0.5,
        verify_clique: bool = True,
    ) -> None:
        if cutoff < 0.0:
            raise ValueError("cutoff must be non-negative")
        self.cutoff: float = float(cutoff)
        self.verify_clique: bool = bool(verify_clique)

    def cluster(
        self,
        pairs: Sequence[tuple[Any, Any]] | pd.MultiIndex,
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]:
        if isinstance(pairs, pd.MultiIndex):
            pair_list: list[tuple[Any, Any]] = list(pairs)
        else:
            pair_list = list(pairs)

        if not pair_list:
            return []

        weights: list[float]
        if edge_weights is None:
            weights = [1.0] * len(pair_list)
        elif isinstance(edge_weights, pd.Series):
            weights = [float(x) for x in edge_weights.to_list()]
        else:
            weights = [float(x) for x in edge_weights]

        if len(weights) != len(pair_list):
            raise ValueError(
                f"Length of edge_weights ({len(weights)}) does not match length of pairs ({len(pair_list)})"
            )

        # Build undirected adjacency map and active vertex set
        adj: dict[Any, dict[Any, float]] = {}
        nodes: set[Any] = set()

        for (u, v), w in zip(pair_list, weights, strict=True):
            if w < self.cutoff:
                continue
            if u == v:
                nodes.add(u)
                if u not in adj:
                    adj[u] = {}
                continue
            nodes.add(u)
            nodes.add(v)
            if u not in adj:
                adj[u] = {}
            if v not in adj:
                adj[v] = {}
            if w > adj[u].get(v, -float("inf")):
                adj[u][v] = w
                adj[v][u] = w

        if not nodes:
            return []

        # Unassigned vertices
        U: set[Any] = set(nodes)

        # Track degree and weighted degree within U
        deg: dict[Any, int] = {u: len(adj.get(u, {})) for u in U}
        w_deg: dict[Any, float] = {u: sum(adj.get(u, {}).values()) for u in U}

        # Priority queue for selecting pivots: (-deg, -w_deg, str(u), counter, u)
        heap: list[tuple[int, float, str, int, Any]] = []
        counter = 0
        for u in U:
            heapq.heappush(heap, (-deg[u], -w_deg[u], str(u), counter, u))
            counter += 1

        clusters: list[set[Any]] = []

        while U:
            # Find next valid pivot from heap
            pivot: Any = None
            while heap:
                neg_d, neg_wd, _, _, cand = heapq.heappop(heap)
                if cand in U and -neg_d == deg[cand] and abs(-neg_wd - w_deg[cand]) < 1e-9:
                    pivot = cand
                    break

            if pivot is None and U:
                pivot = next(iter(U))  # pragma: no cover - defensive fallback for numerical drift

            # Candidate satellites: neighbors of pivot that are still in U
            pivot_neighbors = [v for v in adj.get(pivot, {}) if v in U]
            # Sort satellites by edge weight to pivot descending, tie-break by str(v)
            pivot_neighbors.sort(key=lambda v: (adj[pivot][v], str(v)), reverse=True)

            current_cluster: set[Any] = {pivot}

            for satellite in pivot_neighbors:
                if self.verify_clique:
                    # Satellite must satisfy w(m, satellite) >= cutoff for all existing members m
                    can_join = True
                    for member in current_cluster:
                        if adj.get(member, {}).get(satellite, 0.0) < self.cutoff:
                            can_join = False
                            break
                    if can_join:
                        current_cluster.add(satellite)
                else:
                    current_cluster.add(satellite)

            # Form cluster and remove its members from U
            clusters.append(current_cluster)
            U.difference_update(current_cluster)

            # Update degrees and weighted degrees of remaining nodes in U
            affected_nodes: set[Any] = set()
            for removed_node in current_cluster:
                for neighbor in adj.get(removed_node, {}):
                    if neighbor in U:
                        deg[neighbor] -= 1
                        w_deg[neighbor] -= adj[removed_node][neighbor]
                        affected_nodes.add(neighbor)

            # Push updated stats for affected nodes to heap
            for aff in affected_nodes:
                heapq.heappush(heap, (-deg[aff], -w_deg[aff], str(aff), counter, aff))
                counter += 1

        return clusters

    def resolve(
        self,
        pairs: pd.MultiIndex | Sequence[tuple[Any, Any]],
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]:
        """Resolver method compatible with PluggableResolver and Matcher."""
        return self.cluster(pairs, edge_weights)
