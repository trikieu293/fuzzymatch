"""Connected-components and pluggable resolvers for entity clustering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import networkx as nx
import pandas as pd

ResolverFunc = Callable[[pd.MultiIndex, pd.Series | None], list[set[Any]]]


class ConnectedComponents:
    """Cluster entities by union-find over match edges (networkx).

    Optional `edge_weight_cutoff` drops edges below the threshold before closure
    to mitigate entity-collapse from intransitive matches.
    """

    def __init__(self, edge_weight_cutoff: float | None = None):
        self.cutoff = edge_weight_cutoff

    def resolve(
        self,
        pairs: pd.MultiIndex | Sequence[tuple[Any, Any]],
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]:
        G: nx.Graph[Any] = nx.Graph()
        for i, (u, v) in enumerate(pairs):
            if edge_weights is not None and self.cutoff is not None:
                w = edge_weights.iloc[i] if isinstance(edge_weights, pd.Series) else edge_weights[i]
                if w < self.cutoff:
                    continue
            G.add_edge(u, v)
        return [set(c) for c in nx.connected_components(G)]

    def cluster(
        self,
        pairs: Sequence[tuple[Any, Any]] | pd.MultiIndex,
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]:
        return self.resolve(pairs, edge_weights)


class PluggableResolver:
    """Wrap a user-provided cluster-resolution function.

    Use for custom policies like correlation clustering or MST-cut.
    """

    def __init__(self, func: ResolverFunc):
        self.func = func

    def resolve(
        self,
        pairs: pd.MultiIndex,
        edge_weights: pd.Series | None = None,
    ) -> list[set[Any]]:
        return self.func(pairs, edge_weights)

    def cluster(
        self,
        pairs: Sequence[tuple[Any, Any]] | pd.MultiIndex,
        edge_weights: pd.Series | Sequence[float] | None = None,
    ) -> list[set[Any]]:
        if not isinstance(pairs, pd.MultiIndex):
            pairs_idx = (
                pd.MultiIndex.from_tuples(pairs)
                if pairs
                else pd.MultiIndex.from_tuples([], names=["left", "right"])
            )
        else:
            pairs_idx = pairs
        weights_s = (
            pd.Series(list(edge_weights))
            if edge_weights is not None and not isinstance(edge_weights, pd.Series)
            else edge_weights
        )
        return self.func(pairs_idx, weights_s)
