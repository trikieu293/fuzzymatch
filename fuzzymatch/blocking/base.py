"""Base types for the blocking layer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import pandas as pd


class CandidateIndex:
    """Represents a set of candidate pairs (left_idx, right_idx) for downstream scoring."""

    def __init__(
        self,
        pairs: list[tuple[Any, Any]],
        left_ids: pd.Index,
        right_ids: pd.Index | None = None,
    ):
        self.pairs = pairs
        self.left_ids = left_ids
        self.right_ids = right_ids if right_ids is not None else left_ids

    def to_multiindex(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(self.pairs, names=("left", "right"))

    def __len__(self) -> int:
        return len(self.pairs)

    def __contains__(self, pair: object) -> bool:
        return pair in self.pairs

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        return iter(self.pairs)


class Blocker(Protocol):
    """Protocol for blocking implementations."""

    def block(
        self, df_left: pd.DataFrame, df_right: pd.DataFrame | None = None
    ) -> CandidateIndex: ...
