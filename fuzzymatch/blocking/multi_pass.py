"""Multi-pass candidate generation with candidate budget capping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .base import CandidateIndex


@runtime_checkable
class BaseBlocker(Protocol):
    """Protocol for candidate blocking implementations."""

    def block(
        self, df_left: pd.DataFrame, df_right: pd.DataFrame | None = None
    ) -> CandidateIndex: ...


class MultiPassBlock(BaseBlocker):
    """Executes ordered blocking passes with a candidate budget cap.

    Accumulates novel candidate pairs sequentially from high-priority passes
    (e.g., exact ID -> normalized name -> sorted neighborhood -> shingle/LSH).
    Halts candidate accumulation when the number of accumulated candidate pairs
    reaches `budget_cap`, preventing O(N^2) quadratic candidate explosion while
    recovering records with missing primary keys and typographical errors.
    """

    def __init__(
        self,
        passes: Sequence[BaseBlocker],
        budget_cap: int = 50_000,
    ) -> None:
        if not passes:
            raise ValueError("MultiPassBlock requires at least one blocker pass")
        if budget_cap <= 0:
            raise ValueError("budget_cap must be a positive integer")
        self.passes: list[BaseBlocker] = list(passes)
        self.budget_cap: int = int(budget_cap)

    def block(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame | None = None,
    ) -> CandidateIndex:
        right = df_right if df_right is not None else df_left
        all_pairs: list[tuple[Any, Any]] = []
        seen_pairs: set[tuple[Any, Any]] = set()

        for b in self.passes:
            if len(all_pairs) >= self.budget_cap:
                break
            ci = b.block(df_left, df_right)
            for pair in ci.pairs:
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_pairs.append(pair)
                    if len(all_pairs) >= self.budget_cap:
                        break

        return CandidateIndex(all_pairs, df_left.index, right.index)
