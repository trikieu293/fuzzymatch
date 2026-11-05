"""In-memory pandas backend (default for v1)."""

from __future__ import annotations

import pandas as pd


class PandasBackend:
    """In-memory DataFrame backend using pandas directly.

    Suitable for datasets up to ~1M records; switch to DuckDB for larger.
    """

    name = "pandas"

    def create_candidate_index(self, pairs: list[tuple[int, int]]) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(pairs, names=("left", "right"))
