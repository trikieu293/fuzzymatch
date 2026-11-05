"""Sorted-Neighborhood blocking (Hernandez & Stolfo, 1995)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import CandidateIndex

_MISSING_BLOCKING_KEYS = {"", "nan", "none"}


def _is_valid_blocking_record(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    valid = pd.Series(True, index=df.index)
    for f in fields:
        s = df[f]
        is_missing = s.isna() | s.astype(str).str.strip().str.lower().isin(_MISSING_BLOCKING_KEYS)
        valid &= ~is_missing
    return valid


class SortedNeighborhood:
    """Sliding-window blocking over a sorted blocking key.

    Sorts records by a composite key formed from `fields`, then for each record,
    compares it with up to `window` records on either side. Tolerant to small
    errors in the blocking key but misses global reorderings.

    Supports both self-join (dedupe) and cross-file (link) blocking. Cross-file
    works by merging both tables' keys into one sorted sequence with origin
    tags and pairing only left-origin/right-origin records within the window.
    """

    def __init__(self, fields: list[str], window: int = 3):
        if not fields:
            raise ValueError("SortedNeighborhood requires at least one field")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.fields = fields
        self.window = window

    def _key_series(self, df: pd.DataFrame) -> pd.Series:
        """Composite blocking key for one frame."""
        return (
            df[self.fields]
            .map(lambda x: "" if x is None or pd.isna(x) else str(x))
            .agg("|".join, axis=1)
        )

    def block(self, df_left: pd.DataFrame, df_right: pd.DataFrame | None = None) -> CandidateIndex:
        if df_right is not None:
            return self._block_cross(df_left, df_right)
        valid = _is_valid_blocking_record(df_left, self.fields)
        key = self._key_series(df_left)[valid]
        ids = key.sort_values().index.tolist()
        pairs: list[tuple[Any, Any]] = []
        n = len(ids)
        for i in range(n):
            for offset in range(1, self.window + 1):
                j = i + offset
                if j < n:
                    pairs.append((ids[i], ids[j]))
        return CandidateIndex(pairs, df_left.index, df_left.index)

    def _block_cross(self, df_left: pd.DataFrame, df_right: pd.DataFrame) -> CandidateIndex:
        """Cross-file variant: one merged sort, pairs only across origins."""
        valid_l = _is_valid_blocking_record(df_left, self.fields)
        valid_r = _is_valid_blocking_record(df_right, self.fields)
        keys_l = self._key_series(df_left)
        keys_r = self._key_series(df_right)
        valid_l_arr = valid_l.to_numpy()
        valid_r_arr = valid_r.to_numpy()
        records: list[tuple[str, int, Any]] = [
            (str(k), 0, _idx) for i, (_idx, k) in enumerate(keys_l.items()) if bool(valid_l_arr[i])
        ]
        records.extend(
            (str(k), 1, _idx) for j, (_idx, k) in enumerate(keys_r.items()) if bool(valid_r_arr[j])
        )
        records.sort(key=lambda t: t[0])

        pair_set: set[tuple[Any, Any]] = set()
        total = len(records)
        for a in range(total):
            _ka, oa, ia = records[a]
            for offset in range(1, self.window + 1):
                b = a + offset
                if b >= total:
                    break
                _kb, ob, ib = records[b]
                if oa == ob:
                    continue
                if oa == 0:
                    pair_set.add((ia, ib))
                else:
                    pair_set.add((ib, ia))

        try:
            ordered = sorted(pair_set)
        except TypeError:  # pragma: no cover - unorderable index fallback
            ordered = sorted(pair_set, key=lambda p: (str(p[0]), str(p[1])))
        return CandidateIndex(ordered, df_left.index, df_right.index)
