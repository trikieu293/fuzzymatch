"""Exact blocking on field equality."""

from __future__ import annotations

import pandas as pd

from .base import CandidateIndex

_MISSING_BLOCKING_KEYS = {"", "nan", "none"}


def _is_valid_blocking_key(series: pd.Series) -> pd.Series:
    return series.notna() & ~series.astype(str).str.strip().str.lower().isin(_MISSING_BLOCKING_KEYS)


class ExactBlock:
    """Block on exact equality of one or more fields (hash equi-join)."""

    def __init__(self, fields: list[str]):
        if not fields:
            raise ValueError("ExactBlock requires at least one field")
        self.fields = fields

    def block(self, df_left: pd.DataFrame, df_right: pd.DataFrame | None = None) -> CandidateIndex:
        right = df_right if df_right is not None else df_left
        # Store original index labels in explicit columns without relying on reset_index naming
        left_reset = df_left.assign(_orig_idx_l=list(df_left.index))
        left_mask = pd.Series(True, index=left_reset.index)
        for f in self.fields:
            left_mask &= _is_valid_blocking_key(left_reset[f])
        left_filtered = left_reset[left_mask]

        if df_right is None:
            right_filtered = left_filtered.rename(columns={"_orig_idx_l": "_orig_idx_r"})
        else:
            right_reset = right.assign(_orig_idx_r=list(right.index))
            right_mask = pd.Series(True, index=right_reset.index)
            for f in self.fields:
                right_mask &= _is_valid_blocking_key(right_reset[f])
            right_filtered = right_reset[right_mask]

        merged = left_filtered.merge(
            right_filtered,
            on=self.fields,
            suffixes=("_l", "_r"),
            how="inner",
        )
        pairs = list(zip(merged["_orig_idx_l"], merged["_orig_idx_r"], strict=True))
        if df_right is None:
            pairs = [
                (left_idx, right_idx) for left_idx, right_idx in pairs if left_idx != right_idx
            ]
        return CandidateIndex(pairs, df_left.index, right.index)
