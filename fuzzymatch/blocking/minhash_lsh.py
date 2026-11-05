"""MinHash LSH blocking for set/Jaccard similarity (wraps datasketch)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from datasketch import MinHash
from datasketch import MinHashLSH as DSL

from .base import CandidateIndex


class MinHashLSH:
    """LSH blocking over character n-gram sets, using datasketch MinHashLSH.

    Produces candidate pairs whose Jaccard similarity (over character shingles)
    is approximately >= threshold.
    """

    def __init__(
        self,
        fields: list[str],
        threshold: float = 0.7,
        num_perm: int = 128,
        shingle_size: int = 3,
    ):
        if not fields:
            raise ValueError("MinHashLSH requires at least one field")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        self.fields = fields
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def _shingles(self, text: str) -> list[str]:
        k = self.shingle_size
        if len(text) < k:
            return [text]
        return [text[i : i + k] for i in range(len(text) - k + 1)]

    def _minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm)
        for sh in self._shingles(text):
            mh.update(sh.encode("utf-8"))
        return mh

    def block(self, df_left: pd.DataFrame, df_right: pd.DataFrame | None = None) -> CandidateIndex:
        right = df_right if df_right is not None else df_left
        is_self_join = df_right is None
        lsh = DSL(threshold=self.threshold, num_perm=self.num_perm)
        left_indices = list(df_left.index)
        # Index left records using integer position string to support any index type
        for i, idx in enumerate(left_indices):
            text = " ".join(str(df_left.loc[idx, c]) for c in self.fields)
            mh = self._minhash(text)
            lsh.insert(str(i), mh)
        # Query right records
        pairs: list[tuple[Any, Any]] = []
        for j, ridx in enumerate(right.index):
            text = " ".join(str(right.loc[ridx, c]) for c in self.fields)
            mh = self._minhash(text)
            for hit in lsh.query(mh):
                i = int(hit)
                if is_self_join and i >= j:
                    # Avoid duplicate pairs in self-join
                    continue
                pairs.append((left_indices[i], ridx))
        return CandidateIndex(pairs, df_left.index, right.index)
