"""Tests to close remaining single-line coverage gaps.

Targets:
- backends/pandas_backend.py  line 21  (create_candidate_index)
- blocking/base.py            line 33  (CandidateIndex.__iter__)
- blocking/exact.py           line 17  (empty fields ValueError)
- normalize/pipeline.py       line 32  (skip missing column)
"""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.backends import PandasBackend
from fuzzymatch.blocking.base import CandidateIndex
from fuzzymatch.blocking.exact import ExactBlock
from fuzzymatch.normalize import Normalizer


class TestPandasBackendCandidateIndex:
    """Cover PandasBackend.create_candidate_index (line 21)."""

    def test_create_candidate_index_returns_multiindex(self):
        backend = PandasBackend()
        pairs = [(0, 1), (2, 3)]
        result = backend.create_candidate_index(pairs)
        assert isinstance(result, pd.MultiIndex)
        assert len(result) == 2
        assert result.names == ["left", "right"]
        assert (0, 1) in result
        assert (2, 3) in result

    def test_create_candidate_index_empty(self):
        backend = PandasBackend()
        result = backend.create_candidate_index([])
        assert isinstance(result, pd.MultiIndex)
        assert len(result) == 0


class TestCandidateIndexIter:
    """Cover CandidateIndex.__iter__ (line 33)."""

    def test_iter_yields_pairs(self):
        idx = CandidateIndex(
            pairs=[(0, 1), (2, 3), (4, 5)],
            left_ids=pd.Index([0, 1, 2, 3, 4, 5]),
        )
        collected = list(idx)
        assert collected == [(0, 1), (2, 3), (4, 5)]

    def test_iter_empty(self):
        idx = CandidateIndex(pairs=[], left_ids=pd.Index([]))
        assert list(idx) == []

    def test_for_loop_iteration(self):
        """Ensure __iter__ works in a for-loop context."""
        idx = CandidateIndex(
            pairs=[(10, 20)],
            left_ids=pd.Index([10, 20]),
        )
        pairs = []
        for pair in idx:
            pairs.extend([pair])
        assert pairs == [(10, 20)]


class TestExactBlockEmptyFields:
    """Cover ExactBlock.__init__ ValueError (line 17)."""

    def test_empty_fields_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            ExactBlock([])

    def test_none_like_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one field"):
            ExactBlock(fields=[])


class TestNormalizerMissingColumn:
    """Cover Normalizer.transform skip on missing column (line 32)."""

    def test_missing_column_is_skipped(self):
        norm = Normalizer([])
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        # "address" doesn't exist in df -- should be silently skipped
        result = norm.transform(df, ["name", "address"])
        # Original column untouched
        assert list(result["name"]) == ["Alice", "Bob"]
        # Missing column not added
        assert "address" not in result.columns

    def test_all_columns_missing(self):
        norm = Normalizer([])
        df = pd.DataFrame({"x": [1, 2]})
        result = norm.transform(df, ["y", "z"])
        assert list(result.columns) == ["x"]
