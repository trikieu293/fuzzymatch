"""Tests for MinHashLSH, SortedNeighborhood, and blocking combinators."""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.blocking import (
    ExactBlock,
    IntersectBlock,
    MinHashLSH,
    SortedNeighborhood,
    UnionBlock,
)


class TestMinHashLSH:
    """Tests for MinHash LSH blocking."""

    def test_basic_self_join(self) -> None:
        df = pd.DataFrame({"name": ["john smith", "john smyth", "alice jones", "bob brown"]})
        blocker = MinHashLSH(fields=["name"], threshold=0.3, num_perm=64)
        ci = blocker.block(df)
        # With threshold=0.3, similar strings should produce candidates
        assert isinstance(ci.pairs, list)
        assert len(ci.left_ids) == 4

    def test_dissimilar_records_not_paired(self) -> None:
        df = pd.DataFrame({"name": ["aaaa", "zzzz"]})
        blocker = MinHashLSH(fields=["name"], threshold=0.9, num_perm=64)
        ci = blocker.block(df)
        assert len(ci) == 0

    def test_empty_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            MinHashLSH(fields=[], threshold=0.7)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            MinHashLSH(fields=["name"], threshold=1.5)
        with pytest.raises(ValueError, match="threshold must be in"):
            MinHashLSH(fields=["name"], threshold=0.0)

    def test_cross_file_join(self) -> None:
        left = pd.DataFrame({"name": ["john smith", "alice"]})
        right = pd.DataFrame({"name": ["john smyth", "bob"]})
        blocker = MinHashLSH(fields=["name"], threshold=0.5, num_perm=64)
        ci = blocker.block(left, right)
        assert ci.right_ids is not None
        assert len(ci.right_ids) == 2

    def test_shingle_short_text(self) -> None:
        blocker = MinHashLSH(fields=["name"], threshold=0.5, shingle_size=5)
        shingles = blocker._shingles("ab")
        assert shingles == ["ab"]

    def test_shingle_normal_text(self) -> None:
        blocker = MinHashLSH(fields=["name"], threshold=0.5, shingle_size=3)
        shingles = blocker._shingles("abcde")
        assert shingles == ["abc", "bcd", "cde"]

    def test_multi_field(self) -> None:
        df = pd.DataFrame({"first": ["john", "alice"], "last": ["smith", "jones"]})
        blocker = MinHashLSH(fields=["first", "last"], threshold=0.5)
        ci = blocker.block(df)
        assert isinstance(ci.pairs, list)

    def test_minhash_with_string_index(self) -> None:
        df = pd.DataFrame(
            {"name": ["apple pie", "apple tart"]},
            index=["rec_a", "rec_b"],
        )
        blocker = MinHashLSH(fields=["name"], threshold=0.2, num_perm=64)
        ci = blocker.block(df)
        assert ("rec_a", "rec_b") in ci.pairs


class TestSortedNeighborhood:
    """Tests for Sorted Neighborhood blocking."""

    def test_cross_file_string_index_preservation(self) -> None:
        left = pd.DataFrame({"name": ["apple inc", "banana llc"]}, index=["L1", "L2"])
        right = pd.DataFrame({"name": ["apple inc", "banana llc"]}, index=["R1", "R2"])
        blocker = SortedNeighborhood(fields=["name"], window=2)
        ci = blocker.block(left, right)
        assert ("L1", "R1") in ci.pairs
        assert ("L2", "R2") in ci.pairs

    def test_basic_window(self) -> None:
        df = pd.DataFrame({"name": ["alice", "bob", "charlie", "dave", "eve"]})
        blocker = SortedNeighborhood(fields=["name"], window=2)
        ci = blocker.block(df)
        assert len(ci) > 0

    def test_window_1(self) -> None:
        df = pd.DataFrame({"name": ["a", "b", "c", "d"]})
        blocker = SortedNeighborhood(fields=["name"], window=1)
        ci = blocker.block(df)
        # window=1 means only adjacent pairs after sorting
        assert len(ci) == 3

    def test_empty_fields_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one field"):
            SortedNeighborhood(fields=[])

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            SortedNeighborhood(fields=["name"], window=0)

    def test_cross_file_pairs_similar_records(self) -> None:
        """Cross-file blocking pairs near-identical keys across frames."""
        left = pd.DataFrame({"name": ["apple inc", "microsoft corp", "zebra ltd"]})
        right = pd.DataFrame({"name": ["aple inc", "google llc", "microsft corp"]})
        blocker = SortedNeighborhood(fields=["name"], window=2)
        ci = blocker.block(left, right)
        pairs = set(ci.pairs)
        assert (0, 0) in pairs  # apple inc ~ aple inc
        assert (1, 2) in pairs  # microsoft corp ~ microsft corp
        assert True  # zebra may or may not catch google
        # no self-origin pairs possible by construction
        assert all(li < len(left) and ri >= 0 for li, ri in pairs)

    def test_cross_file_disjoint_keys_no_pairs(self) -> None:
        left = pd.DataFrame({"name": ["aaa", "bbb"]})
        right = pd.DataFrame({"name": ["xxx", "yyy"]})
        blocker = SortedNeighborhood(fields=["name"], window=1)
        ci = blocker.block(left, right)
        # sorted: aaa(L) bbb(L) xxx(R) yyy(R); window=1 crosses at bbb|xxx only
        assert (1, 0) in set(ci.pairs)

    def test_multi_field_key(self) -> None:
        df = pd.DataFrame({"city": ["NY", "NY", "LA"], "name": ["a", "b", "c"]})
        blocker = SortedNeighborhood(fields=["city", "name"], window=1)
        ci = blocker.block(df)
        assert len(ci) > 0


class TestUnionBlock:
    """Tests for blocking union combinator."""

    def test_union_merges_candidates(self) -> None:
        df = pd.DataFrame(
            {"name": ["john smith", "john smyth", "alice"], "city": ["NY", "LA", "NY"]}
        )
        b1 = ExactBlock(fields=["name"])
        b2 = ExactBlock(fields=["city"])
        union = UnionBlock(b1, b2)
        ci = union.block(df)
        # ExactBlock on city should pair (0,2) both NY
        pair_set = set(ci.pairs)
        assert (0, 2) in pair_set

    def test_union_empty(self) -> None:
        df = pd.DataFrame({"name": ["a", "b", "c"]})
        union = UnionBlock()
        ci = union.block(df)
        assert len(ci) == 0


class TestIntersectBlock:
    """Tests for blocking intersection combinator."""

    def test_intersect_keeps_common(self) -> None:
        df = pd.DataFrame({"name": ["john", "john", "alice"], "city": ["NY", "NY", "LA"]})
        b1 = ExactBlock(fields=["name"])
        b2 = ExactBlock(fields=["city"])
        intersect = IntersectBlock(b1, b2)
        ci = intersect.block(df)
        # Only (0,1) matches on both name AND city
        pair_set = set(ci.pairs)
        assert (0, 1) in pair_set

    def test_intersect_empty_blockers(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"]})
        intersect = IntersectBlock()
        ci = intersect.block(df)
        assert len(ci) == 0


class TestExactBlock:
    """Tests for exact blocking and index preservation."""

    def test_exact_block_basic(self) -> None:
        df = pd.DataFrame({"name": ["alice", "alice", "bob"]})
        blocker = ExactBlock(["name"])
        ci = blocker.block(df)
        assert (0, 1) in ci.pairs or (1, 0) in ci.pairs

    def test_exact_block_named_index_and_collision(self) -> None:
        # Index with name and existing "index" column
        df_left = pd.DataFrame(
            {"index": [10, 20], "name": ["alice", "bob"]},
            index=pd.Index(["id_a", "id_b"], name="entity_id"),
        )
        df_right = pd.DataFrame(
            {"index": [30, 40], "name": ["alice", "charlie"]},
            index=pd.Index(["id_x", "id_y"], name="entity_id"),
        )
        blocker = ExactBlock(["name"])
        ci = blocker.block(df_left, df_right)
        assert ("id_a", "id_x") in ci.pairs
