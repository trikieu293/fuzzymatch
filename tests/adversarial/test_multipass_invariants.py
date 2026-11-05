"""Empirical Stress and Invariant Testing for MultiPassBlock and Index Resolution.

Empirically tests AutoMatcher RangeIndex disambiguation across:
1. Shifted RangeIndex (e.g. left [0..N], right [1..N+1], large offsets, partial offsets).
2. Reversed and non-contiguous indices (e.g. left [N..0], right [0..N]).
3. Subset, interleaved, and disjoint indices (e.g. odds vs evens, strict subsets).
4. Mixed integer vs string indices (e.g. integer 0 vs string '0', synthetic prefix 'L_0').
5. Explicit ID columns vs default RangeIndex combinations and fallback on NaN.
6. Non-standard index types (negative ints, large ints, floats, datetime indices).
7. Intransitive triads under shifted indices with StarClusterer clique verification.
8. Hypothesis randomized index permutation invariant harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzymatch.api.auto import AutoMatcher


class TestAdversarialShiftedRangeIndex:
    """Empirical verification of shifted RangeIndex handling in AutoMatcher."""

    def test_shifted_range_index_single_offset_clean(self) -> None:
        """Left [0, 1, 2] and Right [1, 2, 3] matching identical records.

        Distractor at Right[0] ('Omega Technologies Inc') is completely distinct.
        Without proper disambiguation, Right[1] collides with Left[1], and
        Right[2] collides with Left[2], causing StarClusterer to see open path
        0 -> 1 -> 2 -> 3 and sever valid edges.
        """
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]})  # RangeIndex 0, 1, 2
        df_r = pd.DataFrame(
            {"name": ["Omega Technologies Inc", "Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, 1, 2, 3],
        )

        am = AutoMatcher(threshold=0.7, cluster_method="star")
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3, f"Expected 3 matches, got {len(res.matches)}"
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [1, 2, 3]
        assert len(res.clusters) == 3

        # Invariant: Each match has both endpoints in the assigned cluster
        for _, row in res.matches.iterrows():
            cid = row["cluster_id"]
            cluster = res.clusters[cid]
            lid = row["left_id"]
            rid = row["right_id"]
            assert str(lid) in cluster or f"L_{lid}" in cluster
            assert str(rid) in cluster or f"R_{rid}" in cluster

    def test_shifted_range_index_with_competing_distractor_star_resolution(self) -> None:
        """Distractor at Right[0] ('Zeta Ltd') has high token-sort similarity to 'Beta LLC' (0.75).

        Verify that:
        1. RangeIndex collision does not corrupt matching.
        2. Beta LLC (L_1) matches Beta LLC (R_2) with 1.0 > 0.75.
        3. Zeta Ltd (R_0) is safely severed into an isolated cluster by StarClusterer.
        4. Matches dataframe only contains the true 3 matches.
        """
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]})
        df_r = pd.DataFrame(
            {"name": ["Zeta Ltd", "Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, 1, 2, 3],
        )

        am = AutoMatcher(threshold=0.7, cluster_method="star")
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [1, 2, 3]
        # Zeta Ltd forms an isolated cluster {'R_0'}
        assert any(cl == {"R_0"} for cl in res.clusters)
        assert len(res.clusters) == 4

    def test_shifted_range_index_large_offset(self) -> None:
        """Left [0, 1, 2] and Right [1000, 1001, 1002]. Completely disjoint index values."""
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]})
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[1000, 1001, 1002],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [1000, 1001, 1002]
        assert len(res.clusters) == 3

    def test_shifted_range_index_with_distractor_records(self) -> None:
        """Left has 4 records, Right has 7 records with distractors interleaved."""
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Co"]})
        df_r = pd.DataFrame(
            {
                "name": [
                    "Distractor Zero Ltd",
                    "Alpha Corp",
                    "Distractor Two Ltd",
                    "Beta LLC",
                    "Gamma Inc",
                    "Distractor Five Ltd",
                    "Delta Co",
                ]
            },
            index=[0, 1, 2, 3, 4, 5, 6],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 4
        assert list(res.matches["left_id"]) == [0, 1, 2, 3]
        assert list(res.matches["right_id"]) == [1, 3, 4, 6]
        assert len(res.clusters) == 4


class TestAdversarialReversedIndices:
    """Empirical verification with reversed and non-monotonic index configurations."""

    def test_reversed_left_forward_right(self) -> None:
        """Left has reversed index [2, 1, 0], Right has forward index [0, 1, 2]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[2, 1, 0],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, 1, 2],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        # Match Alpha (L:2, R:0), Beta (L:1, R:1), Gamma (L:0, R:2)
        matches = res.matches.set_index("left_id")
        assert matches.loc[2, "right_id"] == 0
        assert matches.loc[1, "right_id"] == 1
        assert matches.loc[0, "right_id"] == 2
        assert len(res.clusters) == 3

    def test_reversed_both_indices_with_shift(self) -> None:
        """Left index [5, 4, 3], Right index [3, 2, 1]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[5, 4, 3],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[3, 2, 1],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[5, "right_id"] == 3
        assert matches.loc[4, "right_id"] == 2
        assert matches.loc[3, "right_id"] == 1

    def test_non_contiguous_custom_integer_indices(self) -> None:
        """Arbitrary non-contiguous indices [10, 42, 999] vs [42, 999, 1000]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[10, 42, 999],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[42, 999, 1000],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[10, "right_id"] == 42
        assert matches.loc[42, "right_id"] == 999
        assert matches.loc[999, "right_id"] == 1000


class TestAdversarialSubsetAndInterleavedIndices:
    """Stress test when indices are subsets or interleaved partitions."""

    def test_strict_subset_indices(self) -> None:
        """Left index is [0, 2, 4], Right index is [0, 1, 2, 3, 4, 5]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, 2, 4],
        )
        df_r = pd.DataFrame(
            {
                "name": [
                    "Alpha Corp",
                    "Noise 1",
                    "Beta LLC",
                    "Noise 3",
                    "Gamma Inc",
                    "Noise 5",
                ]
            },
            index=[0, 1, 2, 3, 4, 5],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[0, "right_id"] == 0
        assert matches.loc[2, "right_id"] == 2
        assert matches.loc[4, "right_id"] == 4
        assert len(res.clusters) == 3

    def test_interleaved_odd_even_indices(self) -> None:
        """Left index is evens [2, 4, 6], Right index is odds [1, 3, 5]. Disjoint sets."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[2, 4, 6],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[1, 3, 5],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[2, "right_id"] == 1
        assert matches.loc[4, "right_id"] == 3
        assert matches.loc[6, "right_id"] == 5


class TestAdversarialMixedIntegerAndStringIndices:
    """Stress test mixed data types across DataFrame index values."""

    def test_integer_index_left_vs_string_index_right(self) -> None:
        """Left has integer index [0, 1, 2], Right has string index ['0', '1', '2'].

        Since str(0) == '0', string overlap detection must correctly trigger
        disambiguation so integer 0 and string '0' do not collide.
        """
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, 1, 2],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=["0", "1", "2"],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches
        assert list(matches["left_id"]) == [0, 1, 2]
        assert list(matches["right_id"]) == ["0", "1", "2"]
        # Contract preservation: left_id types are int, right_id types are str
        for _, row in matches.iterrows():
            assert isinstance(row["left_id"], int | np.integer)
            assert isinstance(row["right_id"], str)
        assert len(res.clusters) == 3

    def test_heterogeneous_mixed_types_within_indices(self) -> None:
        """Index contains mixed types: [0, 'item_1', 2] vs ['0', 'item_1', 99]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[0, "item_1", 2],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=["0", "item_1", 99],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[0, "right_id"] == "0"
        assert matches.loc["item_1", "right_id"] == "item_1"
        assert matches.loc[2, "right_id"] == 99

    def test_index_with_l_and_r_prefixes(self) -> None:
        """Left index already has 'L_0', 'L_1', Right has 'R_0', 'R_1'."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC"]},
            index=["L_0", "L_1"],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC"]},
            index=["R_0", "R_1"],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 2
        matches = res.matches.set_index("left_id")
        assert matches.loc["L_0", "right_id"] == "R_0"
        assert matches.loc["L_1", "right_id"] == "R_1"


class TestAdversarialExplicitIdVsRangeIndexCombinations:
    """Stress test combinations of explicit ID columns vs DataFrame RangeIndex."""

    def test_explicit_id_on_left_range_index_on_right_overlapping_values(self) -> None:
        """Left has explicit column 'id' = [0, 1, 2], Right has no ID column (RangeIndex 0, 1, 2)."""
        df_l = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            }
        )  # RangeIndex 0, 1, 2

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [0, 1, 2]
        assert len(res.clusters) == 3

    def test_range_index_on_left_explicit_id_on_right_overlapping_values(self) -> None:
        """Left has no ID column (RangeIndex 0, 1, 2), Right has explicit column 'id' = [0, 1, 2]."""
        df_l = pd.DataFrame(
            {
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            }
        )  # RangeIndex 0, 1, 2
        df_r = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            }
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [0, 1, 2]
        assert len(res.clusters) == 3

    def test_explicit_id_with_partial_nan_falls_back_to_index(self) -> None:
        """Left has ID column with some None values; should fall back to df.index cleanly."""
        df_l = pd.DataFrame(
            {
                "id": ["ID_A", None, "ID_C"],
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            },
            index=[10, 20, 30],
        )
        df_r = pd.DataFrame(
            {
                "id": [None, "ID_B", "ID_C"],
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            },
            index=[10, 20, 30],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert len(res.clusters) == 3


class TestAdversarialNonStandardIndices:
    """Stress test negative, large, float, and datetime indices."""

    def test_negative_integer_indices(self) -> None:
        """Indices are negative integers [-1, -2, -3] vs [-2, -3, -4]."""
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[-1, -2, -3],
        )
        df_r = pd.DataFrame(
            {"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]},
            index=[-2, -3, -4],
        )

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        assert matches.loc[-1, "right_id"] == -2
        assert matches.loc[-2, "right_id"] == -3
        assert matches.loc[-3, "right_id"] == -4

    def test_large_integer_indices(self) -> None:
        """64-bit large integer indices."""
        idx_l = [10**12 + i for i in range(3)]
        idx_r = [10**12 + i + 1 for i in range(3)]
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]}, index=idx_l)
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]}, index=idx_r)

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        matches = res.matches.set_index("left_id")
        for i in range(3):
            assert matches.loc[idx_l[i], "right_id"] == idx_r[i]

    def test_datetime_indices(self) -> None:
        """DatetimeIndex as DataFrame index."""
        idx_l = pd.date_range("2026-01-01", periods=3, freq="D")
        idx_r = pd.date_range("2026-01-02", periods=3, freq="D")
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]}, index=idx_l)
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]}, index=idx_r)

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == 3
        assert len(res.clusters) == 3


class TestAdversarialIntransitiveTriadWithShiftedIndex:
    """Stress test: Intransitive triad under shifted RangeIndex with StarClusterer."""

    def test_intransitive_triad_shifted_range_index_preserves_clique_protection(self) -> None:
        """
        Left: A (index 0), C (index 2)
        Right: B (index 1)
        A matches B (score 0.85), C matches B (score 0.85), but A !~ C (score 0.0).

        With shifted RangeIndex:
        df_l has index [0, 2], df_r has index [1].
        Because index values 0, 1, 2 overlap numerically across partitions,
        proper disambiguation must:
        1. Keep nodes separate (L_0, R_1, L_2 or 0, 1, 2).
        2. Prevent StarClusterer from placing both A and C in the same cluster.
        3. Maintain res.matches consistency: only the edge in the winning cluster is retained!
        """
        df_l = pd.DataFrame(
            {
                "name": ["Acme Logistics East", "Acme Logistics West"],
            },
            index=[0, 2],
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Logistics"],
            },
            index=[1],
        )

        am = AutoMatcher(threshold=0.60, cluster_method="star")
        res = am.fit_predict(df_l, df_r)

        # Star clusterer must split A and C
        for cl in res.clusters:
            assert not (0 in cl and 2 in cl)
            assert not ("0" in cl and "2" in cl)
            assert not ("L_0" in cl and "L_2" in cl)

        # Consistency: res.matches must only contain edges whose endpoints are in the same cluster
        for _, row in res.matches.iterrows():
            cid = row["cluster_id"]
            cluster_members = res.clusters[cid]
            lid = row["left_id"]
            rid = row["right_id"]
            assert str(lid) in cluster_members or f"L_{lid}" in cluster_members
            assert str(rid) in cluster_members or f"R_{rid}" in cluster_members


class TestHypothesisIndexPermutations:
    """Hypothesis randomized generative testing of RangeIndex disambiguation."""

    @settings(max_examples=25)
    @given(
        n=st.integers(min_value=2, max_value=10),
        offset=st.integers(min_value=-5, max_value=15),
        step=st.integers(min_value=1, max_value=3),
    )
    def test_randomized_range_index_offsets(self, n: int, offset: int, step: int) -> None:
        """Generates random index offsets and steps; verifies all true matches are found."""
        names = [f"Vendor {chr(65 + i)} International Corp" for i in range(n)]

        idx_l = list(range(0, n * step, step))
        idx_r = [i + offset for i in idx_l]

        df_l = pd.DataFrame({"name": names}, index=idx_l)
        df_r = pd.DataFrame({"name": names}, index=idx_r)

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)

        assert len(res.matches) == n
        assert len(res.clusters) == n
        matches = res.matches.set_index("left_id")
        for i in range(n):
            expected_r = idx_r[i]
            assert matches.loc[idx_l[i], "right_id"] == expected_r
