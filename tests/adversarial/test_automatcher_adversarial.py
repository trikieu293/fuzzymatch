"""Adversarial Stress Testing for AutoMatcher and MatchResults.

Adversarially tests AutoMatcher and MatchResults across:
1. Zero-row, 1x1, 1x1000, and zero-column inputs.
2. Disjoint column names, non-string/integer column names, and duplicate headers.
3. Candidate pair accumulation under budget caps (B_max <= 50,000 and tight caps).
4. Intransitive triads (A ~ B, B ~ C, A !~ C) under StarClusterer vs ConnectedComponents
   and Cluster-Match consistency invariants.
5. Bipartite indexing with identical RangeIndex ([0, 1, 2...]), duplicate index, missing IDs.
6. Non-finite / NaN floats, corrupted dates, unicode, emojis, control chars.
7. MatchResults methods: .to_dataframe(), .to_csv(), .to_excel(), .get_explanations().
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzymatch.api.auto import (
    AutoMatcher,
    MatchResults,
)
from fuzzymatch.clustering.connected_components import ConnectedComponents
from fuzzymatch.clustering.star_cluster import StarClusterer


class TestAdversarialZeroAndSingleRowInputs:
    """Stress test boundary shapes: (0, 0), (0, N), (N, 0), (1, 1), (1, 1000), (N, 0)."""

    def test_both_completely_empty(self) -> None:
        am = AutoMatcher()
        res = am.fit_predict(pd.DataFrame(), pd.DataFrame())
        assert isinstance(res, MatchResults)
        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["left_rows"] == 0
        assert res.summary_stats["right_rows"] == 0
        assert res.summary_stats["candidate_pairs"] == 0
        assert res.summary_stats["matched_pairs"] == 0
        assert isinstance(res.to_dataframe(), pd.DataFrame)
        assert res.to_dataframe().empty

    def test_zero_rows_with_columns(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame(columns=["id", "vendor_name", "amount", "date"])
        df_r = pd.DataFrame(columns=["id", "vendor_name", "amount", "date"])
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["left_rows"] == 0
        assert res.summary_stats["right_rows"] == 0

    def test_asymmetric_zero_rows(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame({"id": [1, 2], "name": ["Alpha Corp", "Beta LLC"]})
        df_r = pd.DataFrame(columns=["id", "name"])
        res1 = am.fit_predict(df_l, df_r)
        assert len(res1) == 0
        assert res1.summary_stats["right_rows"] == 0

        res2 = am.fit_predict(df_r, df_l)
        assert len(res2) == 0
        assert res2.summary_stats["left_rows"] == 0

    def test_single_row_1x1_identical(self) -> None:
        am = AutoMatcher(threshold=0.7)
        df_l = pd.DataFrame({"name": ["Global Logistics Inc"]})
        df_r = pd.DataFrame({"name": ["Global Logistics Inc"]})
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert len(res.clusters) == 1
        assert res.matches.iloc[0]["score"] == 1.0
        assert res.matches.iloc[0]["status"] == "match"

    def test_single_row_1x1_disjoint_values(self) -> None:
        am = AutoMatcher(threshold=0.8)
        df_l = pd.DataFrame({"name": ["Alpha"]})
        df_r = pd.DataFrame({"name": ["Omega"]})
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 0
        assert len(res.clusters) == 0

    def test_wide_frame_1x1000_columns(self) -> None:
        """A single row with 1000 columns."""
        cols = [f"col_{i}" for i in range(1000)]
        data_l = {c: [f"val_{i}"] for i, c in enumerate(cols)}
        data_r = {c: [f"val_{i}"] for i, c in enumerate(cols)}
        data_l["col_500"] = ["Apex Solutions Corp"]
        data_r["col_500"] = ["Apex Solutions LLC"]
        df_l = pd.DataFrame(data_l)
        df_r = pd.DataFrame(data_r)

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert res.matches.iloc[0]["score"] >= 0.7

    def test_zero_columns_non_empty_frame(self) -> None:
        """Non-empty DataFrame with 0 columns gracefully returns empty MatchResults."""
        df_l = pd.DataFrame(index=[0, 1])
        df_r = pd.DataFrame(index=[0, 1])
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)
        assert len(res) == 0
        assert len(res.clusters) == 0


class TestAdversarialDisjointHeadersAndColumnTypes:
    """Stress test disjoint headers, integer headers, non-text columns, duplicate headers."""

    def test_completely_disjoint_arbitrary_headers(self) -> None:
        df_l = pd.DataFrame(
            {
                "alpha_xyz": ["Delta Aerospace", "Omega Health"],
                "beta_num": [100.0, 200.0],
            }
        )
        df_r = pd.DataFrame(
            {
                "theta_abc": ["Delta Aerospace Inc", "Omega Health LLC"],
                "gamma_val": [100.0, 200.0],
            }
        )
        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)
        assert len(res) >= 1
        assert res.matches.iloc[0]["score"] >= 0.7

    def test_integer_column_names(self) -> None:
        """Unannotated DataFrames with integer columns 0, 1 are handled gracefully without AttributeError."""
        df_l = pd.DataFrame(
            {
                0: [101, 102],
                1: ["Acme Supply", "Apex Technology"],
            }
        )
        df_r = pd.DataFrame(
            {
                0: [201, 202],
                1: ["Acme Supply LLC", "Apex Technology Inc"],
            }
        )
        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)
        assert len(res) >= 1

    def test_duplicate_column_names(self) -> None:
        """Duplicate column headers are handled gracefully without AttributeError."""
        df_l = pd.DataFrame([["Acme Corp", "Acme Corp"]], columns=["name", "name"])
        df_r = pd.DataFrame([["Acme Corp", "Acme Corp"]], columns=["name", "name"])
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)
        assert len(res) == 1

    def test_purely_numeric_frames_no_text(self) -> None:
        """Frames containing only numeric columns."""
        df_l = pd.DataFrame({"metric_a": [1.0, 2.0, 3.0], "metric_b": [10.0, 20.0, 30.0]})
        df_r = pd.DataFrame({"metric_a": [1.0, 2.0, 3.0], "metric_b": [10.0, 20.0, 30.0]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)
        assert len(res) == 3

    def test_boolean_and_categorical_columns(self) -> None:
        df_l = pd.DataFrame(
            {
                "flag": [True, False],
                "category": pd.Categorical(["Retail", "Wholesale"]),
            }
        )
        df_r = pd.DataFrame(
            {
                "flag": [True, False],
                "category": pd.Categorical(["Retail", "Wholesale"]),
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)


class TestAdversarialCandidateBudgetCap:
    """Stress test budget_cap compliance under combinatorial candidate explosions."""

    def test_budget_cap_tight_under_all_identical_names(self) -> None:
        """500 left rows and 500 right rows with identical names generate 250,000 potential pairs."""
        N = 500
        df_l = pd.DataFrame({"id": range(N), "name": ["Identical Vendor LLC"] * N})
        df_r = pd.DataFrame({"id": range(N), "name": ["Identical Vendor LLC"] * N})

        for cap in [1, 10, 50, 250]:
            am = AutoMatcher(budget_cap=cap)
            res = am.fit_predict(df_l, df_r)
            assert res.summary_stats["candidate_pairs"] <= cap
            assert res.total_candidates <= cap
            assert len(res.decisions) <= cap
            assert res.summary_stats["budget_utilization_pct"] <= 100.0

    def test_budget_cap_upper_bound_50000(self) -> None:
        """Standard budget_cap=50,000 on a large dataset."""
        N = 300
        df_l = pd.DataFrame({"name": [f"Vendor {i % 10}" for i in range(N)]})
        df_r = pd.DataFrame({"name": [f"Vendor {i % 10}" for i in range(N)]})
        am = AutoMatcher(budget_cap=50_000)
        res = am.fit_predict(df_l, df_r)
        assert res.summary_stats["candidate_pairs"] <= 50_000
        assert res.total_candidates <= 50_000


class TestAdversarialIntransitiveTriadsAndClustering:
    """Stress test intransitive triad resolution and Cluster-Match consistency invariants."""

    def test_intransitive_triad_prevention_star_vs_cc(self) -> None:
        """A ~ B (0.90), B ~ C (0.90), but A !~ C (0.0).

        ConnectedComponents must link A, B, C into a single component {A, B, C}.
        StarClusterer with verify_clique=True must NOT allow A and C in the same cluster.
        """
        pairs = [("A", "B"), ("B", "C")]
        weights = [0.90, 0.90]

        cc = ConnectedComponents(edge_weight_cutoff=0.7)
        cc_clusters = cc.cluster(pairs, edge_weights=weights)
        assert len(cc_clusters) == 1
        assert cc_clusters[0] == {"A", "B", "C"}  # Transitive false chaining in CC!

        star = StarClusterer(cutoff=0.7, verify_clique=True)
        star_clusters = star.cluster(pairs, edge_weights=weights)
        for cl in star_clusters:
            assert not ({"A", "C"}.issubset(cl)), f"False transitive chaining detected in {cl}!"

    def test_automatcher_intransitive_triad_clusters_split(self) -> None:
        """AutoMatcher with star clustering separates A and C in res.clusters."""
        df_l = pd.DataFrame(
            {
                "id": ["A", "C"],
                "name": [
                    "Acme Logistics East Coast Division",
                    "Acme Logistics West Coast Division",
                ],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["B"],
                "name": ["Acme Logistics"],
            }
        )
        am_star = AutoMatcher(threshold=0.60, cluster_method="star")
        res_star = am_star.fit_predict(df_l, df_r)

        for cl in res_star.clusters:
            assert not ({"A", "C"}.issubset(cl))
            assert not ({"L_A", "L_C"}.issubset(cl))

    def test_automatcher_cluster_match_inconsistency_defect(self) -> None:
        """Verify cluster-match consistency invariant:
        Pairs severed by StarClusterer (such as A and B) must NOT be present in res.matches.
        Every match in res.matches must have both left_id and right_id belonging to the same cluster.
        """
        df_l = pd.DataFrame(
            {
                "id": ["A", "C"],
                "name": ["Acme Logistics East", "Acme Logistics West"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["B"],
                "name": ["Acme Logistics"],
            }
        )
        am_star = AutoMatcher(threshold=0.60, cluster_method="star")
        res_star = am_star.fit_predict(df_l, df_r)

        # In res_star.clusters, A is isolated from B: [{'C', 'B'}, {'A'}]
        # Therefore, (A, B) must NOT be present in res_star.matches
        assert "A" not in res_star.matches["left_id"].tolist()

        # Any match present must have both endpoints in the assigned cluster
        for _, row in res_star.matches.iterrows():
            assigned_cid = row["cluster_id"]
            cluster_members = res_star.clusters[assigned_cid]
            lid = row["left_id"]
            rid = row["right_id"]
            assert lid in cluster_members or f"L_{lid}" in cluster_members
            assert rid in cluster_members or f"R_{rid}" in cluster_members


class TestAdversarialBipartiteIndexCollision:
    """Stress test when left and right have identical RangeIndex, duplicate index, etc."""

    def test_identical_range_index_disambiguation(self) -> None:
        """Both frames have RangeIndex [0, 1]. Verify no collision occurs."""
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC"]})  # index 0, 1
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC"]})  # index 0, 1

        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 2
        for _, row in res.matches.iterrows():
            assert row["left_id"] in [0, 1]
            assert row["right_id"] in [0, 1]
            assert row["cluster_id"] >= 0

    def test_overlapping_string_ids(self) -> None:
        """Both frames have explicit 'id' column with overlapping keys 'ID_1'."""
        df_l = pd.DataFrame({"id": ["ID_1", "ID_2"], "name": ["Alpha Corp", "Beta LLC"]})
        df_r = pd.DataFrame({"id": ["ID_1", "ID_2"], "name": ["Alpha Corp", "Beta LLC"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 2
        assert len(res.clusters) == 2

    def test_duplicate_indices_in_same_frame(self) -> None:
        """Left frame has duplicate index [0, 0, 1]."""
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Alpha Inc", "Beta LLC"]}, index=[0, 0, 1])
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC"]}, index=[0, 1])
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) >= 2

    def test_missing_ids_collapse_distinct_records_defect(self) -> None:
        """Multiple records with None in ID column do NOT collapse into a single cluster."""
        df_l = pd.DataFrame(
            {
                "id": [None, None],
                "name": ["Alpha Corp", "Beta LLC"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [None, None],
                "name": ["Alpha Corp", "Beta LLC"],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        # Distinct entities Alpha Corp and Beta LLC should be in 2 clusters, not collapsed into 1
        assert len(res.clusters) == 2
        for cl in res.clusters:
            assert "L_None" not in cl
            assert "R_None" not in cl


class TestAdversarialNonFiniteValuesAndSpecialStrings:
    """Stress test NaN, inf, -inf, corrupted dates, unicode, emojis, control chars."""

    def test_infinite_and_nan_numeric_values(self) -> None:
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
                "amount": [np.inf, -np.inf, np.nan],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
                "amount": [100.0, -np.inf, 50.0],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) >= 1
        for _, row in res.matches.iterrows():
            assert math.isfinite(row["score"])
            assert 0.0 <= row["score"] <= 1.0

    def test_corrupted_and_extreme_dates(self) -> None:
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
                "date": ["invalid_date_format", "9999-99-99", None],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
                "date": ["2025-01-01", "2025-01-02", "not_a_date"],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 3
        for _, row in res.matches.iterrows():
            assert math.isfinite(row["score"])
            assert 0.0 <= row["score"] <= 1.0

    def test_unicode_emojis_rtl_and_control_chars(self) -> None:
        df_l = pd.DataFrame(
            {
                "name": [
                    "شركة الأمل للتجارة",  # Arabic RTL
                    "株式会社ソニー・インタラクティブエンタテインメント",  # Japanese
                    "🚀 Rocket Tech 🛰️",  # Emojis
                    "Line1\nLine2\tTabbed\x00NullByte",  # Control characters
                    "   \t  \n   ",  # Whitespace only
                ]
            }
        )
        df_r = pd.DataFrame(
            {
                "name": [
                    "شركة الأمل",
                    "株式会社ソニー",
                    "Rocket Tech",
                    "Line1 Line2 Tabbed",
                    "",
                ]
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert isinstance(res, MatchResults)
        assert len(res.matches) >= 2


class TestAdversarialMatchResultsMethods:
    """Stress test MatchResults public API: to_dataframe, to_csv, to_excel, get_explanations."""

    def test_to_dataframe_immutability(self) -> None:
        df_l = pd.DataFrame({"name": ["Alpha Corp"]})
        df_r = pd.DataFrame({"name": ["Alpha Corp"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        df_out = res.to_dataframe()
        df_out["score"] = 0.0
        assert res.matches.iloc[0]["score"] > 0.0  # Original not mutated

    def test_to_csv_and_to_excel_roundtrip(self, tmp_path: Path) -> None:
        df_l = pd.DataFrame(
            {"name": ["Alpha Corp, with commas\nand newlines", "Beta 'Quotes' LLC"]}
        )
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta Quotes"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        csv_path = tmp_path / "deep" / "nested" / "output.csv"
        xlsx_path = tmp_path / "deep" / "nested" / "output.xlsx"

        res.to_csv(csv_path)
        assert csv_path.exists()
        read_df = pd.read_csv(csv_path)
        assert len(read_df) == len(res)

        try:
            import openpyxl  # noqa: F401

            res.to_excel(xlsx_path)
            assert xlsx_path.exists()
        except ImportError:
            pass

    def test_get_explanations_boundary_values(self) -> None:
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC"]})
        df_r = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        exp_0 = res.get_explanations(top_n=0)
        assert exp_0 == []

        exp_1 = res.get_explanations(top_n=1)
        assert len(exp_1) == 1
        assert "explanation" in exp_1[0]
        assert "score" in exp_1[0]

        exp_all = res.get_explanations(top_n=100)
        assert len(exp_all) == len(res.decisions)
