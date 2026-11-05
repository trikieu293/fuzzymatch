"""Empirical stress test suite for AutoMatcher and MatchResults.

Rigorously tests:
1. Zero-row, 1x1, 1x1000, 1000x1 inputs
2. Disjoint column names, non-string column headers, arbitrary dtypes
3. Strict candidate budget cap enforcement under large data
4. Intransitive triads with StarClusterer vs ConnectedComponents
5. Bipartite index collisions (RangeIndex, duplicate indices, negative indices)
6. Non-finite floats (inf, -inf, NaN), corrupted dates, unicode, emojis
7. MatchResults public methods (.to_dataframe, .to_csv, .to_excel, .get_explanations)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.api.auto import AutoMatcher, MatchResults

# ============================================================================
# 1. Zero-Row, Single-Row, and Asymmetric Dimension Inputs
# ============================================================================


class TestDimensionStress:
    """Stress test boundary shapes and severe dimensional asymmetries."""

    def test_both_zero_rows(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame(columns=["id", "name", "amount", "date"])
        df_r = pd.DataFrame(columns=["id", "name", "amount", "date"])
        res = am.fit_predict(df_l, df_r)

        assert isinstance(res, MatchResults)
        assert len(res) == 0
        assert len(res.clusters) == 0
        assert len(res.decisions) == 0
        assert res.summary_stats["left_rows"] == 0
        assert res.summary_stats["right_rows"] == 0
        assert res.summary_stats["candidate_pairs"] == 0
        assert res.summary_stats["budget_utilization_pct"] == 0.0

        # Method contracts on zero-row results
        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert res.get_explanations() == []

    def test_left_zero_right_non_zero(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame(columns=["name"])
        df_r = pd.DataFrame({"name": [f"Vendor {i}" for i in range(50)]})
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["left_rows"] == 0
        assert res.summary_stats["right_rows"] == 50
        assert res.summary_stats["candidate_pairs"] == 0

    def test_left_non_zero_right_zero(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame({"name": [f"Vendor {i}" for i in range(50)]})
        df_r = pd.DataFrame(columns=["name"])
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["left_rows"] == 50
        assert res.summary_stats["right_rows"] == 0
        assert res.summary_stats["candidate_pairs"] == 0

    def test_single_row_match_and_non_match(self) -> None:
        am = AutoMatcher(threshold=0.75)
        # Match
        res_m = am.fit_predict(
            pd.DataFrame({"name": ["Acme Global Technologies Inc"]}),
            pd.DataFrame({"name": ["Acme Global Technologies"]}),
        )
        assert len(res_m) == 1
        assert len(res_m.clusters) == 1
        assert res_m.matches.iloc[0]["score"] >= 0.75
        assert res_m.matches.iloc[0]["cluster_id"] == 0

        # Non-match
        res_nm = am.fit_predict(
            pd.DataFrame({"name": ["Acme Global Technologies"]}),
            pd.DataFrame({"name": ["Zanzibar Exploration Mining"]}),
        )
        assert len(res_nm) == 0
        assert len(res_nm.clusters) == 0

    def test_extreme_asymmetry_1_x_1000(self) -> None:
        """1 record on left vs 1,000 records on right."""
        am = AutoMatcher(budget_cap=50_000, threshold=0.70)
        df_l = pd.DataFrame({"name": ["Apex Logistics Inc"]})
        # 1 near match + 999 distant records
        names = ["Apex Logistics LLC"] + [f"Random Unrelated Company {i}" for i in range(999)]
        df_r = pd.DataFrame({"name": names})

        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert len(res.clusters) == 1
        assert res.matches.iloc[0]["left_id"] == 0
        assert res.matches.iloc[0]["right_id"] == 0
        assert res.summary_stats["left_rows"] == 1
        assert res.summary_stats["right_rows"] == 1000

    def test_extreme_asymmetry_1000_x_1(self) -> None:
        """1,000 records on left vs 1 record on right."""
        am = AutoMatcher(budget_cap=50_000, threshold=0.70)
        names = ["Apex Logistics LLC"] + [f"Random Unrelated Company {i}" for i in range(999)]
        df_l = pd.DataFrame({"name": names})
        df_r = pd.DataFrame({"name": ["Apex Logistics Inc"]})

        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert len(res.clusters) == 1
        assert res.matches.iloc[0]["left_id"] == 0
        assert res.matches.iloc[0]["right_id"] == 0


# ============================================================================
# 2. Disjoint Column Names, Heterogeneous Headers, and Arbitrary Dtypes
# ============================================================================


class TestDisjointHeadersAndTypes:
    """Stress test tables with no shared schema, arbitrary headers, and unusual dtypes."""

    def test_completely_disjoint_headers(self) -> None:
        """Headers have no semantic keywords and no common substrings."""
        df_l = pd.DataFrame(
            {
                "col_alpha": ["Apex Supplies Co", "Beacon Health"],
                "col_beta": [100.0, 250.0],
            }
        )
        df_r = pd.DataFrame(
            {
                "hdr_x": ["Apex Supplies", "Beacon Health Systems"],
                "hdr_y": [100.0, 250.0],
            }
        )
        am = AutoMatcher(threshold=0.70)
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        assert len(res.clusters) == 2

    def test_purely_numeric_disjoint_columns(self) -> None:
        """Tables have only numeric columns with arbitrary headers."""
        df_l = pd.DataFrame({"val_left": [10.5, 20.0, 30.5]})
        df_r = pd.DataFrame({"val_right": [10.5, 20.2, 99.0]})
        am = AutoMatcher(threshold=0.80)
        res = am.fit_predict(df_l, df_r)

        assert res is not None
        assert isinstance(res.matches, pd.DataFrame)
        assert len(res) >= 1

    def test_purely_date_disjoint_columns(self) -> None:
        """Tables have only date columns with arbitrary headers."""
        df_l = pd.DataFrame({"d_one": ["2025-01-01", "2025-06-15"]})
        df_r = pd.DataFrame({"d_two": ["2025-01-02", "2025-12-31"]})
        am = AutoMatcher(threshold=0.70)
        res = am.fit_predict(df_l, df_r)

        assert res is not None
        assert len(res) >= 1

    def test_categorical_and_boolean_dtypes(self) -> None:
        """Tables with categorical dtypes and boolean flags."""
        df_l = pd.DataFrame(
            {
                "entity": pd.Categorical(["Acme Corp", "Beta LLC"]),
                "is_active": [True, False],
            }
        )
        df_r = pd.DataFrame(
            {
                "firm": pd.Categorical(["Acme Corporation", "Beta Limited"]),
                "flag": [True, True],
            }
        )
        am = AutoMatcher(threshold=0.70)
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        assert len(res.clusters) == 2

    def test_non_string_column_names(self) -> None:
        """Tables with integer column headers (e.g. from read_csv with header=None)."""
        df_l = pd.DataFrame([[1, "Acme Corp"], [2, "Beta LLC"]], columns=[0, 1])
        df_r = pd.DataFrame([[10, "Acme Inc"], [20, "Beta Corp"]], columns=[0, 1])
        # If _classify_column_role assumes col_name is str, does it crash?
        try:
            am = AutoMatcher(threshold=0.70)
            res = am.fit_predict(df_l, df_r)
            assert res is not None
        except AttributeError as e:
            pytest.fail(f"AutoMatcher crashed on integer column headers: {e}")

    def test_whitespace_and_empty_column_names(self) -> None:
        """Columns with whitespace names or punctuation."""
        df_l = pd.DataFrame({"   ": ["Acme Corp"], "...": [100.0]})
        df_r = pd.DataFrame({"   ": ["Acme Inc"], "...": [100.0]})
        am = AutoMatcher(threshold=0.70)
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1


# ============================================================================
# 3. Budget Cap Under Large Data
# ============================================================================


class TestBudgetCapStress:
    """Stress test candidate generation under large inputs and strict budget caps."""

    def test_candidate_budget_cap_saturation(self) -> None:
        """1,000 x 1,000 cartesian candidate space with tight budget_cap=250."""
        n = 1000
        df_l = pd.DataFrame({"name": ["Universal Identical Company"] * n})
        df_r = pd.DataFrame({"name": ["Universal Identical Company"] * n})

        cap = 250
        am = AutoMatcher(budget_cap=cap)
        res = am.fit_predict(df_l, df_r)

        assert res.summary_stats["candidate_pairs"] <= cap
        assert res.total_candidates <= cap
        assert res.summary_stats["budget_utilization_pct"] <= 100.0
        assert len(res.matches) <= cap
        assert len(res.decisions) <= cap

    def test_budget_cap_exact_boundary(self) -> None:
        """Verify behavior when candidate pairs exactly match budget_cap."""
        df_l = pd.DataFrame({"name": [f"Company {i}" for i in range(10)]})
        df_r = pd.DataFrame({"name": [f"Company {i}" for i in range(10)]})
        # Exact matching produces 10 candidate pairs
        am = AutoMatcher(budget_cap=10)
        res = am.fit_predict(df_l, df_r)

        assert res.summary_stats["candidate_pairs"] <= 10
        assert res.summary_stats["budget_utilization_pct"] <= 100.0

    def test_budget_cap_1(self) -> None:
        """Extreme budget_cap=1."""
        df_l = pd.DataFrame({"name": ["Acme 1", "Acme 2", "Acme 3"]})
        df_r = pd.DataFrame({"name": ["Acme 1", "Acme 2", "Acme 3"]})
        am = AutoMatcher(budget_cap=1)
        res = am.fit_predict(df_l, df_r)

        assert res.summary_stats["candidate_pairs"] <= 1
        assert len(res.matches) <= 1


# ============================================================================
# 4. Intransitive Triads: StarClusterer vs ConnectedComponents
# ============================================================================


class TestIntransitiveTriadElimination:
    """Rigorously verify elimination of false transitive chaining on intransitive triads."""

    def test_intransitive_triad_star_vs_connected_components(self) -> None:
        """Graph structure:
        Record A in df_l: "Acme Logistics North"
        Record B in df_r: "Acme Logistics"
        Record C in df_l: "Acme Logistics South"

        A matches B (score >= 0.70)
        C matches B (score >= 0.70)
        A and C do NOT match (score < 0.70)

        ConnectedComponents will falsely merge {A, B, C} into a single cluster.
        StarClusterer MUST break the triad so A and C are never in the same cluster.
        """
        df_l = pd.DataFrame(
            {
                "id": ["A", "C"],
                "name": ["Acme Logistics North", "Acme Logistics South"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["B"],
                "name": ["Acme Logistics"],
            }
        )

        # 1. StarClusterer
        am_star = AutoMatcher(threshold=0.68, cluster_method="star")
        res_star = am_star.fit_predict(df_l, df_r)

        for cl in res_star.clusters:
            # A and C must NEVER be co-clustered
            assert not ({"A", "C"}.issubset(cl)), f"StarClusterer falsely merged {cl}"

        # 2. ConnectedComponents
        am_cc = AutoMatcher(threshold=0.68, cluster_method="connected_components")
        res_cc = am_cc.fit_predict(df_l, df_r)

        # In CC, A and C ARE merged together
        cc_merged = any({"A", "C"}.issubset(cl) for cl in res_cc.clusters)
        assert cc_merged, "ConnectedComponents should show transitive merge"

    def test_multi_satellite_star_graph(self) -> None:
        """Central hub H matches 5 satellites S1..S5, but satellites are mutually dissimilar."""
        df_l = pd.DataFrame(
            {
                "id": [f"S{i}" for i in range(5)],
                "name": [
                    "Omega Alpha Division",
                    "Omega Beta Division",
                    "Omega Gamma Division",
                    "Omega Delta Division",
                    "Omega Epsilon Division",
                ],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["H"],
                "name": ["Omega Division"],
            }
        )

        am = AutoMatcher(threshold=0.65, cluster_method="star")
        res = am.fit_predict(df_l, df_r)

        # In star clustering with clique verification, no two mutually dissimilar satellites can share a cluster
        for cl in res.clusters:
            satellite_count = sum(1 for m in cl if str(m).startswith("S"))
            assert satellite_count <= 1, (
                f"Multiple mutually dissimilar satellites co-clustered: {cl}"
            )


# ============================================================================
# 5. Bipartite Index Collisions and Index Variations
# ============================================================================


class TestBipartiteIndexCollisions:
    """Stress test index collision disambiguation between left and right dataframes."""

    def test_identical_range_indices_no_id_column(self) -> None:
        """Both frames have RangeIndex [0, 1] and no ID column."""
        df_l = pd.DataFrame({"name": ["Acme Corp", "Beta Supplies"]})
        df_r = pd.DataFrame({"name": ["Acme Corp", "Beta Supplies"]})

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        assert len(res.clusters) == 2
        # Cluster IDs must be valid non-negative integers
        for cid in res.matches["cluster_id"]:
            assert cid >= 0, f"Unassigned cluster_id found: {cid}"
        # left_id and right_id must be present
        assert set(res.matches["left_id"]) == {0, 1}
        assert set(res.matches["right_id"]) == {0, 1}

    def test_identical_explicit_id_column(self) -> None:
        """Both frames have column 'id' with identical values [101, 102]."""
        df_l = pd.DataFrame({"id": [101, 102], "name": ["Acme Corp", "Beta Corp"]})
        df_r = pd.DataFrame({"id": [101, 102], "name": ["Acme Inc", "Beta LLC"]})

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        assert len(res.clusters) == 2
        for cid in res.matches["cluster_id"]:
            assert cid >= 0

    def test_negative_and_arbitrary_indices(self) -> None:
        """Frames with negative, non-contiguous indices."""
        df_l = pd.DataFrame({"name": ["Acme Corp", "Beta Corp"]}, index=[-10, -20])
        df_r = pd.DataFrame({"name": ["Acme Inc", "Beta LLC"]}, index=[-10, -20])

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        assert set(res.matches["left_id"]) == {-10, -20}
        assert set(res.matches["right_id"]) == {-10, -20}
        for cid in res.matches["cluster_id"]:
            assert cid >= 0

    def test_duplicate_indices_in_frames(self) -> None:
        """Frames containing duplicate index labels."""
        df_l = pd.DataFrame({"name": ["Acme Corp", "Beta Corp"]}, index=[0, 0])
        df_r = pd.DataFrame({"name": ["Acme Inc", "Beta LLC"]}, index=[0, 0])

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        for cid in res.matches["cluster_id"]:
            assert cid >= 0

    def test_prefix_collision_prevention(self) -> None:
        """IDs that already contain 'L_' and 'R_' prefixes."""
        df_l = pd.DataFrame({"id": ["L_1", "L_2"], "name": ["Acme Corp", "Beta Corp"]})
        df_r = pd.DataFrame({"id": ["L_1", "L_2"], "name": ["Acme Inc", "Beta LLC"]})

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        for cid in res.matches["cluster_id"]:
            assert cid >= 0


# ============================================================================
# 6. Unicode, Emojis, Missing Values, Non-Finite Floats, and Corrupted Dates
# ============================================================================


class TestAdversarialValuesAndDataTypes:
    """Stress test non-finite numbers, corrupted timestamps, multilingual unicode, and emojis."""

    def test_multilingual_and_emoji_strings(self) -> None:
        """Arabic, Chinese, Japanese, Cyrillic, German umlauts, and emojis."""
        df_l = pd.DataFrame(
            {
                "name": [
                    "شركة الأمل للتجارة",
                    "北京科技有限公司 🚀",
                    "東京エレクトロン株式会社 ⚡",
                    "ООО Газпром Нефть 🛢️",  # noqa: RUF001
                    "München Präzisionswerkzeuge GmbH & Co. KG",
                ],
                "city": ["القاهرة", "北京", "東京", "Москва", "München"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": [
                    "شركة الأمل",
                    "北京科技",
                    "東京エレクトロン",
                    "ООО Газпром",  # noqa: RUF001
                    "Munchen Prazisionswerkzeuge",
                ],
                "city": ["القاهرة", "北京", "東京", "Москва", "Munchen"],
            }
        )

        am = AutoMatcher(threshold=0.60)
        res = am.fit_predict(df_l, df_r)
        assert len(res) >= 4

    def test_infinite_and_nan_floats_in_numeric_columns(self) -> None:
        """Numeric column with float('inf'), float('-inf'), and NaN."""
        df_l = pd.DataFrame(
            {
                "name": ["Acme 1", "Acme 2", "Acme 3", "Acme 4"],
                "amount": [100.0, float("inf"), float("-inf"), np.nan],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme 1", "Acme 2", "Acme 3", "Acme 4"],
                "amount": [100.0, 100.0, 100.0, 50.0],
            }
        )

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) >= 1
        # Check that score is finite and in [0, 1]
        for score in res.matches["score"]:
            assert math.isfinite(score)
            assert 0.0 <= score <= 1.0

    def test_corrupted_and_unparseable_dates(self) -> None:
        """Date column with garbage date strings, out-of-bounds dates, and mixed formats."""
        df_l = pd.DataFrame(
            {
                "name": ["Acme 1", "Acme 2", "Acme 3", "Acme 4"],
                "invoice_date": ["2025-01-01", "not_a_date_at_all", "99999-99-99", None],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme 1", "Acme 2", "Acme 3", "Acme 4"],
                "invoice_date": ["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02"],
            }
        )

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) >= 1
        for score in res.matches["score"]:
            assert math.isfinite(score)
            assert 0.0 <= score <= 1.0

    def test_all_missing_values_in_secondary_columns(self) -> None:
        """All-NaN secondary text, amount, and date columns."""
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corp", "Beta LLC"],
                "amount": [np.nan, np.nan],
                "city": [None, None],
                "date": [pd.NaT, pd.NaT],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Inc", "Beta Corp"],
                "amount": [np.nan, np.nan],
                "city": [None, None],
                "date": [pd.NaT, pd.NaT],
            }
        )

        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert len(res) == 2
        for score in res.matches["score"]:
            assert math.isfinite(score)
            assert score > 0.0


# ============================================================================
# 7. MatchResults Interface, Methods, and Serialization
# ============================================================================


class TestMatchResultsContracts:
    """Stress test MatchResults methods and exports."""

    def test_to_dataframe_immutability(self) -> None:
        df_l = pd.DataFrame({"name": ["Acme Corp"]})
        df_r = pd.DataFrame({"name": ["Acme Inc"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        df_copy = res.to_dataframe()
        assert len(df_copy) == 1
        # Mutate the copy
        df_copy.drop(df_copy.index, inplace=True)
        # Original matches must remain unaffected
        assert len(res.matches) == 1

    def test_to_csv_and_to_excel_export(self, tmp_path: Path) -> None:
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Acme Corp 🚀", "Beta 'Special' \"Quotes\""],
                "amount": [123.45, 678.90],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Acme Inc 🚀", "Beta Special Quotes"],
                "amount": [123.45, 678.90],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        csv_file = tmp_path / "nested" / "dir" / "results.csv"
        xlsx_file = tmp_path / "nested" / "dir" / "results.xlsx"

        res.to_csv(csv_file)
        assert csv_file.exists()
        assert csv_file.stat().st_size > 0

        try:
            import openpyxl  # noqa: F401

            res.to_excel(xlsx_file)
            assert xlsx_file.exists()
            assert xlsx_file.stat().st_size > 0
        except ImportError:
            pass

        # Read back CSV
        read_df = pd.read_csv(csv_file)
        assert len(read_df) == 2

    def test_get_explanations_top_n_variations(self) -> None:
        df_l = pd.DataFrame({"name": ["Acme Corp", "Beta LLC", "Gamma Inc"]})
        df_r = pd.DataFrame({"name": ["Acme Corp", "Beta LLC", "Gamma Inc"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        # All explanations
        exp_all = res.get_explanations()
        assert len(exp_all) >= 3

        # top_n = 1
        exp_1 = res.get_explanations(top_n=1)
        assert len(exp_1) == 1
        assert "field_scores" in exp_1[0]["explanation"]

        # top_n = 0
        exp_0 = res.get_explanations(top_n=0)
        assert len(exp_0) == 0

        # top_n > total
        exp_big = res.get_explanations(top_n=9999)
        assert len(exp_big) == len(exp_all)

    def test_repr_and_properties(self) -> None:
        df_l = pd.DataFrame({"name": ["Acme Corp"]})
        df_r = pd.DataFrame({"name": ["Acme Inc"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        rep = repr(res)
        assert "MatchResults" in rep
        assert "matches across" in rep
        assert res.total_candidates >= 1
        assert len(res.pairs) == len(res.matches)
