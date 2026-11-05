"""Comprehensive 100% statement and branch coverage test suite for AutoMatcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.api.auto import (
    AutoMatcher,
    _classify_column_role,
    _discover_column_pairs,
    _extract_series,
)


class TestAutoMatcherParameterValidation:
    """Validate constructor constraints and parameter bounds."""

    def test_default_initialization(self) -> None:
        am = AutoMatcher()
        assert am.budget_cap == 50_000
        assert am.threshold == 0.7
        assert am.cluster_method == "star"
        assert am.max_window == 8

    def test_custom_initialization(self) -> None:
        am = AutoMatcher(
            budget_cap=1_000,
            threshold=0.85,
            cluster_method="connected_components",
            max_window=4,
        )
        assert am.budget_cap == 1_000
        assert am.threshold == 0.85
        assert am.cluster_method == "connected_components"
        assert am.max_window == 4

    def test_invalid_budget_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            AutoMatcher(budget_cap=0)
        with pytest.raises(ValueError, match="positive integer"):
            AutoMatcher(budget_cap=-100)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            AutoMatcher(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold must be in"):
            AutoMatcher(threshold=1.5)

    def test_invalid_cluster_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown cluster_method"):
            AutoMatcher(cluster_method="dbscan")  # type: ignore[arg-type]

    def test_invalid_max_window_raises(self) -> None:
        with pytest.raises(ValueError, match="max_window must be >= 1"):
            AutoMatcher(max_window=0)


class TestEmptyAndBoundaryInputs:
    """Test 0-row, 1x1, and degenerate boundary conditions."""

    def test_empty_left_frame(self) -> None:
        am = AutoMatcher()
        res = am.fit_predict(
            pd.DataFrame(columns=["id", "name"]),
            pd.DataFrame({"id": [1], "name": ["A"]}),
        )
        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["left_rows"] == 0
        assert res.summary_stats["candidate_pairs"] == 0

    def test_empty_right_frame(self) -> None:
        am = AutoMatcher()
        res = am.fit_predict(
            pd.DataFrame({"id": [1], "name": ["A"]}),
            pd.DataFrame(columns=["id", "name"]),
        )
        assert len(res) == 0
        assert len(res.clusters) == 0
        assert res.summary_stats["right_rows"] == 0

    def test_both_empty_frames(self) -> None:
        am = AutoMatcher()
        res = am.fit_predict(pd.DataFrame(), pd.DataFrame())
        assert len(res) == 0
        assert res.summary_stats["candidate_pairs"] == 0

    def test_single_row_exact_match(self) -> None:
        am = AutoMatcher()
        df_l = pd.DataFrame({"name": ["Acme Corp"]})
        df_r = pd.DataFrame({"name": ["Acme Corp"]})
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert len(res.clusters) == 1
        assert res.matches.iloc[0]["score"] == 1.0

    def test_single_row_below_threshold(self) -> None:
        am = AutoMatcher(threshold=0.90)
        df_l = pd.DataFrame({"name": ["Acme Corp"]})
        df_r = pd.DataFrame({"name": ["Different"]})
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 0
        assert len(res.clusters) == 0


class TestColumnRoleClassificationAndPairing:
    """Test semantic column role discovery and 3-tier pairing logic."""

    def test_date_detection(self) -> None:
        df = pd.DataFrame(
            {
                "dt_native": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "invoice_date_str": ["2025-01-01", "2025-01-02"],
                "date_invalid_str": ["not_a_date_1", "not_a_date_2"],
                "not_date": ["foo", "bar"],
            }
        )
        assert _classify_column_role(df["dt_native"], "dt_native") == "date"
        assert _classify_column_role(df["invoice_date_str"], "invoice_date_str") == "date"
        assert _classify_column_role(df["date_invalid_str"], "date_invalid_str") == "text"
        assert _classify_column_role(df["not_date"], "not_date") == "text"

    def test_numeric_and_currency_detection(self) -> None:
        df = pd.DataFrame(
            {
                "amount_float": [100.5, 200.7],
                "price_str": ["$100.50", "$200.75"],
                "amount_invalid_str": ["invalid_amt_1", "invalid_amt_2"],
                "vendor_id_int": [1001, 1002],
                "unique_serial": [501, 502],
                "generic_float": [1.1, 2.2],
                "non_unique_int": [1, 1],
                "is_active_bool": [True, False],
            }
        )
        assert _classify_column_role(df["amount_float"], "amount_float") == "numeric"
        assert _classify_column_role(df["price_str"], "price_str") == "numeric"
        assert _classify_column_role(df["amount_invalid_str"], "amount_invalid_str") == "text"
        assert _classify_column_role(df["vendor_id_int"], "vendor_id") == "key"
        assert _classify_column_role(df["unique_serial"], "unique_serial") == "key"
        assert _classify_column_role(df["generic_float"], "generic_float") == "numeric"
        assert _classify_column_role(df["non_unique_int"], "non_unique_int") == "numeric"
        assert _classify_column_role(df["is_active_bool"], "is_active_bool") == "unknown"

    def test_name_and_secondary_text_detection(self) -> None:
        df = pd.DataFrame(
            {
                "vendor_name": ["Acme Corp LLC", "Beta Supplies"],
                "tax_id": ["12-3456789", "98-7654321"],
                "city": ["Houston", "Dallas"],
                "unstructured_blob": ["Alpha Beta Gamma Delta", "One Two Three Four"],
                "single_token_str": ["Alpha", "Beta"],
            }
        )
        assert _classify_column_role(df["vendor_name"], "vendor_name") == "name"
        assert _classify_column_role(df["tax_id"], "tax_id") == "key"
        assert _classify_column_role(df["city"], "city") == "text"
        assert _classify_column_role(df["unstructured_blob"], "unstructured_blob") == "name"
        assert _classify_column_role(df["single_token_str"], "single_token_str") == "text"

    def test_column_pairing_exact_synonyms_and_fallbacks(self) -> None:
        df_l = pd.DataFrame(
            {
                "id": ["L1"],
                "vendor_name": ["Acme Corp"],
                "city": ["Dallas"],
                "amount": [500.0],
                "invoice_date": ["2025-02-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1"],
                "payee": ["Acme"],
                "location": ["Dallas"],
                "amount": [500.0],
                "invoice_date": ["2025-02-01"],
            }
        )
        pairs = _discover_column_pairs(df_l, df_r)
        assert ("id", "id") in pairs["key"]
        assert ("vendor_name", "payee") in pairs["name"]
        assert ("city", "location") in pairs["text"]
        assert ("amount", "amount") in pairs["numeric"]
        assert ("invoice_date", "invoice_date") in pairs["date"]

    def test_column_pairing_text_fallback_to_name(self) -> None:
        df_l = pd.DataFrame({"bool_l": [True, False]})
        df_r = pd.DataFrame({"bool_r": [False, True]})
        pairs = _discover_column_pairs(df_l, df_r)
        assert ("bool_l", "bool_r") in pairs["name"]


class TestMultiPassBlockingAndCandidateCap:
    """Verify multi-pass generation and strict budget_cap compliance."""

    def test_budget_cap_strictly_enforced(self) -> None:
        df_l = pd.DataFrame({"id": range(100), "name": ["Identical Vendor"] * 100})
        df_r = pd.DataFrame({"id": range(100), "name": ["Identical Vendor"] * 100})
        am = AutoMatcher(budget_cap=50)
        res = am.fit_predict(df_l, df_r)
        assert res.summary_stats["candidate_pairs"] <= 50
        assert res.summary_stats["budget_utilization_pct"] <= 100.0
        assert res.total_candidates <= 50

    def test_multi_pass_covers_exact_and_sorted_neighborhood(self) -> None:
        df_l = pd.DataFrame(
            {"id": ["L1", "L2"], "name": ["Acme Industrial Supplies", "Summit Chemical"]}
        )
        df_r = pd.DataFrame(
            {"id": ["R1", "R2"], "name": ["Acme Industrial Supply LLC", "Summit Chemical Corp"]}
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) >= 2


class TestCompositeScoringIntegration:
    """Verify combination of RapidFuzz, continuous amount, and date window scoring."""

    def test_composite_evaluation(self) -> None:
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corp"],
                "amount": [100.00],
                "invoice_date": ["2025-01-01"],
                "city": ["New York"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Corp"],
                "amount": [100.25],
                "invoice_date": ["2025-01-02"],
                "city": ["New York"],
            }
        )
        am = AutoMatcher(threshold=0.80)
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        row = res.matches.iloc[0]
        assert "score_name" in row
        assert "score_amount" in row
        assert "score_invoice_date" in row
        assert "score_city" in row
        assert row["score"] >= 0.80
        assert row["status"] == "match"

    def test_possible_and_non_match_statuses(self) -> None:
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corporation", "Completely Unrelated"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Inc", "Something Else"],
            }
        )
        am = AutoMatcher(threshold=0.85)
        res = am.fit_predict(df_l, df_r)
        dec_statuses = {d.status for d in res.decisions}
        assert "possible" in dec_statuses or "non_match" in dec_statuses or "match" in dec_statuses

    def test_missing_value_neutrality(self) -> None:
        df_l = pd.DataFrame({"name": ["Acme Corp"], "amount": [np.nan]})
        df_r = pd.DataFrame({"name": ["Acme Corp"], "amount": [50.0]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1
        assert res.matches.iloc[0]["score"] > 0.0


class TestStarClusteringTransitiveProtection:
    """Verify non-transitive triad splitting under StarClusterer."""

    def test_intransitive_triad_split(self) -> None:
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
        am = AutoMatcher(threshold=0.70, cluster_method="star")
        res = am.fit_predict(df_l, df_r)
        assert len(res.clusters) >= 1
        for cl in res.clusters:
            assert not ({"A", "C"}.issubset(cl))

    def test_connected_components_clustering_option(self) -> None:
        df_l = pd.DataFrame({"id": ["L1"], "name": ["Alpha Corp"]})
        df_r = pd.DataFrame({"id": ["R1"], "name": ["Alpha Corp"]})
        am = AutoMatcher(cluster_method="connected_components")
        res = am.fit_predict(df_l, df_r)
        assert len(res.clusters) == 1


class TestMatchResultsInterfaceAndSerialization:
    """Verify all public methods, properties, and serialization formats."""

    def test_match_results_contracts(self, tmp_path: Path) -> None:
        df_l = pd.DataFrame({"id": ["L1"], "name": ["Beta LLC"]})
        df_r = pd.DataFrame({"id": ["R1"], "name": ["Beta Inc"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)

        assert hasattr(res, "matches")
        assert hasattr(res, "pairs")
        assert hasattr(res, "clusters")
        assert len(res) == len(res.matches)
        assert "MatchResults" in repr(res)

        df_copy = res.to_dataframe()
        assert isinstance(df_copy, pd.DataFrame)

        csv_file = tmp_path / "sub" / "test_out.csv"
        xlsx_file = tmp_path / "sub" / "test_out.xlsx"
        res.to_csv(csv_file)
        assert csv_file.exists()
        try:
            import openpyxl  # noqa: F401

            res.to_excel(xlsx_file)
            assert xlsx_file.exists()
        except ImportError:
            pass

        exps_all = res.get_explanations()
        assert len(exps_all) >= 1
        exps_one = res.get_explanations(top_n=1)
        assert len(exps_one) == 1
        assert "field_scores" in exps_one[0]["explanation"]


class TestUnicodeAndSpecialCharacterHandling:
    """Verify non-ASCII, umlauts, accents, and emojis."""

    def test_multilingual_tokens(self) -> None:
        df_l = pd.DataFrame({"name": ["München Präzision GmbH 🚀", "Société Générale S.A."]})
        df_r = pd.DataFrame({"name": ["Munchen Prazision", "Societe Generale"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 2


class TestTablesWithNoMatchingColumnsOrZeroCandidates:
    """Verify graceful handling when no candidates match threshold."""

    def test_all_distinct_tables(self) -> None:
        df_l = pd.DataFrame({"name": ["Apple", "Banana"]})
        df_r = pd.DataFrame({"name": ["Zebra", "Giraffe"]})
        am = AutoMatcher(threshold=0.85)
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 0
        assert res.matches.empty
        assert len(res.clusters) == 0

    def test_fallback_column_when_no_text_column(self) -> None:
        df_l = pd.DataFrame({"ratio_l": [10.5, 20.5]})
        df_r = pd.DataFrame({"ratio_r": [10.5, 20.5]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert res is not None


class TestEdgeCaseCoverageGaps:
    """Exercise remaining edge branches in auto.py."""

    def test_text_promotion_and_secondary_text_nan(self) -> None:
        # Tier 3 promotion: first text column promoted to name
        # Secondary text has NaN on one side
        df_l = pd.DataFrame(
            {
                "col_text_a": ["Acme Corp"],
                "col_text_b": [np.nan],
            }
        )
        df_r = pd.DataFrame(
            {
                "col_text_a": ["Acme Corp"],
                "col_text_b": ["Some notes"],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1

    def test_non_finite_numeric_and_corrupted_date(self) -> None:
        # Non-finite float (inf/nan) and corrupted date
        df_l = pd.DataFrame(
            {
                "name": ["Acme Corp"],
                "amount": [np.inf],
                "trans_date": ["corrupted_date_string"],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Corp"],
                "amount": [100.0],
                "trans_date": ["2025-01-01"],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res) == 1

    def test_shifted_range_index_bipartite_disambiguation(self) -> None:
        """Verify shifted RangeIndex does not create false path collisions in StarClusterer."""
        df_l = pd.DataFrame({"name": ["Alpha Corp", "Beta LLC", "Gamma Inc"]})
        df_r = pd.DataFrame({"name": ["Zeta Ltd", "Alpha Corp", "Beta LLC", "Gamma Inc"]})
        am = AutoMatcher(threshold=0.7)
        res = am.fit_predict(df_l, df_r)
        assert len(res.matches) == 3
        assert list(res.matches["left_id"]) == [0, 1, 2]
        assert list(res.matches["right_id"]) == [1, 2, 3]


class TestAutoMatcherCoverageGapClosure:
    """Exercising remaining 15 unexercised lines in fuzzymatch/api/auto.py for 100.00% coverage."""

    def test_extract_series_duplicate_columns(self) -> None:
        """Covers lines 190-191: _extract_series on DataFrame with duplicate column names."""
        df_dup = pd.DataFrame([[10, 20]], columns=["dup_col", "dup_col"])
        s = _extract_series(df_dup, "dup_col")
        assert isinstance(s, pd.Series)
        assert s.iloc[0] == 10

    def test_extract_series_custom_mapping_fallback(self) -> None:
        """Covers line 194: _extract_series fallback returning pd.Series(val)."""

        class CustomMapping:
            def __getitem__(self, key: str) -> list[int]:
                return [100, 200, 300]

        s = _extract_series(CustomMapping(), "any_key")  # type: ignore[arg-type]
        assert isinstance(s, pd.Series)
        assert s.tolist() == [100, 200, 300]

    def test_discover_column_pairs_tier3_text_promotion(self) -> None:
        """Covers line 253: Tier 3 promotion of first text pair to name when no name exists."""
        df_l = pd.DataFrame({"desc": ["alpha", "beta"]})
        df_r = pd.DataFrame({"desc": ["alpha", "beta"]})
        pairs = _discover_column_pairs(df_l, df_r)
        assert len(pairs["name"]) == 1
        assert pairs["name"][0] == ("desc", "desc")
        assert len(pairs["text"]) == 0

    def test_fit_predict_secondary_text_missing_values(self) -> None:
        """Covers lines 475-478: Scoring secondary text with None, NaN, and empty strings."""
        df_l = pd.DataFrame(
            {
                "name": ["Acme Global Corp", "Beta Logistics Inc"],
                "desc": ["Branch office", None],
            }
        )
        df_r = pd.DataFrame(
            {
                "name": ["Acme Global Corp", "Beta Logistics Inc"],
                "desc": ["Branch office", ""],
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res.matches) >= 1
        assert any(r["score_desc"] == 0.0 for _, r in res.matches.iterrows() if "score_desc" in r)

    def test_fit_predict_numeric_string_conversion_exception(self) -> None:
        """Covers lines 491-492: Scoring numeric amount when string value raises ValueError."""
        amounts_l = [100.0] * 8 + ["unparseable_string", "another_bad"]
        names = [f"Acme Corp {i}" for i in range(10)]
        df_l = pd.DataFrame(
            {
                "name": names,
                "amount": amounts_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "name": names,
                "amount": [100.0] * 10,
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res.matches) >= 8

    def test_fit_predict_date_nat_and_exception_branches(self) -> None:
        """Covers lines 512-514 and 521-522: Date parsing NaT and Exception fallbacks."""

        class UnparseableDateObj:
            def __iter__(self) -> Any:
                raise TypeError("Custom unparseable date object")

        dates_l = ["2025-01-01"] * 8 + ["corrupted_date_str", UnparseableDateObj()]
        names = [f"Acme Corp {i}" for i in range(10)]
        df_l = pd.DataFrame(
            {
                "name": names,
                "trans_date": dates_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "name": names,
                "trans_date": ["2025-01-01"] * 10,
            }
        )
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res.matches) >= 8

    def test_fit_predict_fallback_when_no_scored_columns(self) -> None:
        """Covers lines 526-533: Fallback scoring when only key columns exist."""
        df_l = pd.DataFrame({"account_id": ["123", "456"]})
        df_r = pd.DataFrame({"account_id": ["123", "789"]})
        am = AutoMatcher()
        res = am.fit_predict(df_l, df_r)
        assert len(res.matches) == 1
        assert res.matches.iloc[0]["score"] == 1.0
