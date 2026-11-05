"""Adversarial stress and boundary harness for MultiPassBlock.

Empirically verifies:
1. Candidate budget cap bounding under massive candidate pair explosion.
2. High recall (>= 98%) on dirty benchmark tables with typos and nulls across primary keys.
3. Degenerate and boundary inputs (0-row, 1-row, all-null, duplicate passes, zero-pair passes).
"""

from __future__ import annotations

import random
import string
from typing import Any

import numpy as np
import pandas as pd

from fuzzymatch.blocking import (
    ExactBlock,
    MultiPassBlock,
    SortedNeighborhood,
)


class TestCandidateBudgetCapMassiveExplosion:
    """Stress tests verifying budget cap truncation under combinatorial candidate explosion."""

    def test_low_cardinality_explosion_truncates_strictly(self) -> None:
        """5,000 records with 10 identical low-cardinality keys (2.5 million potential pairs).
        Verify MultiPassBlock(budget_cap=500) strictly truncates to <= 500 pairs.
        """
        n_rows = 5000
        n_keys = 10
        # 10 groups of 500 rows -> 10 * (500 * 499) = 2,495,000 potential pairs
        df = pd.DataFrame(
            {"group_key": [f"GRP_{i % n_keys}" for i in range(n_rows)]},
            index=[f"rec_{i}" for i in range(n_rows)],
        )

        mp = MultiPassBlock(passes=[ExactBlock(["group_key"])], budget_cap=500)
        ci = mp.block(df)

        assert len(ci.pairs) == 500
        assert len(ci) == 500
        # Verify deduplication
        assert len(set(ci.pairs)) == 500
        # Verify all pairs are valid pairs from df index
        for left_idx, right_idx in ci.pairs:
            assert left_idx in df.index
            assert right_idx in df.index
            assert left_idx != right_idx

    def test_single_key_explosion_truncation(self) -> None:
        """1,000 records with identical key (1,000 * 999 = 999,000 potential pairs).
        Verify MultiPassBlock with budget_cap=123 returns exactly 123 pairs.
        """
        df = pd.DataFrame({"key": ["SAME"] * 1000}, index=list(range(1000)))
        mp = MultiPassBlock(passes=[ExactBlock(["key"])], budget_cap=123)
        ci = mp.block(df)

        assert len(ci.pairs) == 123
        assert len(set(ci.pairs)) == 123

    def test_cross_file_explosion_truncation(self) -> None:
        """Cross-file blocking with 2,000 rows on left and 2,000 rows on right (4M potential pairs).
        Verify budget_cap=300 strictly truncates to 300 cross pairs.
        """
        df_left = pd.DataFrame({"key": ["K"] * 2000}, index=[f"L_{i}" for i in range(2000)])
        df_right = pd.DataFrame({"key": ["K"] * 2000}, index=[f"R_{i}" for i in range(2000)])

        mp = MultiPassBlock(passes=[ExactBlock(["key"])], budget_cap=300)
        ci = mp.block(df_left, df_right)

        assert len(ci.pairs) == 300
        assert len(set(ci.pairs)) == 300
        for l_id, r_id in ci.pairs:
            assert l_id.startswith("L_")
            assert r_id.startswith("R_")

    def test_early_pass_exit_prevents_subsequent_pass_explosion(self) -> None:
        """Pass 1 has a high-cardinality exact ID key that fills the budget cap.
        Pass 2 has a catastrophic low-cardinality key (e.g. 5,000 identical rows).
        Verify Pass 2 is never executed because Pass 1 satisfied budget_cap.
        """
        n = 500
        # Pass 1 produces n pairs
        df = pd.DataFrame(
            {
                "id_pair": [
                    f"ID_{i // 2}" for i in range(n)
                ],  # 2 rows per group -> 250 pairs (dedupe -> 500)
                "catastrophic_key": ["ALL_SAME"] * n,
            },
            index=list(range(n)),
        )

        p1 = ExactBlock(["id_pair"])
        p2 = ExactBlock(["catastrophic_key"])

        # Cap at 50 pairs
        mp = MultiPassBlock(passes=[p1, p2], budget_cap=50)
        ci = mp.block(df)

        assert len(ci.pairs) == 50
        # All pairs must be from Pass 1 (id_pair)
        for u, v in ci.pairs:
            assert df.loc[u, "id_pair"] == df.loc[v, "id_pair"]


class TestHighRecallDirtyBenchmark:
    """Verify that MultiPassBlock achieves >= 98% recall on dirty benchmark tables with typos and nulls."""

    def test_multi_pass_achieves_98_percent_recall(self) -> None:
        """Construct 1,000 benchmark entity pairs with realistic dirty corruptions:
        - 35% missing primary tax_id
        - 35% typographical mutations in company name
        - 25% missing postal code
        - Corruptions distributed independently so single pass fails (< 75% recall)
          while multi-pass achieves >= 98.0% recall.
        """
        rng = random.Random(2026)
        n_entities = 1000

        clean_records: list[dict[str, Any]] = []
        dirty_records: list[dict[str, Any]] = []
        ground_truth: set[tuple[str, str]] = set()

        company_types = ["Technologies", "Logistics", "Consulting", "Solutions", "Enterprises"]

        for i in range(n_entities):
            l_id = f"SRC_{i:04d}"
            r_id = f"TGT_{i:04d}"
            tax_id = f"TX-{100000 + i}"
            ctype = company_types[i % len(company_types)]
            base_name = f"Enterprise_{i:04d}_{ctype}"
            postal = f"{10000 + (i % 200):05d}"
            city = f"Metropolis_{i % 30}"

            clean_row = {
                "tax_id": tax_id,
                "name": base_name,
                "postal": postal,
                "city": city,
            }
            dirty_row = dict(clean_row)

            # 35% chance of missing tax_id in dirty target
            if rng.random() < 0.35:
                dirty_row["tax_id"] = np.nan

            # 35% chance of typo in name (realistic token corruption)
            if rng.random() < 0.35:
                chars = list(dirty_row["name"])
                pos = rng.randint(len(dirty_row["name"]) - 5, len(dirty_row["name"]) - 1)
                chars[pos] = rng.choice(string.ascii_letters)
                dirty_row["name"] = "".join(chars)

            # 25% chance of missing postal
            if rng.random() < 0.25:
                dirty_row["postal"] = np.nan

            clean_records.append(clean_row)
            dirty_records.append(dirty_row)
            ground_truth.add((l_id, r_id))

        df_left = pd.DataFrame(clean_records, index=[f"SRC_{i:04d}" for i in range(n_entities)])
        df_right = pd.DataFrame(dirty_records, index=[f"TGT_{i:04d}" for i in range(n_entities)])

        # 1. Single-pass on tax_id
        b_tax = ExactBlock(["tax_id"])
        ci_tax = b_tax.block(df_left, df_right)
        recall_tax = len(set(ci_tax.pairs) & ground_truth) / n_entities
        assert recall_tax < 0.75, f"Tax ID single pass recall unexpectedly high: {recall_tax:.2%}"

        # 2. Single-pass on name
        b_name = ExactBlock(["name"])
        ci_name = b_name.block(df_left, df_right)
        recall_name = len(set(ci_name.pairs) & ground_truth) / n_entities
        assert recall_name < 0.75, f"Name single pass recall unexpectedly high: {recall_name:.2%}"

        # 3. Multi-pass combination: Exact tax_id + Exact name + SortedNeighborhood on name + SortedNeighborhood on postal
        mp = MultiPassBlock(
            passes=[
                ExactBlock(["tax_id"]),
                ExactBlock(["name"]),
                SortedNeighborhood(["name"], window=4),
                SortedNeighborhood(["postal"], window=3),
            ],
            budget_cap=25_000,
        )
        ci_mp = mp.block(df_left, df_right)

        # Invariants:
        # a) Candidate pairs bounded by budget cap
        assert len(ci_mp.pairs) <= 25_000, f"Budget cap exceeded: {len(ci_mp.pairs)} > 25,000"

        # b) Candidate recall achieves >= 98.0%
        retrieved_true = len(set(ci_mp.pairs) & ground_truth)
        recall_mp = retrieved_true / n_entities

        assert recall_mp >= 0.98, (
            f"MultiPass candidate recall failed: {recall_mp:.2%} < 98.0% "
            f"({retrieved_true}/{n_entities} true matches found)"
        )


class TestBoundaryAndDegenerateInputs:
    """Stress tests covering boundary inputs: 0-row, 1-row, all-null, duplicate passes, zero-pair passes."""

    def test_zero_row_dataframe_self_and_cross(self) -> None:
        """0-row DataFrames in self-dedupe and cross-file modes."""
        df_empty = pd.DataFrame(columns=["name", "code"])
        df_non_empty = pd.DataFrame({"name": ["Alpha"], "code": ["C1"]}, index=["NE1"])

        mp = MultiPassBlock(passes=[ExactBlock(["name"])], budget_cap=100)

        # Self-dedupe on empty
        ci_self = mp.block(df_empty)
        assert len(ci_self.pairs) == 0
        assert len(ci_self) == 0

        # Cross-file: empty left, non-empty right
        ci_cross_l = mp.block(df_empty, df_non_empty)
        assert len(ci_cross_l.pairs) == 0

        # Cross-file: non-empty left, empty right
        ci_cross_r = mp.block(df_non_empty, df_empty)
        assert len(ci_cross_r.pairs) == 0

    def test_one_row_dataframe_self_and_cross(self) -> None:
        """1-row DataFrames in self-dedupe and cross-file modes."""
        df1 = pd.DataFrame({"name": ["SingleCompany"]}, index=["REC_1"])
        df2 = pd.DataFrame({"name": ["SingleCompany"]}, index=["REC_2"])
        df_other = pd.DataFrame({"name": ["DifferentCompany"]}, index=["REC_3"])

        mp = MultiPassBlock(passes=[ExactBlock(["name"])], budget_cap=50)

        # Self-dedupe on 1-row DataFrame: no self-pairs (i != j)
        ci_self = mp.block(df1)
        assert len(ci_self.pairs) == 0

        # Cross-file matching keys: returns exactly 1 pair ("REC_1", "REC_2")
        ci_cross_match = mp.block(df1, df2)
        assert len(ci_cross_match.pairs) == 1
        assert ci_cross_match.pairs == [("REC_1", "REC_2")]

        # Cross-file mismatching keys: returns 0 pairs
        ci_cross_mismatch = mp.block(df1, df_other)
        assert len(ci_cross_mismatch.pairs) == 0

    def test_all_null_and_blank_keys(self) -> None:
        """Verify that records with all-null, NaN, None, or blank keys produce 0 pairs."""
        df_nulls = pd.DataFrame(
            {
                "k1": [np.nan, None, "", "nan", "none", "   ", "\t"],
                "k2": [None, np.nan, "   ", "NaN", "None", "", " "],
            },
            index=[f"null_{i}" for i in range(7)],
        )

        mp = MultiPassBlock(
            passes=[ExactBlock(["k1"]), ExactBlock(["k2"])],
            budget_cap=100,
        )
        ci = mp.block(df_nulls)
        assert len(ci.pairs) == 0

    def test_duplicate_passes_deduplication(self) -> None:
        """Verify passing the identical blocker pass multiple times produces no duplicates."""
        df = pd.DataFrame({"k": ["A", "A", "B", "B"]}, index=[0, 1, 2, 3])
        p = ExactBlock(["k"])

        # Identical pass repeated 3 times
        mp = MultiPassBlock(passes=[p, p, p], budget_cap=100)
        ci = mp.block(df)

        assert len(ci.pairs) == 4  # (0,1), (1,0), (2,3), (3,2)
        assert len(set(ci.pairs)) == 4

    def test_passes_with_zero_pairs_followed_by_valid_pass(self) -> None:
        """Pass 1 produces 0 pairs (disjoint), Pass 2 produces 2 pairs."""
        df_left = pd.DataFrame({"k1": ["X1", "X2"], "k2": ["Y", "Y"]}, index=["L1", "L2"])
        df_right = pd.DataFrame({"k1": ["Z1", "Z2"], "k2": ["Y", "Y"]}, index=["R1", "R2"])

        p1_empty = ExactBlock(["k1"])  # disjoint keys -> 0 pairs
        p2_valid = ExactBlock(["k2"])  # matching key 'Y' -> 4 pairs

        mp = MultiPassBlock(passes=[p1_empty, p2_valid], budget_cap=50)
        ci = mp.block(df_left, df_right)

        assert len(ci.pairs) == 4
        assert len(set(ci.pairs)) == 4

    def test_all_passes_produce_zero_pairs(self) -> None:
        """All passes produce 0 pairs -> returns empty CandidateIndex."""
        df_left = pd.DataFrame({"k1": ["A", "B"], "k2": ["C", "D"]}, index=["L1", "L2"])
        df_right = pd.DataFrame({"k1": ["W", "X"], "k2": ["Y", "Z"]}, index=["R1", "R2"])

        mp = MultiPassBlock(
            passes=[ExactBlock(["k1"]), ExactBlock(["k2"])],
            budget_cap=50,
        )
        ci = mp.block(df_left, df_right)
        assert len(ci.pairs) == 0
        assert len(ci) == 0
        assert ci.pairs == []

    def test_candidate_index_api_conformance(self) -> None:
        """CandidateIndex conforms to len, in, iter, to_multiindex."""
        df = pd.DataFrame({"k": ["V", "V"]}, index=["A1", "A2"])
        mp = MultiPassBlock(passes=[ExactBlock(["k"])], budget_cap=10)
        ci = mp.block(df)

        assert len(ci) == 2
        assert ("A1", "A2") in ci
        assert ("A2", "A1") in ci
        assert ("A1", "MISSING") not in ci
        assert set(iter(ci)) == {("A1", "A2"), ("A2", "A1")}

        mi = ci.to_multiindex()
        assert isinstance(mi, pd.MultiIndex)
        assert len(mi) == 2
        assert mi.names == ("left", "right")
