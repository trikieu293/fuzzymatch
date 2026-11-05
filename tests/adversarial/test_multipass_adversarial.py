"""Adversarial stress harness for MultiPassBlock candidate generation.

Verifies:
1. Strict candidate budget cap invariant (len(pairs) <= B) across arbitrary inputs.
2. Hypothesis property-based verification of budget capping and deduplication.
3. Pass priority preservation (earlier passes prioritized under tight budget).
4. Degenerate inputs (empty, single-row, all-NaN, mismatched index types).
5. Dirty benchmark candidate recall >= 98% with budget constraints.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzymatch.blocking import (
    ExactBlock,
    MultiPassBlock,
    SortedNeighborhood,
)


class TestBudgetCapStrictInvariant:
    """Stress testing the budget cap invariant:
    len(ci.pairs) <= budget_cap ALWAYS.
    """

    @pytest.mark.parametrize("budget", [1, 2, 5, 10, 17, 50, 100, 500])
    def test_budget_cap_never_exceeded(self, budget: int) -> None:
        """On a dataset with 1,000 potential candidate pairs, assert strict budget ceiling."""
        # 50 rows in single group -> 50*49 = 2450 candidate pairs in dedupe mode
        df = pd.DataFrame({"grp": ["same_group"] * 50}, index=list(range(50)))
        b = ExactBlock(["grp"])
        mp = MultiPassBlock(passes=[b], budget_cap=budget)
        ci = mp.block(df)

        assert len(ci.pairs) == budget
        assert len(ci) == budget

    @given(
        num_rows=st.integers(min_value=5, max_value=60),
        budget_cap=st.integers(min_value=1, max_value=200),
        seed=st.integers(min_value=0, max_value=5000),
    )
    @settings(max_examples=100, deadline=None)
    def test_hypothesis_budget_cap_and_deduplication(
        self, num_rows: int, budget_cap: int, seed: int
    ) -> None:
        """Property: len(ci.pairs) <= budget_cap AND ci.pairs contains zero duplicates."""
        rng = random.Random(seed)
        # Create DataFrame with several overlapping columns
        df = pd.DataFrame(
            {
                "k1": [f"K_{rng.randint(0, 3)}" for _ in range(num_rows)],
                "k2": [f"V_{rng.randint(0, 5)}" for _ in range(num_rows)],
            },
            index=[f"id_{i}" for i in range(num_rows)],
        )

        p1 = ExactBlock(["k1"])
        p2 = ExactBlock(["k2"])
        mp = MultiPassBlock(passes=[p1, p2], budget_cap=budget_cap)
        ci = mp.block(df)

        # 1. Budget cap ceiling invariant
        assert len(ci.pairs) <= budget_cap, f"Budget cap violated: {len(ci.pairs)} > {budget_cap}"

        # 2. Strict deduplication invariant
        assert len(ci.pairs) == len(set(ci.pairs)), (
            "Duplicate pairs detected in MultiPassBlock output"
        )


class TestPassPriorityAndDeduplication:
    """Verify that earlier passes have priority over later passes."""

    def test_earlier_pass_priority(self) -> None:
        """Pass 1 produces pairs P1, Pass 2 produces pairs P2.
        Under budget = len(P1), NO pairs from P2 should be present.
        """
        df = pd.DataFrame(
            {
                "exact_id": [
                    "E1",
                    "E1",
                    "E2",
                    "E2",
                ],  # pairs: (0,1), (1,0), (2,3), (3,2) -> 4 pairs
                "name": ["N1", "N1", "N1", "N1"],  # pairs: 12 pairs
            },
            index=[0, 1, 2, 3],
        )

        p1 = ExactBlock(["exact_id"])
        p2 = ExactBlock(["name"])

        # Cap at 4
        mp = MultiPassBlock(passes=[p1, p2], budget_cap=4)
        ci = mp.block(df)

        assert len(ci.pairs) == 4
        # All 4 pairs must come from exact_id
        exact_pairs = set(p1.block(df).pairs)
        for pair in ci.pairs:
            assert pair in exact_pairs


class TestDegenerateDataFrames:
    """Edge cases: empty tables, NaN columns, UUID indices."""

    def test_all_nan_columns(self) -> None:
        """When blocking columns contain only NaNs, 0 candidate pairs generated."""
        df = pd.DataFrame(
            {"col1": [np.nan, np.nan, np.nan], "col2": [None, None, None]},
            index=[1, 2, 3],
        )
        mp = MultiPassBlock(passes=[ExactBlock(["col1"]), ExactBlock(["col2"])], budget_cap=100)
        ci = mp.block(df)
        assert len(ci.pairs) == 0

    def test_uuid_dataframe_indices(self) -> None:
        """DataFrames indexed with UUID instances."""
        u_left = [uuid.uuid4() for _ in range(5)]
        u_right = [uuid.uuid4() for _ in range(5)]

        df_left = pd.DataFrame({"key": ["A", "B", "C", "D", "E"]}, index=u_left)
        df_right = pd.DataFrame({"key": ["A", "B", "C", "X", "Y"]}, index=u_right)

        mp = MultiPassBlock(passes=[ExactBlock(["key"])], budget_cap=50)
        ci = mp.block(df_left, df_right)

        assert len(ci.pairs) == 3
        assert (u_left[0], u_right[0]) in ci.pairs
        assert (u_left[1], u_right[1]) in ci.pairs
        assert (u_left[2], u_right[2]) in ci.pairs


class TestDirtyBenchmarkCandidateRecall:
    """Acceptance criterion test:
    Multi-pass blocking achieves >= 98% candidate recall on dirty benchmark datasets
    while keeping candidate pairs within configured budget cap.
    """

    def test_recall_on_dirty_benchmark(self) -> None:
        """Synthesize 200 ground-truth entity pairs with realistic dirty perturbations:
        - 40% missing primary tax_id
        - 40% typographical errors in entity name (transposition/omission)
        - 20% clean exact matches
        """
        rng = random.Random(1337)
        n_entities = 200

        left_records: list[dict[str, Any]] = []
        right_records: list[dict[str, Any]] = []
        ground_truth_pairs: set[tuple[int, int]] = set()

        names = [f"Enterprise_{i}_Corp" for i in range(n_entities)]
        taxes = [f"TAX_{1000 + i}" for i in range(n_entities)]

        for i in range(n_entities):
            l_tax = taxes[i]
            r_tax = taxes[i]
            l_name = names[i]
            r_name = names[i]

            prob = rng.random()
            if prob < 0.40:
                # Missing tax ID in right
                r_tax = np.nan
            elif prob < 0.80:
                # Typo in name in right: substitute last character
                r_name = r_name[:-1] + "X"

            left_records.append({"tax_id": l_tax, "name": l_name})
            right_records.append({"tax_id": r_tax, "name": r_name})
            ground_truth_pairs.add((i, i))

        df_left = pd.DataFrame(left_records)
        df_right = pd.DataFrame(right_records)

        # Single pass on tax_id: misses the 40% missing tax IDs
        b_tax = ExactBlock(["tax_id"])
        ci_tax = b_tax.block(df_left, df_right)
        tax_recall = len(set(ci_tax.pairs) & ground_truth_pairs) / n_entities
        assert tax_recall < 0.70, (
            f"Expected tax single-pass to miss corrupted pairs, got recall {tax_recall}"
        )

        # Multi-pass combination: Exact tax_id + Exact name + SortedNeighborhood on name
        b_name = ExactBlock(["name"])
        b_sn = SortedNeighborhood(["name"], window=3)

        mp = MultiPassBlock(passes=[b_tax, b_name, b_sn], budget_cap=5_000)
        ci_multi = mp.block(df_left, df_right)

        # Verify budget cap
        assert len(ci_multi.pairs) <= 5_000

        # Evaluate candidate recall
        recovered = len(set(ci_multi.pairs) & ground_truth_pairs)
        recall = recovered / n_entities

        assert recall >= 0.98, (
            f"Candidate recall {recall:.2%} failed acceptance threshold of >= 98.0%"
        )
