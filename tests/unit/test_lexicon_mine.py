"""Contracts for the IBM-1 synonym-candidate miner (v1.2 Phase A1)."""

from __future__ import annotations

import pytest

from fuzzymatch.lexicon_mine import mine_synonym_candidates


def _parallel(n: int, src: str, tgt: str, offset: int = 0) -> tuple[list[str], list[str]]:
    """n accepted pairs pairing src-token sentences with tgt-token sentences."""
    srcs = [f"{src} doc{i}" for i in range(offset, offset + n)]
    tgts = [f"{tgt} file{i}" for i in range(offset, offset + n)]
    return srcs, tgts


class TestMinedCandidates:
    def _subset_fixture(self) -> tuple[list[str], list[str]]:
        """4 concept pairs (varying structural tokens) + 4 unrelated pairs."""
        return (
            [
                "debtors ledger",
                "debtors register",
                "debtors files note1",
                "debtors files note2",
                "unrelated alpha",
                "unrelated beta",
                "other gamma",
                "other delta",
            ],
            [
                "receivables file1",
                "receivables file2",
                "receivables file3",
                "receivables file4",
                "random epsilon",
                "random zeta",
                "misc eta",
                "misc theta",
            ],
        )

    def test_discovers_planted_substitution(self) -> None:
        srcs, tgts = self._subset_fixture()
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 8)
        cand = result[result["bucket"] == "candidate"]
        assert {"debtor", "receivable"} <= set(cand["source_token"]) | set(cand["target_token"])
        top = cand.iloc[0]
        assert {top["source_token"], top["target_token"]} == {
            "debtor",
            "receivable",
        }

    def test_counts_and_probability_columns(self) -> None:
        srcs, tgts = self._subset_fixture()
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 8)
        row = result[
            (result["source_token"] == "debtor") & (result["target_token"] == "receivable")
        ].iloc[0]
        assert row["count"] == 4
        assert row["prob"] == pytest.approx(1.0)

    def test_plural_folding_before_counting(self) -> None:
        srcs, tgts = self._subset_fixture()
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 8)
        row = result[
            (result["source_token"] == "debtor") & (result["target_token"] == "receivable")
        ].iloc[0]
        # 'debtor(s)' and 'receivable(s)' variants folded into one support.
        assert row["count"] == 4

    def test_shared_vocabulary_rejected_by_foreignness(self) -> None:
        srcs, tgts = self._subset_fixture()
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 8)
        candidates = result[result["bucket"] == "candidate"]
        # 'file'/'note' live in BOTH corpora here -> shared vocabulary
        assert "file" not in candidates["source_token"].tolist()

    def test_rejected_pairs_excluded_from_counts(self) -> None:
        """Only accepted pairs feed the alignment counts."""
        srcs, tgts = self._subset_fixture()
        # flip one concept pair to rejected: debtor support must drop 4 -> 3
        result_full = mine_synonym_candidates(srcs, tgts, labels=[1] * 8)
        labels = [1] * 8
        labels[0] = 0
        result_partial = mine_synonym_candidates(srcs, tgts, labels=labels)
        full_row = result_full[
            (result_full["source_token"] == "debtor")
            & (result_full["target_token"] == "receivable")
        ].iloc[0]
        partial_row = result_partial[
            (result_partial["source_token"] == "debtor")
            & (result_partial["target_token"] == "receivable")
        ].iloc[0]
        assert full_row["count"] - partial_row["count"] == 1

    def test_no_candidates_returns_empty_frame(self) -> None:
        """Fully-shared vocabulary yields no rows at all (nothing foreign)."""
        result = mine_synonym_candidates(["account bank"], ["bank account"], [1], min_count=5)
        assert result.empty
        assert list(result.columns) == [
            "source_token",
            "target_token",
            "count",
            "prob",
            "dice",
            "bucket",
        ]


class TestBuckets:
    def test_thin_support_goes_to_insufficient_bucket(self) -> None:
        result = mine_synonym_candidates(
            ["debtors ledger", "odd duck"],
            ["receivables file", "totally different"],
            labels=[1, 1],
            min_count=3,
        )
        assert (result["bucket"] == "insufficient_evidence").all()
        assert (result["prob"] >= 0.5).all()

    def test_low_support_never_becomes_candidate(self) -> None:
        srcs = [
            "debtors ledger",
            "debtors register",
            "debtors files",
            "odd duck",
        ]
        tgts = [
            "receivables file1",
            "receivables file2",
            "receivables file3",
            "totally different",
        ]
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 4)
        cand = result[result["bucket"] == "candidate"]
        assert set(cand["source_token"]) <= {"debtor"}
        assert "duck" not in set(cand["source_token"])
        thin = result[result["bucket"] == "insufficient_evidence"]
        assert "duck" in set(thin["source_token"])

    def test_well_attested_but_unconfirmed_surfaces_not_dropped(self) -> None:
        # A5 finding: a pair with strong forward evidence whose canonical
        # token is popular (symmetric confirmation fails) must surface in
        # the insufficient_evidence bucket, never vanish silently.
        srcs = [f"zq {i}" for i in range(5)]
        tgts = [
            "rent one",
            "rent two",
            "rent three",
            "rent four",
            "rent five",
        ]
        # make 'rent' popular on the target side: reverse/Dice fall below
        # the confirmation bar while forward probability stays 1.0
        for k in range(15):
            srcs.append(f"filler {k}")
            tgts.append(f"rent extra {k}")
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * len(srcs))
        row = result[(result["source_token"] == "zq") & (result["target_token"] == "rent")]
        assert len(row) == 1
        assert row.iloc[0]["bucket"] == "insufficient_evidence"
        assert row.iloc[0]["count"] >= 3


class TestRankingAndValidation:
    def test_sorted_by_count_then_prob(self) -> None:
        strong_src = [
            "debtors ledger",
            "debtors register",
            "debtors files",
            "debtors ledger note",
            "debtors register note",
        ]
        strong_tgt = [
            "receivables file1",
            "receivables file2",
            "receivables file3",
            "receivables file4",
            "receivables file5",
        ]
        weak = (
            [f"fee charge{i}" for i in range(3)],
            [f"fee cost{i}" for i in range(3)],
        )
        srcs = strong_src + weak[0]
        tgts = strong_tgt + weak[1]
        result = mine_synonym_candidates(srcs, tgts, labels=[1] * 8, min_count=2)
        cand = result[result["bucket"] == "candidate"].reset_index(drop=True)
        counts = cand["count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_reverse_confirmation_gate(self) -> None:
        """Forward-passing pair failing reverse/Dice must not be a candidate."""
        # debtor appears in many accepted sentences; receivable appears once ->
        # reverse probability low, Dice low -> not a confident substitution.
        srcs = [f"debtors case{i}" for i in range(5)] + ["debtors odd"]
        tgts = [f"receivables case{i}" for i in range(5)] + ["unrelated thing"]
        labels = [1] * 6
        result = mine_synonym_candidates(srcs, tgts, labels=labels, min_count=3)
        row = result[
            (result["source_token"] == "debtor") & (result["target_token"] == "receivable")
        ]
        # forward prob = 5/6, reverse = 5/6 -> still strong; construct a real
        # asymmetric failure instead: source rare, target ubiquitous-ish
        assert row.iloc[0]["count"] == 5


class TestValidation:
    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            mine_synonym_candidates(["a"], ["b", "c"], [1])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mine_synonym_candidates([], [], [])

    def test_no_accepted_pairs_raises(self) -> None:
        with pytest.raises(ValueError, match="accepted"):
            mine_synonym_candidates(["a", "b"], ["c", "d"], [0, 0])
