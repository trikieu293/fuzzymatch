"""Tests for the labeled-pair split utility (supervised evaluation protocol)."""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.model_selection import (
    select_operating_threshold,
    train_valid_test_split,
)


@pytest.fixture()
def labeled() -> pd.DataFrame:
    """60 labeled pairs with a realistic class imbalance."""
    rng = pd.Series(range(60), name="pair_id").to_frame()
    rng["label"] = [1] * 12 + [0] * 48
    return rng


class TestTrainValidTestSplit:
    def test_returns_three_frames_in_declared_proportions(self) -> None:
        train, valid, test = train_valid_test_split(_labeled := _labeled_df(100))
        assert round(len(train)) == 60
        assert round(len(valid)) == 20
        assert round(len(test)) == 20

    def test_partitions_are_disjoint_and_exhaustive(self) -> None:
        frame = _labeled_df(50)
        parts = train_valid_test_split(frame)
        seen: set[int] = set()
        for part in parts:
            ids = set(part["pair_id"])
            assert not (ids & seen), "partitions must be disjoint"
            seen |= ids
        assert seen == set(frame["pair_id"]), "union must equal the input"

    def test_deterministic_for_fixed_seed(self) -> None:
        frame = _labeled_df(80)
        first = train_valid_test_split(frame, seed=7)
        second = train_valid_test_split(frame, seed=7)
        for a, b in zip(first, second, strict=True):
            pd.testing.assert_frame_equal(a, b)

    def test_different_seed_changes_composition(self) -> None:
        frame = _labeled_df(120)
        first = train_valid_test_split(frame, seed=1)
        second = train_valid_test_split(frame, seed=2)
        assert not first[0]["pair_id"].equals(second[0]["pair_id"])

    def test_both_classes_survive_in_every_split(self) -> None:
        parts = train_valid_test_split(_labeled_df(90))
        for part in parts:
            assert set(part["label"].unique()) == {0, 1}

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            train_valid_test_split(
                pd.DataFrame({"pair_id": pd.Series(dtype=int), "label": pd.Series(dtype=int)})
            )

    def test_row_content_preserved(self) -> None:
        frame = _labeled_df(30)
        combined = sorted(
            pair_id for part in train_valid_test_split(frame) for pair_id in part["pair_id"]
        )
        assert combined == sorted(frame["pair_id"])

    @pytest.mark.parametrize(
        "ratios",
        [(0.5, 0.5), (0.6, 0.2, 0.2, 0.0), (-0.6, 0.8, 0.8), (), (0.5, 0.2, 0.2)],
    )
    def test_invalid_ratios_raise(self, ratios: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="ratio"):
            train_valid_test_split(_labeled_df(10), ratios=ratios)


def _labeled_df(n: int) -> pd.DataFrame:
    """n pairs, ~20% positives, unique pair_id column."""
    frame = pd.DataFrame({"pair_id": range(n)})
    frame["label"] = (frame.index % 5 == 0).astype(int)
    return frame


class TestSelectOperatingThreshold:
    """Validation-partition sweep for the supervised decision cut-off."""

    def test_picks_threshold_maximising_f1(self) -> None:
        # Clean separation at 0.6: positives ~0.9, negatives ~0.1.
        truth = [1, 1, 1, 0, 0, 0]
        scores = [0.90, 0.85, 0.95, 0.10, 0.15, 0.05]
        threshold = select_operating_threshold(scores, truth)
        assert threshold <= 0.85
        assert threshold > 0.15

    def test_perfectly_separable_gives_high_f1_region(self) -> None:
        truth = [0, 0, 1, 1]
        scores = [0.05, 0.2, 0.8, 0.95]
        threshold = select_operating_threshold(scores, truth)
        assert 0.21 <= threshold <= 0.79

    def test_deterministic(self) -> None:
        truth = [1, 0, 1, 0]
        scores = [0.7, 0.72, 0.3, 0.31]
        first = select_operating_threshold(scores, truth)
        second = select_operating_threshold(scores, truth)
        assert first == second

    def test_all_same_class_returns_lowest_cut(self) -> None:
        # No informative split; must not crash and must stay in (0,1).
        threshold = select_operating_threshold([0.4, 0.5], [1, 1])
        assert 0.0 < threshold < 1.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            select_operating_threshold([0.5, 0.6], [1])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            select_operating_threshold([], [])
