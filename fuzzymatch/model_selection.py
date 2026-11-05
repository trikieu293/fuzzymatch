"""Deterministic split utilities for supervised evaluation of matchers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_SPLIT_LABEL = "labeled pair split"


def select_operating_threshold(
    scores: Iterable[float],
    truth: Iterable[int],
    grid: int = 100,
) -> float:
    """Pick the score cut-off maximising F1 on a validation partition.

    Sweeps ``grid`` evenly spaced cut-offs in (0, 1) and returns the first
    one achieving the best validation F1 (deterministic tie-breaking).
    Use ONLY validation data — never the test partition.
    """
    s = [float(x) for x in scores]
    y = [int(t) for t in truth]
    if len(s) != len(y):
        raise ValueError(f"scores/truth length mismatch: {len(s)} vs {len(y)}")
    if not s:
        raise ValueError("cannot tune threshold on an empty set")

    def f1_at(cut: float) -> float:
        tp = sum(1 for t, v in zip(y, s, strict=True) if t == 1 and v >= cut)
        fp = sum(1 for t, v in zip(y, s, strict=True) if t == 0 and v >= cut)
        fn = sum(1 for t, v in zip(y, s, strict=True) if t == 1 and v < cut)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    best_cut, best_f1 = 0.5, -1.0
    for i in range(1, grid):
        cut = i / grid
        current = f1_at(cut)
        if current > best_f1:
            best_cut, best_f1 = cut, current
    return float(best_cut)


def train_valid_test_split(
    labeled: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle ``labeled`` and partition it into train/valid/test frames.

    Args:
        labeled: Annotated pairs; one row per labeled pair.
        ratios: Fractions summing to 1.0 (within float tolerance).
        seed: Seed for the deterministic shuffle.

    Returns:
        Three disjoint frames whose union equals the input, with row content
        preserved and sizes following ``ratios`` as closely as integer
        rounding allows.

    Raises:
        ValueError: If ``ratios`` is not three non-negative fractions that
            sum to 1.0 (tolerance 1e-9), or if ``labeled`` is empty.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise ValueError(
            f"{_SPLIT_LABEL} ratios must be three non-negative fractions, got {ratios!r}"
        )
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"{_SPLIT_LABEL} ratios must sum to 1.0, got {sum(ratios)!r}")
    if len(labeled) == 0:
        raise ValueError("cannot split an empty labeled set")

    shuffled = labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    first_cut = round(n * ratios[0])
    second_cut = first_cut + round(n * ratios[1])
    return (
        shuffled.iloc[:first_cut],
        shuffled.iloc[first_cut:second_cut],
        shuffled.iloc[second_cut:],
    )
