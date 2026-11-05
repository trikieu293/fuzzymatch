"""Wrapper around rapidfuzz for string-distance scoring."""

from __future__ import annotations

import re
from typing import Any, cast

import pandas as pd
from rapidfuzz import distance as rfdist
from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

_DIGIT_RUN = re.compile(r"\d+")
_MISSING = {"", "nan", "none"}

# method -> (rapidfuzz scorer, scale factor to reach [0, 1])
_CPDIST_SCORERS = {
    "ratio": (fuzz.ratio, 1 / 100.0),
    "partial_ratio": (fuzz.partial_ratio, 1 / 100.0),
    "token_sort_ratio": (fuzz.token_sort_ratio, 1 / 100.0),
    "token_set_ratio": (fuzz.token_set_ratio, 1 / 100.0),
    "jaro": (rfdist.Jaro.normalized_similarity, 1.0),
    # Explicit prefix_weight locks batch/scalar equivalence if defaults change.
    "jaro_winkler": (
        lambda a, b: rfdist.JaroWinkler.normalized_similarity(a, b, prefix_weight=0.1),
        1.0,
    ),
}


def _is_missing(value: object) -> bool:
    if value is None or bool(pd.isna(cast(Any, value))):
        return True
    return str(value).strip().lower() in _MISSING


def _token_set(value: str) -> set[str]:
    return set(value.lower().split())


def _digit_runs(value: str) -> set[str]:
    return set(_DIGIT_RUN.findall(value))


class RapidFuzzScorer:
    """Wraps rapidfuzz distance functions with a common interface.

    Methods return normalized similarities in [0, 1] (except *_distance methods
    which return integer edit distances).

    Missing-value policy: when either side is empty or a missing marker
    (``""``/``"nan"``/``"none"``, case-insensitive), similarity scorers return
    0.0 so that absent data never masquerades as agreement.
    """

    # Normalized similarity scorers [0, 1]
    def ratio(self, a: str, b: str) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return fuzz.ratio(a, b) / 100.0

    def partial_ratio(self, a: str, b: str) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return fuzz.partial_ratio(a, b) / 100.0

    def token_sort_ratio(self, a: str, b: str) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return fuzz.token_sort_ratio(a, b) / 100.0

    def token_set_ratio(self, a: str, b: str) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return fuzz.token_set_ratio(a, b) / 100.0

    def token_overlap(self, a: str, b: str) -> float:
        """Jaccard containment over whitespace tokens (order/duplicate blind)."""
        if _is_missing(a) or _is_missing(b):
            return 0.0
        tokens_a = _token_set(a)
        tokens_b = _token_set(b)
        shared = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(shared) / len(union)

    def digit_run_similarity(self, a: str, b: str) -> float:
        """Jaccard over digit runs — model-number aware, reorder tolerant.

        Two sides without any digits are neutral (1.0): absence of model
        numbers carries no evidence. Digits on exactly one side score 0.
        Missing markers take precedence over that neutrality.
        """
        if _is_missing(a) or _is_missing(b):
            return 0.0
        runs_a = _digit_runs(a)
        runs_b = _digit_runs(b)
        if not runs_a and not runs_b:
            return 1.0
        if not runs_a or not runs_b:
            return 0.0
        shared = runs_a & runs_b
        union = runs_a | runs_b
        return len(shared) / len(union)

    def jaro(self, a: str, b: str) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return rfdist.Jaro.similarity(a, b)

    def jaro_winkler(self, a: str, b: str, prefix_weight: float = 0.1) -> float:
        if _is_missing(a) or _is_missing(b):
            return 0.0
        return rfdist.JaroWinkler.similarity(a, b, prefix_weight=prefix_weight)

    # Edit-distance scorers (integer; lower is better)
    def levenshtein(self, a: str, b: str, score_cutoff: int | None = None) -> int:
        return rfdist.Levenshtein.distance(a, b, score_cutoff=score_cutoff)

    def damerau_levenshtein(self, a: str, b: str, score_cutoff: int | None = None) -> int:
        return rfdist.DamerauLevenshtein.distance(a, b, score_cutoff=score_cutoff)

    # Batch scoring over pandas Series
    def score_pair(self, method: str, left: pd.Series, right: pd.Series) -> pd.Series:
        """Compute pairwise scores for two equal-length Series using `method`.

        Supported methods: ratio, partial_ratio, token_sort_ratio,
        token_set_ratio, token_overlap, digit_run_similarity, jaro,
        jaro_winkler return similarities in [0, 1]; levenshtein and
        damerau_levenshtein return raw integer edit distances (lower is
        better) — do not threshold them like similarities.

        Fuzz-backed methods run through rapidfuzz's batched ``cpdist`` (C loop)
        with the missing-value mask applied afterwards.
        """
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown scoring method {method!r}. Supported: {SUPPORTED_METHODS}")
        if len(left) != len(right):
            raise ValueError(f"Series length mismatch: {len(left)} vs {len(right)}")

        vectorized = _CPDIST_SCORERS.get(method)
        if vectorized is not None:
            scorer, scale = vectorized
            left_list = ["" if x is None or pd.isna(x) else str(x) for x in left.tolist()]
            right_list = ["" if x is None or pd.isna(x) else str(x) for x in right.tolist()]
            raw = rfprocess.cpdist(
                left_list,
                right_list,
                scorer=cast("Any", scorer),
                processor=None,
            )
            result = pd.Series(raw, index=left.index, dtype=float) * scale
            missing = left.map(_is_missing) | right.map(_is_missing)
            return result.mask(missing.to_numpy(), 0.0)

        fn = getattr(self, method)
        return pd.Series(
            [fn(str(a), str(b)) for a, b in zip(left.tolist(), right.tolist(), strict=True)],
            index=left.index,
        )


# Convenience: list supported scoring methods
SUPPORTED_METHODS: list[str] = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "token_overlap",
    "digit_run_similarity",
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "damerau_levenshtein",
]
