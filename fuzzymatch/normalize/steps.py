"""Normalization steps for text fuzzy matching.

Each step is a callable taking a pandas Series and returning a transformed Series.
Compose them into a Normalizer pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

import pandas as pd

from .legal import DEFAULT_LEGAL_SUFFIXES, strip_legal_suffixes

NormalizationStep = Callable[[pd.Series], pd.Series]

UnicodeForm = Literal["NFC", "NFKC", "NFD", "NFKD"]

__all__ = [
    "DEFAULT_LEGAL_SUFFIXES",
    "NormalizationStep",
    "UnicodeForm",
    "accent_fold",
    "casefold",
    "replace_words",
    "strip_legal_suffixes",
    "strip_punctuation",
    "strip_whitespace",
    "unicode_normalize",
]


def unicode_normalize(form: UnicodeForm) -> NormalizationStep:
    """Apply Unicode normalization (NFC/NFD/NFKC/NFKD).

    NFC is the safe default for general text. NFKC additionally folds compatibility
    characters (ligatures, full-width, superscripts) but loses information, so
    it should only be used for fuzzy keying, not for display.
    """

    def step(series: pd.Series) -> pd.Series:
        return series.str.normalize(form)

    return step


def casefold() -> NormalizationStep:
    """Case-fold using str.casefold() — Unicode-aware, stronger than .lower()."""
    return lambda series: series.str.casefold()


def strip_whitespace() -> NormalizationStep:
    """Collapse internal whitespace and strip leading/trailing whitespace."""
    return lambda series: series.str.replace(r"\s+", " ", regex=True).str.strip()


def strip_punctuation() -> NormalizationStep:
    """Remove ASCII punctuation characters."""
    return lambda series: series.str.replace(
        r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", "", regex=True
    )


def replace_words(mapping: dict[str, str]) -> NormalizationStep:
    """Whole-token replacement for domain synonym lexicons.

    Applied AFTER casefolding: keys and values are matched/compared in
    lowercase. Multi-word keys are supported (greedy longest-match first).
    """

    if not mapping:
        raise ValueError("replace_words requires a non-empty mapping")
    if any(not key.strip() for key in mapping):
        raise ValueError("replace_words mapping keys must be non-empty")

    ordered = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)

    def step(series: pd.Series) -> pd.Series:
        lowered = series.str.lower()
        for key, value in ordered:
            pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
            lowered = lowered.str.replace(pattern, value, regex=True)
        return lowered

    return step


def accent_fold(keep_marks: bool = False) -> NormalizationStep:
    """Strip diacritical marks via NFKD + combining-character filter."""

    def step(series: pd.Series) -> pd.Series:
        import unicodedata

        def _fold(s: str) -> str:
            if keep_marks:
                return s
            nfkd = unicodedata.normalize("NFKD", s)
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        return series.apply(_fold)

    return step
