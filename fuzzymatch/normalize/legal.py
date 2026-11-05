"""Corporate legal entity suffix normalization for entity resolution.

Eliminates international corporate legal entity suffixes (Inc, LLC, Ltd, Corp,
GmbH, S.A., Co., AG, BV, NV, SpA, Oy, AB, etc.) to remove legal form noise
while preserving distinctive brand tokens.

Invariants enforced:
1. Non-empty preservation: If stripping suffixes would produce an empty string,
   the original token is preserved.
2. Suffix-position anchoring: Suffixes are only removed from the end of the text.
   Interior brand tokens (e.g. "Company" in "Acme Tools and Company Supplies")
   are preserved.
3. Strict idempotence: f(f(s)) == f(s).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, overload

import pandas as pd

NormalizationStep = Callable[[pd.Series], pd.Series]

DEFAULT_LEGAL_SUFFIXES: tuple[str, ...] = (
    # Compound multi-word suffixes (ordered longest / most specific first)
    r"gmbh\s*&\s*co\.?\s*kgaa",
    r"gmbh\s*&\s*co\.?\s*kg",
    r"co\.?,?\s*ltd\.?",
    r"pty\.?,?\s*ltd\.?",
    r"pte\.?,?\s*ltd\.?",
    r"s\.a\.?\s*de\s*c\.v\.?",
    r"sa\s+de\s+cv",
    r"sp\.?\s*z\s*o\.?\s*o\.?",
    r"sp\s+z\s+oo",
    r"ltd\.?\s*liability\s*co\.?",
    r"limited\s+liability\s+company",
    r"limited\s+liability\s+partnership",
    r"public\s+limited\s+company",
    r"limited\s+partnership",
    r"ug\s*\(?\s*haftungsbeschr[aä]nkt\s*\)?",
    r"societ[aà]\s+(?:per\s+azioni|a\s+responsabilit[aà]\s+limitata)",
    r"sociedad\s+an[oó]nima\s+de\s+capital\s+variable",
    r"sociedad\s+an[oó]nima",
    r"soci[eé]t[eé]\s+anonyme",
    r"soci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e",
    r"besloten\s+vennootschap",
    r"naamloze\s+vennootschap",
    # Single-word or standard abbreviations
    r"corporation",
    r"incorporated",
    r"incorporation",
    r"company",
    r"limited",
    r"aktiengesellschaft",
    r"kommanditgesellschaft",
    r"gesellschaft\s+mit\s+beschr[aä]nkter\s+haftung",
    r"corp\.?",
    r"inc\.?",
    r"co\.?",
    r"ltd\.?",
    r"l\.t\.d\.?",
    r"llc",
    r"l\.l\.c\.?",
    r"plc",
    r"p\.l\.c\.?",
    r"llp",
    r"l\.l\.p\.?",
    r"lp",
    r"l\.p\.?",
    r"gmbh",
    r"g\.m\.b\.h\.?",
    r"gesmbh",
    r"kgaa",
    r"ag",
    r"a\.g\.?",
    r"kg",
    r"k\.g\.?",
    r"ug",
    r"e\.v\.?",
    r"ev",
    r"s\.a\.r\.l\.?",
    r"sarl",
    r"s\.a\.s\.?",
    r"sas",
    r"s\.a\.?",
    r"sa",
    r"s\.l\.?",
    r"sl",
    r"s\.r\.l\.?",
    r"srl",
    r"b\.v\.?",
    r"bv",
    r"n\.v\.?",
    r"nv",
    r"s\.p\.a\.?",
    r"ltda\.?",
    r"oyj",
    r"oy",
    r"ab",
    r"k\.k\.?",
    r"kk",
)

# Combined terminal suffix regex pattern.
# Requires word boundary or preceding punctuation/whitespace.
_SUFFIX_PATTERN = "|".join(DEFAULT_LEGAL_SUFFIXES)
_TERMINAL_SUFFIX_RE = re.compile(rf"(?i)(?:[\s,./\-_&]+|\b)(?:{_SUFFIX_PATTERN})[\s.]*$")
_TRAILING_PUNCT_RE = re.compile(r"[\s,./\-_&]+$")


def _strip_single_string(text: str) -> str:
    """Strip terminal corporate legal suffixes from a single text string."""
    if not text or not text.strip():
        return text

    # Check if text contains word characters; if purely punctuation/symbols, return unchanged
    if not re.search(r"\w", text):
        return text

    current = text.strip()
    max_passes = 5

    for _ in range(max_passes):
        match = _TERMINAL_SUFFIX_RE.search(current)
        if match is None:
            break

        prefix = current[: match.start()]
        # Strip trailing punctuation from remaining prefix
        prefix_clean = _TRAILING_PUNCT_RE.sub("", prefix).strip()

        # Non-empty preservation: if stripping would leave no word characters,
        # preserve the current token instead of collapsing to empty.
        if not re.search(r"\w", prefix_clean):
            break

        current = prefix_clean

    return current


@overload
def strip_legal_suffixes(text: str) -> str: ...


@overload
def strip_legal_suffixes(text: pd.Series) -> pd.Series: ...


@overload
def strip_legal_suffixes(text: None = None) -> NormalizationStep: ...


def strip_legal_suffixes(
    text: str | pd.Series | None = None,
) -> str | pd.Series | NormalizationStep:
    """Strip international corporate legal suffixes from strings or Series.

    When invoked with:
    - a string: returns the string with terminal legal suffixes stripped.
    - a pandas Series: returns a new Series with legal suffixes stripped.
    - None (or no arguments): returns a NormalizationStep callable for use in Normalizer.

    Invariants:
    - Non-empty preservation: Suffix-only strings (e.g. "Inc", "LLC") are preserved.
    - Idempotent: strip_legal_suffixes(strip_legal_suffixes(s)) == strip_legal_suffixes(s).
    """
    if text is None:

        def step(series: pd.Series) -> pd.Series:
            return series.map(
                lambda s: (
                    _strip_single_string(s)
                    if isinstance(s, str)
                    else ("" if pd.isna(s) else _strip_single_string(str(s)))
                )
            )

        return step

    if isinstance(text, pd.Series):
        return text.map(
            lambda s: (
                _strip_single_string(s)
                if isinstance(s, str)
                else ("" if pd.isna(s) else _strip_single_string(str(s)))
            )
        )

    if isinstance(text, str):
        return _strip_single_string(text)

    # Fallback for other scalar inputs
    val: Any = text
    if pd.isna(val):
        return ""
    return _strip_single_string(str(val))
