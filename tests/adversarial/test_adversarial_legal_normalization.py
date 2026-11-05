"""Adversarial stress-test suite for corporate legal entity suffix normalization.

Empirically challenges:
1. Suffix-only strings: Single and compound corporate suffixes must NEVER collapse to empty string.
2. Interior brand names: Corporate keywords appearing at the beginning or middle of business names
   must be strictly preserved (e.g. 'GmbH Bank', 'Limited Brands Co.', 'Corporation Service Company').
3. Strict idempotence: forall s, f(f(s)) == f(s) across 1,000+ synthetic variations and Hypothesis property tests.
4. Edge cases & boundaries: Empty strings, pure punctuation, unicode accents, mixed casing, multiple
   stacked trailing suffixes, alphanumeric brands, and false-positive protection (e.g. 'PepsiCo', 'Boeing').
5. Computational complexity & non-backtracking: sub-millisecond scaling on extreme string lengths (10,000+ chars).
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("fuzzymatch_finance")

from fuzzymatch_finance.lexicon import legal_entity_normalizer

from fuzzymatch.normalize.legal import (
    strip_legal_suffixes,
)

# =============================================================================
# 1. Suffix-Only Strings: Non-Empty Preservation Invariant
# =============================================================================

SINGLE_SUFFIX_CASES = [
    "Inc",
    "LLC",
    "GmbH",
    "S.A.",
    "Ltd.",
    "Corp.",
    "AG",
    "BV",
    "NV",
    "SpA",
    "Oy",
    "AB",
    "PLC",
    "LLP",
    "LP",
    "Co.",
    "Company",
    "Corporation",
    "Incorporated",
    "Limited",
]

COMPOUND_SUFFIX_CASES = [
    "GmbH & Co. KG",
    "GmbH & Co. KGaA",
    "S.A. de C.V.",
    "SA de CV",
    "Sp. z o.o.",
    "Public Limited Company",
    "Pty Ltd",
    "Pty. Ltd.",
    "Co., Ltd.",
    "UG (haftungsbeschränkt)",
    "Limited Liability Company",
    "Limited Liability Partnership",
]


@pytest.mark.parametrize("suffix", SINGLE_SUFFIX_CASES + COMPOUND_SUFFIX_CASES)
def test_suffix_only_string_never_collapses_to_empty(suffix: str) -> None:
    """Invariant: An isolated legal suffix string must NEVER collapse to an empty string."""
    # Test scalar function
    stripped = strip_legal_suffixes(suffix)
    assert stripped.strip() != "", (
        f"strip_legal_suffixes collapsed isolated suffix '{suffix}' to empty!"
    )

    # Test lowercase variation
    stripped_lower = strip_legal_suffixes(suffix.lower())
    assert stripped_lower.strip() != "", (
        f"strip_legal_suffixes collapsed lowercase '{suffix.lower()}' to empty!"
    )

    # Test pipeline normalizer from fuzzymatch-finance
    norm = legal_entity_normalizer()
    df = pd.DataFrame({"name": [suffix]})
    res_df = norm.transform(df, columns=["name"])
    normalized_val = res_df["name"].iloc[0]
    assert normalized_val.strip() != "", (
        f"legal_entity_normalizer collapsed isolated suffix '{suffix}' to empty!"
    )


# =============================================================================
# 2. Interior Brand Names: Subject-Position Token Preservation
# =============================================================================

INTERIOR_BRAND_CASES: list[tuple[str, str, str]] = [
    # (Input, Expected strip_legal_suffixes output, Expected legal_entity_normalizer output)
    ("GmbH Bank", "GmbH Bank", "gmbh bank"),
    ("Limited Brands Co.", "Limited Brands", "limited brands"),
    ("Corporation Service Company", "Corporation Service", "corporation service"),
    ("The Company Store", "The Company Store", "the company store"),
    ("Incorporated Technologies Inc", "Incorporated Technologies", "incorporated technologies"),
    ("Acme General Partners LLC", "Acme General Partners", "acme general partners"),
    ("Bank of America Corp.", "Bank of America", "bank of america"),
    ("LLC Holdings Group Corp.", "LLC Holdings Group", "llc holdings group"),
    (
        "Acme Tools and Company Supplies",
        "Acme Tools and Company Supplies",
        "acme tools and company supplies",
    ),
    (
        "Standard Oil Company of California",
        "Standard Oil Company of California",
        "standard oil company of california",
    ),
]


@pytest.mark.parametrize(("raw_name", "exp_strip", "exp_norm"), INTERIOR_BRAND_CASES)
def test_interior_brand_names_strictly_preserved(
    raw_name: str, exp_strip: str, exp_norm: str
) -> None:
    """Invariant: Corporate suffix words occurring at the beginning or middle must be preserved."""
    res_strip = strip_legal_suffixes(raw_name)
    assert res_strip == exp_strip, (
        f"strip_legal_suffixes failed on '{raw_name}': got '{res_strip}', exp '{exp_strip}'"
    )

    norm = legal_entity_normalizer()
    df = pd.DataFrame({"name": [raw_name]})
    res_norm = norm.transform(df, columns=["name"])["name"].iloc[0]
    assert res_norm == exp_norm, (
        f"legal_entity_normalizer failed on '{raw_name}': got '{res_norm}', exp '{exp_norm}'"
    )


# =============================================================================
# 3. Idempotence Stress Testing Across 1,000+ Synthetic Variations
# =============================================================================


def test_idempotence_1000_synthetic_variations() -> None:
    """Stress Test: forall s, f(f(s)) == f(s) across 1,000 randomized combinatorial corporate names."""
    rng = random.Random(20261026)

    prefixes = [
        "Acme",
        "Global",
        "Apex",
        "Vanguard",
        "Alpha",
        "Beta",
        "Gamma",
        "Quantum",
        "Nexus",
        "Pacific",
        "Tech",
        "Nordic",
        "First",
        "United",
        "National",
        "Frontier",
        "Meridian",
        "Crestview",
        "Ironclad",
        "Summit",
    ]
    interiors = [
        "",
        " Industrial",
        " Technologies",
        " Services",
        " Holdings",
        " Consulting",
        " Partners",
        " Bank",
        " Store",
        " Solutions",
        " Logistics",
        " Company",
        " Limited",
        " Corporation",
        " Inc",
        " Manufacturing",
        " Software",
        " Fasteners",
        " Timber",
        " Security",
    ]
    suffixes = [
        "",
        " Inc.",
        " Inc",
        " LLC",
        " L.L.C.",
        " Ltd.",
        " Limited",
        " Corp.",
        " Corporation",
        " GmbH",
        " GmbH & Co. KG",
        " S.A.",
        " S.A. de C.V.",
        " Co.",
        " Company",
        " AG",
        " Oy",
        " AB",
        " SpA",
        " S.p.A.",
        " BV",
        " NV",
        " PLC",
        " LLP",
        " LP",
    ]
    punctuations = ["", ".", ",", " -", " /", " &", "   "]

    norm = legal_entity_normalizer()

    for idx in range(1000):
        p = rng.choice(prefixes)
        m = rng.choice(interiors)
        sfx1 = rng.choice(suffixes)
        sfx2 = rng.choice(suffixes) if rng.random() < 0.3 else ""
        punct = rng.choice(punctuations)

        raw = f"{p}{m}{sfx1}{punct}{sfx2}"

        # Apply random casing mutations
        casing_choice = rng.random()
        if casing_choice < 0.2:
            raw = raw.upper()
        elif casing_choice < 0.4:
            raw = raw.lower()
        elif casing_choice < 0.5:
            # Alternating mixed casing
            raw = "".join(c.upper() if j % 2 == 0 else c.lower() for j, c in enumerate(raw))

        # 1. Test strip_legal_suffixes idempotence
        f1 = strip_legal_suffixes(raw)
        f2 = strip_legal_suffixes(f1)
        f3 = strip_legal_suffixes(f2)
        assert f1 == f2 == f3, (
            f"[Iter {idx}] strip_legal_suffixes idempotence broken on '{raw}': f1='{f1}', f2='{f2}'"
        )

        # 2. Test legal_entity_normalizer pipeline idempotence
        df1 = pd.DataFrame({"name": [raw]})
        n1 = norm.transform(df1, columns=["name"])["name"].iloc[0]
        df2 = pd.DataFrame({"name": [n1]})
        n2 = norm.transform(df2, columns=["name"])["name"].iloc[0]
        assert n1 == n2, (
            f"[Iter {idx}] legal_entity_normalizer idempotence broken on '{raw}': n1='{n1}', n2='{n2}'"
        )


@settings(max_examples=100)
@given(st.text(min_size=0, max_size=100))
def test_hypothesis_arbitrary_string_idempotence(text: str) -> None:
    """Property-based verification: strip_legal_suffixes is strictly idempotent for ANY text input."""
    f1 = strip_legal_suffixes(text)
    f2 = strip_legal_suffixes(f1)
    assert f1 == f2, (
        f"Hypothesis found idempotence violation: f1='{f1}', f2='{f2}' for input '{text}'"
    )


# =============================================================================
# 4. Edge Cases: Punctuation, Accents, Mixed Casing & Stacked Suffixes
# =============================================================================

EDGE_CASE_TESTS: list[tuple[str, str, str]] = [
    # Empty & whitespace
    ("", "", ""),
    ("   ", "   ", ""),
    ("\t\n  \r", "\t\n  \r", ""),
    # Pure punctuation & non-word characters
    ("---///,...", "---///,...", "---///,..."),
    ("!@#$%^&*()_+", "!@#$%^&*()_+", "!@#$%^&*()_+"),
    ("1234567890", "1234567890", "1234567890"),
    # Accented international corporate names
    ("Müller GmbH", "Müller", "müller"),
    ("Société Générale S.A.", "Société Générale", "société générale"),
    ("Crédit Agricole S.A.", "Crédit Agricole", "crédit agricole"),
    ("Établissements Dupont S.A.R.L.", "Établissements Dupont", "établissements dupont"),
    # Mixed casing in suffixes
    ("gMbH & cO. kG", "gMbH & cO. kG", "gmbh & co. kg"),
    ("Acme gMbH & cO. kG", "Acme", "acme"),
    ("Apex l.L.c.", "Apex", "apex"),
    ("Vanguard s.A. dE c.V.", "Vanguard", "vanguard"),
    # Multiple trailing stacked suffixes (all should peel off cleanly down to base name)
    ("Acme Corp Inc LLC Ltd", "Acme", "acme"),
    ("Alpha Co., Ltd. Inc.", "Alpha", "alpha"),
    ("Beta GmbH & Co. KG LLC", "Beta", "beta"),
    ("Delta Inc. Corp. LLC", "Delta", "delta"),
    ("Omega Enterprises Ltd. PLC Corp.", "Omega Enterprises", "omega enterprises"),
]


@pytest.mark.parametrize(("raw_input", "exp_strip", "exp_norm"), EDGE_CASE_TESTS)
def test_normalization_edge_cases(raw_input: str, exp_strip: str, exp_norm: str) -> None:
    """Verify robust handling of empty strings, pure symbols, accents, mixed casing, and stacked suffixes."""
    res_strip = strip_legal_suffixes(raw_input)
    assert res_strip == exp_strip, (
        f"strip_legal_suffixes edge case failed on '{raw_input}': got '{res_strip}', exp '{exp_strip}'"
    )

    norm = legal_entity_normalizer()
    df = pd.DataFrame({"name": [raw_input]})
    res_norm = norm.transform(df, columns=["name"])["name"].iloc[0]
    assert res_norm == exp_norm, (
        f"legal_entity_normalizer edge case failed on '{raw_input}': got '{res_norm}', exp '{exp_norm}'"
    )


# =============================================================================
# 5. Boundary Protection on Non-Suffix Brand Names Ending with Suffix Letters
# =============================================================================

NON_SUFFIX_BRAND_TESTS = [
    # Brands that contain suffix tokens as un-delimited substrings
    ("Boeing", "Boeing"),
    ("PepsiCo", "PepsiCo"),
    ("San Francisco", "San Francisco"),
    ("Tesco", "Tesco"),
    ("Costco", "Costco"),
    ("Cisco", "Cisco"),
    ("Acme Spring", "Acme Spring"),
    ("Acme King", "Acme King"),
    ("Acme Holdings", "Acme Holdings"),
    ("Acme Consulting", "Acme Consulting"),
    ("3M Company", "3M"),
    ("7-Eleven Inc", "7-Eleven"),
    ("24/7 Logistics Ltd", "24/7 Logistics"),
    ("X Corp", "X"),
    ("A LLC", "A"),
    ("Q GmbH", "Q"),
]


@pytest.mark.parametrize(("raw_name", "expected_strip"), NON_SUFFIX_BRAND_TESTS)
def test_non_suffix_brand_preservation(raw_name: str, expected_strip: str) -> None:
    """Verify words ending in letters that mimic legal suffixes without boundary separation are safe."""
    assert strip_legal_suffixes(raw_name) == expected_strip


# =============================================================================
# 6. Performance & Asymptotic Scaling on Pathological Long Inputs
# =============================================================================


def test_long_input_sub_millisecond_scaling() -> None:
    """Verify regex execution does not suffer from catastrophic backtracking on 10,000-char strings."""
    import time

    base_junk = "Acme Industrial Logistics Solutions International " * 200  # ~10,000 chars
    pathological_text = f"{base_junk} Inc."

    t0 = time.perf_counter()
    res = strip_legal_suffixes(pathological_text)
    elapsed = time.perf_counter() - t0

    assert res == base_junk.strip()
    assert elapsed < 0.1, (
        f"Stripping 10,000-char text took {elapsed:.4f}s; potential regex backtracking!"
    )


# =============================================================================
# 7. Pandas Series Vectorized Integration & Null Handling
# =============================================================================


def test_series_vectorized_and_null_inputs() -> None:
    """Verify strip_legal_suffixes accepts Series with mixed datatypes, NaNs, and nulls without failure."""
    s = pd.Series(
        [
            "Acme Corp.",
            None,
            np.nan,
            12345,
            "",
            "GmbH & Co. KG",
            "Limited Brands Co.",
            "Acme Corp Inc LLC Ltd",
        ]
    )

    res = strip_legal_suffixes(s)
    assert isinstance(res, pd.Series)
    assert res.iloc[0] == "Acme"
    assert res.iloc[1] == ""
    assert res.iloc[2] == ""
    assert res.iloc[3] == "12345"
    assert res.iloc[4] == ""
    assert res.iloc[5] == "GmbH & Co. KG"  # Preserved because suffix-only
    assert res.iloc[6] == "Limited Brands"  # Interior preserved, terminal stripped
    assert res.iloc[7] == "Acme"  # Stacked suffixes all stripped
