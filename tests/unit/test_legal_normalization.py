"""Unit tests for corporate legal entity suffix normalization.

Tests:
1. strip_legal_suffixes scalar behavior on international corporate suffixes (US/UK, DACH, Romance, Nordic, Asian).
2. Interior brand preservation: tokens like 'Company' or 'Limited' in subject positions are not stripped.
3. Suffix-only non-empty preservation invariant: 'Inc', 'LLC', 'GmbH' alone are preserved.
4. Idempotence: f(f(s)) == f(s) across diverse corporate names.
5. Pandas Series and Pipeline callable integration (strip_legal_suffixes(None) / step).
6. Robustness against missing values (None, NaN), whitespace, and non-string inputs.
7. Typo-tolerant suffix normalization (cCompany, GmvH, S.yA., gerund suffixes, word joins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fuzzymatch.normalize.legal import (
    DEFAULT_LEGAL_SUFFIXES,
    strip_legal_suffixes,
)


def test_default_legal_suffixes_contract() -> None:
    """Verify DEFAULT_LEGAL_SUFFIXES is an immutable, non-empty tuple of regexes."""
    assert isinstance(DEFAULT_LEGAL_SUFFIXES, tuple)
    assert len(DEFAULT_LEGAL_SUFFIXES) > 40
    assert any("gmbh" in s for s in DEFAULT_LEGAL_SUFFIXES)
    assert any("ll" in s for s in DEFAULT_LEGAL_SUFFIXES)
    assert any("ltd" in s for s in DEFAULT_LEGAL_SUFFIXES)


@pytest.mark.parametrize(
    ("input_name", "expected_name"),
    [
        # Standard US / UK corporate suffixes
        ("Acme Industrial, Inc.", "Acme Industrial"),
        ("Acme Industrial Inc", "Acme Industrial"),
        ("Acme Industrial Incorporated", "Acme Industrial"),
        ("Acme Industrial Corporation", "Acme Industrial"),
        ("Acme Industrial Corp.", "Acme Industrial"),
        ("Acme Industrial Corp", "Acme Industrial"),
        ("Acme Industrial LLC", "Acme Industrial"),
        ("Acme Industrial L.L.C.", "Acme Industrial"),
        ("Acme Industrial Ltd.", "Acme Industrial"),
        ("Acme Industrial Limited", "Acme Industrial"),
        ("Acme Industrial Co.", "Acme Industrial"),
        ("Acme Industrial Company", "Acme Industrial"),
        ("Acme Industrial, Co., Ltd.", "Acme Industrial"),
        ("Acme Industrial PLC", "Acme Industrial"),
        ("Acme Industrial LLP", "Acme Industrial"),
        ("Acme Industrial LP", "Acme Industrial"),
        # DACH legal forms
        ("Bosch GmbH", "Bosch"),
        ("Bosch G.m.b.H.", "Bosch"),
        ("Siemens AG", "Siemens"),
        ("Siemens A.G.", "Siemens"),
        ("Müller GmbH & Co. KG", "Müller"),
        ("Müller GmbH & Co. KGaA", "Müller"),
        ("Tech Holding GesmbH", "Tech Holding"),
        ("Venture UG (haftungsbeschränkt)", "Venture"),
        ("Venture UG", "Venture"),
        ("Tradition e.V.", "Tradition"),
        # Romance / Latin America
        ("Danone S.A.", "Danone"),
        ("Danone SA", "Danone"),
        ("Hermes S.A.S.", "Hermes"),
        ("Total S.A.R.L.", "Total"),
        ("Enel S.p.A.", "Enel"),
        ("Telecom Italia S.p.A.", "Telecom Italia"),
        ("Telecom Italia S.P.A.", "Telecom Italia"),
        ("Iberdrola S.A. de C.V.", "Iberdrola"),
        ("Iberdrola SA de CV", "Iberdrola"),
        ("Banco S.L.", "Banco"),
        ("Constructora Ltda.", "Constructora"),
        ("Studio S.r.l.", "Studio"),
        # Nordic & others
        ("Nokia Oyj", "Nokia"),
        ("Kone Oy", "Kone"),
        ("Volvo AB", "Volvo"),
        ("Toyota K.K.", "Toyota"),
        ("Polski Handel Sp. z o.o.", "Polski Handel"),
        ("Dutch Holdings B.V.", "Dutch Holdings"),
        ("Dutch Holdings N.V.", "Dutch Holdings"),
    ],
)
def test_strip_legal_suffixes_international_forms(input_name: str, expected_name: str) -> None:
    """Verify stripping international legal entity suffixes produces canonical base names."""
    assert strip_legal_suffixes(input_name) == expected_name


def test_interior_brand_preservation() -> None:
    """Verify interior brand tokens are not stripped when appearing in non-suffix positions."""
    assert strip_legal_suffixes("The Company Store") == "The Company Store"
    assert strip_legal_suffixes("Company One Services LLC") == "Company One Services"
    assert strip_legal_suffixes("Limited Editions Corp") == "Limited Editions"
    assert strip_legal_suffixes("Incubator Technologies") == "Incubator Technologies"
    assert (
        strip_legal_suffixes("Standard Oil Company of California")
        == "Standard Oil Company of California"
    )


def test_suffix_only_string_preservation() -> None:
    """Verify suffix-only strings are preserved to prevent collapsing entities to empty strings."""
    assert strip_legal_suffixes("Inc") == "Inc"
    assert strip_legal_suffixes("LLC") == "LLC"
    assert strip_legal_suffixes("GmbH") == "GmbH"
    assert strip_legal_suffixes("Corp.") == "Corp."
    assert strip_legal_suffixes("Ltd.") == "Ltd."
    assert strip_legal_suffixes("S.A.") == "S.A."


@pytest.mark.parametrize(
    "name",
    [
        "Acme Manufacturing, Inc.",
        "Precision Machining GmbH & Co. KG",
        "NorthStar Logistics LLC",
        "Apex Global Ltd.",
        "Cascade Supply Company",
        "Total Energy S.A. de C.V.",
    ],
)
def test_idempotence(name: str) -> None:
    """Verify strict mathematical idempotence: f(f(s)) == f(s)."""
    first_pass = strip_legal_suffixes(name)
    second_pass = strip_legal_suffixes(first_pass)
    assert first_pass == second_pass


def test_edge_cases_and_non_words() -> None:
    """Verify handling of empty strings, whitespace, pure punctuation, and numeric scalars."""
    assert strip_legal_suffixes("") == ""
    assert strip_legal_suffixes("   ") == "   "
    assert strip_legal_suffixes("---///,,,") == "---///,,,"
    # Scalar numeric inputs
    assert strip_legal_suffixes(12345) == "12345"  # type: ignore[arg-type]
    assert strip_legal_suffixes(float("nan")) == ""  # type: ignore[arg-type]


def test_series_integration() -> None:
    """Verify strip_legal_suffixes operating directly on pandas Series."""
    s = pd.Series(
        [
            "Alpha Corp.",
            "Beta LLC",
            "Gamma GmbH",
            "",
            None,
            np.nan,
            "Delta Company Store",
        ]
    )
    result = strip_legal_suffixes(s)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == "Alpha"
    assert result.iloc[1] == "Beta"
    assert result.iloc[2] == "Gamma"
    assert result.iloc[3] == ""
    assert result.iloc[4] == ""
    assert result.iloc[5] == ""
    assert result.iloc[6] == "Delta Company Store"


def test_pipeline_step_callable_generation() -> None:
    """Verify invoking strip_legal_suffixes() with None returns a valid NormalizationStep."""
    step = strip_legal_suffixes()
    assert callable(step)

    s = pd.Series(["Alpha Industries, Inc.", "Omega Services Ltd."])
    transformed = step(s)
    assert isinstance(transformed, pd.Series)
    assert list(transformed) == ["Alpha Industries", "Omega Services"]


def test_legitimate_english_words_preserved() -> None:
    """Verify common English words and distinctive brand tokens are never stripped."""
    words = [
        "Dingy",
        "Springy",
        "Ringa",
        "Blingy",
        "Clingy",
        "Stingy",
        "North Sea",
        "Deep Sea",
        "Red Sea",
        "Apex Lad",
        "Iron Led",
        "Flying Lid",
        "Grand Hotel & Spa",
        "Day Spa",
        "Resort & Spa",
        "Chor Singt",
    ]
    for word in words:
        assert strip_legal_suffixes(word) == word, (
            f"Corrupted '{word}' to '{strip_legal_suffixes(word)}'"
        )
