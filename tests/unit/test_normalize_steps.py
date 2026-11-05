"""Tests for normalization steps."""

from __future__ import annotations

import pandas as pd

from fuzzymatch.normalize.steps import (
    accent_fold,
    casefold,
    strip_punctuation,
    strip_whitespace,
    unicode_normalize,
)


class TestUnicodeNormalize:
    def test_nfc(self) -> None:
        step = unicode_normalize("NFC")
        s = pd.Series(["café"])  # e + combining acute
        result = step(s)
        assert result.iloc[0] == "café"  # NFC composed

    def test_nfkd(self) -> None:
        step = unicode_normalize("NFKD")
        s = pd.Series(["ﬁ"])  # fi ligature
        result = step(s)
        assert result.iloc[0] == "fi"


class TestCasefold:
    def test_basic(self) -> None:
        step = casefold()
        s = pd.Series(["Hello WORLD"])
        result = step(s)
        assert result.iloc[0] == "hello world"

    def test_german_sharp_s(self) -> None:
        step = casefold()
        s = pd.Series(["Straße"])
        result = step(s)
        assert result.iloc[0] == "strasse"


class TestStripWhitespace:
    def test_collapses_spaces(self) -> None:
        step = strip_whitespace()
        s = pd.Series(["  hello   world  "])
        result = step(s)
        assert result.iloc[0] == "hello world"

    def test_tabs_and_newlines(self) -> None:
        step = strip_whitespace()
        s = pd.Series(["hello\t\n world"])
        result = step(s)
        assert result.iloc[0] == "hello world"


class TestStripPunctuation:
    def test_removes_punctuation(self) -> None:
        step = strip_punctuation()
        s = pd.Series(["hello, world! how's it?"])
        result = step(s)
        assert "," not in result.iloc[0]
        assert "!" not in result.iloc[0]
        assert "'" not in result.iloc[0]
        assert "?" not in result.iloc[0]

    def test_keeps_alphanumeric(self) -> None:
        step = strip_punctuation()
        s = pd.Series(["abc 123"])
        result = step(s)
        assert result.iloc[0] == "abc 123"


class TestAccentFold:
    def test_strips_accents(self) -> None:
        step = accent_fold()
        s = pd.Series(["café", "naïve", "üñ"])
        result = step(s)
        assert result.iloc[0] == "cafe"
        assert result.iloc[1] == "naive"
        assert result.iloc[2] == "un"

    def test_keep_marks(self) -> None:
        step = accent_fold(keep_marks=True)
        s = pd.Series(["café"])
        result = step(s)
        assert result.iloc[0] == "café"  # unchanged
