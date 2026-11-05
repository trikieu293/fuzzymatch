"""Token-replacement normalization step contracts (synonym lexicons)."""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.normalize import Normalizer
from fuzzymatch.normalize.steps import casefold, replace_words


class TestReplaceWords:
    def test_maps_known_tokens(self) -> None:
        step = replace_words({"debtors": "receivables"})
        series = pd.Series(["debtors control", "total debtors"])
        assert step(series).tolist() == ["receivables control", "total receivables"]

    def test_unknown_tokens_untouched(self) -> None:
        step = replace_words({"debtors": "receivables"})
        series = pd.Series(["cash at bank"])
        assert step(series).tolist() == ["cash at bank"]

    def test_composes_with_casefold_in_pipeline(self) -> None:
        normalizer = Normalizer([casefold(), replace_words({"stock": "inventory"})])
        out = normalizer.transform(pd.DataFrame({"v": ["Stock In Hand"]}), ["v"])
        assert out["v"].iloc[0] == "inventory in hand"

    def test_multi_word_replacement_allowed(self) -> None:
        step = replace_words({"p and l": "profit loss"})
        series = pd.Series(["p and l account"])
        assert step(series).tolist() == ["profit loss account"]

    @pytest.mark.parametrize("bad", [{}, {"": "x"}])
    def test_invalid_mapping_raises(self, bad: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="mapping"):
            replace_words(bad)
