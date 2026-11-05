"""Composable normalization pipeline."""

from __future__ import annotations

import pandas as pd

from .steps import NormalizationStep


class Normalizer:
    """Apply an ordered list of NormalizationStep functions to specified columns.

    Usage:
        normalizer = Normalizer([
            unicode_normalize("NFC"),
            casefold(),
            strip_whitespace(),
        ])
        df_clean = normalizer.transform(df, columns=["name", "address"])
    """

    def __init__(self, steps: list[NormalizationStep]):
        self.steps = steps

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Apply normalization steps to specified columns; return a new DataFrame."""
        result = df.copy()
        for col in columns:
            if col not in result.columns:
                continue
            series = result[col].astype(str)
            for step in self.steps:
                series = step(series)
            result[col] = series
        return result
