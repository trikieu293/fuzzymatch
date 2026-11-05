"""Pre-match data profiling: descriptive stats plus advisory config hints."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_DIGIT = re.compile(r"\d")


def profile_table(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Describe candidate matching columns.

    Args:
        df: The table to profile.
        columns: Restrict profiling to these columns (default: all).

    Returns:
        ``{column: profile}`` where text columns carry ``non_null_rate``,
        ``n_unique``, ``has_digits`` and ``token_count_mean``; numeric columns
        omit token statistics.

    Raises:
        ValueError: If the frame has no rows.
    """
    if len(df) == 0:
        raise ValueError("cannot profile an empty frame")
    chosen = columns if columns is not None else list(df.columns)
    profiles: dict[str, dict[str, Any]] = {}
    for column in chosen:
        series = df[column]
        non_null = int(series.notna().sum())
        if pd.api.types.is_numeric_dtype(series):
            profiles[column] = {
                "non_null_rate": round(non_null / len(df), 4),
                "n_unique": int(series.dropna().nunique()),
                "numeric": True,
                "has_digits": False,
            }
            continue
        as_text = series.dropna().astype(str)
        tokens = as_text.str.split().apply(len)
        profiles[column] = {
            "non_null_rate": round(non_null / len(df), 4),
            "n_unique": int(series.dropna().nunique()),
            "numeric": False,
            "has_digits": bool(as_text.str.contains(_DIGIT, regex=True).any()),
            "token_count_mean": float(tokens.mean()) if len(tokens) else 0.0,
        }
    return profiles


def suggest_matching_config(
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Turn a :func:`profile_table` result into advisory matching hints.

    Advisory only — never applied automatically.
    """
    text_columns = [
        (name, p)
        for name, p in profiles.items()
        if not p.get("numeric", False) and p["non_null_rate"] >= 0.5
    ]
    if not text_columns:
        return {"key_field": None, "scoring_fields": [], "scoring_methods": {}}

    key_field = max(text_columns, key=lambda kv: kv[1]["n_unique"])[0]

    scoring_fields = [name for name, _ in text_columns]
    scoring_methods = {name: "digit_run_similarity" for name, p in text_columns if p["has_digits"]}
    return {
        "key_field": key_field,
        "scoring_fields": scoring_fields,
        "scoring_methods": scoring_methods,
    }
