"""Contracts for the pre-match data profiling module."""

from __future__ import annotations

import pandas as pd
import pytest

from fuzzymatch.profiling import profile_table, suggest_matching_config


@pytest.fixture()
def messy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_name": ["Acme Corp", "acme corp", None, "Beta Ltd 12"],
            "erp_code": ["ACC-0001", "ACC-0002", "ACC-0003", "ACC-0004"],
            "amount": [1.5, 2.5, None, 4.0],
        }
    )


class TestProfileTable:
    def test_non_null_rates(self, messy: pd.DataFrame) -> None:
        profiles = profile_table(messy)
        assert profiles["account_name"]["non_null_rate"] == pytest.approx(0.75)
        assert profiles["amount"]["non_null_rate"] == pytest.approx(0.75)
        assert profiles["erp_code"]["non_null_rate"] == 1.0

    def test_unique_counts(self, messy: pd.DataFrame) -> None:
        profiles = profile_table(messy)
        # 'acme corp' dedupes against itself after casefold in uniqueness?
        # Contract: raw distinct non-null values.
        assert profiles["account_name"]["n_unique"] == 3
        assert profiles["erp_code"]["n_unique"] == 4

    def test_digit_flag(self, messy: pd.DataFrame) -> None:
        profiles = profile_table(messy)
        assert profiles["erp_code"]["has_digits"] is True
        assert profiles["account_name"]["has_digits"] is True  # 'Beta Ltd 12'
        assert profiles["amount"]["has_digits"] is False

    def test_token_stats_on_text_columns_only(self, messy: pd.DataFrame) -> None:
        profiles = profile_table(messy)
        assert "token_count_mean" not in profiles["amount"]
        assert profiles["account_name"]["token_count_mean"] == pytest.approx(7 / 3)

    def test_column_subset_respected(self, messy: pd.DataFrame) -> None:
        profiles = profile_table(messy, columns=["erp_code"])
        assert set(profiles) == {"erp_code"}

    def test_empty_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            profile_table(pd.DataFrame({"a": []}))


class TestSuggestMatchingConfig:
    def test_suggests_high_cardinality_text_key(self, messy: pd.DataFrame) -> None:
        hints = suggest_matching_config(profile_table(messy))
        assert hints["key_field"] == "erp_code"

    def test_scoring_fields_exclude_numeric(self, messy: pd.DataFrame) -> None:
        hints = suggest_matching_config(profile_table(messy))
        assert "amount" not in hints["scoring_fields"]
        assert "account_name" in hints["scoring_fields"]

    def test_numeric_only_table_has_no_key(self) -> None:
        frame = pd.DataFrame({"amount": [1.0, 2.0]})
        hints = suggest_matching_config(profile_table(frame))
        assert hints["key_field"] is None
        assert hints["scoring_fields"] == []

    def test_digit_heavy_fields_get_digit_method(self) -> None:
        frame = pd.DataFrame({"model_no": ["abc-123", "abc-456"]})
        hints = suggest_matching_config(profile_table(frame))
        assert hints["scoring_methods"]["model_no"] == "digit_run_similarity"
