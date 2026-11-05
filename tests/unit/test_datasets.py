"""Unit tests for the datasets module."""

from __future__ import annotations

from pathlib import Path

import pytest

from fuzzymatch.datasets import available_datasets, load


def test_available_datasets():
    names = available_datasets()
    assert isinstance(names, list)
    assert "Abt-Buy" in names
    assert "DBLP-ACM" in names
    assert len(names) >= 5


def test_load_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("Nonexistent-Dataset")


def test_load_caches(tmp_path: Path):
    """Loading twice should hit the cache, not re-download."""
    pytest.skip("Requires network access; covered by integration test")
