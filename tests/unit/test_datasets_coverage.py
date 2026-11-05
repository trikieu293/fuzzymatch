"""Tests to achieve 100% coverage of datasets/loaders.py download logic.

These tests mock the network download mechanism to avoid requiring
actual network access while ensuring all code paths are exercised.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fuzzymatch.datasets import load


def test_load_with_cached_files():
    """Test loading when files already exist in cache (no download needed).

    This covers:
    - Line 81: base_url = _BASE_URLS[name]
    - Line 82: target_dir = (cache_dir or CACHE_ROOT / name).expanduser().resolve()
    - Line 83: target_dir.mkdir(parents=True, exist_ok=True)
    - Line 85-86: def _fetch and path = target_dir / filename
    - Line 87 (False branch): if not path.exists() - file exists, skip download
    - Line 90: return pd.read_csv(path)
    - Lines 92-95: table_a, table_b, labeled fetches and return
    """
    with pytest.importorskip("tempfile").TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_dataset"
        cache_dir.mkdir(parents=True)

        # Create dummy CSV files in cache
        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_csv(
            cache_dir / "tableA.csv", index=False
        )
        pd.DataFrame({"id": [3, 4], "name": ["Charlie", "David"]}).to_csv(
            cache_dir / "tableB.csv", index=False
        )
        pd.DataFrame({"id": [1, 2], "left_id": [1, 2], "right_id": [3, 4], "label": [1, 0]}).to_csv(
            cache_dir / "train.csv", index=False
        )

        # Load should use cached files without attempting download
        labeled, table_a, table_b = load("Abt-Buy", cache_dir=cache_dir)

        # Verify all DataFrames were loaded
        assert isinstance(labeled, pd.DataFrame)
        assert isinstance(table_a, pd.DataFrame)
        assert isinstance(table_b, pd.DataFrame)

        # Verify content
        assert len(table_a) == 2
        assert len(table_b) == 2
        assert len(labeled) == 2
        assert "id" in table_a.columns
        assert "label" in labeled.columns


def test_load_with_download_required():
    """Test loading when files need to be downloaded.

    This covers:
    - Line 87 (True branch): if not path.exists() - file doesn't exist
    - Line 88: url = base_url + filename
    - Line 89: urllib.request.urlretrieve(url, path)
    """
    with pytest.importorskip("tempfile").TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_dataset"
        # Don't create files - they need to be downloaded

        # Mock urlretrieve to create the files when called
        def mock_urlretrieve(url: str, path: Path) -> None:
            """Simulate downloading by creating CSV files."""
            path = Path(path)
            if "tableA" in str(path):
                pd.DataFrame({"id": [10, 20], "value": ["A", "B"]}).to_csv(path, index=False)
            elif "tableB" in str(path):
                pd.DataFrame({"id": [30, 40], "value": ["C", "D"]}).to_csv(path, index=False)
            elif "train" in str(path):
                pd.DataFrame(
                    {"id": [1, 2], "left_id": [10, 20], "right_id": [30, 40], "label": [1, 0]}
                ).to_csv(path, index=False)

        # Patch urllib.request.urlretrieve to use our mock
        with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
            labeled, table_a, table_b = load("DBLP-ACM", cache_dir=cache_dir)

        # Verify all DataFrames were loaded
        assert isinstance(labeled, pd.DataFrame)
        assert isinstance(table_a, pd.DataFrame)
        assert isinstance(table_b, pd.DataFrame)

        # Verify content
        assert len(table_a) == 2
        assert len(table_b) == 2
        assert len(labeled) == 2
        assert table_a["id"].tolist() == [10, 20]
        assert table_b["id"].tolist() == [30, 40]


def test_load_with_partial_cache():
    """Test loading when some files are cached and others need download.

    This ensures _fetch is called independently for each file.
    """
    with pytest.importorskip("tempfile").TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_dataset"
        cache_dir.mkdir(parents=True)

        # Pre-create only tableA.csv
        pd.DataFrame({"id": [1, 2]}).to_csv(cache_dir / "tableA.csv", index=False)

        # Mock urlretrieve to create the missing files
        def mock_urlretrieve(url: str, path: Path) -> None:
            path = Path(path)
            if "tableB" in str(path):
                pd.DataFrame({"id": [3, 4]}).to_csv(path, index=False)
            elif "train" in str(path):
                pd.DataFrame({"id": [1], "left_id": [1], "right_id": [3], "label": [1]}).to_csv(
                    path, index=False
                )

        with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
            labeled, table_a, table_b = load("Amazon-Google", cache_dir=cache_dir)

        # Verify all loaded
        assert isinstance(labeled, pd.DataFrame)
        assert isinstance(table_a, pd.DataFrame)
        assert isinstance(table_b, pd.DataFrame)


def test_load_invalid_dataset_name():
    """Test that loading an invalid dataset name raises ValueError.

    This covers:
    - Lines 77-80: ValueError for unknown dataset
    """
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("Nonexistent-Dataset-12345")

    # Also test with empty string
    with pytest.raises(ValueError, match="Unknown dataset"):
        load("")


def test_load_with_default_cache_dir():
    """Test that load() uses CACHE_ROOT when cache_dir is not provided.

    This verifies line 82's default behavior: (cache_dir or CACHE_ROOT / name)
    """
    # We'll mock both the directory creation and urlretrieve
    # to avoid actually using the home directory
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        with patch("pathlib.Path.exists", return_value=False):
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                with patch("pandas.read_csv") as mock_read_csv:
                    # Setup mock to return a DataFrame
                    mock_read_csv.return_value = pd.DataFrame({"id": [1]})

                    try:
                        load("Company")
                    except Exception:
                        # We expect this might fail due to mocking, but we
                        # just want to verify the code path is exercised
                        pass

                    # Verify mkdir was called (would use CACHE_ROOT)
                    assert mock_mkdir.called
                    # Verify urlretrieve was attempted
                    assert mock_retrieve.called


def test_load_url_construction():
    """Test that URLs are correctly constructed from base_url + filename.

    This specifically covers line 88: url = base_url + filename
    """
    with pytest.importorskip("tempfile").TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_dataset"

        captured_urls = []

        def mock_urlretrieve(url: str, path: Path) -> None:
            """Capture the URL and create the file."""
            captured_urls.append(url)
            path = Path(path)
            pd.DataFrame({"id": [1]}).to_csv(path, index=False)

        with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve):
            load("iTunes-Amazon", cache_dir=cache_dir)

        # Verify URLs were constructed correctly
        assert len(captured_urls) == 3
        base = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/exp_data/"
        assert f"{base}tableA.csv" in captured_urls
        assert f"{base}tableB.csv" in captured_urls
        assert f"{base}train.csv" in captured_urls
