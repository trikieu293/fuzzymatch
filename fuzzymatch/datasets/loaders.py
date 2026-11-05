"""Load ER benchmark datasets (ER-Magellan suite from DeepMatcher).

All datasets are downloaded once into a local cache directory
(~/.cache/fuzzymatch/datasets/<name>/) and reused on subsequent calls.

Supported datasets:
  - Abt-Buy        (9,575 pairs; textual product matching)
  - DBLP-ACM       (12,363 pairs; structured citation matching)
  - DBLP-Scholar   (28,707 pairs; structured citation matching)
  - Amazon-Google  (11,460 pairs; structured software product matching)
  - Walmart-Amazon (10,242 pairs; structured product matching)
  - iTunes-Amazon  (539 pairs; structured music product matching)
  - Fodors-Zagats  (946 pairs; structured restaurant matching)
  - BeerAdvo-RateBeer (450 pairs; structured beer matching)
  - Company        (112,632 pairs; textual company name matching)

Note: only a subset of datasets have public download URLs that we can reach.
If a URL is unreachable, the loader raises an HTTPError with a clear message.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

_BASE_URLS = {
    # The DeepMatcher / anhaidgroup repository hosts the canonical ER-Magellan
    # benchmark files used in Mudgal et al. SIGMOD 2018.
    # NOTE: Dataset files were removed from the GitHub repo; they are now
    # served from UW-Madison CS department pages. Availability verified by
    # HEAD probes on 2026-08-22; DBLP-Scholar and BeerAdvo-RateBeer could not
    # be located on the new host and keep their historical (dead) URLs.
    "Abt-Buy": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/",
    "DBLP-ACM": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data/",
    "DBLP-Scholar": "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/data/dblp_scholar/",
    "Amazon-Google": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/",
    "Walmart-Amazon": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty/Walmart-Amazon/exp_data/",
    "iTunes-Amazon": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/exp_data/",
    "Fodors-Zagats": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data/",
    "BeerAdvo-RateBeer": "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/data/beeradvo_ratebeer/",
    "Company": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Company/exp_data/",
}

CACHE_ROOT = Path.home() / ".cache" / "fuzzymatch" / "datasets"


def available_datasets() -> list[str]:
    """List the dataset names supported by `load()`."""
    return list(_BASE_URLS.keys())


def load(
    name: str, cache_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load an ER benchmark dataset.

    Parameters
    ----------
    name : str
        One of the names returned by `available_datasets()`.
    cache_dir : Path, optional
        Local directory to cache downloaded CSV files. Defaults to
        `~/.cache/fuzzymatch/datasets/<name>/`.

    Returns
    -------
    (labeled_pairs, table_a, table_b) : tuple of pd.DataFrame
        - labeled_pairs: DataFrame with columns `ltable_id, rtable_id, label`
          (label is 1 for matches, 0 for non-matches; pairs are DeepMatcher's
          training partition of the full annotation set)
        - table_a, table_b: the two source tables to be matched

    Raises
    ------
    ValueError
        If `name` is not in the supported list.
    urllib.error.URLError
        If the dataset cannot be downloaded.
    """
    if name not in _BASE_URLS:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {available_datasets()}")
    base_url = _BASE_URLS[name]
    target_dir = (cache_dir or CACHE_ROOT / name).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(filename: str) -> pd.DataFrame:
        path = target_dir / filename
        if not path.exists():
            url = base_url + filename
            urllib.request.urlretrieve(url, path)
        return pd.read_csv(path)

    table_a = _fetch("tableA.csv")
    table_b = _fetch("tableB.csv")
    labeled = _fetch("train.csv")
    return labeled, table_a, table_b
