"""Chunked cross-file fitting: identical results, bounded memory."""

from __future__ import annotations

import pandas as pd
import pytest

import fuzzymatch as fm


def _tables(n_left: int = 200, n_right: int = 400) -> tuple[pd.DataFrame, pd.DataFrame]:
    def name(i: int) -> str:
        return f"brand{i % 40} widget{i % 53} v{i % 31}"

    left = pd.DataFrame({"id": range(n_left), "name": [name(i) for i in range(n_left)]})
    right = pd.DataFrame(
        {
            "id": range(n_right),
            "name": [
                name(i % n_left).upper() + " x" if i % 3 == 0 else name(1_000_000 + i)
                for i in range(n_right)
            ],
        }
    )
    return left, right


def _matcher(**kwargs) -> fm.Matcher:
    return fm.Matcher(
        link_type="link",
        blocking=fm.MinHashLSH(["name"], threshold=0.5, num_perm=32),
        normalizer=fm.Normalizer([fm.casefold(), fm.strip_whitespace()]),
        classifier=fm.FSClassifier(),
        clusterer=fm.ConnectedComponents(),
        scoring_fields=["name"],
        **kwargs,
    )


class TestChunkedFit:
    def test_chunked_matches_plain_fit_decisions(self) -> None:
        """Same candidate pairs and scores regardless of chunking."""
        left, right = _tables()
        plain = _matcher()
        plain.fit(left, right)
        chunked = _matcher()
        chunked.fit(left, right, chunked=True, num_chunks=5)

        def as_dict(m: fm.Matcher) -> dict:
            return {
                (int(d.left_id), int(d.right_id)): (
                    d.status,
                    round(float(d.score), 9),
                )
                for d in m.decisions()
            }

        assert as_dict(chunked) == as_dict(plain)

    def test_chunked_right_labels_reference_original_frame(self) -> None:
        left, right = _tables()
        chunked = _matcher()
        chunked.fit(left, right, chunked=True, num_chunks=4)
        pairs = chunked._candidates.pairs
        assert pairs, "expected candidates"
        max_right_label = max(int(r) for _, r in pairs)
        assert max_right_label < len(right), "right ids must be original labels"

    @pytest.mark.parametrize("bad_chunks", [0, -2])
    def test_invalid_num_chunks_raises(self, bad_chunks: int) -> None:
        left, right = _tables()
        matcher = _matcher()
        with pytest.raises(ValueError, match="num_chunks"):
            matcher.fit(left, right, chunked=True, num_chunks=bad_chunks)
