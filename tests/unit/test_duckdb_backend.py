"""DuckDB-backed candidate generation contracts."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from fuzzymatch import FSClassifier, Matcher
from fuzzymatch.blocking import ExactBlock, MinHashLSH, SortedNeighborhood


def _frames(n: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(
        {
            "grp": [f"g{i % 7}" for i in range(n)],
            "name": [f"acct {i * 13 % n} llc" for i in range(n)],
        }
    )
    right = pd.DataFrame(
        {
            "grp": [f"g{i % 5}" for i in range(n)],
            "name": [f"acct {i * 11 % n} inc" for i in range(n)],
        }
    )
    return left, right


def _norm(pairs: list[tuple]) -> set[tuple]:
    return {(str(a), str(b)) for a, b in pairs}


class TestDuckBackendParity:
    def test_exact_block_parity(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames()
        expected = _norm(ExactBlock(["grp"]).block(left, right).pairs)
        got = _norm(DuckDBBackend().block_exact(left, right, ["grp"]))
        assert got == expected

    def test_sorted_neighborhood_parity(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames()
        blocker = SortedNeighborhood(["name"], window=4)
        expected = _norm(blocker.block(left, right).pairs)
        got = _norm(DuckDBBackend().block_sorted_neighborhood(left, right, ["name"], 4))
        assert got == expected

    def test_non_default_index_labels_preserved(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames(20)
        left.index = [f"L{i}" for i in range(len(left))]
        right.index = [f"R{i}" for i in range(len(right))]
        expected = _norm(ExactBlock(["grp"]).block(left, right).pairs)
        got = _norm(DuckDBBackend().block_exact(left, right, ["grp"]))
        assert got == expected

    def test_missing_marker_blocking_parity(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left = pd.DataFrame(
            {
                "grp": ["valid_a", "", "   ", "nan", "NaN", "none", "None", None, "valid_b"],
                "name": [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "eta",
                    "theta",
                    "iota",
                ],
            }
        )
        right = pd.DataFrame(
            {
                "grp": ["valid_a", "", "nan", "none", None, "valid_b", "extra"],
                "name": ["alpha", "beta", "delta", "zeta", "theta", "iota", "kappa"],
            }
        )
        expected_exact = _norm(ExactBlock(["grp"]).block(left, right).pairs)
        got_exact = _norm(DuckDBBackend().block_exact(left, right, ["grp"]))
        assert got_exact == expected_exact
        assert got_exact == {("0", "0"), ("8", "5")}
        assert DuckDBBackend().count_block_exact(left, right, ["grp"]) == len(expected_exact)

        sn_blocker = SortedNeighborhood(["grp"], window=2)
        expected_sn = _norm(sn_blocker.block(left, right).pairs)
        got_sn = _norm(DuckDBBackend().block_sorted_neighborhood(left, right, ["grp"], 2))
        assert got_sn == expected_sn
        assert DuckDBBackend().count_sorted_neighborhood(left, right, ["grp"], 2) == len(
            expected_sn
        )

    def test_unsupported_blocker_rejected(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames(10)
        with pytest.raises(NotImplementedError, match="MinHashLSH"):
            DuckDBBackend().block_for_blocker(left, right, MinHashLSH(fields=["grp"]))

    def test_missing_duckdb_install_message(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "duckdb", None)
        import importlib

        from fuzzymatch.backends import duckdb_backend

        importlib.reload(duckdb_backend)
        with pytest.raises(ImportError, match="fuzzymatch\\[duckdb\\]"):
            duckdb_backend.DuckDBBackend()


class TestMatcherWiring:
    def _matcher(self, **kw) -> Matcher:
        return Matcher(
            link_type="link",
            blocking=ExactBlock(["grp"]),
            classifier=FSClassifier(),
            scoring_fields=["name"],
            **kw,
        )

    def test_blocking_backend_default_pandas(self) -> None:
        assert self._matcher().blocking_backend == "pandas"

    def test_duckdb_candidates_identical_decisions(self) -> None:
        left, right = _frames(40)
        base = self._matcher().fit(left.copy(), right.copy())
        duck = self._matcher(blocking_backend="duckdb").fit(left.copy(), right.copy())
        assert len(base.decisions()) > 0
        assert _norm(base._candidates.pairs) == _norm(duck._candidates.pairs)

    def test_unsupported_blocker_via_matcher_raises(self) -> None:
        m = Matcher(
            link_type="link",
            blocking=MinHashLSH(fields=["name"]),
            classifier=FSClassifier(),
            scoring_fields=["name"],
            blocking_backend="duckdb",
        )
        left, right = _frames(10)
        with pytest.raises(NotImplementedError):
            m.fit(left, right)


class TestCoverageGuards:
    """Coverage guards for DuckDB backend error and edge cases."""

    def test_invalid_blocking_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="blocking_backend"):
            Matcher(
                link_type="link",
                blocking=ExactBlock(["grp"]),
                classifier=FSClassifier(),
                scoring_fields=["name"],
                blocking_backend="snowflake",
            )

    def test_duckdb_dedupe_rejected(self) -> None:
        from fuzzymatch import ExactBlock as _EB

        m = Matcher(
            link_type="dedupe",
            blocking=_EB(["grp"]),
            classifier=FSClassifier(),
            scoring_fields=["grp"],
            blocking_backend="duckdb",
        )
        left, _ = _frames(10)
        with pytest.raises(NotImplementedError, match="dedupe"):
            m.fit(left)

    def test_decision_probabilities_before_fit_raises(self) -> None:
        m = Matcher(
            link_type="link",
            blocking=ExactBlock(["grp"]),
            classifier=FSClassifier(),
            scoring_fields=["name"],
        )
        with pytest.raises(RuntimeError, match="decision_probabilities"):
            m.decision_probabilities()

    def test_count_helpers_and_close(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames(30)
        b = DuckDBBackend()
        exact_n = b.count_block_exact(left, right, ["grp"])
        sn_n = b.count_sorted_neighborhood(left, right, ["name"], 3)
        assert exact_n == len(
            set((int(a), int(c)) for a, c in ExactBlock(["grp"]).block(left, right).pairs)
        )
        assert sn_n > 0
        b.close()

    def test_dispatch_sorted_neighborhood(self) -> None:
        # [coverage guard] block_for_blocker SN dispatch branch
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames(30)
        b = DuckDBBackend()
        pairs = b.block_for_blocker(left, right, SortedNeighborhood(["name"], window=4))
        assert pairs
        b.close()

    def test_sn_window_below_one_rejected(self) -> None:
        from fuzzymatch.backends.duckdb_backend import DuckDBBackend

        left, right = _frames(10)
        b = DuckDBBackend()
        with pytest.raises(ValueError, match="window"):
            b.block_sorted_neighborhood(left, right, ["name"], 0)
        with pytest.raises(ValueError, match="window"):
            b.count_sorted_neighborhood(left, right, ["name"], 0)
        b.close()
