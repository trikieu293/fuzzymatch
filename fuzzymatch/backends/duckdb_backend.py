"""DuckDB-backed candidate generation (v1.3 Phase B).

Pushes exact-key and sorted-neighborhood blocking into SQL so cross-file
joins scale past comfortable pandas memory. Pairs are returned with the
frames' original index labels, byte-compatible with the pandas blockers.
MinHashLSH is not pushed down (Python-side only, documented scope).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_KEY = "__fm_key"
_LIDX = "__fm_lidx"
_RIDX = "__fm_ridx"
_ORIGIN = "__fm_origin"
_ROWNUM = "__fm_rn"


def _register(con: Any, df: pd.DataFrame, view: str, idx_col: str) -> None:
    frame = df.reset_index(names=idx_col)
    con.register(view, frame)


def _valid_key_sql(alias: str, col: str) -> str:
    return (
        f"{alias}.{col} IS NOT NULL AND "
        f"LOWER(TRIM(CAST({alias}.{col} AS VARCHAR))) NOT IN ('', 'nan', 'none')"
    )


class DuckDBBackend:
    """SQL candidate generation over DuckDB-registered frames."""

    name = "duckdb"

    def __init__(self) -> None:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "DuckDBBackend requires the 'duckdb' package. "
                "Install with: pip install fuzzymatch[duckdb]"
            ) from exc
        self._con = duckdb.connect()

    def _key_expr(self, fields: list[str]) -> str:
        return " || '|' || ".join(f"CAST({f} AS VARCHAR)" for f in fields)

    def block_exact(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        fields: list[str],
    ) -> list[tuple[Any, Any]]:
        """Cross join on exact equality of the concatenated key."""
        _register(self._con, left, "fm_left", _LIDX)
        _register(self._con, right, "fm_right", _RIDX)
        on = " AND ".join(f"CAST(l.{f} AS VARCHAR) = CAST(r.{f} AS VARCHAR)" for f in fields)
        valid = " AND ".join(
            f"{_valid_key_sql('l', f)} AND {_valid_key_sql('r', f)}" for f in fields
        )
        rel = self._con.sql(
            f"""
            SELECT l.{_LIDX} AS li, r.{_RIDX} AS ri
            FROM fm_left l JOIN fm_right r ON {on}
            WHERE {valid}
            """
        )
        return [(li, ri) for li, ri in rel.fetchall()]

    def _sn_pair_query(self, fields: list[str], window: int) -> str:
        key = self._key_expr(fields)
        where_l = " AND ".join(_valid_key_sql("fm_left", f) for f in fields)
        where_r = " AND ".join(_valid_key_sql("fm_right", f) for f in fields)
        return f"""
            WITH merged AS MATERIALIZED (
                SELECT {_LIDX} AS idx, 0 AS org, {key} AS k,
                       ROW_NUMBER() OVER () AS pos FROM fm_left
                WHERE {where_l}
                UNION ALL
                SELECT {_RIDX} AS idx, 1 AS org, {key} AS k,
                       ROW_NUMBER() OVER () AS pos FROM fm_right
                WHERE {where_r}
            ),
            ordered AS MATERIALIZED (
                SELECT idx, org,
                       ROW_NUMBER() OVER (
                            ORDER BY k, org, pos
                        ) AS {_ROWNUM}
                FROM merged
            ),
            forward AS (
                -- left-origin rows pairing right-origin rows ranked after them
                SELECT a.idx AS li, b.idx AS ri
                FROM ordered a JOIN ordered b
                  ON a.org = 0 AND b.org = 1
                 AND b.{_ROWNUM} > a.{_ROWNUM}
                 AND b.{_ROWNUM} <= a.{_ROWNUM} + {int(window)}
            ),
            backward AS (
                -- right-origin rows ranked before a left-origin row
                SELECT a.idx AS li, b.idx AS ri
                FROM ordered a JOIN ordered b
                  ON a.org = 0 AND b.org = 1
                 AND b.{_ROWNUM} < a.{_ROWNUM}
                 AND b.{_ROWNUM} >= a.{_ROWNUM} - {int(window)}
            )
            SELECT li, ri FROM forward
            UNION ALL
            SELECT li, ri FROM backward
        """

    def block_sorted_neighborhood(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        fields: list[str],
        window: int,
    ) -> list[tuple[Any, Any]]:
        """Sorted-neighborhood via one merged ordering + directional joins.

        Mirrors SortedNeighborhood's cross-file semantics: records from both
        sides are merged into a single key-sorted sequence and paired only
        across origins within ``window`` neighbours (both directions).
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        _register(self._con, left, "fm_left", _LIDX)
        _register(self._con, right, "fm_right", _RIDX)
        rel = self._con.sql(self._sn_pair_query(fields, window))
        return [(li, ri) for li, ri in rel.fetchall()]

    def count_block_exact(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        fields: list[str],
    ) -> int:
        """Exact-block pair count without materializing pairs."""
        _register(self._con, left, "fm_left", _LIDX)
        _register(self._con, right, "fm_right", _RIDX)
        on = " AND ".join(f"CAST(l.{f} AS VARCHAR) = CAST(r.{f} AS VARCHAR)" for f in fields)
        valid = " AND ".join(
            f"{_valid_key_sql('l', f)} AND {_valid_key_sql('r', f)}" for f in fields
        )
        rows = self._con.execute(
            f"SELECT COUNT(*) FROM fm_left l JOIN fm_right r ON {on} WHERE {valid}"
        ).fetchall()
        assert rows
        return int(rows[0][0])

    def count_sorted_neighborhood(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        fields: list[str],
        window: int,
    ) -> int:
        """Sorted-neighborhood pair count without materializing pairs."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        _register(self._con, left, "fm_left", _LIDX)
        _register(self._con, right, "fm_right", _RIDX)
        # Forward and backward branches are disjoint (rank direction), so
        # counting them separately avoids the planner pathology seen when
        # wrapping their UNION ALL in an aggregate.
        key = self._key_expr(fields)
        where_l = " AND ".join(_valid_key_sql("fm_left", f) for f in fields)
        where_r = " AND ".join(_valid_key_sql("fm_right", f) for f in fields)
        base = f"""
            CREATE OR REPLACE TEMP TABLE fm_sn_ord AS
            SELECT idx, org, ROW_NUMBER() OVER (ORDER BY k, org, pos) AS {_ROWNUM}
            FROM (
                SELECT {_LIDX} AS idx, 0 AS org, {key} AS k,
                        ROW_NUMBER() OVER () AS pos FROM fm_left
                WHERE {where_l}
                UNION ALL
                SELECT {_RIDX} AS idx, 1 AS org, {key} AS k,
                        ROW_NUMBER() OVER () AS pos FROM fm_right
                WHERE {where_r}
            )
        """
        self._con.execute(base)
        forward = self._con.execute(
            f"""
            SELECT COUNT(*) FROM fm_sn_ord a JOIN fm_sn_ord b
              ON a.org = 0 AND b.org = 1
             AND b.{_ROWNUM} > a.{_ROWNUM}
             AND b.{_ROWNUM} <= a.{_ROWNUM} + {int(window)}
            """
        ).fetchall()
        backward = self._con.execute(
            f"""
            SELECT COUNT(*) FROM fm_sn_ord a JOIN fm_sn_ord b
              ON a.org = 0 AND b.org = 1
             AND b.{_ROWNUM} < a.{_ROWNUM}
             AND b.{_ROWNUM} >= a.{_ROWNUM} - {int(window)}
            """
        ).fetchall()
        assert forward and backward
        return int(forward[0][0]) + int(backward[0][0])

    def block_for_blocker(
        self, left: pd.DataFrame, right: pd.DataFrame, blocker: Any
    ) -> list[tuple[Any, Any]]:
        """Dispatch on blocker type; unsupported strategies are rejected."""
        from ..blocking import ExactBlock, SortedNeighborhood

        if isinstance(blocker, ExactBlock):
            return self.block_exact(left, right, blocker.fields)
        if isinstance(blocker, SortedNeighborhood):
            return self.block_sorted_neighborhood(left, right, blocker.fields, blocker.window)
        raise NotImplementedError(
            f"DuckDB backend does not support {type(blocker).__name__}; "
            "use blocking_backend='pandas' for this strategy."
        )

    def close(self) -> None:
        self._con.close()
