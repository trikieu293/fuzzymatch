"""Backends: data storage abstractions (pandas and DuckDB)."""

from .duckdb_backend import DuckDBBackend
from .pandas_backend import PandasBackend

__all__ = ["DuckDBBackend", "PandasBackend"]
