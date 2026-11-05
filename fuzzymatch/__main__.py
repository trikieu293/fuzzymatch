"""Executable module entry point for `python -m fuzzymatch`."""

from __future__ import annotations

import sys

from fuzzymatch.cli import main

if __name__ == "__main__":
    sys.exit(main())
