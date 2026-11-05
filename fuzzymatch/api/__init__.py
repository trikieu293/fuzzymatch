"""High-level API: Matcher class (unified entry point for all modes)."""

from .auto import AutoMatcher, MatchResults
from .matcher import Matcher

__all__ = ["AutoMatcher", "MatchResults", "Matcher"]
