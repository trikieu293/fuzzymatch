"""fuzzymatch - industrial-scale text fuzzy mapping library.

5-stage pipeline: normalize -> block -> score -> classify -> cluster.
Supports deduplication, record linkage, and fuzzy join modes.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import datasets
from .api import AutoMatcher, Matcher, MatchResults
from .backends import DuckDBBackend, PandasBackend
from .blocking import (
    BaseBlocker,
    CandidateIndex,
    ExactBlock,
    IntersectBlock,
    MinHashLSH,
    MultiPassBlock,
    SortedNeighborhood,
    UnionBlock,
)
from .classification import FSClassifier, SklearnClassifier
from .clustering import BaseClusterer, ConnectedComponents, PluggableResolver, StarClusterer
from .config import PipelineConfig
from .decisions import Explanation, FieldScore, MatchDecision, MatchStatus
from .lexicon_mine import mine_synonym_candidates
from .metrics import block_quality_metrics, cluster_metrics, pairwise_metrics
from .model_selection import select_operating_threshold, train_valid_test_split
from .monitoring import drift_report
from .normalize import (
    Normalizer,
    accent_fold,
    casefold,
    replace_words,
    strip_legal_suffixes,
    strip_punctuation,
    strip_whitespace,
    unicode_normalize,
)
from .profiling import profile_table, suggest_matching_config
from .scoring import (
    DateWindowScorer,
    NumericProximityScorer,
    RapidFuzzScorer,
    amount_proximity_score,
    date_window_score,
)

__all__ = [
    "AutoMatcher",
    "BaseBlocker",
    "BaseClusterer",
    "CandidateIndex",
    "ConnectedComponents",
    "DateWindowScorer",
    "DuckDBBackend",
    "ExactBlock",
    "Explanation",
    "FSClassifier",
    "FieldScore",
    "IntersectBlock",
    "MatchDecision",
    "MatchResults",
    "MatchStatus",
    "Matcher",
    "MinHashLSH",
    "MultiPassBlock",
    "Normalizer",
    "NumericProximityScorer",
    "PandasBackend",
    "PipelineConfig",
    "PluggableResolver",
    "RapidFuzzScorer",
    "SklearnClassifier",
    "SortedNeighborhood",
    "StarClusterer",
    "UnionBlock",
    "accent_fold",
    "amount_proximity_score",
    "block_quality_metrics",
    "casefold",
    "cluster_metrics",
    "datasets",
    "date_window_score",
    "drift_report",
    "mine_synonym_candidates",
    "pairwise_metrics",
    "profile_table",
    "replace_words",
    "select_operating_threshold",
    "strip_legal_suffixes",
    "strip_punctuation",
    "strip_whitespace",
    "suggest_matching_config",
    "train_valid_test_split",
    "unicode_normalize",
]
