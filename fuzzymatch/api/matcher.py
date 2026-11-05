"""High-level Matcher: unified entry point for dedup / link / fuzzy_join modes."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from .. import __version__
from ..blocking import Blocker, CandidateIndex
from ..classification import FSClassifier, SklearnClassifier
from ..clustering import ConnectedComponents
from ..config import PipelineConfig
from ..decisions import Explanation, FieldScore, MatchDecision, MatchStatus
from ..normalize import Normalizer
from ..scoring import RapidFuzzScorer

LinkType = str  # "dedupe" | "link" | "fuzzy_join"


class Matcher:
    """High-level unified entry point for text fuzzy mapping.

    Three task modes:
      - "dedupe":      self-join within one DataFrame; outputs clusters.
      - "link":        cross-file join between two DataFrames; outputs pairs.
      - "fuzzy_join":  SQL-style merge on a fuzzy key; outputs merged DataFrame.

    Example:
        matcher = fm.Matcher(
            link_type="dedupe",
            blocking=fm.ExactBlock(["surname"]),
            classifier=fm.FSClassifier(),
            clusterer=fm.ConnectedComponents(),
        )
        matcher.fit(df)
        pairs = matcher.predict(threshold="match")
        clusters = matcher.clusters()
    """

    def __init__(
        self,
        link_type: LinkType = "dedupe",
        blocking: Blocker | None = None,
        normalizer: Normalizer | None = None,
        scorer: RapidFuzzScorer | None = None,
        classifier: FSClassifier | SklearnClassifier | None = None,
        clusterer: ConnectedComponents | None = None,
        scoring_method: str = "token_sort_ratio",
        scoring_fields: list[str] | None = None,
        scoring_methods: dict[str, str] | None = None,
        term_frequency: bool = False,
        blocking_backend: str = "pandas",
    ):
        if link_type not in ("dedupe", "link", "fuzzy_join"):
            raise ValueError(f"Unknown link_type: {link_type}")
        if blocking is None:
            raise ValueError("Matcher requires a blocking strategy")
        if classifier is None and link_type in ("dedupe", "link"):
            raise ValueError("Matcher requires a classifier for dedupe/link modes")
        from ..scoring.rapidfuzz_wrapper import SUPPORTED_METHODS

        for field_name, method in (scoring_methods or {}).items():
            if method not in SUPPORTED_METHODS:
                raise ValueError(
                    f"Unknown scoring method {method!r} for field {field_name!r}. "
                    f"Supported: {SUPPORTED_METHODS}"
                )

        self.link_type = link_type
        self.blocking = blocking
        self.normalizer = normalizer
        self.scorer = scorer or RapidFuzzScorer()
        self.classifier = classifier
        self.clusterer = clusterer or ConnectedComponents()
        self.scoring_method = scoring_method
        self.scoring_fields = scoring_fields or []
        self.scoring_methods = dict(scoring_methods) if scoring_methods else {}
        self.term_frequency = term_frequency
        self.term_frequencies_: dict[str, dict[str, float]] = {}
        if blocking_backend not in ("pandas", "duckdb"):
            raise ValueError(
                f"Unknown blocking_backend: {blocking_backend!r} (expected 'pandas' or 'duckdb')"
            )
        self.blocking_backend = blocking_backend

        self._df_left: pd.DataFrame | None = None
        self._df_right: pd.DataFrame | None = None
        self._candidates: CandidateIndex | None = None
        self._features: pd.DataFrame | None = None
        self._proba: pd.Series | None = None

    def pipeline_config(self) -> PipelineConfig:
        """Return a serializable summary of this matcher configuration."""
        normalizer_steps: tuple[str, ...]
        if self.normalizer is None:
            normalizer_steps = ()
        else:
            normalizer_steps = tuple(
                getattr(step, "__name__", type(step).__name__) for step in self.normalizer.steps
            )

        return PipelineConfig(
            link_type=self.link_type,
            blocker=type(self.blocking).__name__,
            normalizer_steps=normalizer_steps,
            scorer=type(self.scorer).__name__,
            scoring_method=self.scoring_method,
            scoring_fields=tuple(self.scoring_fields),
            classifier=type(self.classifier).__name__ if self.classifier is not None else None,
            clusterer=type(self.clusterer).__name__ if self.clusterer is not None else None,
            library_version=__version__,
        )

    def fit(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame | None = None,
        labels: pd.Series | None = None,
        chunked: bool = False,
        num_chunks: int = 8,
    ) -> Matcher:
        """Run blocking + (optional) scoring + (optional) classifier training.

        For dedupe: pass only df_left.
        For link / fuzzy_join: pass df_left and df_right.
        labels is optional supervised signal for the classifier.

        With ``chunked=True`` (cross-file only) the right frame is split into
        ``num_chunks`` deterministic partitions blocked independently; the
        union of candidates is identical to an unchunked fit while peak memory
        stays bounded by one chunk's worth of right rows.
        """
        if chunked and num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        if self.normalizer is not None:
            # Use union of blocking field names as default columns
            cols = self.scoring_fields or list(df_left.columns)
            df_left = self.normalizer.transform(df_left, cols)
            if df_right is not None:
                df_right = self.normalizer.transform(df_right, cols)

        self._df_left = df_left
        self._df_right = df_right
        if chunked and df_right is not None:
            self._candidates = self._block_chunked(df_left, df_right, num_chunks)
        else:
            self._candidates = self._generate_candidates(df_left, df_right)

        # Score candidate pairs (if scoring fields are set and scorer provided)
        if self.scoring_fields:
            self._features = self._build_feature_matrix()
            if self.classifier is not None:
                self.classifier.fit(self._features, labels)
                self._proba = self.classifier.predict_proba(self._features)
            self._compute_term_frequencies()
            if self.term_frequency:
                self._apply_term_frequency_adjustment()

        return self

    def _compute_term_frequencies(self) -> None:
        """Per-field normalized-value frequency shares over the right corpus."""
        from ..classification.term_frequency import value_frequencies

        right = self._df_right if self._df_right is not None else self._df_left
        assert right is not None and self._features is not None
        base_fields = [c for c in self._features.columns if not c.endswith("__missing")]
        self.term_frequencies_ = {
            f: value_frequencies(right[f]) for f in base_fields if f in right.columns
        }

    def _apply_term_frequency_adjustment(self) -> None:
        """Splink-style TF correction: rare right-value collisions add bits.

        Per pair, fields whose similarity reaches the agreement bar earn a
        bonus of ``tf_bonus(freq(right_value))`` bits in log2-odds space.
        """
        from ..classification.term_frequency import (
            TF_AGREEMENT_BAR,
            logit2,
            sigmoid2,
            tf_bonus,
        )

        right = self._df_right if self._df_right is not None else self._df_left
        assert right is not None and self._features is not None
        base_fields = [c for c in self._features.columns if not c.endswith("__missing")]
        assert self._proba is not None and self._candidates is not None

        agreement_bar = TF_AGREEMENT_BAR
        adjusted = []
        pairs = self._candidates.pairs
        right_index = {label: pos for pos, label in enumerate(right.index)}
        for i, (_, right_label) in enumerate(pairs):
            pos = right_index[right_label]
            x = logit2(float(self._proba.iloc[i]))
            # Max (not sum) across agreeing fields: rarity of one shared
            # value is the evidence; stacking correlated fields would
            # triple-count it.
            field_bonuses = [
                tf_bonus(
                    self.term_frequencies_.get(f, {}).get(
                        str(right[f].iloc[pos]).strip().casefold(), 0.0
                    )
                )
                for f in base_fields
                if float(self._features[f].iloc[i]) >= agreement_bar
            ]
            if field_bonuses:
                x += max(field_bonuses)
            adjusted.append(sigmoid2(x))
        self._proba = pd.Series(adjusted, index=self._proba.index, name="proba")

    def decision_probabilities(self) -> list[float]:
        """Posterior match probabilities for the candidate pairs."""
        if self._proba is None:
            raise RuntimeError("Call fit() before decision_probabilities()")
        return [float(p) for p in self._proba]

    def _generate_candidates(
        self, df_left: pd.DataFrame, df_right: pd.DataFrame | None
    ) -> CandidateIndex:
        """Candidate generation via the configured backend.

        'duckdb' pushes exact/sorted-neighborhood blocking into SQL; any
        other strategy (or backend) falls through to the pandas blockers.
        """
        if self.blocking_backend == "duckdb":
            if df_right is None:
                raise NotImplementedError(
                    "DuckDB backend currently supports cross-file (link) "
                    "blocking only; use blocking_backend='pandas' for dedupe."
                )
            from ..backends.duckdb_backend import DuckDBBackend

            backend = DuckDBBackend()
            try:
                pairs = backend.block_for_blocker(df_left, df_right, self.blocking)
            finally:
                backend.close()
            return CandidateIndex(pairs, df_left.index, df_right.index)
        return self.blocking.block(df_left, df_right)

    def _block_chunked(
        self, df_left: pd.DataFrame, df_right: pd.DataFrame, num_chunks: int
    ) -> CandidateIndex:
        """Block left x right in right-frame partitions; union is exact."""
        row_hashes = pd.util.hash_pandas_object(
            df_right.astype(str).apply("|".join, axis=1), index=False
        )
        # Deterministic interleaved partitioning keeps near-duplicate-heavy
        # ranges from concentrating in one chunk.
        order = row_hashes.sort_values(kind="stable").index.tolist()
        partitions = [order[i::num_chunks] for i in range(num_chunks)]

        all_pairs: list[tuple[int, int]] = []
        for part_labels in partitions:
            sub_right = df_right.loc[part_labels]
            part_index = self.blocking.block(df_left, sub_right)
            all_pairs.extend(part_index.pairs)
        return CandidateIndex(all_pairs, df_left.index, df_right.index)

    def _build_feature_matrix(self) -> pd.DataFrame:
        """Compute pairwise similarity scores for each candidate pair."""
        if self._candidates is None or self._df_left is None:
            raise RuntimeError("Call fit() before _build_feature_matrix")
        left = self._df_left
        right = self._df_right if self._df_right is not None else self._df_left
        pairs = self._candidates.pairs
        left_labels = [li for li, _ in pairs]
        right_labels = [ri for _, ri in pairs]
        # One vectorized reindex per field instead of per-pair .loc scalars.
        col_scores: dict[str, pd.Series] = {}
        missing_flags: dict[str, pd.Series] = {}
        for field in self.scoring_fields:
            if field not in left.columns:
                continue
            left_vals = pd.Series(
                left[field].reindex(left_labels).to_numpy(),
                index=range(len(pairs)),
            )
            right_vals = pd.Series(
                right[field].reindex(right_labels).to_numpy(),
                index=range(len(pairs)),
            )
            method = self.scoring_methods.get(field, self.scoring_method)
            col_scores[field] = self.scorer.score_pair(method, left_vals, right_vals)
            missing_flags[f"{field}__missing"] = (
                left_vals.str.strip().str.lower().isin(["", "nan", "none"])
                | right_vals.str.strip().str.lower().isin(["", "nan", "none"])
            ).astype(int)  # mirrors scoring._MISSING markers
        features = pd.DataFrame(col_scores)
        for name, flags in missing_flags.items():
            if flags.any():
                features[name] = flags.to_numpy()
        return features

    def _status_for_score(self, score: float) -> MatchStatus:
        """Convert a numeric score to a generic threshold band."""
        if self.classifier is None:
            if score >= 0.8:
                return "match"
            if score >= 0.5:
                return "possible"
            return "non_match"
        if score >= self.classifier.match_thresh:
            return "match"
        if score >= self.classifier.possible_lower:
            return "possible"
        return "non_match"

    def _blocking_rule_names(self) -> tuple[str, ...]:
        """Return stable names for the configured blocker."""
        blocker = self.blocking
        nested = getattr(blocker, "blockers", None)
        if nested is not None:
            return tuple(type(item).__name__ for item in nested)
        return (type(blocker).__name__,)

    def decisions(self) -> list[MatchDecision]:
        """Return generic match decisions for all candidate pairs.

        The returned objects are domain-neutral and can be interpreted by
        downstream packages such as finance, manufacturing, or supply-chain
        workflow layers.
        """
        if self._candidates is None or self._df_left is None:
            raise RuntimeError("Call fit() before decisions()")
        if self._features is None:
            raise RuntimeError(
                "Matcher.decisions() requires scoring_fields so explanations can be built"
            )

        right = self._df_right if self._df_right is not None else self._df_left
        if self._proba is not None:
            scores = (
                self._proba.astype(float)
                if self._proba is not None
                else self._features.mean(axis=1).astype(float)
            )
        else:
            base_cols = [c for c in self._features.columns if not c.endswith("__missing")]
            scores = self._features[base_cols].mean(axis=1).astype(float)
        return self._build_decisions(
            pairs=self._candidates.pairs,
            left=self._df_left,
            right=right,
            features=self._features,
            scores=scores,
        )

    def _build_decisions(
        self,
        pairs: list[tuple[Any, Any]],
        left: pd.DataFrame,
        right: pd.DataFrame,
        features: pd.DataFrame,
        scores: pd.Series,
    ) -> list[MatchDecision]:
        """Shared decision construction over an arbitrary candidate set."""
        scores_arr = scores.to_numpy()
        left_labels = [li for li, _ in pairs]
        right_labels = [ri for _, ri in pairs]

        # Precompute per-field arrays once; no per-pair .loc/iloc scalars.
        field_arrays: dict[str, tuple[list[object], list[object], list[float]]] = {}
        for field in features.columns:
            if field.endswith("__missing"):
                continue
            left_vals = left[field].reindex(left_labels).to_numpy().tolist()
            right_vals = right[field].reindex(right_labels).to_numpy().tolist()
            feat_vals = features[field].astype(float).to_numpy().tolist()
            field_arrays[field] = (left_vals, right_vals, feat_vals)
        field_names = list(field_arrays)

        decisions: list[MatchDecision] = []
        blocking_rules = self._blocking_rule_names()
        weights = getattr(self.classifier, "weights_", None)
        classifier_weights = dict(weights) if weights else None

        append = decisions.append
        for i, (left_pos, right_pos) in enumerate(pairs):
            score = float(scores_arr[i])
            status = self._status_for_score(score)
            field_scores = tuple(
                FieldScore(
                    field=field,
                    left_value=field_arrays[field][0][i],
                    right_value=field_arrays[field][1][i],
                    score=float(field_arrays[field][2][i]),
                )
                for field in field_names
            )
            explanation = Explanation(
                field_scores=field_scores,
                threshold_band=status,
                blocking_rules=blocking_rules,
                classifier_weights=classifier_weights,
            )
            append(
                MatchDecision(
                    left_id=left_pos,
                    right_id=right_pos,
                    status=status,
                    score=score,
                    explanation=explanation,
                )
            )
        return decisions

    def match_right_batch(self, df_new: pd.DataFrame) -> list[MatchDecision]:
        """Score a new right-frame batch against the stored left frame.

        Reuses the fitted classifier — no retraining. The new batch is
        normalized with the matcher's own normalizer and blocked against the
        stored left side; returned decisions reference the ORIGINAL left
        labels and positional indices within ``df_new`` as right ids.

        Requires a prior cross-file ``fit()`` (both frames provided).
        """
        if self._df_left is None or self._df_right is None:
            raise RuntimeError("Call fit(df_left, df_right) before match_right_batch()")
        df_new_local = df_new.copy()
        if self.normalizer is not None:
            cols = self.scoring_fields or list(df_new_local.columns)
            df_new_local = self.normalizer.transform(df_new_local, cols)

        part_index = self.blocking.block(self._df_left, df_new_local)
        left = self._df_left
        pairs = part_index.pairs
        left_labels = [li for li, _ in pairs]
        right_labels = [ri for _, ri in pairs]

        col_scores: dict[str, pd.Series] = {}
        for field in self.scoring_fields:
            if field not in left.columns:
                continue
            left_vals = pd.Series(
                left[field].reindex(left_labels).to_numpy(),
                index=range(len(pairs)),
            )
            right_vals = pd.Series(
                df_new_local[field].reindex(right_labels).to_numpy(),
                index=range(len(pairs)),
            )
            method = self.scoring_methods.get(field, self.scoring_method)
            col_scores[field] = self.scorer.score_pair(method, left_vals, right_vals)
        features = pd.DataFrame(col_scores)
        proba = self.classifier.predict_proba(features) if self.classifier is not None else None
        scores = proba.astype(float) if proba is not None else features.mean(axis=1).astype(float)
        return self._build_decisions(
            pairs=pairs,
            left=left,
            right=df_new_local,
            features=features,
            scores=scores,
        )

    def iter_decision_chunks(self, chunk_size: int = 50_000) -> Iterator[list[MatchDecision]]:
        """Yield decisions in bounded batches without full materialisation.

        Equivalent to :meth:`decisions()` but memory-bounded: suitable when
        candidate counts reach millions and only aggregated/thresholded
        consumption is needed. Requires the same preconditions as
        :meth:`decisions`.
        """
        if self._candidates is None or self._df_left is None:
            raise RuntimeError("Call fit() before iter_decision_chunks()")
        if self._features is None or chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        right = self._df_right if self._df_right is not None else self._df_left
        scores = (
            self._proba.astype(float)
            if self._proba is not None
            else self._features[[c for c in self._features.columns if not c.endswith("__missing")]]
            .mean(axis=1)
            .astype(float)
        )
        pairs = self._candidates.pairs
        for start in range(0, len(pairs), chunk_size):
            sl = slice(start, start + chunk_size)
            yield self._build_decisions(
                pairs=pairs[sl],
                left=self._df_left,
                right=right,
                features=self._features.iloc[sl].reset_index(drop=True),
                scores=scores.iloc[sl].reset_index(drop=True),
            )

    def iter_decisions(self, chunk_size: int = 50_000) -> Iterator[MatchDecision]:
        """Yield individual decisions lazily (see iter_decision_chunks)."""
        for batch in self.iter_decision_chunks(chunk_size=chunk_size):
            yield from batch

    def predict(self, threshold: str = "match") -> pd.MultiIndex:
        """Return candidate pairs filtered by the classifier threshold."""
        if self._candidates is None:
            raise RuntimeError("Call fit() before predict()")
        if self._proba is None or self.classifier is None:
            return self._candidates.to_multiindex()
        if self._features is None:
            raise RuntimeError("Call fit() before predict()")
        mask = self.classifier.predict(self._features, threshold=threshold)
        kept_pairs = [
            self._candidates.pairs[i] for i in range(len(self._candidates.pairs)) if mask.iloc[i]
        ]
        return pd.MultiIndex.from_tuples(kept_pairs, names=("left", "right"))

    def clusters(self, threshold: str = "match") -> list[set[int]]:
        """Cluster matched pairs into entity clusters (dedupe mode)."""
        if self.link_type != "dedupe":
            raise ValueError("clusters() only valid for link_type='dedupe'")
        idx = self.predict(threshold)
        return self.clusterer.resolve(idx)

    def join(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        on: str,
        threshold: float = 0.5,
        labels: pd.Series | None = None,
    ) -> pd.DataFrame:
        """fuzzy_join mode: return merged DataFrame on fuzzy key `on`.

        Each left row maps to its best-matching right row above `threshold`.
        Scoring uses the matcher's configured ``scoring_fields``; ``on`` names
        the logical join key and must exist in both frames.

        ``labels`` is an optional supervised signal aligned with the candidate
        pair order produced by the blocker (1 = match, 0 = non-match).
        """
        if self.link_type != "fuzzy_join":
            raise ValueError("join() requires link_type='fuzzy_join'")
        if on not in df_left.columns or on not in df_right.columns:
            raise ValueError(f"join key {on!r} must exist in both frames")
        if self.classifier is None:
            raise ValueError("join() requires a classifier to score candidate pairs")
        if not self.scoring_fields:
            raise ValueError("join() requires scoring_fields to score candidate pairs")
        self.fit(df_left, df_right, labels=labels)
        candidates = self._candidates
        if candidates is None:
            raise RuntimeError("Call fit() before join()")
        # Validations above guarantee fit() produced probabilities.
        assert self._proba is not None
        mask = self._proba >= threshold
        kept = [candidates.pairs[i] for i in range(len(candidates.pairs)) if mask.iloc[i]]
        overlap = [c for c in df_left.columns if c in df_right.columns]
        left_renamed = df_left.rename(columns={c: f"{c}_left" for c in overlap})
        right_renamed = df_right.rename(columns={c: f"{c}_right" for c in overlap})
        left_pos = [left_pos for left_pos, _ in kept]
        right_pos = [right_pos for _, right_pos in kept]
        merged = pd.merge(
            left_renamed.iloc[left_pos].reset_index(drop=True),
            right_renamed.iloc[right_pos].reset_index(drop=True),
            left_index=True,
            right_index=True,
        )
        return merged
