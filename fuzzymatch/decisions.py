"""Generic domain-neutral match decision and explanation artifacts.

Provides typed, frozen data classes for representing match/non-match
decisions together with their human-readable explanations, suitable for
any domain (deduplication, record linkage, fuzzy join, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MatchStatus = Literal["match", "possible", "non_match"]
"""Three-zone match status for Fellegi-Sunter style classification."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldScore:
    """Score for a single field-level comparison.

    Attributes:
        field: Name of the field being compared (e.g. ``"name"``).
        left_value: Raw value from the left-hand record.
        right_value: Raw value from the right-hand record.
        score: Similarity score in [0.0, 1.0] (normalised by the scorer).
    """

    field: str
    left_value: object
    right_value: object
    score: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "field": self.field,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "score": self.score,
        }


@dataclass(frozen=True)
class Explanation:
    """Human-readable explanation for why a pair received its match status.

    Attributes:
        field_scores: One :class:`FieldScore` per compared field.
        threshold_band: The match/possible/non_match band that was applied.
        blocking_rules: Names of the blocking rules that brought this pair
            into the candidate set (empty for non-blocked comparisons).
        classifier_weights: Per-field weights used by the classifier, if any.
    """

    field_scores: tuple[FieldScore, ...]
    threshold_band: MatchStatus
    blocking_rules: tuple[str, ...] = ()
    classifier_weights: dict[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary representation."""
        result: dict[str, object] = {
            "field_scores": [fs.to_dict() for fs in self.field_scores],
            "threshold_band": self.threshold_band,
            "blocking_rules": list(self.blocking_rules),
        }
        if self.classifier_weights is not None:
            result["classifier_weights"] = self.classifier_weights
        return result


@dataclass(frozen=True)
class MatchDecision:
    """Final match decision for a record pair, including its explanation.

    Attributes:
        left_id: Identifier of the left-hand record.
        right_id: Identifier of the right-hand record.
        status: The three-zone match status.
        score: Aggregated similarity score in [0.0, 1.0].
        explanation: Full :class:`Explanation` for the decision.
    """

    left_id: object
    right_id: object
    status: MatchStatus
    score: float
    explanation: Explanation

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "status": self.status,
            "score": self.score,
            "explanation": self.explanation.to_dict(),
        }
