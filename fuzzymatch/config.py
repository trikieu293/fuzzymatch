"""Serializable configuration summaries for matching pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Domain-neutral summary of a configured matching pipeline."""

    link_type: str
    blocker: str
    normalizer_steps: tuple[str, ...]
    scorer: str
    scoring_method: str
    scoring_fields: tuple[str, ...]
    classifier: str | None
    clusterer: str | None
    library_version: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "link_type": self.link_type,
            "blocker": self.blocker,
            "normalizer_steps": list(self.normalizer_steps),
            "scorer": self.scorer,
            "scoring_method": self.scoring_method,
            "scoring_fields": list(self.scoring_fields),
            "classifier": self.classifier,
            "clusterer": self.clusterer,
            "library_version": self.library_version,
        }
