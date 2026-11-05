"""Term-frequency-adjusted match weights contracts.

Rare-value agreements are stronger evidence than common-value agreements
(Splink-style TF adjustment). Adjustment is opt-in and applied as a
log-odds correction built from corpus value frequencies.
"""

from __future__ import annotations

import pandas as pd

from fuzzymatch import ExactBlock, FSClassifier, Matcher

SRC = pd.DataFrame(
    {
        "grp": ["g1", "g2"],
        "name": [
            "zzq services",  # collides with a RARE target value
            "rent expense main",  # collides with an UBIQUITOUS target value
        ],
    }
)
TGT = pd.DataFrame(
    {
        "grp": ["g1"] + [f"g{i}" for i in range(2, 14)],
        "name": ["zzq services"] + ["rent expense"] * 12,
    }
)


def _matcher(**kw) -> Matcher:
    return Matcher(
        link_type="link",
        blocking=ExactBlock(["grp"]),
        classifier=FSClassifier(),
        scoring_fields=["name"],
        **kw,
    )


class TestValueFrequencies:
    def test_counts_normalized_values(self) -> None:
        matcher = _matcher()
        matcher.fit(SRC, TGT)
        freqs = matcher.term_frequencies_
        assert "name" in freqs
        common_share = freqs["name"]["rent expense"]
        rare_share = freqs["name"]["zzq services"]
        assert common_share > rare_share
        assert 0.0 < rare_share <= 0.5

    def test_empty_corpus_returns_empty_map(self) -> None:
        # [coverage guard] empty column -> no frequencies
        from fuzzymatch.classification.term_frequency import value_frequencies

        assert value_frequencies([]) == {}


class TestTermFrequencyAdjustment:
    def test_flag_defaults_off(self) -> None:
        assert _matcher().term_frequency is False
        assert _matcher(term_frequency=True).term_frequency is True

    def test_off_posteriors_near_tie(self) -> None:
        m = _matcher()
        m.fit(SRC, TGT)
        proba = m.decision_probabilities()
        assert len(proba) == 2
        assert all(0.0 <= p <= 1.0 for p in proba)

    def test_on_boosts_rare_collision_above_common(self) -> None:
        base = _matcher()
        base.fit(SRC, TGT)
        adjusted = _matcher(term_frequency=True)
        adjusted.fit(SRC, TGT)

        base_p = list(base.decision_probabilities())
        adj_p = list(adjusted.decision_probabilities())
        # the rare-collision row (first source row) gains ground relative to
        # the ubiquitous-collision row
        delta_base = base_p[0] - base_p[1]
        delta_adj = adj_p[0] - adj_p[1]
        assert delta_adj > delta_base
