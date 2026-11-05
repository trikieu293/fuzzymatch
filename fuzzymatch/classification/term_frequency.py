"""Term-frequency adjustments for match posteriors (v1.3 Phase A).

Fellegi-Sunter treats agreement levels as exchangeable across values, but a
collision on a rare value ("zz-q4-9911") is stronger evidence than a collision
on a ubiquitous one ("rent expense"). Following Splink's term-frequency
customization, we correct match posteriors in log-odds space using corpus
value frequencies: rare right-side values add bits, common values add ~none.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

_EPS = 1e-9

# Minimum per-field similarity for a field to contribute a TF bonus.
# Chosen by ER-Magellan A/B sweep (experiments/tf_bar_sweep.py): 0.9
# dominates lower bars on every dataset without triggering the correlated-
# field regressions seen at 0.7.
TF_AGREEMENT_BAR = 0.9


def value_frequencies(values: Iterable[object]) -> dict[str, float]:
    """Normalized-value frequency shares over one column's corpus.

    Values are casefolded/whitespace-stripped; shares sum to ~1.0.
    """
    counts: Counter[str] = Counter(str(v).strip().casefold() for v in values if v is not None)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {value: count / total for value, count in counts.items()}


def logit2(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return float(np_log2(p / (1.0 - p)))


def sigmoid2(x: float) -> float:
    return float(1.0 / (1.0 + 2.0 ** (-x)))


def tf_bonus(freq: float, max_bits: float = 3.0) -> float:
    """Evidence bonus (bits) for agreeing on a value of corpus frequency freq.

    Ubiquitous values contribute ~0; the rarest values cap at ``max_bits``.
    """
    return min(max_bits, max(0.0, float(np_log2(1.0 / (max(freq, 0.0) + _EPS)))))


def np_log2(x: float) -> float:
    from math import log2

    return log2(max(x, _EPS))
