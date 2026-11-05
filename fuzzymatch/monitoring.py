"""Score-distribution drift monitoring for recurring matching jobs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

# Standard PSI interpretation bands: <0.1 stable, 0.1-0.25 moderate,
# >0.25 significant shift.
DEFAULT_PSI_WARNING = 0.25
# Below this many observations per window, distributional statistics are
# noise; the report carries an explicit insufficient-data status instead.
MIN_OBSERVATIONS = 300
_EPS = 1e-6


def _js_divergence(ref_pct: Any, cur_pct: Any) -> float:
    """Jensen-Shannon divergence (base 2, bounded [0, 1]) of two histograms."""
    m: Any = (ref_pct + cur_pct) / 2.0

    def _kl(p: Any, q: Any) -> float:
        mask: Any = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return round(0.5 * _kl(ref_pct, m) + 0.5 * _kl(cur_pct, m), 6)


def drift_report(
    reference: Iterable[float],
    current: Iterable[float],
    bins: int = 10,
    psi_warning: float = DEFAULT_PSI_WARNING,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, object]:
    """Population Stability Index between two score samples.

    Bin edges are the reference sample's equal-frequency quantiles, so the
    reference is uniform by construction and PSI measures how far ``current``
    drifted from it.

    Windows smaller than ``min_observations`` produce
    ``status="insufficient_data"`` with ``psi=None`` and no warning —
    distributional estimates at that size are dominated by sampling noise.

    Args:
        reference: Baseline score sample (e.g. last good run's decisions).
        current: New score sample to compare.
        bins: Number of quantile bins over the reference.
        psi_warning: PSI at or above which ``warning`` is True.
        min_observations: Minimum per-window count for a full report.

    Returns:
        ``{"status": "ok" | "insufficient_data", "psi": float | None,
        "js_divergence": float | None, "warning": bool,
        "n_reference": int, "n_current": int}``.

    Raises:
        ValueError: If either sample is empty or ``bins`` < 2.
    """
    ref = np.asarray(list(reference), dtype=float)
    cur = np.asarray(list(current), dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("drift_report requires non-empty samples")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")

    n_reference = int(ref.size)
    n_current = int(cur.size)
    if n_reference < min_observations or n_current < min_observations:
        return {
            "status": "insufficient_data",
            "psi": None,
            "js_divergence": None,
            "warning": False,
            "n_reference": n_reference,
            "n_current": n_current,
        }

    raw_edges = np.quantile(ref, [i / bins for i in range(1, bins)])
    # Degenerate/discrete references yield duplicate edges; collapsing them
    # keeps the histogram well-formed and PSI finite.
    edges = np.unique(raw_edges)
    ref_counts = np.histogram(ref, bins=[-np.inf, *edges, np.inf])[0]
    cur_counts = np.histogram(cur, bins=[-np.inf, *edges, np.inf])[0]

    n_bins = len(ref_counts)
    ref_pct = (ref_counts + _EPS) / (ref_counts.sum() + _EPS * n_bins)
    cur_pct = (cur_counts + _EPS) / (cur_counts.sum() + _EPS * n_bins)

    psi = float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))
    return {
        "status": "ok",
        "psi": round(psi, 6),
        "js_divergence": _js_divergence(ref_pct, cur_pct),
        "warning": bool(psi >= psi_warning),
        "n_reference": n_reference,
        "n_current": n_current,
    }
