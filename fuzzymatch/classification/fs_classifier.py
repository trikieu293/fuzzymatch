"""Fellegi-Sunter probabilistic classifier with three-zone thresholding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_SCHEMA = "fuzzymatch.fs_classifier"
_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# EM constants
# ---------------------------------------------------------------------------

_N_BINS = 3
_BIN_QUANTILES = (0.5, 0.8)
_SMOOTHING = 0.005
_MAX_ITER = 100
_TOL = 1e-6
_SCORE_CLIP = 30.0
_M_INIT: tuple[float, float, float] = (0.15, 0.35, 0.55)
_LAMBDA_INIT = 0.1
# Below this many candidates a mixture model is unidentifiable (no population
# statistics); fall back to the legacy agreement-mean estimator.
_MIN_ROWS_FOR_EM = 10


_MISSING_SUFFIX = "__missing"


def _presence_matrix(
    n_rows: int,
    base_cols: list[str],
    features: pd.DataFrame,
) -> NDArray[np.float64]:
    """Per-row multipliers (n x k): 0 where the field was missing."""
    presence = np.ones((n_rows, len(base_cols)), dtype=float)
    for j, col in enumerate(base_cols):
        flag_col = col + _MISSING_SUFFIX
        if flag_col in features.columns:
            presence[:, j] -= features[flag_col].to_numpy(dtype=float)
    return np.clip(presence, 0.0, 1.0)


class FSClassifier:
    """Fellegi-Sunter probabilistic classifier.

    Unsupervised mode (labels=None): proper Fellegi-Sunter with EM (Winkler),
    over binned agreement levels. Each field's candidate scores are binned into
    3 ordered levels at that field's own q50/q80 quantiles (label-free, adapts
    per dataset and per field). EM alternates:

      E-step: p_i = sigmoid2(log2(lam/(1-lam)) + sum_f log2(m_f(b_if)/u_f(b_if)))
      M-step: lam = mean(p); m_f(b) ~ weighted share among matches;
              u_f(b) ~ weighted share among non-matches

    Fields whose score distribution carries no signal end up with m ~ u across
    bins and thus ~zero weight, so unreliable fields are down-weighted
    automatically instead of dragging an aggregate mean down. Because agreeing
    fields contribute several bits each, the default match band (>= 0.8) is
    reachable out-of-box.

    Supervised mode (labels provided): m/u estimated directly from labeled
    agreement means (legacy behaviour, kept for backward compatibility).

    Three-zone thresholding:
      - match:      P >= match_thresh  (positive decision)
      - possible:   possible_lower <= P < match_thresh (manual review)
      - non-match:  P < possible_lower (negative decision)

    Public attributes ``m_``, ``u_``, ``weights_`` report the highest-bin
    (strong-agreement) estimates; the full per-bin tables are kept privately.
    """

    def __init__(
        self,
        threshold: tuple[float, float, float] = (0.8, 0.5, 0.2),
    ):
        self.match_thresh = threshold[0]
        self.possible_lower = threshold[1]
        self.possible_upper = threshold[2]
        self.m_: dict[str, float] = {}
        self.u_: dict[str, float] = {}
        self.weights_: dict[str, float] = {}
        self.lambda_: float = 0.1  # prior probability of match
        self._em_mode: bool = False
        self._edges: dict[str, NDArray[np.float64]] = {}
        self._m_table: dict[str, NDArray[np.float64]] = {}
        self._u_table: dict[str, NDArray[np.float64]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | None = None,
    ) -> FSClassifier:
        """Estimate m, u per feature from labels (supervised) or via EM (unsupervised)."""
        self._em_mode = labels is None and len(features) >= _MIN_ROWS_FOR_EM
        self._edges = {}
        self._m_table = {}
        self._u_table = {}

        for col in features.columns:
            vals = features[col].astype(float)
            if labels is not None:
                matches = vals[labels == 1]
                non_matches = vals[labels == 0]
                if len(matches) > 0:
                    self.m_[col] = float(np.clip(matches.mean(), 0.5, 1.0))
                else:
                    self.m_[col] = 0.8
                if len(non_matches) > 0:
                    self.u_[col] = float(np.clip(non_matches.mean(), 1e-6, 0.5))
                else:
                    self.u_[col] = 0.05
            else:
                # Legacy agreement-mean estimator (also the small-n fallback).
                mean = float(vals.mean())
                self.m_[col] = float(np.clip(mean, 0.5, 1.0))
                self.u_[col] = float(np.clip(1.0 - mean, 0.01, 0.5))
            ratio = self.m_[col] / max(self.u_[col], 1e-10)
            self.weights_[col] = float(np.log2(ratio)) if ratio > 0 else 0.0

        if self._em_mode:
            self._fit_em(features)

        return self

    def _fit_em(self, features: pd.DataFrame) -> None:
        """Run the binned Fellegi-Sunter EM on the candidate features."""
        cols = [c for c in features.columns if not c.endswith(_MISSING_SUFFIX)]
        n = len(features)
        presence = _presence_matrix(n, cols, features)

        levels = np.empty((n, len(cols)), dtype=np.intp)
        marginals = np.empty((len(cols), _N_BINS), dtype=float)
        for j, col in enumerate(cols):
            vals = features[col].astype(float).to_numpy()
            obs_mask = presence[:, j] > 0
            if np.any(obs_mask):
                edges = np.asarray(np.quantile(vals[obs_mask], _BIN_QUANTILES), dtype=np.float64)
            else:
                edges = np.asarray(_BIN_QUANTILES, dtype=np.float64)
            self._edges[col] = edges
            levels[:, j] = np.digitize(vals, edges)
            if np.any(obs_mask):
                counts = np.bincount(levels[obs_mask, j], minlength=_N_BINS).astype(float)
                counts = (counts + _SMOOTHING) / (counts.sum() + _SMOOTHING * _N_BINS)
                marginals[j] = counts
            else:
                marginals[j] = np.full(_N_BINS, 1.0 / _N_BINS, dtype=float)

        # Init: u starts at the observed bin marginals (most candidates are
        # non-matches); m starts optimistic toward higher bins.
        m_tab = np.tile(np.array(_M_INIT, dtype=float), (len(cols), 1))
        u_tab = marginals.copy()

        lam = _LAMBDA_INIT
        for _ in range(_MAX_ITER):
            p = self._posterior(levels, m_tab, u_tab, lam, presence=presence)
            new_lam = float(np.clip(p.mean(), _SMOOTHING, 1.0 - _SMOOTHING))
            new_m = np.empty_like(m_tab)
            new_u = np.empty_like(u_tab)
            for j in range(len(cols)):
                lv = levels[:, j]
                obs = presence[:, j]
                pos = float((p * obs).sum())
                neg = float(((1.0 - p) * obs).sum())
                for b in range(_N_BINS):
                    mask = (lv == b).astype(float)
                    new_m[j, b] = (float((p * obs * mask).sum()) + _SMOOTHING) / (
                        pos + _SMOOTHING * _N_BINS
                    )
                    new_u[j, b] = (float(((1.0 - p) * obs * mask).sum()) + _SMOOTHING) / (
                        neg + _SMOOTHING * _N_BINS
                    )
            delta = max(
                abs(new_lam - lam),
                float(np.abs(new_m - m_tab).max()),
                float(np.abs(new_u - u_tab).max()),
            )
            m_tab, u_tab, lam = new_m, new_u, new_lam
            if delta < _TOL:
                break

        self.lambda_ = lam
        for j, col in enumerate(cols):
            self._m_table[col] = m_tab[j]
            self._u_table[col] = u_tab[j]
            self.m_[col] = float(m_tab[j][_N_BINS - 1])
            self.u_[col] = float(u_tab[j][_N_BINS - 1])
            ratio = m_tab[j][_N_BINS - 1] / max(u_tab[j][_N_BINS - 1], 1e-10)
            self.weights_[col] = float(np.log2(ratio))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _bin_columns(self, features: pd.DataFrame) -> NDArray[np.intp]:
        """Bin features with stored edges -> int levels array (n, k)."""
        cols = list(self._edges.keys())
        levels = np.empty((len(features), len(cols)), dtype=np.intp)
        for j, col in enumerate(cols):
            vals = features[col].astype(float).to_numpy()
            levels[:, j] = np.digitize(vals, self._edges[col])
        return levels

    def _posterior(
        self,
        levels: NDArray[np.intp],
        m_tab: NDArray[np.float64],
        u_tab: NDArray[np.float64],
        lam: float,
        presence: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """E-step: P(match | agreement pattern) for every candidate row.

        ``presence`` (n x k, values in [0, 1]) down-weights fields that were
        missing for a row so they contribute no likelihood evidence.
        """
        prior = float(np.log2(lam / (1.0 - lam)))
        score = np.full(levels.shape[0], prior, dtype=float)
        for j in range(levels.shape[1]):
            lr = np.log2(m_tab[j][levels[:, j]] / u_tab[j][levels[:, j]])
            if presence is not None:
                lr = lr * presence[:, j]
            score += lr
        score = np.clip(score, -_SCORE_CLIP, _SCORE_CLIP)
        return cast(NDArray[np.float64], 2.0**score / (1.0 + 2.0**score))

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Compute Fellegi-Sunter match probability for each row."""
        if not self.weights_:
            raise RuntimeError("FSClassifier must be fit() before predict_proba")
        if self._em_mode:
            levels = self._bin_columns(features)
            cols = list(self._edges.keys())
            m_tab = np.stack([self._m_table[c] for c in cols])
            u_tab = np.stack([self._u_table[c] for c in cols])
            presence = _presence_matrix(len(features), cols, features)
            proba = self._posterior(levels, m_tab, u_tab, self.lambda_, presence=presence)
            return pd.Series(proba, index=features.index).clip(0.0, 1.0)
        score = features.apply(
            lambda row: sum(row.get(c, 0.0) * w for c, w in self.weights_.items()),
            axis=1,
        )
        prior = np.log2(self.lambda_ / max(1.0 - self.lambda_, 1e-10))
        score = score + prior
        proba = 2.0**score / (1.0 + 2.0**score)
        return cast("pd.Series[float]", proba.clip(0, 1))

    def predict(self, features: pd.DataFrame, threshold: str = "match") -> pd.Series:
        """Return boolean mask. threshold in {'match', 'possible', 'non-match'}."""
        proba = self.predict_proba(features)
        if threshold == "match":
            return proba >= self.match_thresh
        elif threshold == "possible":
            return (proba >= self.possible_lower) & (proba < self.match_thresh)
        elif threshold == "non_match" or threshold == "non-match":
            return proba < self.possible_lower
        else:
            raise ValueError(f"Unknown threshold: {threshold}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Serialize the fitted state to JSON with a versioned schema."""
        payload: dict[str, Any] = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "em_mode": self._em_mode,
            "thresholds": [
                self.match_thresh,
                self.possible_lower,
                self.possible_upper,
            ],
            "lambda_": self.lambda_,
            "m_": self.m_,
            "u_": self.u_,
            "weights_": self.weights_,
            "edges": {k: v.tolist() for k, v in self._edges.items()},
            "m_table": {k: v.tolist() for k, v in self._m_table.items()},
            "u_table": {k: v.tolist() for k, v in self._u_table.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> FSClassifier:
        """Restore a classifier saved via :meth:`save`.

        Raises:
            ValueError: If the file's schema name or version is unknown.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != _SCHEMA:
            raise ValueError(f"Unknown schema {payload.get('schema')!r}; expected {_SCHEMA!r}")
        version = payload.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {version!r}; expected {_SCHEMA_VERSION}")
        clf = cls(threshold=tuple(payload["thresholds"]))
        clf._em_mode = bool(payload["em_mode"])
        clf.lambda_ = float(payload["lambda_"])
        clf.m_ = dict(payload["m_"])
        clf.u_ = dict(payload["u_"])
        clf.weights_ = dict(payload["weights_"])
        clf._edges = {k: np.asarray(v, dtype=float) for k, v in payload["edges"].items()}
        clf._m_table = {k: np.asarray(v, dtype=float) for k, v in payload["m_table"].items()}
        clf._u_table = {k: np.asarray(v, dtype=float) for k, v in payload["u_table"].items()}
        return clf
