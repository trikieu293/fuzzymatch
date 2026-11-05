"""Mathematical and property-based invariant verification for core fuzzymatch.

Verifies metric axioms, probability simplex conservation, index equivariance,
missingness neutrality, drift divergence bounds, and DuckDB-Pandas isomorphism.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fuzzymatch.backends.duckdb_backend import DuckDBBackend
from fuzzymatch.blocking import ExactBlock, SortedNeighborhood
from fuzzymatch.classification.fs_classifier import FSClassifier
from fuzzymatch.clustering.star_cluster import StarClusterer
from fuzzymatch.monitoring import drift_report
from fuzzymatch.scoring.date import DateWindowScorer
from fuzzymatch.scoring.numeric import NumericProximityScorer
from fuzzymatch.scoring.rapidfuzz_wrapper import RapidFuzzScorer

# ---------------------------------------------------------------------------
# 1. Metric and Similarity Axioms
# ---------------------------------------------------------------------------


class TestMetricAndSimilarityAxioms:
    """Rigorous verification of metric properties and similarity bounds."""

    @given(st.text(), st.text())
    @settings(max_examples=100)
    def test_similarity_boundedness(self, a: str, b: str) -> None:
        """Similarity functions must map strictly into [0.0, 1.0]."""
        scorer = RapidFuzzScorer()
        for method in [
            scorer.ratio,
            scorer.jaro,
            scorer.jaro_winkler,
            scorer.token_sort_ratio,
            scorer.token_set_ratio,
            scorer.token_overlap,
        ]:
            val = method(a, b)
            assert -1e-12 <= val <= 1.0 + 1e-12

    @given(st.text(), st.text())
    @settings(max_examples=100)
    def test_similarity_symmetry(self, a: str, b: str) -> None:
        """Similarity functions must be symmetric: s(a, b) == s(b, a)."""
        scorer = RapidFuzzScorer()
        for method in [
            scorer.ratio,
            scorer.jaro,
            scorer.jaro_winkler,
            scorer.token_sort_ratio,
            scorer.token_set_ratio,
            scorer.token_overlap,
        ]:
            assert math.isclose(method(a, b), method(b, a), abs_tol=1e-12)

    @given(st.text(min_size=1).filter(lambda s: s.strip().lower() not in {"", "nan", "none"}))
    @settings(max_examples=100)
    def test_identity_of_indiscernibles_non_missing(self, s: str) -> None:
        """For non-empty non-missing strings, s(x, x) == 1.0."""
        scorer = RapidFuzzScorer()
        for method in [
            scorer.ratio,
            scorer.jaro,
            scorer.jaro_winkler,
            scorer.token_sort_ratio,
            scorer.token_set_ratio,
            scorer.token_overlap,
        ]:
            assert math.isclose(method(s, s), 1.0, abs_tol=1e-12)

    @given(st.text(max_size=30), st.text(max_size=30), st.text(max_size=30))
    @settings(max_examples=100)
    def test_levenshtein_metric_triangle_inequality(self, a: str, b: str, c: str) -> None:
        """Levenshtein distance satisfies d(a, c) <= d(a, b) + d(b, c)."""
        scorer = RapidFuzzScorer()
        d_ab = scorer.levenshtein(a, b)
        d_bc = scorer.levenshtein(b, c)
        d_ac = scorer.levenshtein(a, c)
        assert d_ac <= d_ab + d_bc
        assert d_ab >= 0
        assert scorer.levenshtein(a, a) == 0
        assert scorer.levenshtein(a, b) == scorer.levenshtein(b, a)


# ---------------------------------------------------------------------------
# 2. Fellegi-Sunter EM Invariants
# ---------------------------------------------------------------------------


class TestFellegiSunterInvariants:
    """Verification of EM probability simplex, posterior bounds, and index equivariance."""

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=15,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_em_probability_simplex_conservation(self, pairs: list[tuple[float, float]]) -> None:
        """m and u tables must remain valid discrete probability distributions summing to 1."""
        df = pd.DataFrame(pairs, columns=["f1", "f2"])
        clf = FSClassifier()
        clf.fit(df)
        assert clf._em_mode is True
        assert 0.0 < clf.lambda_ < 1.0

        for col in ["f1", "f2"]:
            m_sum = float(np.sum(clf._m_table[col]))
            u_sum = float(np.sum(clf._u_table[col]))
            assert math.isclose(m_sum, 1.0, abs_tol=1e-9)
            assert math.isclose(u_sum, 1.0, abs_tol=1e-9)
            assert (clf._m_table[col] >= 0.0).all()
            assert (clf._u_table[col] >= 0.0).all()

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=15,
            max_size=40,
        )
    )
    @settings(max_examples=100)
    def test_em_index_equivariance_and_boundedness(self, pairs: list[tuple[float, float]]) -> None:
        """EM predict_proba must preserve index and return values in [0, 1]."""
        custom_idx = [f"id_{i}" for i in range(len(pairs))]
        df = pd.DataFrame(pairs, columns=["f1", "f2"], index=pd.Index(custom_idx))
        clf = FSClassifier()
        clf.fit(df)
        proba = clf.predict_proba(df)

        assert proba.index.equals(df.index)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()
        assert not proba.isna().any()

    def test_missingness_neutrality_in_posterior(self) -> None:
        """Rows with all-missing features receive baseline prior with zero LR evidence."""
        df = pd.DataFrame(
            {
                "f1": [0.95, 0.90, 0.85, 0.92, 0.88, 0.96, 0.10, 0.12, 0.08, 0.11, 0.09, 0.13],
                "f2": [0.90, 0.94, 0.85, 0.92, 0.88, 0.95, 0.15, 0.09, 0.11, 0.14, 0.08, 0.12],
            }
        )
        clf = FSClassifier()
        clf.fit(df)

        # Candidate with both fields completely missing
        eval_df = pd.DataFrame(
            {
                "f1": [0.0],
                "f2": [0.0],
                "f1__missing": [1.0],
                "f2__missing": [1.0],
            }
        )
        proba = clf.predict_proba(eval_df)
        # When all features are missing, score == prior = log2(lam / (1 - lam))
        # Thus posterior must equal exactly lambda_
        assert math.isclose(float(proba.iloc[0]), clf.lambda_, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 3. Information-Theoretic Drift Invariants (monitoring.py)
# ---------------------------------------------------------------------------


class TestDriftInvariants:
    """Verification of divergence non-negativity, zero-divergence identity, and bounds."""

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=30,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_identity_zero_divergence(self, scores: list[float]) -> None:
        """PSI(r, r) and JSD(r, r) must be identically 0.0 up to float precision."""
        report = drift_report(scores, list(scores), bins=10, min_observations=30)
        assert report["status"] == "ok"
        psi = report["psi"]
        jsd = report["js_divergence"]
        assert isinstance(psi, float) and psi < 1e-9
        assert isinstance(jsd, float) and jsd < 1e-9
        assert report["warning"] is False

    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=30,
            max_size=50,
        ),
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=30,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_divergence_non_negativity_and_bounds(self, ref: list[float], cur: list[float]) -> None:
        """PSI >= 0 and 0 <= JSD <= 1.0 (base-2 JSD is bounded by log2(2) == 1.0)."""
        report = drift_report(ref, cur, bins=10, min_observations=30)
        assert report["status"] == "ok"
        psi = report["psi"]
        jsd = report["js_divergence"]
        assert isinstance(psi, float) and psi >= 0.0
        assert isinstance(jsd, float) and 0.0 <= jsd <= 1.0 + 1e-6

    def test_drift_default_min_observations(self) -> None:
        """Deterministic check for default min_observations=300."""
        rng = np.random.default_rng(42)
        ref = rng.uniform(0.0, 1.0, size=300).tolist()
        cur = rng.uniform(0.0, 1.0, size=300).tolist()
        report = drift_report(ref, cur, bins=10)
        assert report["status"] == "ok"
        assert report["psi"] is not None


# ---------------------------------------------------------------------------
# 4. DuckDB-Pandas Set Isomorphism
# ---------------------------------------------------------------------------


class TestBackendSetIsomorphism:
    """Exact candidate set equivalence between Pandas and DuckDB backends."""

    @given(
        st.lists(
            st.sampled_from(["cat", "dog", "", "  ", "nan", "none", "None", None, "zeta"]),
            min_size=10,
            max_size=30,
        ),
        st.lists(
            st.sampled_from(["cat", "dog", "", "  ", "nan", "none", "None", None, "omega"]),
            min_size=10,
            max_size=30,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_exact_block_set_isomorphism(
        self, left_keys: list[str | None], right_keys: list[str | None]
    ) -> None:
        """C_DuckDB(L, R) == C_Pandas(L, R) for exact blocking over arbitrary values."""
        left = pd.DataFrame(
            {"grp": left_keys, "val": [f"L_{i}" for i in range(len(left_keys))]},
            index=[f"left_{i}" for i in range(len(left_keys))],
        )
        right = pd.DataFrame(
            {"grp": right_keys, "val": [f"R_{j}" for j in range(len(right_keys))]},
            index=[f"right_{j}" for j in range(len(right_keys))],
        )

        duck_backend = DuckDBBackend()
        expected = {(str(a), str(b)) for a, b in ExactBlock(["grp"]).block(left, right).pairs}
        got = {(str(a), str(b)) for a, b in duck_backend.block_exact(left, right, ["grp"])}
        assert got == expected

        count = duck_backend.count_block_exact(left, right, ["grp"])
        assert count == len(expected)
        duck_backend.close()

    @given(
        st.lists(
            st.sampled_from(["apple", "banana", "", "nan", "none", None, "cherry"]),
            min_size=10,
            max_size=25,
        ),
        st.lists(
            st.sampled_from(["apple", "banana", "", "nan", "none", None, "date"]),
            min_size=10,
            max_size=25,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_sorted_neighborhood_set_isomorphism(
        self, left_keys: list[str | None], right_keys: list[str | None]
    ) -> None:
        """C_DuckDB(L, R) == C_Pandas(L, R) for sorted neighborhood blocking."""
        left = pd.DataFrame(
            {"grp": left_keys, "val": [f"L_{i}" for i in range(len(left_keys))]},
            index=[f"left_{i}" for i in range(len(left_keys))],
        )
        right = pd.DataFrame(
            {"grp": right_keys, "val": [f"R_{j}" for j in range(len(right_keys))]},
            index=[f"right_{j}" for j in range(len(right_keys))],
        )

        duck_backend = DuckDBBackend()
        blocker = SortedNeighborhood(["grp"], window=2)
        expected = {(str(a), str(b)) for a, b in blocker.block(left, right).pairs}
        got = {
            (str(a), str(b))
            for a, b in duck_backend.block_sorted_neighborhood(left, right, ["grp"], 2)
        }
        assert got == expected

        count = duck_backend.count_sorted_neighborhood(left, right, ["grp"], 2)
        assert count == len(expected)
        duck_backend.close()


# ---------------------------------------------------------------------------
# 5. Continuous Scorers Invariants
# ---------------------------------------------------------------------------


class TestContinuousScorersInvariants:
    """Hypothesis property tests for continuous numeric and date scorers."""

    @settings(max_examples=100)
    @given(
        x=st.floats(allow_nan=True, allow_infinity=True),
        y=st.floats(allow_nan=True, allow_infinity=True),
        eps=st.floats(min_value=1e-3, max_value=1e4),
    )
    def test_numeric_proximity_invariants(self, x: float, y: float, eps: float) -> None:
        """Verification of bounds in [0.0, 1.0], symmetry, and reflexivity."""
        scorer = NumericProximityScorer(epsilon=eps)
        s = scorer.score(x, y)
        assert 0.0 <= s <= 1.0
        assert math.isclose(s, scorer.score(y, x), abs_tol=1e-9)
        if math.isfinite(x):
            assert scorer.score(x, x) == 1.0

    @settings(max_examples=100)
    @given(
        base=st.floats(min_value=-1e6, max_value=1e6),
        d1=st.floats(min_value=0.0, max_value=1e4),
        d2=st.floats(min_value=0.0, max_value=1e4),
        eps=st.floats(min_value=1e-2, max_value=1e4),
    )
    def test_numeric_proximity_monotonicity(
        self, base: float, d1: float, d2: float, eps: float
    ) -> None:
        """As distance |x - y| increases, similarity score decreases monotonically."""
        scorer = NumericProximityScorer(epsilon=eps)
        near = min(d1, d2)
        far = max(d1, d2)
        assert scorer.score(base, base + near) >= scorer.score(base, base + far) - 1e-12

    @settings(max_examples=100)
    @given(
        days_offset1=st.integers(min_value=0, max_value=3650),
        days_offset2=st.integers(min_value=0, max_value=3650),
        delta=st.floats(min_value=0.5, max_value=60.0),
    )
    def test_date_window_invariants(
        self, days_offset1: int, days_offset2: int, delta: float
    ) -> None:
        """Verification of date window score bounds, symmetry, reflexivity, and monotonicity."""
        base_date = datetime.date(2020, 1, 1)
        d1 = base_date + datetime.timedelta(days=days_offset1)
        d2 = base_date + datetime.timedelta(days=days_offset2)

        scorer = DateWindowScorer(delta_days=delta)
        s12 = scorer.score(d1, d2)
        assert 0.0 <= s12 <= 1.0
        assert math.isclose(s12, scorer.score(d2, d1), abs_tol=1e-9)
        assert scorer.score(d1, d1) == 1.0


# ---------------------------------------------------------------------------
# 6. Star Clustering Mathematical Invariants
# ---------------------------------------------------------------------------


class TestStarClusteringMathematicalInvariants:
    """Hypothesis property tests verifying non-transitivity and clique guarantees."""

    @settings(max_examples=100)
    @given(
        w_ab=st.floats(min_value=0.5, max_value=1.0),
        w_bc=st.floats(min_value=0.5, max_value=1.0),
        w_ac=st.floats(min_value=0.0, max_value=0.49),
    )
    def test_star_clusterer_eliminates_transitive_chaining(
        self, w_ab: float, w_bc: float, w_ac: float
    ) -> None:
        """Formal verification: A ~ B and B ~ C does NOT imply A ~ C when w(A, C) < cutoff."""
        pairs = [("A", "B"), ("B", "C"), ("A", "C")]
        weights = [w_ab, w_bc, w_ac]
        sc = StarClusterer(cutoff=0.5, verify_clique=True)
        clusters = sc.cluster(pairs, weights)
        for clu in clusters:
            assert not ({"A", "C"}.issubset(clu)), f"Transitive chaining failure: {clu}"
