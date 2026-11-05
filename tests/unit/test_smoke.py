"""Smoke test for fuzzymatch: verify the package imports and the high-level
Matcher works on a tiny synthetic dedupe example."""

import pandas as pd

import fuzzymatch as fm


def test_imports():
    """All public symbols must import cleanly."""
    assert hasattr(fm, "Matcher")
    assert hasattr(fm, "Normalizer")
    assert hasattr(fm, "ExactBlock")
    assert hasattr(fm, "FSClassifier")
    assert hasattr(fm, "RapidFuzzScorer")
    assert hasattr(fm, "ConnectedComponents")
    assert hasattr(fm, "replace_words")
    assert hasattr(fm, "drift_report")
    assert hasattr(fm, "profile_table")
    assert hasattr(fm, "suggest_matching_config")
    assert hasattr(fm.normalize, "replace_words")


def test_normalizer():
    norm = fm.Normalizer([fm.unicode_normalize("NFC"), fm.casefold(), fm.strip_whitespace()])
    df = pd.DataFrame({"name": ["  Jose  ", "MARIA"]})
    out = norm.transform(df, ["name"])
    assert out["name"].iloc[0] == "jose"
    assert out["name"].iloc[1] == "maria"


def test_exact_block_self_join():
    df = pd.DataFrame({"key": ["A", "A", "B"]})
    pairs = fm.ExactBlock(["key"]).block(df)
    assert (0, 1) in pairs
    assert (1, 2) not in pairs  # different keys


def test_rapidfuzz_scorer():
    s = fm.RapidFuzzScorer()
    sim = s.token_sort_ratio("john smith", "smith john")
    assert sim > 0.95  # should be near-identical after token sort


def test_matcher_dedupe_smoke():
    """End-to-end: dedupe a tiny DataFrame with exact blocking + FSClassifier."""
    df = pd.DataFrame(
        {
            "name": ["Alice Smith", "Alice Smith", "Bob Jones", "Robert Jones"],
            "id": [1, 2, 3, 4],
        }
    )
    matcher = fm.Matcher(
        link_type="dedupe",
        blocking=fm.ExactBlock(["name"]),
        scorer=fm.RapidFuzzScorer(),
        classifier=fm.FSClassifier(),
        clusterer=fm.ConnectedComponents(),
        scoring_fields=["name"],
        scoring_method="ratio",
    )
    matcher.fit(df)
    pairs = matcher.predict(threshold="possible")
    # With "name" equality, we should get the (0,1) pair (Alice Smith x2)
    # and (2,3) pair (Bob/Robert Jones — close but not equal).
    # ExactBlock will produce at least the exact-match pair.
    assert isinstance(pairs, pd.MultiIndex)


def test_duckdb_backend_import():
    """DuckDBBackend must be importable from fuzzymatch.backends and top-level."""
    from fuzzymatch.backends import DuckDBBackend as DuckDBBackendFromBackends

    assert DuckDBBackendFromBackends is not None
    assert hasattr(fm, "DuckDBBackend")
    assert fm.DuckDBBackend is DuckDBBackendFromBackends


def test_duckdb_backend_readme_snippet():
    """Verify DuckDBBackend readme snippet works as advertised."""
    df_a = pd.DataFrame({"name": ["Alice", "Bob"], "company_code": ["1", "2"]})
    df_b = pd.DataFrame({"name": ["Alice", "Bobby"], "company_code": ["1", "2"]})

    backend = fm.DuckDBBackend()
    candidates = backend.block_exact(df_a, df_b, ["company_code"])
    assert len(candidates) == 2
    sorted_candidates = backend.block_sorted_neighborhood(df_a, df_b, ["name"], window=5)
    assert len(sorted_candidates) >= 1
    backend.close()


def test_matcher_sklearn_classifier_end_to_end():
    """Verify Matcher works with SklearnClassifier across all core workflows:
    fit(), predict(), decisions(), iter_decisions(), and match_right_batch().
    """
    from sklearn.linear_model import LogisticRegression

    df_left = pd.DataFrame(
        {
            "group": [1, 1, 1, 1],
            "name": ["Alice Smith", "Bob Jones", "Charlie Brown", "David Green"],
        }
    )
    df_right = pd.DataFrame(
        {
            "group": [1, 1, 1, 1],
            "name": ["Alice Smith", "Robert Jones", "Charlie Brown", "David White"],
        }
    )

    clf = fm.SklearnClassifier(LogisticRegression(), threshold=(0.8, 0.4, 0.2))
    matcher = fm.Matcher(
        link_type="link",
        blocking=fm.ExactBlock(["group"]),
        scorer=fm.RapidFuzzScorer(),
        classifier=clf,
        scoring_fields=["name"],
        scoring_method="ratio",
    )

    # 16 candidate pairs from group=1 exact block.
    # Provide balanced binary labels for LogisticRegression.
    labels = pd.Series([1 if i % 2 == 0 else 0 for i in range(16)])
    matcher.fit(df_left, df_right, labels=labels)

    # 1. predict()
    pairs = matcher.predict(threshold="match")
    assert isinstance(pairs, pd.MultiIndex)

    # 2. decisions() - ensures no AttributeError on classifier.weights_
    decisions = matcher.decisions()
    assert len(decisions) == 16
    for d in decisions:
        assert d.explanation.classifier_weights is None
        assert d.status in ("match", "possible", "non_match")

    # 3. iter_decisions()
    lazy_decisions = list(matcher.iter_decisions(chunk_size=4))
    assert len(lazy_decisions) == 16
    assert lazy_decisions[0].left_id == decisions[0].left_id

    # 4. match_right_batch()
    df_new = pd.DataFrame(
        {
            "group": [1, 1],
            "name": ["Alice Smith", "David White"],
        }
    )
    new_decisions = matcher.match_right_batch(df_new)
    assert len(new_decisions) > 0
    for nd in new_decisions:
        assert nd.explanation.classifier_weights is None


if __name__ == "__main__":
    test_imports()
    test_normalizer()
    test_exact_block_self_join()
    test_rapidfuzz_scorer()
    test_matcher_dedupe_smoke()
    test_duckdb_backend_import()
    test_duckdb_backend_readme_snippet()
    test_matcher_sklearn_classifier_end_to_end()
    print("All smoke tests passed.")
