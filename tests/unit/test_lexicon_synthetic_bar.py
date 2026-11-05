"""A5 falsifiable bar: miner recovers >=90% planted synonyms at <=10% FP."""

from __future__ import annotations

from tests.helpers.synthetic_lexicon import evaluate


def test_planted_synonym_bar_across_seeds():
    results = [evaluate(seed) for seed in (7, 21, 99)]
    for r in results:
        assert r["bar_met"], f"bar failed for seed {r['seed']}: {r}"
        assert r["best_f1"] >= 0.9, f"F1 too low for seed {r['seed']}: {r}"
