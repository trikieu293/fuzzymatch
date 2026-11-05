"""Synthetic lexicon generation and evaluation harness for tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fuzzymatch.lexicon_mine import mine_synonym_candidates

FILLER = [
    "acme",
    "global",
    "services",
    "holdings",
    "group",
    "office",
    "central",
    "north",
    "south",
    "east",
    "west",
    "main",
    "united",
    "general",
    "prime",
    "core",
    "metro",
    "summit",
    "harbor",
    "canyon",
]
VOCAB = [
    "rent",
    "wages",
    "fuel",
    "insurance",
    "audit",
    "legal",
    "marketing",
    "travel",
    "utilities",
    "software",
    "hardware",
    "training",
    "repairs",
    "freight",
    "storage",
    "packaging",
    "printing",
    "postage",
    "cleaning",
    "catering",
    "security",
    "recruiting",
    "interest",
    "dividends",
]


def _name(rng: Any, tokens: list[str]) -> str:
    n = rng.choice([2, 3])
    picked = rng.choice(len(tokens), size=n, replace=False)
    return " ".join(tokens[i] for i in sorted(picked))


def generate(
    seed: int, n_planted: int = 15, reps: int = 8
) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Return (frame, planted_pairs)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    planted: list[tuple[str, str]] = []
    used_targets: set[str] = set()
    while len(planted) < n_planted:
        t_tok = VOCAB[rng.integers(len(VOCAB))]
        s_tok = f"x{len(planted)}{t_tok[:3]}"
        if t_tok in used_targets:
            continue
        used_targets.add(t_tok)
        planted.append((s_tok, t_tok))

    # target corpus: names from vocab (+filler), planted targets included
    target_names = [_name(rng, VOCAB + FILLER) for _ in range(120)]
    rows: list[dict[str, Any]] = []
    for s_tok, t_tok in planted:
        for _ in range(reps):
            filler = list(rng.choice(FILLER, size=rng.integers(1, 3)))
            src_tokens = [*filler, s_tok]
            tgt_tokens = [
                *list(rng.choice(FILLER, size=int(rng.integers(1, 3)))),
                t_tok,
            ]
            rows.append(
                {
                    "source": " ".join(sorted(src_tokens)),
                    "target": " ".join(sorted(tgt_tokens)),
                    "label": 1,
                }
            )
    # background accepted pairs (no planted source tokens)
    vocab_names = [_name(rng, VOCAB + FILLER) for _ in range(80)]
    rows.extend(
        {
            "source": vocab_names[int(rng.integers(len(vocab_names)))],
            "target": target_names[int(rng.integers(len(target_names)))],
            "label": 1,
        }
        for _ in range(120)
    )
    rows.extend(
        {
            "source": _name(rng, VOCAB + FILLER),
            "target": target_names[int(rng.integers(len(target_names)))],
            "label": 0,
        }
        for _ in range(60)
    )
    frame = pd.DataFrame(rows)
    return frame, set(planted)


def _fold(token: str) -> str:
    # Mirrors lexicon_mine._fold (trailing-s plural folding) so ground
    # truth compares against folded token names.
    return token[:-1] if token.endswith("s") else token


def pr_sweep(candidates: pd.DataFrame, planted: set[tuple[str, str]]) -> list[dict[str, float]]:
    """PR curve over probability thresholds on the curation-ranked list."""
    cand = candidates[candidates["bucket"] == "candidate"]
    thin = candidates[candidates["bucket"] == "insufficient_evidence"].sort_values(
        ["count", "prob"], ascending=False, kind="stable"
    )
    ranked = pd.concat([cand, thin], ignore_index=True)
    pairs = list(zip(ranked["source_token"], ranked["target_token"], strict=True))
    probs = list(ranked["prob"])
    truth = {(_fold(s), _fold(t)) for s, t in planted}
    found_flags = [p in truth for p in pairs]
    order = sorted(range(len(pairs)), key=lambda i: -probs[i])

    points: list[dict[str, float]] = []
    tp = 0
    fp = 0
    for idx in order:
        if found_flags[idx]:
            tp += 1
        else:
            fp += 1
        points.append(
            {
                "threshold": float(probs[idx]),
                "recall": tp / len(truth) if truth else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
            }
        )
    return points


def evaluate(seed: int) -> dict[str, Any]:
    frame, planted = generate(seed)
    cands = mine_synonym_candidates(frame["source"], frame["target"], frame["label"])
    points = pr_sweep(cands, planted)
    passing = [p for p in points if p["recall"] >= 0.90 and p["precision"] >= 0.90]
    best_f1 = (
        max(
            (
                2 * p["precision"] * p["recall"] / (p["precision"] + p["recall"])
                if p["precision"] + p["recall"]
                else 0.0
            )
            for p in points
        )
        if points
        else 0.0
    )
    return {
        "seed": seed,
        "bar_met": bool(passing),
        "best_f1": round(best_f1, 4),
        "n_points": len(points),
    }
