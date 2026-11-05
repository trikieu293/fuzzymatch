"""Mine synonym candidates from human-accepted match pairs (IBM-1 counts).

Each accepted pair is treated as a parallel sentence; every source token
co-occurs with every target token of the same sentence (IBM Model 1
alignment assumption). Candidates are filtered by:

* support:        co-occurrence count >= ``min_count``
* forward prob:   P(target | source) >= ``min_prob``
* confirmation:   reverse probability OR Dice coefficient passes ``min_prob``
* foreignness:    source token frequency in the TARGET corpus <=
                  ``max_target_frequency`` — shared vocabulary is not a
                  substitution candidate.

Tokens are lowercased and simple-plural folded before counting so
morphological variants do not fragment support. Output carries a ``bucket``
column: ``candidate`` for confident pairs, ``insufficient_evidence`` for
high-probability pairs with thin support.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

_EPS = 1e-9


def _fold(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str, fold: bool = True) -> set[str]:
    out = {t.lower() for t in str(text).split()}
    return {_fold(t) for t in out} if fold else out


def mine_synonym_candidates(
    source_texts: Iterable[str],
    target_texts: Iterable[str],
    labels: Iterable[int],
    *,
    min_count: int = 3,
    min_prob: float = 0.5,
    max_target_frequency: float = 0.05,
) -> pd.DataFrame:
    """Rank substitution candidates from accepted match pairs.

    Args:
        source_texts: Left-side names of labeled pairs.
        target_texts: Right-side names of labeled pairs.
        labels: 1 for human-accepted matches, 0 otherwise. Mining uses only
            accepted pairs; at least one is required.
        min_count: Minimum sentence co-occurrence count for the candidate
            bucket; below it, high-probability pairs land in the
            ``insufficient_evidence`` bucket.
        min_prob: Minimum forward probability (and reverse-or-Dice
            confirmation bar).
        max_target_frequency: Maximum share of target sentences that may
            contain a source token for it to count as foreign.

    Returns:
        DataFrame ``[source_token, target_token, count, prob, dice,
        bucket]`` ranked by count then probability descending.

    Raises:
        ValueError: On length mismatch, empty input, no accepted pairs, or
            non-binary labels.
    """
    src_list = [str(x) for x in source_texts]
    tgt_list = [str(x) for x in target_texts]
    lab_list = [int(x) for x in labels]
    if len(src_list) != len(tgt_list) or len(src_list) != len(lab_list):
        raise ValueError("source/target/labels length mismatch")
    if not src_list:
        raise ValueError("mine_synonym_candidates requires non-empty input")
    if not any(lab == 1 for lab in lab_list):
        raise ValueError("requires at least one accepted pair (label=1)")

    fold = True
    src_tok = [_tokens(s, fold) for s in src_list]
    tgt_tok = [_tokens(t, fold) for t in tgt_list]

    # --- counts over ACCEPTED sentences -----------------------------------
    co: dict[tuple[str, str], int] = {}
    src_totals: dict[str, int] = {}
    tgt_totals: dict[str, int] = {}
    for i, label in enumerate(lab_list):
        if label != 1:
            continue
        for s in src_tok[i]:
            src_totals[s] = src_totals.get(s, 0) + 1
            for t in tgt_tok[i]:
                co[(s, t)] = co.get((s, t), 0) + 1
        for t in tgt_tok[i]:
            tgt_totals[t] = tgt_totals.get(t, 0) + 1

    # --- foreignness over the FULL target corpus --------------------------
    tgt_corpus_freq: dict[str, float] = {}
    all_target_counts: dict[str, int] = {}
    for toks in tgt_tok:
        for t in toks:
            all_target_counts[t] = all_target_counts.get(t, 0) + 1
    total_targets_all = len(tgt_list)
    for token in all_target_counts:
        tgt_corpus_freq[token] = all_target_counts[token] / max(total_targets_all, 1)

    rows: list[dict[str, object]] = []
    for (s, t), count in co.items():
        src_total = max(src_totals.get(s, 0), 1)
        tgt_total = max(tgt_totals.get(t, 0), 1)
        prob = count / src_total
        reverse = count / tgt_total
        dice = (2 * count) / (src_total + tgt_total)
        confirmed = max(reverse, dice) >= min_prob
        foreign = tgt_corpus_freq.get(s, 0.0) <= max_target_frequency

        if prob >= min_prob and confirmed and foreign:
            bucket = "candidate" if count >= min_count else "insufficient_evidence"
        elif prob >= min_prob and foreign:
            # Sparsity/confirmation fallback: strong forward evidence that
            # fails symmetric confirmation (or thin support) surfaces for
            # human curation instead of vanishing.
            bucket = "insufficient_evidence"
        else:
            continue
        rows.append(
            {
                "source_token": s,
                "target_token": t,
                "count": count,
                "prob": round(prob, 6),
                "dice": round(dice, 6),
                "bucket": bucket,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "source_token",
            "target_token",
            "count",
            "prob",
            "dice",
            "bucket",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(["count", "prob"], ascending=False, kind="stable").reset_index(
        drop=True
    )
