# fuzzymatch

[![CI](https://github.com/trikieu293/fuzzymatch/actions/workflows/ci.yml/badge.svg)](https://github.com/trikieu293/fuzzymatch/actions)
[![Type Checking](https://img.shields.io/badge/types-mypy%20--strict-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)](https://github.com/trikieu293/fuzzymatch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Industrial-scale text fuzzy mapping library for Python — deduplication, record linkage, and approximate string joins over millions of records.

Built around the universal 5-stage pipeline:

```text
normalize → block → score → classify → cluster
```

with three task modes (`dedupe` / `link` / `fuzzy_join`) accessible from a single high-level `Matcher`.

## Highlights

- **MIT-licensed** — no copyleft traps. Safe for commercial and research use.
- **Compiled hot loop** — wraps [`rapidfuzz`](https://github.com/maxbachmann/RapidFuzz) (MIT C++ extension) for high-performance string distance calculations.
- **Layered, swappable pipeline** — every stage is a distinct module; choose the blocking strategy, classifier, and clusterer that fit your dataset.
- **Fellegi–Sunter + EM** probabilistic classifier with three-zone thresholding — label-free, interpretable, with a calibrated log-odds → probability transform, quantile conditioning on observed pairs, and index equivariance.
- **Out-of-core SQL candidate generation** — `DuckDBBackend` enables exact, composite, and sorted-neighborhood blocking at multi-million scale with complete Pandas missing-marker parity.
- **Term-frequency-adjusted posteriors** — opt-in TF weighting discounts frequent tokens to reduce false positives.
- **Cluster-level metrics** — B-cubed, ARI, V-measure for deduplication evaluation.
- **Block-quality metrics** — Reduction Ratio, Pairs Completeness, Pairs Quality out of the box.
- **Production drift telemetry** — Population Stability Index (PSI) and Jensen-Shannon Divergence (JSD) monitoring with guaranteed discrete probability simplex conservation.
- **Formal mathematical verification** — Invariant suite verified under `hypothesis` covering metric axioms, EM simplex bounds, and backend set isomorphism.

## Installation

```bash
pip install fuzzymatch
```

Or for development (editable install with dev and duckdb extras):

```bash
git clone https://github.com/trikieu293/fuzzymatch
cd fuzzymatch
pip install -e ".[dev,duckdb]"
```

Runtime dependencies are installed automatically: `pandas`, `numpy`, `scikit-learn`, `datasketch`, `networkx`, `rapidfuzz`. Optional DuckDB backend: `pip install fuzzymatch[duckdb]`.

## Quick start

```python
import fuzzymatch as fm
import pandas as pd

df = pd.DataFrame(
    {
        "name": ["Alice Smith", "Alice Smith", "Bob Jones", "Robert Jones"],
        "id": [1, 2, 3, 4],
    }
)

normalizer = fm.Normalizer(
    [
        fm.unicode_normalize("NFC"),
        fm.casefold(),
        fm.strip_whitespace(),
    ]
)

matcher = fm.Matcher(
    link_type="dedupe",
    normalizer=normalizer,
    blocking=fm.ExactBlock(["name"]),
    scorer=fm.RapidFuzzScorer(),
    classifier=fm.FSClassifier(),
    clusterer=fm.ConnectedComponents(),
    scoring_fields=["name"],
    scoring_method="ratio",
)

matcher.fit(df)
pairs = matcher.predict(threshold="match")  # candidate pairs
clusters = matcher.clusters()  # entity clusters
```

### Three task modes

```python
# 1. Deduplication (self-join → entity clusters)
matcher = fm.Matcher(link_type="dedupe", blocking=..., clusterer=...)

# 2. Record linkage (cross-file → matched pairs)
matcher = fm.Matcher(link_type="link", blocking=..., classifier=...)
matcher.fit(df_a, df_b)
pairs = matcher.predict(threshold="match")

# 3. Fuzzy join (SQL-style merge on a fuzzy key)
matcher = fm.Matcher(
    link_type="fuzzy_join", blocking=..., classifier=fm.FSClassifier(), scoring_fields=["name"]
)
merged = matcher.join(df_a, df_b, on="name", threshold=0.5)
```

## Architecture

| Stage        | Module                       | Built-in options                                                                                             |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1. Normalize | `fuzzymatch.normalize`       | `unicode_normalize`, `casefold`, `strip_whitespace`, `strip_punctuation`, `accent_fold`, `replace_words` |
| 2. Block     | `fuzzymatch.blocking`        | `ExactBlock`, `SortedNeighborhood`, `MinHashLSH`, `UnionBlock`, `IntersectBlock`                             |
| 3. Score     | `fuzzymatch.scoring`         | `RapidFuzzScorer` (Levenshtein, Jaro, Jaro-Winkler, token-sort, token-set, partial ratio, Hamming)           |
| 4. Classify  | `fuzzymatch.classification`  | `FSClassifier` (Fellegi-Sunter + EM, label-free) or `SklearnClassifier` (any sklearn estimator)              |
| 5. Cluster   | `fuzzymatch.clustering`      | `ConnectedComponents` (networkx union-find) or `PluggableResolver` (user-provided)                           |

## Enterprise & Production Capabilities

`fuzzymatch` is domain-neutral. It exposes reusable matching artifacts that can be interpreted by downstream systems:

- `MatchDecision` — one candidate-pair decision with `match`, `possible`, or `non_match` status
- `Explanation` — per-field scores, threshold band, blocking rule names, and classifier weights
- `FieldScore` — similarity evidence for one compared field
- `PipelineConfig` — serializable summary of a configured matching pipeline
- `block_quality_metrics` — Reduction Ratio, Pairs Completeness, and Pairs Quality for blocker evaluation
- `Matcher.decisions()` — returns `list[MatchDecision]` with explanations for every candidate pair
- `Matcher.iter_decisions()` — streaming generator yielding decisions without buffering large candidate sets in memory
- `Matcher.pipeline_config()` — returns `PipelineConfig` describing the active pipeline

### DuckDB SQL Backend (Out-of-Core Candidate Generation)

For datasets that exceed memory limits, the `DuckDBBackend` runs candidate generation in DuckDB:

```python
from fuzzymatch.backends import DuckDBBackend

backend = DuckDBBackend()
candidates = backend.block_exact(df_a, df_b, ["company_code"])
sorted_candidates = backend.block_sorted_neighborhood(df_a, df_b, ["name"], window=5)
```

### Term-Frequency Weighting

Opt-in term frequency weighting discounts common tokens:

```python
matcher = fm.Matcher(
    link_type="link",
    term_frequency=True,  # discounts high-frequency tokens
    scoring_fields=["name"],
    scoring_method="token_set_ratio",
)
```

### Population Drift Telemetry

Monitor production score distributions against a historical baseline using Population Stability Index (PSI) and Jensen-Shannon Divergence (JSD):

```python
from fuzzymatch.monitoring import drift_report

report = drift_report(baseline_scores, current_scores, bins=10)
# {'status': 'ok', 'psi': 0.012, 'js_divergence': 0.004, 'warning': False, 'n_reference': 1200, 'n_current': 1500}
```

### Table Profiling & Advisory Configuration

Profile dataset missingness and cardinality before matching, and generate advisory configuration:

```python
from fuzzymatch.profiling import profile_table, suggest_matching_config

table_profile = profile_table(df)
config_hints = suggest_matching_config(table_profile)
```

### Model Selection & Persistence

```python
from fuzzymatch.classification import FSClassifier
from fuzzymatch.model_selection import select_operating_threshold, train_valid_test_split

train_df, valid_df, test_df = train_valid_test_split(labeled_pairs, ratios=(0.6, 0.2, 0.2))
best_threshold = select_operating_threshold(scores=val_scores, truth=val_labels)

# Save and load fitted classifiers
clf = FSClassifier()
clf.fit(features)
clf.save("fs_model.json")
loaded_clf = FSClassifier.load("fs_model.json")
```

### Explain candidate decisions

```python
matcher.fit(df)
decisions = matcher.decisions()

first = decisions[0]
print(first.status)  # "match", "possible", or "non_match"
print(first.score)  # probability or mean feature score
print(first.explanation.field_scores[0])  # per-field evidence

# All decisions are serializable to plain dicts
first.to_dict()
```

### Blocking quality

```python
from fuzzymatch.metrics import block_quality_metrics

metrics = block_quality_metrics(
    candidate_pairs=[(0, 0), (0, 1), (1, 1)],
    true_pairs=[(0, 0), (1, 1), (2, 2)],
    total_possible_pairs=9,
)
# {'reduction_ratio': 0.67, 'pairs_completeness': 0.67, 'pairs_quality': 0.67}
```

## Companion domain packages

The core library intentionally avoids domain-specific workflow classes. Domain packages depend on `fuzzymatch` and translate generic match decisions into business-specific workflows.

Released companion package:

- [`fuzzymatch-finance`](https://github.com/trikieu293/fuzzymatch-finance) (v1.0.0) — FP&A and financial controlling workflow package for rule-first finance mapping, fuzzy fallback, structured evidence, review queues, ISO 7064 / Luhn identifier validation, and machine-readable audit logs.

Boundary rule:

> Anything useful across domains belongs in `fuzzymatch`. Anything that only makes sense for FP&A or financial controlling belongs in `fuzzymatch-finance`.

## Benchmarks and datasets

```python
# Load ER-Magellan benchmarks
from fuzzymatch.datasets import load

labeled, table_a, table_b = load("Abt-Buy")  # 9,575 pairs; product matching
```

Supported: Abt-Buy, DBLP-ACM, DBLP-Scholar, Amazon-Google, Walmart-Amazon, iTunes-Amazon, Fodors-Zagats, BeerAdvo-RateBeer, Company.

```python
# Evaluate pairwise predictions against gold-standard labels
from fuzzymatch.metrics import pairwise_metrics

metrics = pairwise_metrics(y_true, y_pred, y_score)
```

## Development & Verification

### Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev,duckdb]"
```

### Tests & Coverage Gate

```bash
pytest tests/unit/ tests/adversarial/ -v                      # fast unit & adversarial invariant suites
pytest tests/integration -m slow                             # integration benchmarks
pytest --cov=fuzzymatch --cov-report=term --cov-fail-under=95 # enforced 95% coverage gate
```

### Lint and Type-check

```bash
ruff check fuzzymatch tests
mypy
```

## Citation

If you use `fuzzymatch` in academic or industrial research, please cite:

```bibtex
@software{fuzzymatch2026,
  author = {Tri Kieu},
  title = {fuzzymatch: Industrial-Scale Text Fuzzy Mapping & Probabilistic Record Linkage},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/trikieu293/fuzzymatch}
}
```

See [CITATION.cff](CITATION.cff) and [.zenodo.json](.zenodo.json) for full metadata.

## License

MIT. See [LICENSE](LICENSE).
