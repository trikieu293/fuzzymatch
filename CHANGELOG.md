# Changelog

All notable changes to **fuzzymatch** are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [1.0.0] - 2026-11-05

Initial production release of `fuzzymatch` — industrial-scale text fuzzy mapping, record linkage, and deduplication library.

### Added
- **5-Stage Pipeline**: Architecture consisting of Normalization, Blocking, Scoring, Classification, and Clustering.
- **Three Task Modes**: Single high-level `Matcher` interface supporting Deduplication (`dedupe`), Record Linkage (`link`), and Fuzzy Join (`fuzzy_join`).
- **Compiled Scoring Engine**: RapidFuzz C++ extension integration with Levenshtein, Jaro, Jaro-Winkler, token-sort, token-set, Hamming, and partial ratio metrics.
- **Continuous Domain Scorers**:
  - `NumericProximityScorer` / `amount_proximity_score` with bounded linear and exponential decay for financial and numeric amounts.
  - `DateWindowScorer` / `date_window_score` with calendar tolerance windows for invoice and transaction alignment.
- **Flexible Blocker Algorithms**:
  - `ExactBlock`: Exact key blocking with composite column support.
  - `SortedNeighborhood`: Merged-sort sliding window blocking for self-join and cross-file linkage.
  - `MinHashLSH`: Locality-sensitive hashing for set-based token similarity.
  - `MultiPassBlock`: Cascaded multi-rule candidate generation with union or deduplication semantics and early exit.
  - `UnionBlock` and `IntersectBlock` combinators.
- **Probabilistic & Supervised Classifiers**:
  - `FSClassifier`: Binned Fellegi-Sunter EM probabilistic classifier with quantile binning conditioned on observed record pairs, term frequency log2-odds adjustments, and model persistence (`save` / `load`).
  - `SklearnClassifier`: Wrapper for arbitrary scikit-learn estimators.
- **Graph Clustering**:
  - `ConnectedComponents`: Union-find graph partitioning for transitive entity groups.
  - `StarClusterer`: Degree-ordered hub expansion with deterministic tie-breaking to prevent giant-component chaining on dense networks.
  - `PluggableResolver`: Custom cluster resolution callbacks.
- **Zero-Config AutoMatcher & Profiling**:
  - `AutoMatcher`: Automated column role detection, dynamic blocking rule induction, and composite scoring pipeline configuration.
  - `profile_table` & `suggest_matching_config`: Dataset missingness and cardinality profiling with heuristic config recommendations.
- **High-Performance Backends**:
  - `PandasBackend`: Standard in-memory candidate generation.
  - `DuckDBBackend`: Out-of-core SQL candidate generation and streaming pair counting with full missing-marker parity (`NULL`, `""`, `"nan"`, `"none"`).
- **Monitoring & Quality Telemetry**:
  - `drift_report`: Population Stability Index (PSI) and Jensen-Shannon Divergence (JSD) score distribution drift monitoring with probability simplex conservation.
  - `block_quality_metrics`: Reduction Ratio, Pairs Completeness, and Pairs Quality.
  - `cluster_metrics`: B-cubed, ARI, and V-measure.
  - `pairwise_metrics`: Precision, recall, F1, and AUC evaluation against gold standards.
- **Lexicon Mining**:
  - `mine_synonym_candidates`: IBM-1 translation miner with foreignness filtering and symmetric confirmation.
- **CLI & Explainability**:
  - `fuzzymatch` command-line interface for table profiling, deduplication, and record linkage.
  - `MatchDecision` and `Explanation` data models providing per-field similarity waterfalls and rule firing traces.
  - `iter_decisions` and `iter_decision_chunks` for memory-bounded streaming decision consumption.
