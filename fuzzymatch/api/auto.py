"""Zero-configuration AutoMatcher engine and MatchResults container.

Profiles unannotated input tables, infers column roles, configures
multi-pass disjunctive candidate blocking, computes continuous similarity
scores, and resolves non-transitive clusters using Star/Pivot clustering.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..blocking import ExactBlock, MultiPassBlock, SortedNeighborhood
from ..clustering import ConnectedComponents, StarClusterer
from ..decisions import Explanation, FieldScore, MatchDecision, MatchStatus
from ..normalize import strip_legal_suffixes
from ..scoring import DateWindowScorer, NumericProximityScorer, RapidFuzzScorer

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class MatchResults:
    """Container for AutoMatcher execution results.

    Attributes:
        matches: DataFrame of matched pairs with cluster IDs, similarity scores,
                 and field comparison values.
        clusters: List of disjoint entity clusters (sets of record identifiers).
        decisions: List of detailed MatchDecision objects for explainability.
        summary_stats: Execution metrics (record counts, candidate pairs, budget %, timing).
        left_df: Original left DataFrame.
        right_df: Original right DataFrame.
    """

    matches: pd.DataFrame
    clusters: list[set[Any]]
    decisions: list[MatchDecision]
    summary_stats: dict[str, Any]
    left_df: pd.DataFrame
    right_df: pd.DataFrame

    @property
    def pairs(self) -> pd.DataFrame:
        """Alias for matches DataFrame for compatibility."""
        return self.matches

    @property
    def total_candidates(self) -> int:
        """Total candidate pairs generated during blocking."""
        return int(self.summary_stats.get("candidate_pairs", len(self.matches)))

    def to_dataframe(self) -> pd.DataFrame:
        """Return matches DataFrame."""
        return self.matches.copy()

    def to_csv(self, path: str | Path, index: bool = False, **kwargs: Any) -> None:
        """Export matches DataFrame to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.matches.to_csv(path, index=index, **kwargs)

    def to_excel(self, path: str | Path, index: bool = False, **kwargs: Any) -> None:
        """Export matches DataFrame to Excel (.xlsx)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.matches.to_excel(path, index=index, **kwargs)

    def get_explanations(self, top_n: int | None = None) -> list[dict[str, Any]]:
        """Return structured human-readable explanations for match decisions."""
        decs = self.decisions if top_n is None else self.decisions[:top_n]
        return [d.to_dict() for d in decs]

    def __len__(self) -> int:
        return len(self.matches)

    def __repr__(self) -> str:
        n_m = len(self.matches)
        n_c = len(self.clusters)
        util = self.summary_stats.get("budget_utilization_pct", 0.0)
        return (
            f"<MatchResults: {n_m} matches across {n_c} clusters | "
            f"candidate_budget_used={util:.1f}%>"
        )


# ---------------------------------------------------------------------------
# Semantic Column Classifier & Matcher
# ---------------------------------------------------------------------------

_KEY_KEYWORDS = {
    "id",
    "code",
    "tax_id",
    "ein",
    "vat",
    "ssn",
    "key",
    "number",
    "no",
    "uuid",
    "guid",
    "ref",
    "vendor_id",
    "ap_id",
    "cust_id",
    "customer_id",
    "vendor_code",
    "erp_code",
    "account_no",
    "account_id",
}

_NAME_KEYWORDS = {
    "name",
    "payee",
    "vendor",
    "company",
    "customer",
    "account",
    "supplier",
    "firm",
    "entity",
    "client",
    "title",
    "organization",
    "vendor_name",
    "account_name",
    "cust_name",
    "business_name",
}

_SECONDARY_TEXT_KEYWORDS = {
    "city",
    "location",
    "address",
    "state",
    "country",
    "zip",
    "postal",
    "street",
    "memo",
    "desc",
    "description",
    "category",
    "type",
    "dept",
}

_AMOUNT_KEYWORDS = {
    "amount",
    "total",
    "balance",
    "price",
    "cost",
    "value",
    "payment",
    "cleared",
    "invoice_amount",
    "cleared_amount",
    "net",
    "gross",
    "fee",
}

_DATE_KEYWORDS = {
    "date",
    "time",
    "timestamp",
    "invoice_date",
    "cleared_date",
    "txn_date",
    "trans_date",
    "created_at",
    "posted_date",
    "due_date",
}


def _classify_column_role(series: pd.Series, col_name: Any) -> str:
    """Classify a column into semantic role: 'key', 'name', 'text', 'numeric', 'date', or 'unknown'."""
    lower_name = str(col_name).lower().strip()
    name_tokens = set(re.findall(r"[a-z0-9]+", lower_name))

    def matches_keywords(k_set: set[str]) -> bool:
        if bool(name_tokens & k_set) or lower_name in k_set:
            return True
        return any(k in lower_name for k in k_set if "_" in k)

    # 1. Date Detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if matches_keywords(_DATE_KEYWORDS):
        sample = series.dropna().head(20)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().sum() / len(sample) >= 0.7:
                return "date"

    # 2. Numeric / Amount Detection
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if matches_keywords(_AMOUNT_KEYWORDS):
            return "numeric"
        if any(
            lower_name.endswith(k) or lower_name == k for k in ("id", "code", "num", "no", "key")
        ):
            return "key"
        if pd.api.types.is_float_dtype(series):
            return "numeric"
        # Integer column fallback: high uniqueness implies key, low implies numeric
        n_unique = series.nunique()
        if len(series) > 0 and (n_unique / len(series)) >= 0.8:
            return "key"
        return "numeric"

    # String with potential currency formatting
    as_str = series.dropna().astype(str).head(20)
    if matches_keywords(_AMOUNT_KEYWORDS) and len(as_str) > 0:
        cleaned = as_str.str.replace(r"[$€£,\s]", "", regex=True)
        numeric_test = pd.to_numeric(cleaned, errors="coerce")
        if numeric_test.notna().sum() / len(as_str) >= 0.7:
            return "numeric"

    # 3. Key / Identifier Detection
    if matches_keywords(_KEY_KEYWORDS):
        return "key"

    # 4. Primary Name vs Secondary Text Detection
    if matches_keywords(_NAME_KEYWORDS):
        return "name"
    if matches_keywords(_SECONDARY_TEXT_KEYWORDS):
        return "text"

    # 5. Default by string token statistics
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        tokens = as_str.str.split().apply(len)
        mean_tokens = float(tokens.mean()) if len(tokens) > 0 else 0.0
        if mean_tokens >= 1.8:
            return "name"
        return "text"

    return "unknown"


def _extract_series(df: pd.DataFrame, col: Any) -> pd.Series:
    val = df[col]
    if isinstance(val, pd.DataFrame):
        res: pd.Series = val.iloc[:, 0]
        return res
    if isinstance(val, pd.Series):
        return val
    return pd.Series(val)


def _discover_column_pairs(
    df_l: pd.DataFrame, df_r: pd.DataFrame
) -> dict[str, list[tuple[Any, Any]]]:
    """Pair columns between df_left and df_right by exact match, synonyms, and roles."""
    roles_l = {
        col: _classify_column_role(
            _extract_series(df_l, col),
            col,
        )
        for col in df_l.columns
    }
    roles_r = {
        col: _classify_column_role(
            _extract_series(df_r, col),
            col,
        )
        for col in df_r.columns
    }

    pairs: dict[str, list[tuple[Any, Any]]] = {
        "key": [],
        "name": [],
        "text": [],
        "numeric": [],
        "date": [],
    }

    used_l: set[Any] = set()
    used_r: set[Any] = set()

    # Tier 1: Exact Name Match (case-insensitive)
    cols_r_lower: dict[str, Any] = {str(c).lower().strip(): c for c in df_r.columns}
    for col_l in df_l.columns:
        cl_low = str(col_l).lower().strip()
        if cl_low in cols_r_lower:
            col_r = cols_r_lower[cl_low]
            role = roles_l[col_l]
            if role in pairs and col_l not in used_l and col_r not in used_r:
                pairs[role].append((col_l, col_r))
                used_l.add(col_l)
                used_r.add(col_r)

    # Tier 2: Keyword Synonym Match across common roles
    for role in ("name", "key", "numeric", "date", "text"):
        unpaired_l = [c for c, r in roles_l.items() if r == role and c not in used_l]
        unpaired_r = [c for c, r in roles_r.items() if r == role and c not in used_r]
        for cl in unpaired_l:
            for cr in unpaired_r:
                if cl not in used_l and cr not in used_r:
                    pairs[role].append((cl, cr))
                    used_l.add(cl)
                    used_r.add(cr)
                    break

    # Tier 3: Promote first text pair to name if no name was discovered
    if not pairs["name"] and pairs["text"]:
        pairs["name"].append(pairs["text"].pop(0))

    # Tier 4: Unpaired text fallback to name if no name was discovered
    if not pairs["name"]:
        unpaired_l_text = [
            c for c in df_l.columns if c not in used_l and roles_l[c] in ("name", "text", "unknown")
        ]
        unpaired_r_text = [
            c for c in df_r.columns if c not in used_r and roles_r[c] in ("name", "text", "unknown")
        ]
        if unpaired_l_text and unpaired_r_text:
            pairs["name"].append((unpaired_l_text[0], unpaired_r_text[0]))
            used_l.add(unpaired_l_text[0])
            used_r.add(unpaired_r_text[0])

    return pairs


# ---------------------------------------------------------------------------
# AutoMatcher Engine
# ---------------------------------------------------------------------------


class AutoMatcher:
    """Zero-configuration automated record linkage and entity resolution engine.

    Automatically profiles arbitrary input tables, discovers key/text/numeric/date
    attributes, selects optimal multi-pass candidate blocking passes, computes
    continuous proximity scores, and generates non-transitive match clusters
    using Star/Pivot clustering.
    """

    def __init__(
        self,
        budget_cap: int = 50_000,
        threshold: float = 0.7,
        cluster_method: Literal["star", "connected_components"] = "star",
        max_window: int = 8,
    ) -> None:
        if budget_cap <= 0:
            raise ValueError(f"budget_cap must be a positive integer, got {budget_cap}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        if cluster_method not in ("star", "connected_components"):
            raise ValueError(
                f"Unknown cluster_method: {cluster_method!r}. Expected 'star' or 'connected_components'"
            )
        if max_window < 1:
            raise ValueError(f"max_window must be >= 1, got {max_window}")

        self.budget_cap = int(budget_cap)
        self.threshold = float(threshold)
        self.cluster_method = cluster_method
        self.max_window = int(max_window)
        self.scorer = RapidFuzzScorer()

    def fit_predict(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
    ) -> MatchResults:
        """Profile inputs, execute multi-pass blocking, score, and cluster matches."""
        # 1. Pre-flight Boundary Checks
        if (
            len(df_left) == 0
            or len(df_right) == 0
            or len(df_left.columns) == 0
            or len(df_right.columns) == 0
        ):
            return MatchResults(
                matches=pd.DataFrame(
                    columns=["cluster_id", "left_id", "right_id", "score", "status"]
                ),
                clusters=[],
                decisions=[],
                summary_stats={
                    "left_rows": len(df_left),
                    "right_rows": len(df_right),
                    "candidate_pairs": 0,
                    "budget_cap": self.budget_cap,
                    "budget_utilization_pct": 0.0,
                    "matched_pairs": 0,
                    "clusters": 0,
                    "threshold": self.threshold,
                    "cluster_method": self.cluster_method,
                },
                left_df=df_left,
                right_df=df_right,
            )

        # 2. Automated Column Discovery & Pairing
        col_pairs = _discover_column_pairs(df_left, df_right)

        # 3. Construct Normalized Working Frames with Reset Index
        work_l = df_left.reset_index(drop=True)
        work_r = df_right.reset_index(drop=True)

        def _get_series(df: pd.DataFrame, col: Any) -> pd.Series:
            return _extract_series(df, col)

        def _get_val(row: pd.Series, col: Any) -> Any:
            v = row[col]
            return v.iloc[0] if isinstance(v, pd.Series) else v

        # Resolve primary key columns for output labelling
        left_id_col = col_pairs["key"][0][0] if col_pairs["key"] else None
        right_id_col = col_pairs["key"][0][1] if col_pairs["key"] else None

        passes: list[Any] = []

        # Pass 1: Exact ID / Key Block (if paired key found)
        if col_pairs["key"]:
            kl, kr = col_pairs["key"][0]
            work_l["_auto_key"] = _get_series(work_l, kl).astype(str).str.strip().str.lower()
            work_r["_auto_key"] = _get_series(work_r, kr).astype(str).str.strip().str.lower()
            passes.append(ExactBlock(["_auto_key"]))

        # Pass 2 & 3: Normalized Name Exact Block & Sorted Neighborhood
        if col_pairs["name"]:
            nl, nr = col_pairs["name"][0]
            work_l["_auto_clean_name"] = (
                _get_series(work_l, nl)
                .apply(strip_legal_suffixes)
                .astype(str)
                .str.strip()
                .str.lower()
            )
            work_r["_auto_clean_name"] = (
                _get_series(work_r, nr)
                .apply(strip_legal_suffixes)
                .astype(str)
                .str.strip()
                .str.lower()
            )
            passes.append(ExactBlock(["_auto_clean_name"]))

            effective_window = min(self.max_window, max(len(work_l), len(work_r)))
            passes.append(SortedNeighborhood(["_auto_clean_name"], window=max(1, effective_window)))

        # Pass 4: Secondary Text Exact Block
        if col_pairs["text"]:
            tl, tr = col_pairs["text"][0]
            work_l["_auto_sec_text"] = _get_series(work_l, tl).astype(str).str.strip().str.lower()
            work_r["_auto_sec_text"] = _get_series(work_r, tr).astype(str).str.strip().str.lower()
            passes.append(ExactBlock(["_auto_sec_text"]))

        # Fallback if no passes were constructed: SortedNeighborhood on first column
        if not passes:
            fallback_col_l = df_left.columns[0]
            fallback_col_r = df_right.columns[0]
            work_l["_auto_fallback"] = (
                _get_series(work_l, fallback_col_l).astype(str).str.strip().str.lower()
            )
            work_r["_auto_fallback"] = (
                _get_series(work_r, fallback_col_r).astype(str).str.strip().str.lower()
            )
            effective_window = min(self.max_window, max(len(work_l), len(work_r)))
            passes.append(SortedNeighborhood(["_auto_fallback"], window=max(1, effective_window)))

        # 4. Multi-Pass Candidate Generation bounded by budget_cap
        blocker = MultiPassBlock(passes=passes, budget_cap=self.budget_cap)
        candidate_index = blocker.block(work_l, work_r)
        candidate_pairs = list(candidate_index.pairs)

        # 5. Continuous Scoring across Paired Attributes
        num_scorers: dict[tuple[Any, Any], NumericProximityScorer] = {}
        for nl, nr in col_pairs["numeric"]:
            s_l = _get_series(df_left, nl)
            s_r = _get_series(df_right, nr)
            vals = pd.to_numeric(pd.concat([s_l, s_r]), errors="coerce")
            finite_vals = vals[np.isfinite(vals)]
            std_val = (
                float(finite_vals.std())
                if len(finite_vals) > 1 and not math.isnan(float(finite_vals.std()))
                else 1.0
            )
            epsilon = max(1.0, 0.05 * std_val)
            num_scorers[(nl, nr)] = NumericProximityScorer(epsilon=epsilon)

        date_scorers: dict[tuple[Any, Any], DateWindowScorer] = {
            (dl, dr): DateWindowScorer(delta_days=5.0) for dl, dr in col_pairs["date"]
        }

        decisions: list[MatchDecision] = []
        scored_pairs: list[tuple[Any, Any]] = []
        pair_scores: list[float] = []
        match_records: list[dict[str, Any]] = []

        # Check for potential ID overlap between left and right
        all_l_ids = {
            str(_get_val(df_left.iloc[i], left_id_col))
            if left_id_col is not None and pd.notna(_get_val(df_left.iloc[i], left_id_col))
            else (str(df_left.index[i]) if pd.notna(df_left.index[i]) else f"L_{i}")
            for i in range(len(df_left))
        }
        all_r_ids = {
            str(_get_val(df_right.iloc[j], right_id_col))
            if right_id_col is not None and pd.notna(_get_val(df_right.iloc[j], right_id_col))
            else (str(df_right.index[j]) if pd.notna(df_right.index[j]) else f"R_{j}")
            for j in range(len(df_right))
        }
        has_id_overlap = bool(all_l_ids & all_r_ids)

        for l_pos, r_pos in candidate_pairs:
            row_l = df_left.iloc[l_pos]
            row_r = df_right.iloc[r_pos]

            if left_id_col is not None:
                raw_lid = _get_val(row_l, left_id_col)
                left_id = (
                    raw_lid
                    if pd.notna(raw_lid)
                    else (df_left.index[l_pos] if pd.notna(df_left.index[l_pos]) else f"L_{l_pos}")
                )
            else:
                left_id = df_left.index[l_pos] if pd.notna(df_left.index[l_pos]) else f"L_{l_pos}"

            if right_id_col is not None:
                raw_rid = _get_val(row_r, right_id_col)
                right_id = (
                    raw_rid
                    if pd.notna(raw_rid)
                    else (
                        df_right.index[r_pos] if pd.notna(df_right.index[r_pos]) else f"R_{r_pos}"
                    )
                )
            else:
                right_id = (
                    df_right.index[r_pos] if pd.notna(df_right.index[r_pos]) else f"R_{r_pos}"
                )

            field_scores_list: list[FieldScore] = []
            weights: list[float] = []

            # Score Name
            if col_pairs["name"]:
                nl, nr = col_pairs["name"][0]
                val_l = _get_val(row_l, nl)
                val_r = _get_val(row_r, nr)
                vl = str(val_l) if pd.notna(val_l) else ""
                vr = str(val_r) if pd.notna(val_r) else ""
                s_name = self.scorer.token_sort_ratio(
                    strip_legal_suffixes(vl), strip_legal_suffixes(vr)
                )
                field_scores_list.append(
                    FieldScore(field=str(nl), left_value=vl, right_value=vr, score=s_name)
                )
                weights.append(0.60)

            # Score Secondary Text
            if col_pairs["text"]:
                tl, tr = col_pairs["text"][0]
                val_l = _get_val(row_l, tl)
                val_r = _get_val(row_r, tr)
                has_l = pd.notna(val_l) and str(val_l).strip() != ""
                has_r = pd.notna(val_r) and str(val_r).strip() != ""
                if has_l and has_r:
                    vl = str(val_l)
                    vr = str(val_r)
                    s_text = self.scorer.token_sort_ratio(vl, vr)
                    field_scores_list.append(
                        FieldScore(field=str(tl), left_value=vl, right_value=vr, score=s_text)
                    )
                    weights.append(0.15)
                else:
                    vl = str(val_l) if pd.notna(val_l) else ""
                    vr = str(val_r) if pd.notna(val_r) else ""
                    field_scores_list.append(
                        FieldScore(field=str(tl), left_value=vl, right_value=vr, score=0.0)
                    )
                    weights.append(0.0)

            # Score Numeric Amount
            for (nl, nr), n_scorer in num_scorers.items():
                val_l = _get_val(row_l, nl)
                val_r = _get_val(row_r, nr)
                is_num_valid = pd.notna(val_l) and pd.notna(val_r)
                if is_num_valid:
                    try:
                        f_l = float(val_l)
                        f_r = float(val_r)
                        if not (math.isfinite(f_l) and math.isfinite(f_r)):
                            is_num_valid = False
                    except (ValueError, TypeError):
                        is_num_valid = False

                if is_num_valid:
                    s_num = n_scorer.score(val_l, val_r)
                    field_scores_list.append(
                        FieldScore(field=str(nl), left_value=val_l, right_value=val_r, score=s_num)
                    )
                    weights.append(0.20)
                else:
                    field_scores_list.append(
                        FieldScore(field=str(nl), left_value=val_l, right_value=val_r, score=0.0)
                    )
                    weights.append(0.0)

            # Score Date
            for (dl, dr), d_scorer in date_scorers.items():
                val_l = _get_val(row_l, dl)
                val_r = _get_val(row_r, dr)
                is_date_valid = pd.notna(val_l) and pd.notna(val_r)
                if is_date_valid:
                    try:
                        p_l = pd.to_datetime(val_l, errors="coerce")
                        p_r = pd.to_datetime(val_r, errors="coerce")
                        if pd.isna(p_l) or pd.isna(p_r):
                            is_date_valid = False
                    except Exception:
                        is_date_valid = False

                if is_date_valid:
                    s_date = d_scorer.score(val_l, val_r)
                    field_scores_list.append(
                        FieldScore(field=str(dl), left_value=val_l, right_value=val_r, score=s_date)
                    )
                    weights.append(0.15)
                else:
                    field_scores_list.append(
                        FieldScore(field=str(dl), left_value=val_l, right_value=val_r, score=0.0)
                    )
                    weights.append(0.0)

            # Fallback (if no standard columns paired or all weights are 0.0)
            if not weights or sum(weights) == 0.0:
                fl, fr = df_left.columns[0], df_right.columns[0]
                val_l = _get_val(row_l, fl)
                val_r = _get_val(row_r, fr)
                vl = str(val_l) if pd.notna(val_l) else ""
                vr = str(val_r) if pd.notna(val_r) else ""
                s_fb = self.scorer.token_sort_ratio(vl, vr)
                field_scores_list.append(
                    FieldScore(field=str(fl), left_value=vl, right_value=vr, score=s_fb)
                )
                weights.append(1.0)

            total_weight = sum(weights)
            composite_score = (
                sum(fs.score * w for fs, w in zip(field_scores_list, weights, strict=True))
                / total_weight
                if total_weight > 0
                else 0.0
            )

            status: MatchStatus = (
                "match"
                if composite_score >= self.threshold
                else (
                    "possible"
                    if composite_score >= max(0.0, self.threshold - 0.20)
                    else "non_match"
                )
            )

            explanation = Explanation(
                field_scores=tuple(field_scores_list),
                threshold_band=status,
                blocking_rules=("MultiPassBlock",),
            )

            decision = MatchDecision(
                left_id=left_id,
                right_id=right_id,
                status=status,
                score=composite_score,
                explanation=explanation,
            )
            decisions.append(decision)

            if composite_score >= self.threshold:
                # Disambiguate bipartite identifiers if overlapping
                node_l = f"L_{left_id}" if has_id_overlap else str(left_id)
                node_r = f"R_{right_id}" if has_id_overlap else str(right_id)
                scored_pairs.append((node_l, node_r))
                pair_scores.append(composite_score)

                rec: dict[str, Any] = {
                    "left_id": left_id,
                    "source_id": left_id,
                    "right_id": right_id,
                    "target_id": right_id,
                    "score": composite_score,
                    "fuzzy_score": composite_score,
                    "status": status,
                }
                for fs in field_scores_list:
                    rec[f"score_{fs.field}"] = fs.score
                    rec[f"left_{fs.field}"] = fs.left_value
                    rec[f"right_{fs.field}"] = fs.right_value
                match_records.append(rec)

        # 6. Clustering
        clusters: list[set[Any]] = []
        if scored_pairs:
            if self.cluster_method == "star":
                clusterer = StarClusterer(cutoff=self.threshold, verify_clique=True)
                clusters = clusterer.cluster(scored_pairs, edge_weights=pair_scores)
            else:
                cc = ConnectedComponents(edge_weight_cutoff=self.threshold)
                clusters = cc.cluster(scored_pairs, edge_weights=pair_scores)

        # 7. Map Cluster IDs back to Matches DataFrame
        node_to_cluster: dict[Any, int] = {}
        for c_idx, cl in enumerate(clusters):
            for member in cl:
                node_to_cluster[member] = c_idx

        consistent_match_records: list[dict[str, Any]] = []
        for rec in match_records:
            node_l = f"L_{rec['left_id']}" if has_id_overlap else str(rec["left_id"])
            node_r = f"R_{rec['right_id']}" if has_id_overlap else str(rec["right_id"])
            c_l = node_to_cluster.get(node_l, -1)
            c_r = node_to_cluster.get(node_r, -1)
            if c_l != -1 and c_l == c_r:
                rec["cluster_id"] = c_l
                consistent_match_records.append(rec)

        matches_df = pd.DataFrame(consistent_match_records)
        if matches_df.empty:
            matches_df = pd.DataFrame(
                columns=[
                    "cluster_id",
                    "left_id",
                    "source_id",
                    "right_id",
                    "target_id",
                    "score",
                    "fuzzy_score",
                    "status",
                ]
            )
        else:
            lead_cols = ["cluster_id", "left_id", "right_id", "score", "status"]
            other_cols = [c for c in matches_df.columns if c not in lead_cols]
            matches_df = matches_df[lead_cols + other_cols]

        summary_stats = {
            "left_rows": len(df_left),
            "right_rows": len(df_right),
            "candidate_pairs": len(candidate_pairs),
            "budget_cap": self.budget_cap,
            "budget_utilization_pct": (len(candidate_pairs) / self.budget_cap) * 100.0,
            "matched_pairs": len(matches_df),
            "clusters": len(clusters),
            "threshold": self.threshold,
            "cluster_method": self.cluster_method,
        }

        return MatchResults(
            matches=matches_df,
            clusters=clusters,
            decisions=decisions,
            summary_stats=summary_stats,
            left_df=df_left,
            right_df=df_right,
        )
