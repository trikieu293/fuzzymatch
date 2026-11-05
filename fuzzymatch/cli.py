"""Command line interface for core fuzzymatch operations.

Provides the `fuzzymatch match` command for zero-configuration entity
resolution, record linkage, and deduplication across tabular datasets (CSV and Excel).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .api.auto import AutoMatcher, MatchResults
from .decisions import MatchDecision


def load_tabular_file(path_str: str) -> pd.DataFrame:
    """Load tabular data from CSV or Excel file formats with validation.

    Args:
        path_str: Path to the tabular input file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format cannot be parsed or decoded.
    """
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: '{path_str}'")

    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls", ".xlsm"):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read tabular file '{path_str}': {exc}") from exc


def format_summary_table(
    left_count: int,
    right_count: int,
    candidate_count: int,
    budget_cap: int,
    cluster_count: int,
    match_count: int,
    elapsed_sec: float,
) -> str:
    """Format matching execution summary statistics into a clean ASCII table."""
    budget_usage_pct = (candidate_count / budget_cap * 100.0) if budget_cap > 0 else 0.0
    min_records = min(left_count, right_count)
    match_rate_pct = (match_count / min_records * 100.0) if min_records > 0 else 0.0

    rows: list[tuple[str, str]] = [
        ("Total records (Left)", f"{left_count:,}"),
        ("Total records (Right)", f"{right_count:,}"),
        ("Candidate pairs generated", f"{candidate_count:,}"),
        ("Candidate budget cap", f"{budget_cap:,}"),
        ("Budget cap usage", f"{budget_usage_pct:.2f}%"),
        ("Match clusters identified", f"{cluster_count:,}"),
        ("Matched record pairs", f"{match_count:,}"),
        ("Match rate (vs min records)", f"{match_rate_pct:.2f}%"),
        ("Execution time", f"{elapsed_sec:.3f}s"),
    ]

    col1_w = max(len(label) for label, _ in rows) + 2
    col2_w = max(len(val) for _, val in rows) + 2
    total_w = col1_w + col2_w + 5

    divider = "+" + "-" * (col1_w + 1) + "+" + "-" * (col2_w + 1) + "+"
    header_divider = "=" * total_w

    lines = [
        "",
        header_divider,
        "FUZZYMATCH RECORD LINKAGE SUMMARY".center(total_w),
        header_divider,
        divider,
        f"| {'Metric'.ljust(col1_w)}| {'Value'.ljust(col2_w)}|",
        divider,
    ]
    for label, val in rows:
        lines.append(f"| {label.ljust(col1_w)}| {val.ljust(col2_w)}|")
    lines.append(divider)
    lines.append("")
    return "\n".join(lines)


def format_explanations(
    decisions: Sequence[MatchDecision],
    threshold: float,
    max_display: int = 50,
) -> str:
    """Format human-readable breakdown of match decisions and field scores."""
    lines: list[str] = [
        "",
        "=" * 72,
        "                        MATCH EXPLANATIONS                        ",
        "=" * 72,
    ]

    matched_decisions = [d for d in decisions if d.status == "match"]
    if not matched_decisions:
        matched_decisions = list(decisions[:max_display])
    else:
        matched_decisions = matched_decisions[:max_display]

    if not matched_decisions:
        lines.append("  No candidate matches available to explain.")
        lines.append("=" * 72)
        lines.append("")
        return "\n".join(lines)

    for i, dec in enumerate(matched_decisions, 1):
        lines.append(f"Pair #{i}: Left [{dec.left_id}] <-> Right [{dec.right_id}]")
        lines.append(
            f"  Composite Score: {dec.score:.4f}  (Threshold: >={threshold:.4f}, Band: {dec.status.upper()})"
        )
        if dec.explanation.blocking_rules:
            rules_str = ", ".join(dec.explanation.blocking_rules)
            lines.append(f"  Blocking Passes: {rules_str}")
        if dec.explanation.field_scores:
            lines.append("  Field-Level Breakdown:")
            for fs in dec.explanation.field_scores:
                weight_str = ""
                if (
                    dec.explanation.classifier_weights
                    and fs.field in dec.explanation.classifier_weights
                ):
                    w = dec.explanation.classifier_weights[fs.field]
                    weight_str = f" [weight: {w:.2f}]"
                lines.append(
                    f"    * {fs.field}: {fs.left_value!r} vs {fs.right_value!r} -> score: {fs.score:.4f}{weight_str}"
                )
        lines.append("-" * 72)

    total_candidates = len(decisions)
    if total_candidates > len(matched_decisions):
        rem = total_candidates - len(matched_decisions)
        lines.append(f"  ... and {rem} additional evaluated pair(s) omitted from terminal display.")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def write_output_file(
    results: MatchResults,
    output_path_str: str,
) -> int:
    """Serialize matching output records to CSV or Excel with directory creation."""
    out_path = Path(output_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(results, "to_dataframe") and callable(results.to_dataframe):
        out_df = results.to_dataframe()
    elif hasattr(results, "matches") and isinstance(results.matches, pd.DataFrame):
        out_df = results.matches
    else:
        cluster_map: dict[Any, int] = {}
        clusters = getattr(results, "clusters", [])
        for c_idx, members in enumerate(clusters):
            for mem in members:
                cluster_map[mem] = c_idx

        records: list[dict[str, Any]] = []
        decisions = getattr(results, "decisions", [])
        for d in decisions:
            if d.status == "match":
                c_id = cluster_map.get(d.left_id, cluster_map.get(d.right_id, -1))
                records.append(
                    {
                        "cluster_id": c_id,
                        "left_id": d.left_id,
                        "right_id": d.right_id,
                        "match_score": d.score,
                        "status": d.status,
                    }
                )
        out_df = pd.DataFrame(records)

    suffix = out_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        out_df.to_excel(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    return len(out_df)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with subcommands and flags."""
    parser = argparse.ArgumentParser(
        prog="fuzzymatch",
        description="Industrial-scale text fuzzy mapping and record linkage CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    match_parser = subparsers.add_parser(
        "match",
        help="Perform automated zero-configuration entity resolution between two tabular files.",
    )
    match_parser.add_argument(
        "left",
        nargs="?",
        default=None,
        help="Path to left dataset (CSV or Excel).",
    )
    match_parser.add_argument(
        "right",
        nargs="?",
        default=None,
        help="Path to right dataset (CSV or Excel).",
    )
    match_parser.add_argument(
        "-l",
        "--left",
        dest="left_opt",
        default=None,
        help="Path to left dataset (optional if specified as positional).",
    )
    match_parser.add_argument(
        "-r",
        "--right",
        dest="right_opt",
        default=None,
        help="Path to right dataset (optional if specified as positional).",
    )
    match_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Destination path for matched records output (.csv or .xlsx).",
    )
    match_parser.add_argument(
        "--budget-cap",
        type=int,
        default=50_000,
        help="Candidate pair generation budget cap (default: 50,000).",
    )
    match_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Composite similarity score match threshold (default: 0.7).",
    )
    match_parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="Display detailed human-readable match explanations in terminal.",
    )
    match_parser.add_argument(
        "--cluster-method",
        choices=["star", "connected_components"],
        default="star",
        help="Clustering algorithm for entity resolution (default: 'star').",
    )

    return parser


def run_match_command(args: argparse.Namespace) -> int:
    """Execute the `fuzzymatch match` pipeline workflow."""
    left_path = args.left_opt or args.left
    right_path = args.right_opt or (args.right if args.left_opt is None else args.left)

    if not left_path or not right_path:
        print(
            "Error: Both left and right datasets must be specified (e.g. "
            "'fuzzymatch match left.csv right.csv' or 'fuzzymatch match --left left.csv --right right.csv').",
            file=sys.stderr,
        )
        return 2

    try:
        df_left = load_tabular_file(left_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        df_right = load_tabular_file(right_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    start_time = time.perf_counter()
    try:
        matcher = AutoMatcher(
            budget_cap=args.budget_cap,
            threshold=args.threshold,
            cluster_method=args.cluster_method,
        )
        results = matcher.fit_predict(df_left, df_right)
    except Exception as exc:
        print(f"Error during entity resolution: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start_time

    left_count = len(df_left)
    right_count = len(df_right)
    candidate_count = getattr(
        results, "total_candidates", len(results.pairs) if hasattr(results, "pairs") else 0
    )
    cluster_count = len(results.clusters) if hasattr(results, "clusters") else 0
    match_count = (
        len(results.matches)
        if hasattr(results, "matches") and results.matches is not None
        else (len(results.pairs) if hasattr(results, "pairs") else 0)
    )

    table_str = format_summary_table(
        left_count=left_count,
        right_count=right_count,
        candidate_count=candidate_count,
        budget_cap=args.budget_cap,
        cluster_count=cluster_count,
        match_count=match_count,
        elapsed_sec=elapsed,
    )
    print(table_str)

    if args.explain:
        decisions = getattr(results, "decisions", [])
        exp_str = format_explanations(decisions, threshold=args.threshold)
        print(exp_str)

    if args.output:
        try:
            n_written = write_output_file(results, args.output)
            print(f"Matched records successfully saved to '{args.output}' ({n_written} records).")
        except Exception as exc:
            print(f"Error writing output to '{args.output}': {exc}", file=sys.stderr)
            return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point for console scripts and programmatic execution."""
    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]

    try:
        args = parser.parse_args(raw_args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0

    if args.subcommand == "match":
        return run_match_command(args)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
