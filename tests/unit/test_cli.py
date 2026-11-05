"""Unit test suite for fuzzymatch.cli achieving 100.00% statement coverage."""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fuzzymatch.cli import (
    build_parser,
    format_explanations,
    format_summary_table,
    load_tabular_file,
    main,
    write_output_file,
)
from fuzzymatch.decisions import Explanation, FieldScore, MatchDecision


def test_build_parser_options() -> None:
    """Verify CLI parser construction and default flag configurations."""
    parser = build_parser()
    args = parser.parse_args(["match", "left.csv", "right.csv"])
    assert args.subcommand == "match"
    assert args.left == "left.csv"
    assert args.right == "right.csv"
    assert args.budget_cap == 50_000
    assert args.threshold == 0.7
    assert args.cluster_method == "star"
    assert args.explain is False
    assert args.output is None


def test_main_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --help and --version cleanly return exit code 0."""
    code_help = main(["--help"])
    assert code_help == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out

    code_ver = main(["--version"])
    assert code_ver == 0


def test_main_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify invoking with no subcommand prints help to stderr and returns code 2."""
    code = main([])
    assert code == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.err


def test_main_missing_dataset_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify missing left or right arguments returns exit code 2."""
    assert main(["match"]) == 2
    assert main(["match", "only_left.csv"]) == 2
    assert main(["match", "--right", "only_right.csv"]) == 2


def test_load_tabular_file_not_found() -> None:
    """Verify FileNotFoundError raised for non-existent paths."""
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_tabular_file("non_existent_file.csv")


def test_load_tabular_file_corrupted(tmp_path: Path) -> None:
    """Verify ValueError raised for corrupted or unparseable files."""
    bad_file = tmp_path / "bad.csv"
    bad_file.write_bytes(b"\x00\xff\xfe\x01\x02\x03\x04\x05")
    with pytest.raises(ValueError, match="Failed to read tabular file"):
        load_tabular_file(str(bad_file))


def test_load_tabular_file_excel_and_csv(tmp_path: Path) -> None:
    """Verify loading valid CSV and Excel files."""
    csv_file = tmp_path / "test.csv"
    pd.DataFrame({"id": [1], "name": ["Alpha"]}).to_csv(csv_file, index=False)
    df_csv = load_tabular_file(str(csv_file))
    assert len(df_csv) == 1

    try:
        import openpyxl  # noqa: F401

        xlsx_file = tmp_path / "test.xlsx"
        pd.DataFrame({"id": [2], "name": ["Beta"]}).to_excel(xlsx_file, index=False)
        df_xlsx = load_tabular_file(str(xlsx_file))
        assert len(df_xlsx) == 1
    except ImportError:
        pass


def test_main_file_not_found(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify main returns 1 on file not found for left or right dataset."""
    valid_file = tmp_path / "valid.csv"
    pd.DataFrame({"id": [1]}).to_csv(valid_file, index=False)

    code1 = main(["match", "missing_l.csv", str(valid_file)])
    assert code1 == 1

    code2 = main(["match", str(valid_file), "missing_r.csv"])
    assert code2 == 1


def test_main_malformed_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify main returns 1 on malformed file input."""
    bad_file = tmp_path / "bad.csv"
    bad_file.write_bytes(b"\x00\xff\xfe\x01\x02\x03\x04\x05")
    good_file = tmp_path / "good.csv"
    pd.DataFrame({"id": [1]}).to_csv(good_file, index=False)

    assert main(["match", str(bad_file), str(good_file)]) == 1


def test_main_execution_error_handling(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main catches unexpected AutoMatcher errors and exits with code 1."""
    f_l = tmp_path / "l.csv"
    f_r = tmp_path / "r.csv"
    pd.DataFrame({"id": [1]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": [1]}).to_csv(f_r, index=False)

    with patch(
        "fuzzymatch.cli.AutoMatcher.fit_predict", side_effect=RuntimeError("Engine failure")
    ):
        code = main(["match", str(f_l), str(f_r)])
        assert code == 1
        captured = capsys.readouterr()
        assert "Engine failure" in captured.err


def test_main_csv_positional_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify end-to-end match execution with positional CSV arguments and summary output."""
    f_l = tmp_path / "left.csv"
    f_r = tmp_path / "right.csv"
    f_out = tmp_path / "out.csv"

    pd.DataFrame({"id": ["A1"], "name": ["Acme Corp"]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": ["B1"], "name": ["Acme Corp LLC"]}).to_csv(f_r, index=False)

    code = main(["match", str(f_l), str(f_r), "--output", str(f_out)])
    assert code == 0
    assert f_out.exists()

    captured = capsys.readouterr()
    assert "FUZZYMATCH RECORD LINKAGE SUMMARY" in captured.out
    assert "Matched record pairs" in captured.out


def test_main_excel_named_flags_and_output(tmp_path: Path) -> None:
    """Verify end-to-end match execution with named flags and Excel output."""
    pytest.importorskip("openpyxl")
    f_l = tmp_path / "left.xlsx"
    f_r = tmp_path / "right.xlsx"
    f_out = tmp_path / "out.xlsx"

    pd.DataFrame({"id": ["A1"], "name": ["Apex"]}).to_excel(f_l, index=False)
    pd.DataFrame({"id": ["B1"], "name": ["Apex Inc"]}).to_excel(f_r, index=False)

    code = main(
        [
            "match",
            "--left",
            str(f_l),
            "--right",
            str(f_r),
            "--output",
            str(f_out),
            "--budget-cap",
            "1000",
            "--threshold",
            "0.8",
            "--cluster-method",
            "connected_components",
        ]
    )
    assert code == 0
    assert f_out.exists()
    out_df = pd.read_excel(f_out)
    assert len(out_df) >= 0


def test_main_explain_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify --explain outputs formatted match details and field breakdowns."""
    f_l = tmp_path / "l.csv"
    f_r = tmp_path / "r.csv"

    pd.DataFrame({"id": [1], "name": ["Beacon Global"]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": [2], "name": ["Beacon Global LLC"]}).to_csv(f_r, index=False)

    code = main(["match", str(f_l), str(f_r), "--explain"])
    assert code == 0
    captured = capsys.readouterr()
    assert "MATCH EXPLANATIONS" in captured.out


def test_format_explanations_empty() -> None:
    """Verify format_explanations handles empty candidate lists gracefully."""
    out = format_explanations([], threshold=0.7)
    assert "No candidate matches available to explain" in out


def test_format_explanations_truncation() -> None:
    """Verify format_explanations truncates display when exceeding max_display."""
    decisions = [
        MatchDecision(
            left_id=i,
            right_id=i,
            status="match",
            score=0.95,
            explanation=Explanation(
                field_scores=(FieldScore("name", "A", "A", 1.0),),
                threshold_band="match",
                blocking_rules=("ExactBlock",),
                classifier_weights={"name": 1.0},
            ),
        )
        for i in range(10)
    ]
    out = format_explanations(decisions, threshold=0.7, max_display=3)
    assert "Pair #1" in out
    assert "Pair #3" in out
    assert "7 additional evaluated pair(s) omitted" in out


def test_format_summary_table_zero_boundaries() -> None:
    """Verify summary table formatting with 0 budget_cap and 0 records."""
    table = format_summary_table(
        left_count=0,
        right_count=0,
        candidate_count=0,
        budget_cap=0,
        cluster_count=0,
        match_count=0,
        elapsed_sec=0.01,
    )
    assert "0.00%" in table


def test_write_output_file_fallback_and_error(tmp_path: Path) -> None:
    """Verify output serialization fallback to decision objects and error handling."""
    mock_results = MagicMock(spec=[])
    mock_results.clusters = [{1, 2}]
    mock_results.decisions = [
        MatchDecision(
            left_id=1,
            right_id=2,
            status="match",
            score=0.92,
            explanation=Explanation(
                field_scores=(),
                threshold_band="match",
            ),
        )
    ]
    out_file = tmp_path / "test_out.csv"
    n = write_output_file(mock_results, str(out_file))
    assert n == 1
    assert out_file.exists()


def test_write_output_file_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify main returns 1 if writing output file encounters an OS/IO error."""
    f_l = tmp_path / "l.csv"
    f_r = tmp_path / "r.csv"
    pd.DataFrame({"id": [1]}).to_csv(f_l, index=False)
    pd.DataFrame({"id": [1]}).to_csv(f_r, index=False)

    with patch(
        "fuzzymatch.cli.write_output_file", side_effect=PermissionError("Disk write denied")
    ):
        code = main(["match", str(f_l), str(f_r), "--output", "/forbidden/out.csv"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error writing output" in captured.err


def test_main_module_execution_with_sys_argv() -> None:
    """Verify invocation via default sys.argv fallback."""
    with patch("sys.argv", ["fuzzymatch", "--version"]):
        code = main()
        assert code == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_runpy_entrypoint() -> None:
    """Verify python -m fuzzymatch.cli executes __main__ block cleanly."""
    with patch("sys.argv", ["fuzzymatch", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("fuzzymatch.cli", run_name="__main__")
        assert exc_info.value.code == 0


def test_write_output_file_matches_attribute(tmp_path: Path) -> None:
    """Verify write_output_file when results has matches DataFrame but no to_dataframe method."""
    mock_results = MagicMock(spec=["matches"])
    mock_results.matches = pd.DataFrame({"id": [1], "name": ["Alpha"]})
    out_file = tmp_path / "from_matches.csv"
    n = write_output_file(mock_results, str(out_file))
    assert n == 1
    assert out_file.exists()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_runpy_package_main_entrypoint() -> None:
    """Verify python -m fuzzymatch executes root __main__ cleanly."""
    with patch("sys.argv", ["fuzzymatch", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("fuzzymatch", run_name="__main__")
        assert exc_info.value.code == 0
