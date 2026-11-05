"""Packaging contract tests: release mechanics enforced as executable policy.

These tests pin repository-level policies (version single-sourcing, coverage
gate, CI workflow shape, docs tracking) so they cannot silently regress.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import fuzzymatch

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


class TestVersionSingleSourcing:
    """As a maintainer I want the version defined once, in code."""

    def test_version_is_dynamic_not_duplicated(self) -> None:
        project = _load_pyproject()["project"]
        assert "version" in project.get("dynamic", []), (
            'pyproject must declare dynamic=["version"] instead of a static '
            "copy that can drift from fuzzymatch.__version__"
        )
        assert "version" not in project, "static version must be removed"

    def test_runtime_version_matches_installed_distribution(self) -> None:
        from importlib.metadata import version

        assert fuzzymatch.__version__ == version("fuzzymatch")


class TestDistributionMetadata:
    """As a PyPI consumer I want complete distribution metadata."""

    def test_readme_wired_as_long_description(self) -> None:
        project = _load_pyproject()["project"]
        assert project.get("readme") == "README.md"

    def test_license_declared(self) -> None:
        lic = _load_pyproject()["project"]["license"]
        assert lic == "MIT" or (isinstance(lic, dict) and lic.get("text") == "MIT")


class TestCoverageGate:
    """As a reviewer I want CI to fail when coverage drops."""

    def test_fail_under_configured_at_or_above_95(self) -> None:
        tool = _load_pyproject().get("tool", {})
        fail_under = tool.get("coverage", {}).get("report", {}).get("fail_under")
        assert isinstance(fail_under, (int, float)), (
            "[tool.coverage.report] fail_under must be set in pyproject.toml"
        )
        assert fail_under >= 95


class TestCiWorkflow:
    """As a contributor I want lint/typecheck/tests enforced on every PR."""

    def _workflow(self) -> dict:
        import yaml

        wf_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert wf_path.exists(), "ci.yml must exist"
        wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        assert isinstance(wf, dict), "ci.yml must parse to a mapping"
        return wf

    def test_defines_lint_typecheck_test_jobs(self) -> None:
        jobs = self._workflow().get("jobs", {})
        assert {"lint", "typecheck", "test"} <= set(jobs)

    def test_commands_present(self) -> None:
        import json

        text = json.dumps(self._workflow())
        for command in ("ruff check", "mypy", "pytest"):
            assert command in text, f"CI must run {command!r}"

    def test_test_job_matrix_covers_windows_and_ubuntu(self) -> None:
        test_job = self._workflow()["jobs"]["test"]
        os_list = test_job["strategy"]["matrix"]["os"]
        joined = ";".join(str(os_name) for os_name in os_list).lower()
        assert "windows" in joined
        assert "ubuntu" in joined


class TestLocalResearchIgnored:
    """As a maintainer I want local research and docs excluded from release git trees."""

    def test_local_research_in_gitignore(self) -> None:
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.exists(), ".gitignore must exist"
        content = gitignore_path.read_text(encoding="utf-8")
        assert "docs/" in content or "docs" in content, "docs must be gitignored"
        assert "experiments/" in content or "experiments" in content, (
            "experiments must be gitignored"
        )

    def test_local_research_not_tracked_in_git(self) -> None:
        result = subprocess.run(  # noqa: PLW1510
            ["git", "ls-files", "docs", "experiments"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.stdout.strip() == "", "docs/ and experiments/ must not be tracked in git"

    def test_local_research_pruned_in_manifest(self) -> None:
        manifest_path = REPO_ROOT / "MANIFEST.in"
        assert manifest_path.exists(), "MANIFEST.in must exist"
        content = manifest_path.read_text(encoding="utf-8")
        assert "prune docs" in content, "MANIFEST.in must prune docs"
        assert "prune experiments" in content, "MANIFEST.in must prune experiments"
