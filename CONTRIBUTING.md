# Contributing to fuzzymatch

Thank you for your interest in contributing to `fuzzymatch`! We welcome contributions to core algorithms, performance optimizations, bug fixes, documentation, and mathematical tests.

---

## Code of Conduct & Principles

1. **Mathematical Rigor**: All algorithms and scorers must be deductively sound and map cleanly to formal mathematical specifications.
2. **Cheat-Resistant Implementation**: General algorithms only. Never hardcode return values, pattern-match specific test literals, or mock away core logic.
3. **Property-Based Verification**: New features must include property-based invariant tests under `hypothesis` and boundary stress tests.

---

## Development Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/trikieu293/fuzzymatch.git
cd fuzzymatch
pip install -e ".[dev,duckdb]"
```

---

## Code Quality Standards

Before submitting a Pull Request, ensure that all quality gates pass locally:

### 1. Code Formatting and Linting
We use [Ruff](https://github.com/astral-sh/ruff) with strict rule configurations:

```bash
ruff check fuzzymatch tests
ruff format --check fuzzymatch tests
```

To auto-format:
```bash
ruff format fuzzymatch tests
ruff check --fix fuzzymatch tests
```

### 2. Static Type Checking
We enforce strict typing with zero errors under `mypy`:

```bash
mypy
```

### 3. Unit and Adversarial Test Suites
Run the full test suite with coverage enforcement:

```bash
pytest tests/unit tests/adversarial tests/integration --cov=fuzzymatch --cov-report=term-missing --cov-fail-under=95
```

### 4. Packaging Verification
Verify that the package builds and validates strictly:

```bash
python -m build
twine check --strict dist/*
```

---

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature or algorithm enhancement
- `fix:` A bug fix or invariant defect resolution
- `docs:` Documentation improvements or whitepapers
- `test:` Adding or updating property-based or adversarial tests
- `refactor:` Code refactoring without behavior change
- `perf:` Performance or memory optimization
- `chore:` Release tasks, dependency updates, packaging
