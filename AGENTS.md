# Repository Guidelines

## Start Here

- Read `CLAUDE.md` first: it contains the project intro, required reading (e.g., `README.md`, `ROADMAP.md`), and the mandatory coding standards/policies for this repo.

## Project Structure & Module Organization

- `src/esper/`: main Python package (domain modules: `kasmina/`, `leyline/`, `tamiyo/`, `tolaria/`, `simic/`, `nissa/`, `karn/`, plus `scripts/`).
- `tests/`: pytest suite (unit-style tests at repo root and module folders; integration tests in `tests/integration/`; property-based tests in `tests/properties/`).
- `docs/`: architecture notes and implementation plans.
- `scripts/`: helper shell entrypoints (e.g., `scripts/train_cifar.sh`).
- `data/` and `telemetry/`: local datasets and run artifacts (gitignored; only `.gitkeep` is tracked).

## Build, Test, and Development Commands

- Install deps (preferred): `uv sync`
- Run training CLI:
  - Heuristic: `PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar10 --episodes 1`
  - PPO: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar10 --episodes 10 --n-envs 4`
- Optional dashboard deps: `uv sync --extra dashboard` (then add `--dashboard` to the train command).
- Run tests: `uv run pytest` (common: `uv run pytest -m "not slow"`)
- CI parity checks:
  - Lint: `ruff check src/ tests/`
  - Types: `mypy src/`

## Coding Style & Naming Conventions

- Python 3.11+; 4-space indentation; prefer type hints and `from __future__ import annotations` on new modules.
- Keep names conventional: `snake_case.py`, `snake_case()` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Shared contracts (enums/constants/schemas) belong in `src/esper/leyline/` to avoid cross-domain coupling.
- Follow the repo guardrails in `CLAUDE.md` (notably: no compatibility shims/legacy code; `hasattr()` requires explicit, documented authorization).

## Testing Guidelines

- Tests follow `test_*.py` naming and use pytest markers from `pytest.ini` (e.g., `integration`, `slow`, `property`).
- Hypothesis uses profiles via `HYPOTHESIS_PROFILE` (`dev` default; `ci`/`thorough` available).
- Coverage: CI enforces ≥80% when run with `pytest --cov` (see `.github/workflows/test-suite.yml`).

## Commit & Pull Request Guidelines

- Use Conventional Commits seen in history: `feat(scope): …`, `fix: …`, `docs: …`, `refactor: …`, `chore: …`.
- PRs should include: intent + summary, commands run (tests/lint), linked issues if any, and screenshots for TUI/dashboard changes.
- Do not commit generated artifacts from `data/` or `telemetry/`.
