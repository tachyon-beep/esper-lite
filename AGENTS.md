# Repository Guidelines

## Agent Crib Sheet

- When starting any task, skim docs/architecture_summary.md first. It summarizes subsystem responsibilities, Leyline contracts, safety/telemetry patterns, and key integration points. Treat it as the canonical quick reference when navigating code or adding features.

## Project Structure & Module Organization

- Source modules live under `src/esper/`; create subsystem packages (e.g. `tolaria`, `kasmina`, `tamiyo`) in line with the detailed-design docs.
- Tests live in `tests/`, mirroring the code; slower integration suites belong under `tests/integration`.
- Specifications sit in `docs/design/detailed_design/` (legacy `old/` remains authoritative); planning materials live in `docs/project/`.

## Build, Test, and Development Commands

- Create a local env with `python3.11 -m venv .venv && source .venv/bin/activate`; keep the virtualenv out of Git.
- Install or update dependencies as you introduce them, ideally via a `pyproject.toml` or pinned `requirements.txt` committed with the feature.
- Run the fast test suite with `pytest tests`.
- Lint before committing: `pylint --rcfile .codacy/tools-configs/pylint.rc src/esper`.
- CI parity checks run through `.codacy/cli.sh ...`, which bootstraps the toolchain from `.codacy/codacy.yaml`.

## Coding Style & Naming Conventions

- Python is the default language; follow PEP 8 with four-space indentation, type hints, and docstrings that cite the relevant design section.
- Packages and modules use `snake_case`, classes `PascalCase`, and constants `UPPER_SNAKE_CASE`.
- Organize subsystem code by lifecycle phases (e.g. `esper/kasmina/lifecycle.py`) and expose public APIs in `__init__.py`.

## Testing Guidelines

- Prefer `pytest` fixtures for shared infrastructure fakes (Redis Streams, Prometheus exporters) and keep mocks lightweight.
- Unit tests should mirror the source path (`tests/kasmina/test_lifecycle.py`) and assert both success paths and guard-rail failures from the design docs.
- Add integration flows for the control loop and blueprint pipeline as slices land; target ≥80% coverage and property tests for Leyline serialization.

## Commit & Pull Request Guidelines

- Match existing history: short, imperative, Title-Case subjects (`Add Tamiyo Risk Gates`), with bodies that explain why and reference doc sections.
- Every PR needs a “Changes” summary, test command list, linked planning ticket, and notes on telemetry or contract impacts.
- Request reviews from the subsystem owner named in `docs/project/prototype_charter.md`; attach logs or screenshots for observability work.

## Design & Operations Reference

- Update `docs/` whenever behaviour diverges from the canonical `old/` design files, and surface new operational learnings in `docs/project/backlog.md`.
- Never commit secrets; document required environment variables in a checked-in `README` or `.env.example` within the relevant subsystem folder.
- Keep docker-compose or deployment manifests beside the service code once infrastructure scaffolding begins.

### Prototype Delta Guidance

- We maintain a prototype scope that differs from the full detailed design; see `docs/prototype-delta/README.md` and its per‑subsystem deltas.
- When implementing changes, optimise for the “green for prototype” acceptance (prototype‑delta) rather than the full design unless the delta calls for a full feature.
- Use `docs/prototype-delta/rubric.md` when assessing completeness; reference delta docs in PRs where appropriate.
