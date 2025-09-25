# Repository Guidelines

## Agent Crib Sheet

- When starting any task, skim docs/architecture_summary.md first. It summarizes subsystem responsibilities, Leyline contracts, safety/telemetry patterns, and key integration points. Treat it as the canonical quick reference when navigating code or adding features.
- If a task requires network access or any other sandbox-breaking action, pause and obtain explicit authorization from the user before proceeding.

## Project Structure & Module Organization

- Source modules live under `src/esper/`; create subsystem packages (e.g. `tolaria`, `kasmina`, `tamiyo`) in line with the detailed-design docs.
- Tests live in `tests/`, mirroring the code; slower integration suites belong under `tests/integration`.
- Specifications sit in `docs/design/detailed_design/`; planning materials live in `docs/project/`.

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

- Update `docs/` whenever behaviour diverges from the canonical design files, and surface new operational learnings in `docs/project/backlog.md`.
- Never commit secrets; document required environment variables in a checked-in `README` or `.env.example` within the relevant subsystem folder.
- Keep docker-compose or deployment manifests beside the service code once infrastructure scaffolding begins.

### Prototype Delta Guidance

- We maintain a prototype scope that differs from the full detailed design; see `docs/prototype-delta/README.md` and its per‑subsystem deltas.
- When implementing changes, optimise for the “green for prototype” acceptance (prototype‑delta) rather than the full design unless the delta calls for a full feature.
- Use `docs/prototype-delta/rubric.md` when assessing completeness; reference delta docs in PRs where appropriate.

## Strict Dependencies & Failure Policy (Prototype)

The prototype optimises for simplicity and determinism over compatibility layers. Apply these principles across all subsystems and PRs:

- Strict dependencies (no pseudo-optional deps)
  - Do not label a dependency “optional” unless it is truly install-time optional and fully guarded behind feature gates. Otherwise, treat it as mandatory and declare it in `pyproject.toml`.
  - Avoid `try/except ImportError` patterns in live code for mandatory deps. Fail fast on import or at service preflight.
  - Validate core deps at service startup (preflight): imports, basic connectivity (e.g., Elasticsearch ping), and GPU prerequisites (NVML when CUDA is available).
  - Tests must provide mandatory dependencies explicitly (e.g., pass an `UrzaLibrary` to `TamiyoService`). Do not rely on hidden defaults.

- No backwards compatibility pre‑1.0
  - We are a pre‑1.0 prototype. Prefer breaking changes that simplify the code and remove dead paths; deprecation layers are not required.
  - When breaking behaviour, update tests and docs in the same PR. Keep the repo “green for prototype”.

- No “helpful masking” of failures
  - Do not silently create defaults or fallbacks that hide serious configuration/infrastructure errors (e.g., auto‑constructing Urza, falling back to stub datastores). Such masking leads to fragile systems.
  - Emit clear, actionable errors and, where appropriate, telemetry explaining the failure. Fail early rather than operating in an undefined state.

- No partial degradation
  - Assume all subsystems are available and healthy; otherwise treat the system as fully degraded. Do not add “partial availability” branches inside core paths.
  - Weatherlight emits a `system_mode` indicator (`operational` or `degraded`) derived from worker health/backoff to help operators. Startup preflight rejects missing deps outright.

- Code hygiene and dead code
  - Remove unused toggles, stale fallbacks, and “just in case” branches when finalising a feature. Keep code paths minimal and explicit.
  - Prefer one authoritative implementation over multiple conditional paths.

- Testing & CI posture
  - Unit tests: cover success paths and guard‑rail failures; don’t depend on masked defaults. Provide test doubles/fakes explicitly (e.g., FakeRedis, minimal UrzaLibrary roots).
  - Integration tests: exercise cross‑subsystem flows with real contracts (Leyline) and the strict preflight on.

- Documentation duties
  - When enforcing strictness (e.g., making ES mandatory, requiring Urza), update the relevant `docs/prototype-delta/*/README.md` and the operator runbook.
  - Record cross‑system policy and execution status under `docs/prototype-delta/cross-system/` (e.g., STRICT_DEPENDENCIES_PLAN.md).

These policies are intended to keep the prototype tight, remove ambiguity, and avoid hidden states that slow iteration. When in doubt: make the dependency explicit, fail fast, and document the requirement.
