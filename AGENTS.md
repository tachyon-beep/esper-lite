# Repository Guidelines

## Start Here

- YOU MUST NOT JUST REVERT CHANGES YOU DID NOT MAKE YOURSELF. IF YOU SPOT A NEW OR CHANGED FILE, THE USER CHANGED THAT FILE FOR A REASON. ASK FIRST. ALL DESTRUCTIVE GIT OPERATIONS REQUIRE APPROVAL.
- STOP AND READ `CLAUDE.md` FIRST. Do not proceed without it. It contains the project intro, required reading (e.g., `README.md`, `ROADMAP.md`), and mandatory coding standards/policies.
- `CLAUDE.md` is the source of truth for non-negotiables (e.g., no legacy/backwards-compat code, no defensive programming, and required metaphors). If anything conflicts, `CLAUDE.md` wins.

## PROHIBITION ON "DEFENSIVE PROGRAMMING" PATTERNS

No Bug-Hiding Patterns: This codebase prohibits defensive patterns that mask bugs instead of fixing them. Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception handling to suppress errors from nonexistent attributes, malformed data, or incorrect types. A common anti-pattern is when an LLM hallucinates a variable or field name, the code fails, and the "fix" is wrapping it in getattr(obj, "hallucinated_field", None) to silence the error—this hides the real bug. When code fails, fix the actual cause: correct the field name, migrate the data source to emit proper types, or fix the broken integration. Typed dataclasses with discriminator fields serve as contracts; access fields directly (obj.field) not defensively (obj.get("field")). If code would fail without a defensive pattern, that failure is a bug to fix, not a symptom to suppress.

### Legitimate Uses

This prohibition does not exclude genuine uses of type checking or error handling where appropriate, such as:

- **PyTorch tensor operations** (9): Converting tensors to scalars, device moves
- **Device type normalization** (6): `str` → `torch.device` conversion
- **Typed payload discrimination** (40): Distinguishing between typed dataclass payloads (correct migration pattern)
- **Enum/bool/int validation** (10): Rejecting bool as int in coercion
- **Numeric field type guards** (20): Conditional rendering of optional numeric fields
- **NN module initialization** (2): Layer type detection for weight init
- **Serialization polymorphism** (3): Enum, datetime, Path handling

For absence of doubt, when using these ask yourself "is this defensive programming to hide a bug that should not be possible in a well designed system, or is this legitimate type handling?' If the former, remove it and fix the underlying issue.

## Project Structure & Module Organization

- `src/esper/`: main Python package (core domains: `kasmina/`, `leyline/`, `tamiyo/`, `tolaria/`, `simic/`, `nissa/`, `karn/`, plus `runtime/` and `scripts/`).
- `tests/`: pytest suite grouped mostly by domain (`tests/<domain>/`), with cross-cutting suites in `tests/integration/`, `tests/stress/`, and `tests/**/properties/`.
- `src/esper/karn/overwatch/web/`: Vite/Vue frontend for the Overwatch dashboard, including Playwright/Vitest coverage for the web UI.
- `docs/`: architecture notes and implementation plans.
- `scripts/`: repository helper scripts and CI guardrails (for example the custom lint checks used in GitHub Actions).
- `data/` and `telemetry/`: local datasets and run artifacts (gitignored; only `.gitkeep` is tracked).

## Build, Test, and Development Commands

- Install deps (preferred): `uv sync --group dev`
- Add optional extras when needed: `uv sync --group dev --extra dashboard` for Overwatch, `uv sync --group dev --extra wandb` for Weights & Biases
- Run training CLI:
  - Heuristic: `PYTHONPATH=src uv run python -m esper.scripts.train heuristic --task cifar_baseline --episodes 1`
  - PPO: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --task cifar_baseline --rounds 100 --envs 4 --episode-length 150`
- Developer UIs: add `--sanctum` for the Textual debugger UI or `--overwatch` for the web dashboard (`--overwatch-port` is optional)
- Run tests: `uv run pytest` (defaults to `-m "not integration and not stress"` via `pytest.ini`)
- Run property tests: `HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest -m property -v -x --hypothesis-show-statistics`
- Run integration tests: `PYTHONPATH=src uv run pytest -m integration -v -x`
- CI parity checks:
  - Policy guardrails: `uv run python scripts/lint_leyline_types.py`, `uv run python scripts/lint_defensive_patterns.py`, `uv run python scripts/lint_gpu_sync.py`
  - Lint: `uv run ruff check src/ tests/`
  - Types: `MYPYPATH=src uv run mypy -p esper`

## Coding Style & Naming Conventions

- Python 3.11+; 4-space indentation; prefer type hints and `from __future__ import annotations` on new modules.
- Keep names conventional: `snake_case.py`, `snake_case()` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Shared contracts (enums/constants/schemas) belong in `src/esper/leyline/` to avoid cross-domain coupling.
- Follow the repo guardrails in `CLAUDE.md` (notably: no compatibility shims/legacy code; `hasattr()` requires explicit, documented authorization).

## Testing Guidelines

- Tests follow `test_*.py` naming; module suites live under `tests/<domain>/`, with stress tests in `tests/stress/` and property suites in `tests/**/properties/`.
- Pytest markers come from `pytest.ini` (`integration`, `slow`, `stress`, `property`, `e2e`, `benchmark`, `tamiyo`, `simic`, `no_torch_seeding`).
- Hypothesis uses profiles via `HYPOTHESIS_PROFILE` (`dev` default; `ci` and `thorough` are used in CI/nightly flows).
- Coverage: the CI unit-test job enforces `75%` on the reduced suite (`not integration and not stress and not property`); full local coverage with all test types is roughly `81%` (see `.github/workflows/test-suite.yml` and `pytest.ini`).

## Commit & Pull Request Guidelines

- Use Conventional Commits seen in history: `feat(scope): …`, `fix: …`, `docs: …`, `refactor: …`, `chore: …`.
- PRs should include: intent + summary, commands run (tests/lint), linked issues if any, and screenshots for TUI/dashboard changes.
- Do not commit generated artifacts from `data/` or `telemetry/`.

<!-- filigree:instructions:v2.2.0:9dff6e6d -->
## Filigree Issue Tracker

`filigree` tracks tasks for this project. Data lives in `.filigree/`. Prefer
the MCP tools (`mcp__filigree__*`) when available; fall back to the `filigree`
CLI otherwise.

### Workflow

```bash
# At session start
filigree session-context                            # ready / in-progress / critical path

# Pick up the next startable issue (atomic claim + transition into its working status)
filigree start-next-work --assignee <name>
# ...or claim a specific issue
filigree start-work <id> --assignee <name>

# Do the work, commit, then
filigree close <id>
```

Use the atomic claim+transition verbs — `start_work` / `start_next_work`
(MCP) or `start-work` / `start-next-work` (CLI). Do **not** chain
`claim_issue` (MCP) or `filigree claim` (CLI) with a subsequent status
update — the two-step form races against other agents; the combined verb is
atomic.

**Ready ≠ startable.** The working status is type-specific (tasks →
`in_progress`, features → `building`). Bugs start at `triage`, which has no
single-hop transition into work (`triage → confirmed → fixing`), so a triage
bug is *ready* but not directly *startable*: `start_work` on one returns
`INVALID_TRANSITION` naming the next status, and `start_next_work` skips it.
`get_ready` items carry a `startable` flag (plus a `next_action` hint when
false). Pass `advance=true` (MCP) / `--advance` (CLI) to walk the soft
transitions to the nearest working status automatically.

### Observations: when (and when not) to use them

`observe` is a fire-and-forget scratchpad for *incidental* defects — things
you notice *outside the scope of your current task* (a code smell in a
neighbouring file, a stale TODO, a missing test for an edge case you happened
to spot). Notes expire after 14 days unless promoted. Include `file_path` and
`line` when relevant. At session end, skim `list_observations` and either
`dismiss_observation` or `promote_observation` for what has accumulated.

**You fix bugs in your currently defined scope. You do NOT use observations
to finish work prematurely.** If a defect, gap, or follow-up belongs to your
current task, you own it — handle it as part of that task: fix it now, expand
the task's scope, file a proper issue with a dependency, or surface it to the
user. Filing it as an observation and closing the task is *not* completing
the task; it is shipping known-broken work and hiding the debt in a 14-day
expiring scratchpad. The test is "would I have noticed this even if I weren't
working on this task?" If no, it's task scope, not an observation.

### Priority scale

- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog

### Reaching for tools

MCP tool schemas describe each tool; `filigree --help` and `filigree <verb>
--help` are the authoritative CLI reference. You do not need to memorise
either catalogue. The verbs you will reach for most:

- **Find work:** `get_ready`, `get_blocked`, `list_issues`, `search_issues`
- **Claim work:** `start_work`, `start_next_work`
- **Update:** `add_comment`, `add_label`, `update_issue`, `close_issue`
- **Admin (irreversible):** `delete_issue` (MCP) / `delete-issue` (CLI) —
  hard-deletes a terminal issue and its rows; `undo_last` cannot reverse it.
- **Scratchpad:** `observe`, `list_observations`, `promote_observation`, `dismiss_observation`
- **Cross-product entity bindings (ADR-029):** `add_entity_association`,
  `remove_entity_association`, `list_entity_associations`,
  `list_associations_by_entity`. Used when a sibling tool (e.g.
  Clarion) needs to bind a Filigree issue to a function, class, or
  module identifier it owns. The `entity_id` is an opaque string
  from Filigree's perspective; the consumer (the sibling tool's read
  path) does drift detection against the stored
  `content_hash_at_attach`. `list_associations_by_entity` is the
  reverse-lookup surface — given a Clarion entity ID, return every
  Filigree issue bound to it (project isolation is by DB file). Also
  reachable over HTTP as
  `GET/POST /api/issue/{issue_id}/entity-associations`,
  `DELETE /api/issue/{issue_id}/entity-associations?entity_id=…`,
  and `GET /api/entity-associations?entity_id=…`.
- **Health:** `get_stats`, `get_metrics`, `get_mcp_status`

Pass `--actor <name>` (CLI) so events attribute to your agent identity. It
works in either position — before the verb (`filigree --actor X update …`) or
after it (`filigree update … --actor X`); the post-verb value overrides the
group-level one.

### Error handling

Errors return `{error: str, code: ErrorCode, details?: dict}`. Switch on
`code`, not on message text. Codes: `VALIDATION`, `NOT_FOUND`, `CONFLICT`,
`INVALID_TRANSITION`, `PERMISSION`, `NOT_INITIALIZED`, `IO`,
`INVALID_API_URL`, `FILE_REGISTRY_DISPLACED`, `REGISTRY_UNAVAILABLE`,
`CLARION_REGISTRY_VERSION_MISMATCH`, `BRIEFING_BLOCKED`, `STOP_FAILED`,
`SCHEMA_MISMATCH`, `INTERNAL`.

On `INVALID_TRANSITION`, call `get_valid_transitions` (MCP) or
`filigree transitions <id>` to see what the workflow allows from here.

Two failure modes deserve a specific response:

- **`SCHEMA_MISMATCH`** — the installed `filigree` is older than the project
  database. The error message contains upgrade guidance. Surface it to the
  user; do not retry.
- **`ForeignDatabaseError`** — filigree found a parent project's database
  but no local `.filigree.conf`. Run `filigree init` in the current
  directory. Do **not** `cd` upward to a different project unless that was
  the actual intent.
<!-- /filigree:instructions -->
