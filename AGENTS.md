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

<!-- filigree:instructions -->
## Filigree Issue Tracker

Use `filigree` for all task tracking in this project. Data lives in `.filigree/`.

### Quick Reference

```bash
# Finding work
filigree ready                              # Show issues ready to work (no blockers)
filigree list --status=open                 # All open issues
filigree list --status=in_progress          # Active work
filigree show <id>                          # Detailed issue view

# Creating & updating
filigree create "Title" --type=task --priority=2          # New issue
filigree update <id> --status=in_progress                # Claim work
filigree close <id>                                      # Mark complete
filigree close <id> --reason="explanation"               # Close with reason

# Dependencies
filigree add-dep <issue> <depends-on>       # Add dependency
filigree remove-dep <issue> <depends-on>    # Remove dependency
filigree blocked                            # Show blocked issues

# Comments & labels
filigree add-comment <id> "text"            # Add comment
filigree get-comments <id>                  # List comments
filigree add-label <id> <label>             # Add label
filigree remove-label <id> <label>          # Remove label

# Workflow templates
filigree types                              # List registered types with state flows
filigree type-info <type>                   # Full workflow definition for a type
filigree transitions <id>                   # Valid next states for an issue
filigree packs                              # List enabled workflow packs
filigree validate <id>                      # Validate issue against template
filigree guide <pack>                       # Display workflow guide for a pack

# Atomic claiming
filigree claim <id> --assignee <name>            # Claim issue (optimistic lock)
filigree claim-next --assignee <name>            # Claim highest-priority ready issue

# Batch operations
filigree batch-update <ids...> --priority=0      # Update multiple issues
filigree batch-close <ids...>                    # Close multiple with error reporting

# Planning
filigree create-plan --file plan.json            # Create milestone/phase/step hierarchy

# Event history
filigree changes --since 2026-01-01T00:00:00    # Events since timestamp
filigree events <id>                             # Event history for issue
filigree explain-state <type> <state>            # Explain a workflow state

# All commands support --json and --actor flags
filigree --actor bot-1 create "Title"            # Specify actor identity
filigree list --json                             # Machine-readable output

# Project health
filigree stats                              # Project statistics
filigree search "query"                     # Search issues
filigree doctor                             # Health check
```

### Workflow
1. `filigree ready` to find available work
2. `filigree show <id>` to review details
3. `filigree transitions <id>` to see valid state changes
4. `filigree update <id> --status=in_progress` to claim it
5. Do the work, commit code
6. `filigree close <id>` when done

### Priority Scale
- P0: Critical (drop everything)
- P1: High (do next)
- P2: Medium (default)
- P3: Low
- P4: Backlog
<!-- /filigree:instructions -->
