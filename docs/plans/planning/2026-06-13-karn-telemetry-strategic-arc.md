# Karn Telemetry Quality Strategic Arc

```yaml
# Plan Metadata
id: karn-telemetry-quality-arc
title: Karn Telemetry Quality Strategic Arc
type: planning
created: 2026-06-13
updated: 2026-06-13
owner: Codex

urgency: high
value: Make Karn a trustworthy operator and CI surface before deeper Simic/Tamiyo/Tolaria work resumes.

complexity: L
risk: medium
risk_notes: Karn touches UI, telemetry contracts, MCP views, web build tooling, and CI. A broad rewrite could destabilize the recovery baseline, so work must stay sliced and merge through small validated PRs.

depends_on:
  - dependency-drain
soft_depends:
  - telemetry-domain-sep
blocks:
  - reward-efficiency
  - phase3-tinystories

status_notes: Strategic arc selected after green-state recovery and dependency PR triage exposed recurring Karn/Sanctum/Overwatch quality gaps. Sprint 1 is drafted in `docs/plans/planning/2026-06-13-karn-telemetry-sprint-1.md`.
percent_complete: 0

reviewed_by: []
```

## Strategic Intent

Karn should become the next quality-upgrade package for Esper. It is the organism's operator-visible nervous system: Sanctum, Overwatch, MCP analytics, and the telemetry contracts that tell us whether morphogenesis is behaving. The recovery effort keeps finding Karn-adjacent failures: a CI-only Sanctum state bug, Overwatch TypeScript contract drift, stale dependency PRs around web tooling, and telemetry surfaces that are hard to use as proof.

The intent is not to add new dashboard features. The intent is to make Karn reliable enough that later work on Simic rewards, Tamiyo policy behavior, Tolaria execution, and Phase 3 validation can be judged from trustworthy signals instead of ad hoc local inspection.

## Why Karn Before Simic

Simic and Tamiyo remain the strategic core, but they should not be the next refactor package. The current bottleneck is evidence quality. The architecture constitution says sensors must match capabilities; if the policy cannot see a signal, it cannot optimize it. The same applies to operators and CI: if Karn cannot build, test, and render the signal contracts consistently, deeper training changes become harder to validate and easier to misread.

Karn is also better shaped for a first post-recovery package:

- It is high value because it improves every later validation loop.
- It is bounded: `src/esper/karn/`, selected `src/esper/leyline/` contracts, `tests/karn/`, and the Overwatch web workspace.
- It has concrete current failures, so success can be falsified.
- It does not require changing RL algorithm behavior during the stabilization window.

## Now / Next / Later

### Now: Sprint 1 - Stabilize the Evidence Surface

Goal: land the dependency-drain PR or successor, remove stale dependency PR noise, reproduce and fix the Sanctum CI-only state failure, and make Overwatch build failure explicit with a repair path.

Primary outcomes:

- `main` is green again.
- Open dependency PRs are either merged through the consolidation PR or closed as superseded/deferred.
- The Sanctum policy-group toggle state transition is deterministic under the CI unit command.
- Overwatch contract drift is inventoried with exact missing Python/TypeScript sources.

### Next: Sprint 2 - Contract Repair and Web Build

Goal: repair generated contract drift and make Overwatch build a dependable validation target.

Primary outcomes:

- `npm run build` passes in `src/esper/karn/overwatch/web`.
- Missing types are either generated from current Python contracts or removed from consumers by updating the actual data model.
- Web package lock changes are folded into a passing build PR.
- Contract generation has a documented local command and expected outputs.

### Next: Sprint 3 - Telemetry Contract Consolidation

Goal: reduce duplicate shapes and move genuinely shared telemetry contracts into Leyline while keeping UI-only view models inside Karn.

Primary outcomes:

- Shared event/snapshot contracts have one source of truth.
- Karn UI view models are explicit projections, not silent fallbacks.
- No backwards-compatibility shims or defensive bug-hiding access patterns are introduced.
- Karn tests cover both contract ingestion and UI projection behavior.

### Later: Operator Experience Hardening

Goal: improve Karn's operator utility once the evidence surface is stable.

Candidate outcomes:

- Sanctum help and navigation polish.
- Better run comparison and A/B display.
- MCP query affordances for current training investigations.
- Additional telemetry panels only when the source signal already exists and is trustworthy.

## Scope Boundaries

In scope:

- `src/esper/karn/sanctum/`
- `src/esper/karn/overwatch/`
- `src/esper/karn/mcp/`
- `tests/karn/`
- `src/esper/leyline/` only for shared contracts that Karn consumes across subsystem boundaries
- `src/esper/karn/overwatch/web/` build and generated contract files

Out of scope:

- RL algorithm redesign in `src/esper/simic/`
- Tamiyo policy architecture changes
- Tolaria execution-loop refactors
- New dashboard features without fixing the underlying contract/build reliability
- Major dependency jumps such as `transformers 5.0.0rc3` until separately planned

## Operating Rules

- Keep PRs small and mergeable.
- Protect the current green baseline; do not batch speculative refactors into recovery fixes.
- Treat telemetry as an API: typed payloads, explicit projections, no silent fallback.
- Move shared contracts to Leyline only when more than one domain truly owns the shape.
- Use Loomweave before broad contract edits to map callers and references.
- Run Wardline when touching SQL, HTTP, file input, telemetry ingestion, or other trust boundaries.
- Use CI-parity commands under Python 3.11 for final Karn validation.

## Success Measures

- PR #91 or its successor merges and `main` is green.
- `uv run --python 3.11 pytest -m "not integration and not stress and not property" -x --cov=src --cov-report=json` passes locally or its remaining failure is documented as outside Karn.
- `uv run --python 3.11 pytest tests/karn -q` passes.
- `npm ci --legacy-peer-deps` and `npm run build` pass in `src/esper/karn/overwatch/web`.
- Open dependency PR count is reduced to only intentionally deferred work.
- Karn contract drift has no untracked missing types between Python and TypeScript.
