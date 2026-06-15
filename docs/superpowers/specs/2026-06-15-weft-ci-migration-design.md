# Design: Retire Homegrown CI → Weft Toolkit

**Date:** 2026-06-15
**Status:** Approved (design); not yet planned/implemented
**Author:** shadow-review session (with John)

## Goal

Retire esper-lite's homegrown CI *checks* and replace them with the Weft
federation (filigree, loomweave, wardline, warpline, legis) — which was written by
this team based on this work. Strategy: **start at Phase A (like-for-like linter
swap), advance to Phase C (full governance + change-impact re-architecture) only
if A proves the toolkit can carry the weight.** Evidence-gated, not faith-gated.

## What this is NOT

Weft replaces the *category of custom static-analysis/governance checks we
hand-rolled* — not the test/lint engines. `ruff`, `mypy`, `pytest` (all tiers),
the `npm`/Playwright web tests, and the 75% coverage gate **stay**. Weft can later
*orchestrate and select* them (Phase C), but it does not reimplement them.

## Current homegrown CI (baseline)

`.github/workflows/test-suite.yml` jobs: `lint` (3 custom AST linters + ruff),
`typecheck` (mypy), `property-tests`, `unit-and-integration-tests` (+coverage
gate), `overwatch-web-tests`, `nightly-full-suite`. The custom linters live in
`scripts/` with companion `.yaml` whitelists.

## Scope — per-check mapping

| Homegrown check | Weft replacement | Confidence | Retire in A? |
|---|---|---|---|
| `lint_defensive_patterns.py` + `defensive_patterns.yaml` | **wardline scan** (yaml → baseline + waivers) | High (wardline's core purpose) | ✅ |
| `test_import_cycle.py` | **loomweave** `module_circular_import_list` | High (direct equivalent) | ✅ |
| `lint_leyline_types.py` + `leyline_boundaries.yaml` | **loomweave** import-edge/boundary check (→ **legis** `@policy_boundary` in C) | Medium (rule must be expressed) | ✅ as loomweave check |
| `lint_gpu_sync.py` + `gpu_sync_whitelist.yaml` | *(no native Weft equivalent — bespoke CUDA-sync rule)* | Low | ❌ **keep homegrown** |
| `ruff`, `mypy`, `pytest`, web tests, coverage | engines — not Weft territory | — | ❌ stay |

**Decision (approved):** A retires **3 of 4** custom linters. `gpu_sync` stays
homegrown — it detects PyTorch CUDA-sync points (`.item()`/`.cpu()`) against a
whitelist, which is genuinely domain-specific. Declaring 100% retirement would be
an over-claim.

## Validation strategy (approved): local proof, then shadow burn-in

Sequenced, both methods:

1. **Local parity harness (off-CI, same-day).** For each retirable check, run
   *both* the homegrown linter and its Weft replacement against (a) the current
   tree and (b) a sample of recent commits. Normalize both outputs to
   `(file, line, rule)` findings and emit a three-way diff: **matches /
   Weft-only / homegrown-only**.
   - **"Carries the weight" criterion:** *zero homegrown-only findings* (Weft
     misses nothing the homegrown linter caught); Weft-only findings are triaged
     as real-defect or waiver.
2. **Shadow-parallel CI.** Ship the survivors as **non-blocking** CI jobs
   alongside the existing homegrown linters. The diff runs on every PR for a
   burn-in window. Zero coverage gap during transition.
3. **Retirement gate (A → retire, per-check, independent).** Delete a homegrown
   linter + its `.yaml` only after **≥20 PRs or 2 weeks (whichever first) with
   zero homegrown-only findings.** Each check retires independently — defensive
   patterns may retire before leyline boundaries.

## CI architecture — evolves by phase

- **Phase A:** invoke Weft **CLIs directly** as added jobs in `test-suite.yml`
  (`wardline scan --fail-on ERROR`, `loomweave analyze` + query for cycles/
  boundaries). Minimal; fastest path to parity evidence. Shadow jobs are
  non-blocking until their retirement gate passes, then flipped to blocking.
- **Phase C:** evolve to **legis as the CI/governance orchestrator** (its stated
  role). legis drives wardline/loomweave, emits a single governance verdict, and
  records SEI-keyed audit that survives rename/move.

## Phase A deliverables

1. Parity-harness script (per-check: run both, normalize, diff, report).
2. Parity report artifact (the same-day "can it carry the weight" read).
3. Shadow-parallel CI jobs added to `test-suite.yml` (non-blocking).
4. Documented retirement gate + per-check status tracker.

## Phase C preview (destination, gated on A)

- **legis** governance gates: override-rate, `@policy_boundary` evidence.
- Analyzer findings → **filigree** issues with closure gates
  (`legis filigree_closure_gate`).
- **warpline** change-impact → run only impacted pytest tiers (the real CI-time
  payoff: stop running the whole suite on every change).

## Risks / known blockers

- **warpline MCP not wired up** (skill installed, no server/tools). Required for
  Phase C selective tests; not needed for A. Must be connected before C.
- **wardline MCP server stale** (running pre-upgrade code; needs restart). Does
  *not* affect A's CLI invocation in CI (fresh each run), but fix locally.
- **legis unconfigured** for policy-cells + wardline-routing. Only matters at C.
- **leyline-boundary rule expression** is the medium-confidence mapping; the
  local parity proof is where we confirm loomweave's import-edge view actually
  reproduces `lint_leyline_types.py`'s rule.
- **Federation layout:** the suite now uses a unified `.weft/` state dir +
  `weft.toml` — confirm CI provisions/initializes this in the runner.

## Out of scope / flagged upstream

- **gpu_sync as a Weft rule:** flagged to the Weft project as a *candidate*
  feature (a domain-specific CUDA-sync rule). Weft may decline it as out of
  scope. Either way, esper-lite keeps `lint_gpu_sync.py` until/unless such a rule
  exists and passes the same parity gate. Not a blocker.

## Open decisions (resolve during planning)

- Sample size / commit range for the local parity proof's historical pass.
- Whether the leyline-boundary check lands as a pure loomweave query in A or goes
  straight to a legis `@policy_boundary` policy.
- Exact normalization format for cross-tool finding comparison.
