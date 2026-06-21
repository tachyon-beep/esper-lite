# Weft Phase A CI Migration

## Plan Metadata

```yaml
id: weft-phase-a-ci-migration
title: Weft Phase A CI Migration
type: in-progress
created: 2026-06-18
updated: 2026-06-18
owner: John
urgency: high
value: Adds Wardline, Loomweave, Legis, and Warpline evidence to CI without removing existing gates before parity is proven.
complexity: M
risk: medium
risk_notes: The Weft tools do not yet prove replacement parity for every homegrown check; retirement must be evidence-gated.
depends_on: []
soft_depends:
  - docs/superpowers/specs/2026-06-15-weft-ci-migration-design.md
blocks: []
status_notes: Phase A shadow CI scaffold is implemented on codex/weft-phase-a-ci-migration. No homegrown check is retired in the setup PR.
percent_complete: 55
reviewed_by:
  - reviewer: python-engineering
    date: 2026-06-18
    verdict: approved-with-changes
    notes: Keep shadow CI non-blocking and preserve custom gates until zero homegrown-only parity evidence exists.
```

## Scope

Phase A adds a non-blocking `weft-shadow` job and a local parity report. It does
not replace `ruff`, `mypy`, pytest tiers, coverage, web tests, proof-packet
checks, or the GPU-sync linter.

The shadow job is a narrow owner-approved exception to the no-dual-implementation
policy in `CLAUDE.md`. It is CI verification only, introduces no runtime dual
path, and expires per check after 20 PRs or 14 days with zero homegrown-only
findings.

## Implementation

- Keep existing blocking lint steps unchanged: Leyline type boundaries,
  defensive-pattern whitelist, GPU-sync whitelist, and Ruff.
- Add `weft-shadow` to `.github/workflows/test-suite.yml` with
  `continue-on-error: true` and artifact upload from `$RUNNER_TEMP/weft-shadow`.
- Pin the first shadow toolchain to Wardline 1.0.1, Loomweave 1.1.0,
  Loomweave Python plugin 1.1.0rc5, Legis 1.0.0, and Warpline 1.0.0.
- Write report artifacts only to `$RUNNER_TEMP`; ignored `.weft/` stores may be
  initialized in the ephemeral CI checkout for tools that require local state.
  Do not emit `findings.jsonl` into the repo tree.
- Build the parity report with `scripts/ci_weft_parity.py`.

## Replacement Readiness

- `lint_defensive_patterns.py`: stays blocking until Wardline emits a mapped
  defensive-pattern rule family AND a real homegrown-vs-Wardline comparison is
  implemented that shows zero homegrown-only findings. The pinned Wardline
  toolchain has no defensive-pattern rule and emits no line-anchored findings, so
  no comparison is performed today; the parity report marks this check
  `comparison: "deferred"`, and "zero homegrown-only findings" alone (the
  homegrown linter merely being quiet) does NOT constitute parity.
- `lint_leyline_types.py`: stays blocking until Loomweave exposes per-kind
  class-contract metadata for enum, dataclass, protocol, TypedDict, and NamedTuple
  placement, including stale-whitelist semantics. Loomweave 1.x exposes none of
  these, so the parity report marks this check `comparison: "deferred"` and
  leyline readiness is never signalled until a real Loomweave surface lands.
- `scripts/test_import_cycle.py`: is not a current CI gate and checks lazy import
  side effects, so it is not counted as retired by a Loomweave module-cycle
  artifact.
- `lint_gpu_sync.py`: remains homegrown. The current Weft toolchain has no
  equivalent CUDA-sync policy.

## Acceptance

- Existing blocking CI behavior remains intact.
- `weft-shadow` publishes Wardline, Loomweave, Legis, Warpline, and parity
  artifacts on every pull request.
- The parity report marks replacement readiness false throughout Phase A; it
  exposes `shadow_signal_ready` separately from the later retirement decision.
- Retirement happens only in a later per-check PR after the burn-in window.
