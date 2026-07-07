# PDR-0043 — Stage-2 MAJOR-1 acceptance: review remediation checkpoint

Date: 2026-07-07   Status: accepted (state checkpoint; delivery remains in progress)
Author: Codex (GPT-5)   Owner sign-off: n/a (autonomous checkpoint within grant; no push/freeze/A/B)
Related: PDR-0039, PDR-0040, PDR-0041, PDR-0042, task esper-lite-2a4b56e719,
epic esper-lite-f25b71c165.
Code commits: `7c6ec9da`, `0a2f6b9c`, `f0ee27fe`.

## Context

PDR-0042 recorded Stage-2 S7 as built but review-gated. Subsequent review rounds found
acceptance-packet blockers in the evidence contracts: stale/spec-controlled G2 host-parameter
evidence, incomplete or stale terminal env coverage, duplicate `TRAINING_STARTED` / `runs` rows,
missing required telemetry fields, stale S7 diagnostic availability, leyline boundary violations,
G3 fossilize churn omission, and misleading return-variance diagnostics.

Those known findings are now remediated in code and tests through `f0ee27fe`. This checkpoint
records the state transition without claiming that the MAJOR-1 experiment has been run or that
the gate has been frozen.

## Options considered

1. **Leave PDR-0042 as the latest state.** Rejected: the workspace would keep telling the next
   session that S7 review findings are still outstanding even though the known findings have been
   fixed and committed.
2. **Mark Stage-2 accepted/frozen.** Rejected: the paired fresh-init A/B has not run, and the
   owner has not frozen the gate document after the review-remediation commits.
3. **Checkpoint review remediation only.** Chosen: record that the known review findings are fixed
   while preserving the product gates for final review, freeze, and A/B launch.

## The call

Treat the Stage-2 acceptance harness as **review-remediated but still in progress**. The next
decision is freeze readiness, not value acceptance.

Durable changes now in scope:

- G2 `host_params` evidence is checked against telemetry frozen config before scoring.
- Env completeness validates terminal per-env evidence, not just distinct env presence.
- Duplicate `runs` / `TRAINING_STARTED` rows fail loud instead of mixing provenance.
- Missing required terminal and churn fields fail loud instead of being skipped by SQL `AVG`.
- S7 diagnostic scalars are selected, stored, rendered, and covered by tests.
- G3 includes fossilize churn in the materially-elevated safety gate.
- Leyline type-boundary additions are routed through the leyline boundary.
- Return-variance confound diagnostics name the return-variance denominator.

## Verification recorded

- `uv run pytest tests/simic/telemetry/test_stage2_acceptance_io.py tests/simic/telemetry/test_stage2_acceptance_packet.py tests/scripts/test_stage2_packet.py -q`
  — 129 passed.
- Broader focused set including Karn views, Simic reward/agent tests, and leyline round-trip
  — 185 passed.
- Touched-file `ruff`, `scripts/lint_leyline_types.py`, and `git diff --check` passed.
- `wardline scan . --fail-on ERROR` passed, with the existing warning that taint boundaries are inert.
- Full `mypy`, full `ruff`, defensive-pattern lint, and GPU-sync lint still fail on existing
  non-Stage-2 files and remain branch baseline risks rather than acceptance-packet regressions.

## Rationale

This keeps product state aligned with git while avoiding the build trap. The code has been
hardened against the review findings, but the product value has not landed until the gate is
frozen and the paired ON/OFF run produces a verdict.

## Reversal trigger

- If a final net-diff review finds another acceptance-packet blocker, keep Stage-2 in progress and
  write a follow-up PDR after remediation.
- If the owner declines to freeze the gate with the current thresholds or evidence definitions,
  update the gate and product state before any A/B.
- If any baseline red full-repo gate is judged merge-blocking for this branch, promote it from
  baseline risk to active Stage-2 blocker before closeout.
