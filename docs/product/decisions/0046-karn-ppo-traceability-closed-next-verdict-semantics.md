# PDR-0046 — Karn PPO traceability closed; verdict semantics and proof lane unblocked

Date: 2026-07-08   Status: accepted
Author: Codex (GPT-5)   Owner sign-off: n/a (tracker/product checkpoint; no push/freeze/A/B)
Related: PDR-0045, work package esper-lite-a2abff5ec5, task esper-lite-570d98c451,
task esper-lite-9678c6d05a, task esper-lite-c3cb5338c4.
Code commit: `0838cc40`.

## Context

PDR-0045 made esper-lite-570d98c451 the next autonomous reward-efficiency task after the
PPO proof-provenance close. That task is now implemented, verified, committed, and closed.
Karn exposes PPO update counts, skipped status, NaN/Inf counts, robust value fields,
low-return-variance counts, missing/nonfinite counts, and finite/range validation through a
public `ppo_traceability_evidence` view. The proof packet renders that evidence before ROI
verdict math and blocks missing, nonfinite, skipped, and impossible finite evidence.

Closing esper-lite-570d98c451 unblocked three tasks:

- esper-lite-9678c6d05a — Reconcile reward-efficiency ROI verdict semantics (P1).
- esper-lite-c3cb5338c4 — Add CI-safe PPO learnability proof lane (P1).
- esper-lite-a36f3311a3 — Surface oracle proof trace in Sanctum TUI (P2).

## The call

For serial product work, take esper-lite-9678c6d05a next. The packet now has proof-grade
PPO traceability, but reward-efficiency `CONTINUE` semantics still need to match the
roadmap: baseline-subtracted same-seed ROI over Shaped plus final-accuracy evidence over
Control, not best-available `final_accuracy / param_ratio` rows.

If parallel agents are available, run esper-lite-c3cb5338c4 alongside it. The proof-lane
task is now startable and Filigree's critical-path command currently selects that branch
(`c3cb5338c4 -> 9f30f62993 -> 2b077204e9`), but it does not replace the need to fix ROI
semantics before reading reward-efficiency verdicts.

Stage-2 still has priority if the owner is ready to freeze the gate and launch the paired
fresh-init HRA ON/OFF A/B. That remains an owner-gated experiment launch, not an automatic
next action.

## Verification recorded

- Targeted RED tests failed for missing Karn view/catalog/columns, missing packet evidence
  section/blockers, nonfinite aggregate undercount, missing robust-value rendering, and
  impossible finite robust-value ranges.
- Focused Karn/MCP/proof suites: 131 passed.
- Full default `uv run pytest -q`: 5291 passed, 5 skipped, 448 deselected, 33 warnings.
- Touched-file `ruff`, `git diff --check`, and added-line defensive/GPU-sync scans passed.
- `wardline scan . --fail-on ERROR` passed with 0 active findings; caveat: Wardline
  reported the taint gate inert because no trust boundaries are declared.
- `scripts/lint_defensive_patterns.py` and `scripts/lint_gpu_sync.py` still report
  existing non-touched violations outside this task.

## Reversal trigger

- If reward-efficiency ROI semantics cannot be reconciled without new telemetry, file or
  claim the telemetry blocker before changing verdict math.
- If the CI-safe proof lane reveals that traceability evidence is not emitted by a real
  short PPO fixture, reopen or file a blocker against the traceability surface before the
  rehearsal task.
- If the owner freezes and launches Stage-2 first, write the frozen threshold values and
  launch spec before any ON run.
