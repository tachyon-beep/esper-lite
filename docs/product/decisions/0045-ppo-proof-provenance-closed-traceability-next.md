# PDR-0045 — PPO proof provenance closed; traceability next

Date: 2026-07-08   Status: accepted
Author: Codex (GPT-5)   Owner sign-off: n/a (tracker/product checkpoint; no push/freeze/A/B)
Related: PDR-0044, work package esper-lite-a2abff5ec5, task esper-lite-441fbc6810,
task esper-lite-570d98c451.
Code commit: `7ebaff72ac909d067654af014003564d0d13069c`.

## Context

PDR-0044 left the autonomous fallback path for PPO Learning Gate / Reward-Efficiency
Statistics at esper-lite-441fbc6810 if the owner did not freeze and launch the Stage-2
paired HRA ON/OFF A/B. That task is now implemented, verified, committed, and closed. The
proof packet fails closed on missing PPO updates, missing required PPO fields, and missing
precision provenance before reward-efficiency verdict math can run.

Filigree reports esper-lite-570d98c451 ("Add Karn PPO traceability evidence section") as
newly unblocked and as the first item on the live critical path toward reward-efficiency
ROI verdicts.

## Options considered

1. **Treat PPO Learning Gate as ready for rehearsal now.** Rejected: the packet can now
   block missing evidence, but the next traceability section still has to expose the
   proof-critical counts and validation summary before ROI semantics should be read.
2. **Switch back to Stage-2 launch unconditionally.** Rejected: Stage-2 remains the Now
   bet, but launch still requires owner freeze of the remaining threshold slots and owner
   approval for the paired A/B.
3. **Advance branch survivor cleanup next.** Deferred: it is unblocked, but merge,
   jettison, push, and branch deletion remain owner-gated and do not validate the active
   EV-stabilization value hypothesis.
4. **Keep EV-stabilization as Now; make esper-lite-570d98c451 the next autonomous
   reward-efficiency task if the owner does not freeze/launch Stage-2.** Chosen.

## The call

Task esper-lite-441fbc6810 is closed at `7ebaff72`. The next autonomous product move is
esper-lite-570d98c451: add the Karn PPO traceability evidence section so proof packets
summarize update counts, missing fields, skipped/nonfinite evidence, low-return-variance
counts, and finite/range validation before reward-efficiency ROI semantics run.

Stage-2 still has priority if the owner is ready to freeze the gate and launch the paired
fresh-init HRA ON/OFF A/B. That remains an owner-gated experiment launch, not an automatic
next action.

## Verification recorded

- Targeted red/green proof/vectorized/train tests: 14 passed after the fix.
- Affected proof/Karn/vectorized/train/oracle suites: 151 passed.
- Full default `uv run pytest -q`: 5264 passed, 5 skipped, 448 deselected, 33 warnings.
- Touched-file `ruff`, `git diff --check`, and added-line defensive/GPU-sync scan passed.
- `wardline scan . --fail-on ERROR` passed with 0 active findings; caveat: Wardline
  reported the taint gate inert because no trust boundaries are declared.
- `scripts/lint_defensive_patterns.py` and `scripts/lint_gpu_sync.py` still report
  existing non-touched violations outside this task.

## Rationale

This preserves the product distinction between "the blocker was fixed" and "the value bet
is ready to read." The proof packet is stricter now, but the product still needs an
inspectable evidence section before a bounded reward-efficiency rehearsal can make an ROI
verdict. Keeping Stage-2 owner gates explicit prevents an implementation close from being
mistaken for permission to launch or claim value.

## Reversal trigger

- If the owner freezes and launches Stage-2 before the reward-efficiency path continues,
  write a new PDR with the frozen threshold values and launch spec.
- If esper-lite-570d98c451 reveals missing telemetry that belongs to the just-closed proof
  provenance boundary, reopen or file a blocking task before continuing to ROI semantics.
- If reward-efficiency traceability lands cleanly, proceed to the next critical-path item,
  esper-lite-9678c6d05a, to reconcile reward-efficiency ROI verdict semantics.
