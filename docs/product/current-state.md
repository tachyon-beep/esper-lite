# Current State — Esper        Checkpoint: 2026-07-07 23:39 AEST · code commit `3669934d` · checkpoint #30 (PDR-0044; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet) has its **Stage-2 MAJOR-1
acceptance harness implementation closed** (esper-lite-2a4b56e719, `closed`) in
`3669934d`. The harness is **7/7 slices built, review-remediated, and tracker-closed**,
including OFF calibration validity, run-log traceback validity, advantage-floor reporting,
and the stale gate doc fix. It is **not an experimental read, not frozen, not pushed, and
not A/B-ready**.

The metric this leg ultimately moves remains host-accuracy contribution via the
Stage-2 MAJOR-1 gate; no paired fresh-init HRA ON/OFF result has been read yet.

## In flight
- **Stage-2 acceptance harness (esper-lite-2a4b56e719, closed):** pure scorer (PDR-0037),
  wrapper pure layer (PDR-0039), S5 reader + review fixes (PDR-0040), S6 packet/env-count
  assembler (PDR-0041), S7 emission/provenance/readers/G4/CLI (PDR-0042), and the
  follow-on review-remediation commits `7c6ec9da`, `0a2f6b9c`, `f0ee27fe` (PDR-0043)
  plus final-review closeout `3669934d` (PDR-0044).
  Latest hardening covers telemetry-derived G2 `host_params`, terminal per-env evidence
  coverage, duplicate `runs` row rejection, fail-loud required-field completeness, S7
  diagnostic scalar plumbing, G3 fossilize churn, clearer return-variance diagnostics,
  final-review OFF calibration validity, spec-scoped traceback/run-log invalidation, and
  `advantage_std_floored` contamination reporting.
- **Stage-2 experiment gate (owner-gated):** freeze thresholds, then owner-approved paired
  fresh-init HRA ON/OFF A/B and packet read. This is the remaining value-validation step.
- **Branch survivor pick (esper-lite-1f1e55f58f, open/unblocked):** unchanged
  (PDR-0038). Do not merge/delete/push or jettison a branch without explicit owner approval.
- **Reward-efficiency statistics (esper-lite-a2abff5ec5, Next bet):** the live critical path
  starts at esper-lite-441fbc6810 if the owner does not want to launch Stage-2 next.

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039.** Caller-supplied pairing, telemetry-verified
  ON/OFF signature, type-enforced freeze order, and training-touch-last ordering stand.
- **S5/S6 reader and assembler seams settled — PDR-0040/PDR-0041.** G1/G2 are
  per-env-terminal; env coverage is assembly-layer validity; `build_report` never
  calibrates; `BuiltReport.validity` must be threaded into `render_packet`.
- **S7 built, then review-remediated and closed — PDR-0042/PDR-0043/PDR-0044.** The harness
  is no longer blocked on the known S7 review findings or the stale-gate/final-review validity
  blockers, but code-complete is not value-landed. No A/B before owner freeze.
- **Verification baseline remains mixed:** focused Stage-2 + adjacent regression suites pass;
  touched-file ruff, leyline lint, diff check, and wardline pass. Full mypy, full ruff,
  defensive-pattern lint, and GPU-sync lint still report existing non-Stage-2 failures.
- **Branch:** stay on `feat/ev-stab-stage2-hra`. Do not push without an explicit owner ask.

## Open questions / blocked-on-owner
- **Gate freeze:** owner must freeze the remaining Stage-2 threshold slots before any ON/OFF
  paired A/B. This is a product/owner gate, not an automatic consequence of code passing.
- **Launch:** paired fresh-init HRA ON/OFF A/B remains owner-launched only.
- **Threshold placeholders:** `Δparam_max`, burn-in `W`, G3/G4 materiality thresholds,
  floored-EV asymmetry policy, north-star target/date, rent ceiling, and host-accuracy floor
  still need owner-set falsifiable values where the gate requires them.
- **Standing owner gates:** no push, tag, release, branch deletion, telemetry deletion, or remote
  action without explicit approval.

## Last checkpoint did (checkpoint #30)
- Committed the final-review Stage-2 fixset as `3669934d` and closed Filigree task
  esper-lite-2a4b56e719 with that commit anchor.
- Recorded PDR-0044: implementation is closed, but the Stage-2 value read remains
  owner-gated on freeze + paired HRA ON/OFF A/B.
- Reoriented next product work: Stage-2 launch is first if the owner freezes it; otherwise
  the autonomous critical path starts at esper-lite-441fbc6810.

## Previous checkpoint did (checkpoint #29)
- Brought the stale Stage-2 gate doc current with the review-remediated wrapper/packet reality
  while preserving "not frozen / no A/B" owner gates.
- Added final-review fixes for OFF calibration validity, spec-scoped traceback/run-log validity,
  and `advantage_std_floored` packet reporting.

## Earlier checkpoint did (checkpoint #28)
- **PDR-0043** recorded that the S7 review gate moved from "findings outstanding" to
  "known findings remediated and committed." The latest commit is `f0ee27fe`.
- Updated `metrics.md` and `roadmap.md` so the product workspace no longer says only S7 is pending.
- No roadmap horizon changed: EV-stabilization remains Now; reward-efficiency statistics remain Next.
- No new experimental metric reading landed; Stage-2 MAJOR-1 is still pending a frozen paired run.

## Next session, start here
Ask the owner whether to spend the next session freezing and launching the Stage-2 paired
fresh-init HRA ON/OFF A/B. If yes, freeze `Δparam_max`, exact `W`, floored-EV asymmetry,
G3/G4 materiality, and advantage-floor policy before any ON run. If no, start the P1
reward-efficiency critical path at esper-lite-441fbc6810. Branch survivor cleanup is
unblocked but remains owner-gated for any merge/jettison/push action.
