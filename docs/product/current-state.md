# Current State — Esper        Checkpoint: 2026-07-08 03:42 AEST · code commit `89b50a16` · checkpoint #33 (PDR-0047; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet) has its **Stage-2 MAJOR-1
acceptance harness implementation closed** (esper-lite-2a4b56e719, `closed`) in
`3669934d`, and its escrow telescoping correctness break fixed in `89b50a16`
(esper-lite-3defe42928). The harness is **7/7 slices built, review-remediated, and
tracker-closed**, and the reward path no longer clips escrow potential deltas behind a
non-Markov observation. It is **not an experimental read, not frozen, not pushed, and not
A/B-ready**.

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
- **Escrow telescoping fix (esper-lite-3defe42928, implemented):** PDR-0047 removed the
  reward-path `escrow_delta_clip`, exposed min-over-window stable validation accuracy and
  per-slot escrow credit in Obs V3 schema v2, bumped PPO checkpoint compatibility to v4,
  and updated the Obs V3 contract to 120 non-blueprint dims / 132 full network input dims.
  This is a correctness fix, not an experiment result.
- **Stage-2 experiment gate (owner-gated):** freeze thresholds, then owner-approved paired
  fresh-init HRA ON/OFF A/B and packet read. This is the remaining value-validation step.
- **Branch survivor pick (esper-lite-1f1e55f58f, open/unblocked):** unchanged
  (PDR-0038). Do not merge/delete/push or jettison a branch without explicit owner approval.
- **Reward-efficiency statistics (esper-lite-a2abff5ec5, Next bet):** first proof-provenance
  blocker esper-lite-441fbc6810 is closed in `7ebaff72`; Karn PPO traceability
  esper-lite-570d98c451 is closed in `0838cc40`. The newly startable P1s are
  esper-lite-9678c6d05a (ROI verdict semantics) and esper-lite-c3cb5338c4 (CI-safe PPO
  learnability proof lane). If work proceeds serially, take ROI verdict semantics next;
  the proof lane is a parallel-startable critical-path sibling.

## Facts the next session must not relitigate
- **Wrapper architecture settled — PDR-0039.** Caller-supplied pairing, telemetry-verified
  ON/OFF signature, type-enforced freeze order, and training-touch-last ordering stand.
- **S5/S6 reader and assembler seams settled — PDR-0040/PDR-0041.** G1/G2 are
  per-env-terminal; env coverage is assembly-layer validity; `build_report` never
  calibrates; `BuiltReport.validity` must be threaded into `render_packet`.
- **S7 built, then review-remediated and closed — PDR-0042/PDR-0043/PDR-0044.** The harness
  is no longer blocked on the known S7 review findings or the stale-gate/final-review validity
  blockers, but code-complete is not value-landed. No A/B before owner freeze.
- **Escrow shaping state settled — PDR-0047.** Escrow credit deltas are no longer clipped in
  the reward path; the Markov state needed to predict the dynamic potential is visible through
  Obs V3 schema v2. Checkpoints older than v4 are intentionally incompatible with the new
  observation contract.
- **Verification baseline remains mixed:** full default pytest passes at `89b50a16`; touched-file
  ruff, diff check, and leyline type lint pass. Defensive-pattern lint, GPU-sync lint, and full
  mypy still report existing non-touched failures outside this task.
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

## Last checkpoint did (checkpoint #33)
- Closed stale verification-only task esper-lite-e6382020d2 against the current branch evidence.
- Committed the escrow telescoping fix as `89b50a16` and closed esper-lite-3defe42928 with
  that commit anchor.
- Removed `escrow_delta_clip` from contribution reward shaping, added escrow stable accuracy
  and per-slot escrow credit to Obs V3 schema v2, rebased deterministic PPO/HRA goldens, bumped
  checkpoint compatibility to v4, and updated specs/roadmap/docs for the 120/132-dim contract.
- Verified with focused feature/reward/golden/checkpoint suites, full default pytest, touched-file
  ruff, diff check, leyline type lint, and full static guardrails with the known pre-existing
  non-touched failures still outstanding.

## Previous checkpoint did (checkpoint #32)
- Committed Karn PPO traceability evidence as `0838cc40` and closed Filigree task
  esper-lite-570d98c451 with that commit anchor.
- Added `ppo_traceability_evidence` as a Karn public proof surface, expanded `ppo_updates`
  with inf-gradient/update-skip/update-count/robust-value evidence, rendered `## PPO
  Traceability Evidence`, and blocked missing, nonfinite, skipped, and impossible finite
  PPO evidence before reward-efficiency ROI verdict math.
- Recorded PDR-0046: traceability is closed; ROI verdict semantics
  (esper-lite-9678c6d05a) and the CI-safe PPO proof lane (esper-lite-c3cb5338c4) are both
  startable, with ROI semantics the serial recommendation and proof-lane work parallelable.
- Verified with targeted red/green tests, focused Karn/MCP/proof suites, full default
  pytest, touched-file ruff, diff checks, Wardline, and added-line guardrail scans.

## Earlier checkpoint did (checkpoint #31)
- Committed PPO proof-provenance/missing-evidence fixes as `7ebaff72` and closed Filigree
  task esper-lite-441fbc6810 with that commit anchor.
- Recorded PDR-0045: PPO reward-efficiency remains Next; its autonomous critical path now
  starts at esper-lite-570d98c451, while Stage-2 freeze/launch remains owner-gated.
- Verified the proof fix with targeted red/green coverage, affected suites, full default
  pytest, touched-file ruff, diff checks, Wardline, and an added-line guardrail scan.

## Earlier checkpoint did (checkpoint #30)
- Committed the final-review Stage-2 fixset as `3669934d` and closed Filigree task
  esper-lite-2a4b56e719 with that commit anchor.
- Recorded PDR-0044: implementation is closed, but the Stage-2 value read remains
  owner-gated on freeze + paired HRA ON/OFF A/B.
- Reoriented next product work: Stage-2 launch is first if the owner freezes it; otherwise
  the autonomous critical path starts at esper-lite-441fbc6810.

## Earlier checkpoint did (checkpoint #29)
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
fresh-init HRA ON/OFF A/B. If yes, freeze `Δparam_max`, exact `W`, floored-EV
asymmetry, G3/G4 materiality, and advantage-floor policy before any ON run. If no, start
esper-lite-9678c6d05a (reconcile reward-efficiency ROI verdict semantics) as the serial
reward-efficiency move; esper-lite-c3cb5338c4 can run in parallel if another agent is
available and remains the Filigree critical path. Branch survivor cleanup is unblocked but
owner-gated for any merge/jettison/push action.
