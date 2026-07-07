# PDR-0044 — Stage-2 harness closed; next product gate

Date: 2026-07-07   Status: accepted
Author: Codex (GPT-5)   Owner sign-off: n/a (tracker/product checkpoint; no push/freeze/A/B)
Related: PDR-0037, PDR-0039, PDR-0040, PDR-0041, PDR-0042, PDR-0043,
task esper-lite-2a4b56e719, epic esper-lite-f25b71c165, work package esper-lite-a2abff5ec5.
Code commit: `3669934d56aebc14be3a26c5fb894eb0661c1bd1`.

## Context

PDR-0043 checkpointed Stage-2 as review-remediated but still in progress. A final review
found four remaining blockers: stale gate/product language, OFF calibration evidence not
fully validated before freezing delta, traceback/run-log failures not invalidating the
packet, and missing `advantage_std_floored` reporting. Those blockers were remediated in
code, tests, and product docs in `3669934d`, then Filigree task esper-lite-2a4b56e719 was
closed with the commit anchor.

This does not mean the Stage-2 MAJOR-1 value hypothesis has passed. The gate document is
still not owner-frozen, no paired fresh-init HRA ON/OFF A/B has run, and no Stage-2 packet
has produced an experimental verdict.

## Options considered

1. **Close Stage-2 implementation and immediately treat EV-stabilization as accepted.**
   Rejected: that would bank output as outcome. The acceptance harness is closed, but the
   paired experiment is not read.
2. **Launch the paired HRA ON/OFF A/B now.** Rejected for this checkpoint: owner freeze is
   still required for the remaining threshold slots and launch posture before any ON run.
3. **Move directly to branch survivor cleanup.** Deferred: the cleanup task is now
   unblocked, but it contains owner-gated merge/jettison/branch deletion work and does not
   itself validate the EV-stabilization product bet.
4. **Keep EV-stabilization as the Now bet, with the next owner gate explicit; use the
   reward-efficiency critical path as the next autonomous product track if the owner does
   not freeze/launch Stage-2 immediately.** Chosen.

## The call

Stage-2 implementation task esper-lite-2a4b56e719 is **closed**. EV-stabilization remains
the Now bet until the Stage-2 gate is frozen and the paired HRA ON/OFF A/B is read.

Next product stance:

- If the owner is ready to continue the active EV-stabilization read, first freeze the
  remaining Stage-2 gate slots: `Δparam_max`, exact burn-in `W`, floored-EV asymmetry
  policy, G3/G4 materiality thresholds, and any hard `advantage_std_floored` cutoff.
- Only after freeze, prepare the owner-approved paired fresh-init HRA ON/OFF A/B and score
  the Stage-2 MAJOR-1 packet.
- If the owner does not want to launch that experiment now, the next autonomous product
  implementation track is the P1 critical path under PPO Learning Gate / Reward-Efficiency
  Statistics, starting with esper-lite-441fbc6810.
- Branch survivor cleanup (esper-lite-1f1e55f58f) is unblocked but should stay behind the
  owner-gated Stage-2 launch decision unless the owner explicitly prioritizes branch
  hygiene. No merge, branch deletion, push, tag, or remote action without approval.

## Verification recorded

- `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_stage2_acceptance_io.py tests/simic/telemetry/test_stage2_acceptance_packet.py tests/scripts/test_stage2_packet.py -q`
  — 141 passed.
- Touched-file `ruff`, `git diff --check`, `scripts/lint_leyline_types.py`, and
  `wardline scan . --fail-on ERROR` passed.
- `scripts/lint_defensive_patterns.py` remains red on existing non-touched files and stale
  whitelist entries; no new prohibited pattern was introduced in the Stage-2 patch.
- Legis Filigree closure gate could not run because the binding ledger is not enabled
  (`LEGIS_HMAC_KEY` missing); the local Filigree close still recorded the commit anchor.

## Rationale

This keeps the product state honest: implementation is closed, but the value claim is not.
The next irreversible or outward-facing acts remain owner-gated, while the tracker still
has a P1 autonomous path available if the owner chooses not to spend the next session on
Stage-2 experiment launch.

## Reversal trigger

- If the owner freezes the gate and launches the paired A/B, write a new PDR with the
  exact frozen thresholds and launch spec.
- If the Stage-2 packet rejects or lands inconclusive, keep EV-stabilization open and
  decide whether to refold counterfactual credit through DPBA, continue Stage-1/per-head
  variance work, or park the bet.
- If the owner prioritizes branch survivor cleanup before the A/B, record that as a
  separate owner call because it crosses merge/jettison/push-adjacent authority boundaries.
