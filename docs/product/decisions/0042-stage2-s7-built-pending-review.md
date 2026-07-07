# PDR-0042 — Stage-2 MAJOR-1 acceptance: S7 built, pending specialist review before freeze

Date: 2026-07-07   Status: accepted (state checkpoint only; delivery remains in review)
Author: Codex (GPT-5)   Owner sign-off: n/a (autonomous checkpoint within grant; no release/push/freeze)
Related: PDR-0039 (wrapper slice map), PDR-0040 (S5 reader + review fixes), PDR-0041
(S6 packet/env-count/assembler), task esper-lite-2a4b56e719, epic esper-lite-f25b71c165.
Code commit: `9666fe0c`.

Forward pointer: state advanced by PDR-0043 and the final-review working-tree fixset after
`6d2dd07e`; known S7 findings and final-review validity blockers were remediated, but freeze
and the paired HRA ON/OFF A/B remain owner-gated.

## Context

The product workspace checkpoint #26 said Stage-2 acceptance was **6/7 slices done** and that S7
remained. Live tracker comments and git now show S7 was built afterward at `9666fe0c`: §9
diagnostic emission, `actor_advantage_source` provenance, `read_run_meta`, G4 guard-channel
reading, explicit G3/G4 threshold shapes, and the runnable `scripts/stage2_packet.py` CLI.

That delivery is **not accepted and not freezable yet**. The Filigree task remains
`in_progress`, and the last tracker comment says a three-agent review is running: holistic review
of changes since `f5f4c7f3`, plus `pytorch-expert`, plus `drl-expert`, including confirmation or
rejection of the definitional items around G1/G2/G3/G4 and env-count completeness.

## Options considered

1. **Leave the workspace at checkpoint #26 until the review finishes.** Rejected: RESUME would keep
   telling the next owner to build S7 even though the code is already at HEAD.
2. **Checkpoint the state transition without accepting the delivery.** Chosen: the product state
   now says S7 is code-complete, but acceptance is still gated on specialist review and finding
   remediation.

## The call

Record the Stage-2 task as **7/7 slices code-complete, pending review**. Do not close the task, do
not freeze the gate, and do not start the paired HRA ON/OFF A/B until:

- the holistic review and the `pytorch-expert` / `drl-expert` reviews are complete;
- any findings are fixed with targeted tests;
- the OFF-leg byte-identity claim, G1/G2/G3/G4 definitions, G4 reader, and env-count threshold are
  explicitly accepted or revised;
- the gate doc is frozen against the reviewed code.

The tactical tracker remains the source of truth for status: esper-lite-2a4b56e719 stays
`in_progress`.

## Rationale

This is a continuity checkpoint, not a delivery acceptance. It removes stale instructions from
`current-state.md` while preserving the product gate: code-complete is not value-landed, and
pre-registration is not frozen until review hardens the S7 training-code touch.

## Reversal trigger

- If any of the three reviews finds a blocker or major issue, keep Stage-2 in progress and revise
  the S7 implementation or gate definitions before freeze.
- If the OFF-leg byte-identity evidence does not survive review, rework the emission approach before
  any A/B.
- If the definitional review rejects the current G1/G2/G3/G4 or env-count hard-reject shape, update
  the scorer/wrapper and this product state before accepting the gate.
