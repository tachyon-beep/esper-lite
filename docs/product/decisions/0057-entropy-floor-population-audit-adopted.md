# PDR-0057 — Entropy-floor population audit adopted as front-of-TIP concept

Date: 2026-07-10
Status: accepted (within grant: write specs, prioritize)

## Context

A three-round external review (owner-relayed, "Sol") of the PDR-0053→0055
penalty-schedule confound arc surfaced a genuinely new concern: the optimiser's
entropy-floor penalty may regularise a population that conflates policy
uncertainty with decision density. Each round was verified against the codebase
in-session before adoption (no performative agreement): the floor averages
legal-set-normalised per-step entropy over the AVAILABILITY mask, counting
single-legal-action steps as exact zeros and including forced steps
(`ppo_update.py:158`, `action_masks.py:595`, `causal_masks.py:66`). The exact
identity `H_avail = d × H_choice_avail` holds; `d < floor` makes a head's floor
mathematically unattainable; and loss magnitude decouples from gradient
pressure as density falls (persistent-loss/weak-gradient regime). Existing
telemetry cannot exactly reconstruct the sparse-head floor population (three
gating differences vs `choice_conditional_head_entropy`); the op head's
`H_avail` is exactly reconstructable now.

## Options

- Audit now (during Stage-2 freeze) — rejected: heavy telemetry reads and any
  training-path change were freeze-blocked; also a mid-experiment objective
  change would contaminate the A/B.
- Adopt as a spec'd concept, run post-freeze after the Stage-2 read — chosen.
- Dismiss as speculative — rejected: the structural premises are verified in
  code, and the project already paid for one population artifact (the false
  entropy-collapse alarm).

## The call

Adopt the audit as a concept at the front of the telemetry programme
(`docs/plans/concepts/2026-07-10-entropy-floor-population-audit.md`, tracker
row beside TIP). Feasibility-margin (`d − floor`) first; five-case
classification (genuine collapse / density activation / structural
infeasibility / population mismatch / persistent-loss-weak-gradient); new
per-head floor-population telemetry contract with a reconciliation assertion.
Three-track sequencing discipline: schedule fix (PDR-0055), observability
addition, and any objective change are SEPARATE — the objective change needs
its own PDR, pre-frozen hypotheses, and an experiment. Stage-2 implication:
common-mode at assignment; the floor is a possible treatment MEDIATOR, not a
confound (recorded wording in the concept doc §9).

## Reversal trigger

If the post-freeze audit lands every head in the benign case (H_avail ≥ floor
throughout; positive feasibility margins; no case B/C/D/E findings), the audit
demotes to documentation and NO objective change is licensed. Conversely, a
case-C finding (structurally unattainable floor) on any head escalates the
objective-change track to the owner immediately.
