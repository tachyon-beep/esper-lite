# PDR-0076 — Refocus APPROVED: "make permanence visible" is the Now bet, with binding anti-fossil-farm conditions

Date: 2026-07-14   Status: accepted (owner-approved this session, subject to the round-12 reviewer conditions below)
Supersedes: **PDR-0075** (its `proposed` status → APPROVED, and its wording "build a valid causal measure of a permanent seed (hard)" → the layered, existing-data-first L1/L2/H7 plan). Builds on PDR-0074 (diagnosis). Related: `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md` (round-11 + H6, and this round's design brief). Round-12 reviewers: gpt-prime, claude-prime; owner ratified all three decisions "subject to roadblocks from the primes."

## Context
PDR-0074 resolved the commitment defect to a permanent-value measurement gap; round-11/H6 confirmed it poisons the OBSERVATION too (`features.py:830` `None→0`) and retired the critic as a symptom. The owner approved (1) the refocus, (2) an ESCROW safety guard, (3) a drl-expert L1/L2 design + reads — each "subject to the primes' roadblocks." Those roadblocks are conditions, recorded here so they bind the work.

## What does this buy?  (REQUIRED — PDR-0068)
It points the epic at the single binding defect and removes a silent training lie ("permanence = worthless"). Metric: a genuinely-contributing seed retains its measured value across FOSSILIZE (no `→0`). Guardrail (the new failure mode this decision must not create): the fix does NOT FARM fossils.

## The call (owner-approved)
1. **Now bet = "make permanence visible":** make missingness EXPLICIT in observation + reward + settlement (L1 freeze-don't-zero with lifecycle scoping; L2 boundary settlement), test proxy-adequacy (H7), then re-ask commitment learning. Roadmap band moved; `vision.md` reinforced against the SNR + Goodhart anti-goals.
2. **ESCROW guard:** scoped, fail-closed (raise on construction when ESCROW active + FOSSILIZE reachable + no settlement configured); H4 test; promote to hard error if clawback confirmed. Not a universal escrow ban.
3. **drl-expert design (not land) + reads H3/H4/H5/H7 + offline replay.** Dispatched this session.

## Binding conditions (the round-12 roadblocks — violate these and the approval lapses)
- **H7 GATES L1.** H7 (proxy-validity/drift probe) measures the fossil-contribution decay = the freshness `γ` L1 needs. Design L1's decay AFTER H7, not before. L1 without H7 ships a guessed decay constant.
- **Design against the FOSSIL-FARM adversary.** L1+L2 naively = revenue ~100 vs cost ~0.12 ≈ 800:1 → commit-everything-at-the-spike. Three hazards: sell-at-the-spike (policy picks the timing of a max over a noisy estimator; mitigations necessary, not sufficient), decay-lie (a zero-lie swapped for a decay-lie if `γ` mis-set), double-counting (settle AND keep paying). L2 revenue must ≈ the discounted forfeited stream (−113), not exceed it.
- **L1 is NOT a one-line `None→last`.** Canonical contribution-state struct (last-valid + status{NEVER_MEASURED/FRESH/STALE/FROZEN_AT_FOSSILIZE} + freshness + lifecycle-id) consumed by observation+reward+settlement; distinguish true-0 from absence; CLEAR on germinate/prune/slot-reuse (else the carry-forward artifact recurs inside the live policy); schema version bump.
- **L2 is an accounting-equivalence candidate, not truth.** Pre-registered invariant `G_stay ≈ G_fossilize` for a constant-contribution seed; full spec (discount, clip/norm order, negative contribution, one-shot-bonus interaction, double-payment, never-measured, too-stale, terminal). Evaluate by OFFLINE REWARD REPLAY before any run.
- **DESIGN-don't-land** (except the ESCROW guard); **no GPU run**; **floor + critic held CONSTANT** through the first L1/L2 experiment (remeasure the critic AFTER; a residual returns then, not now).
- **Premise discipline:** "under-fossilization is a defect" is UNMEASURED, not wrong. Do not swing to "commitment is bad."
- **Two gates:** training-legibility (does L1/L2 remove the `→0` discontinuity?) vs product-validity (is the last-valid LOO a valid durable-value proxy?). H7 decides legibility-adequacy, NOT causality; a hard L3 causal instrument may still be needed (H7's own verdict).
- **P0 harness independent:** K>1 (K=1 leaves PPO with no operative trust-region guard, Read B), checkpointing, per-decision telemetry — correct regardless of how the epic resolves.

## Reversal trigger
- If offline replay shows an L2 candidate creates an early-fossilization windfall that no timing/staleness mitigation removes → that candidate is a farm subsidy; do not run it.
- If H7 finds the last-valid LOO does NOT predict retained product value → L1/L2 are a legibility bridge only and the hard L3 causal instrument becomes the bet.
- If, after L1/L2 remove the `→0` discontinuity, a controlled read shows FOSSILIZE still has reliably negative PRODUCT return (not proxy-credit) → under-fossilization may be correct and the epic's premise reverses.
