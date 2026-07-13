# PDR-0075 — Refocus the commitment/EV-stab epic to causal permanent-seed instrumentation (PROPOSED — owner-gated)

Date: 2026-07-14   Status: **proposed — owner sign-off pending** (a strategy/Now-bet re-scope = vision-level; not enacted; `vision.md`/roadmap band NOT rewritten). Owner has SIGNALED the direction (offered to change the measurement + rerun, this session) but has not formally ratified the epic re-scope.
Related: PDR-0074 (the diagnosis this acts on), PDR-0073/0071/0069, epic esper-lite-f25b71c165. Reviewers converged (both primes, gpt-prime's owner-direction structure).

## Context
PDR-0074 resolved the commitment defect to a MEASUREMENT gap: a fossil's contribution is structurally unmeasured, so the reward can only pay a seed while provisional. The floor ρ-sweep (PDR-0069/0073) is contraindicated and coupled; ESCROW/de-shape do not address a missing input; the critic failure is secondary.

## What does this buy?  (REQUIRED — PDR-0068)
It points the epic at the binding constraint. Metric it moves: whether a genuinely-contributing seed can be committed WITHOUT forfeiting its credit — i.e. whether fossilization becomes a policy choice rather than a floor artifact. Success = the corrected objective makes commitment scoreable, then the policy's fossilization behaviour reflects value, not the floor.

## Options considered
1. **Instrument-first (measure/settle a permanent seed's contribution).** Chosen (proposed). The naive fossil-ablation is invalid (host-damage artifact — the code's own rationale), so the likely-cheapest valid measure is **settle-at-fossilize**: carry the seed's last valid HOLDING contribution forward as its permanent credit (this is what the `hindsight_credit`/escrow settlement channel was for). Alternatives: retrain-without counterfactual (gold standard, expensive), influence/Fisher estimate (cheap, approximate). drl-expert to design/review (CLAUDE.md mandate).
2. **Reward-redesign (ESCROW / de-shape).** Rejected: ESCROW would claw back credit on the fossilize measurement-zero (catastrophic); de-shaping removes a term whose INPUT, not form, is missing.
3. **Floor-first (ρ-sweep).** Deferred: contraindicated + coupled; retained as a later experiment once the reward ledger is corrected.

## The call (PROPOSED)
Refocus the epic Now bet toward **causal permanent-seed instrumentation**; **defer** the floor ρ-sweep (coupled, comes after); keep **critic calibration secondary** and held CONSTANT in the first reward experiment (else attribution is destroyed). Sequencing before any GPU arm: drl-expert designs the valid measure → add it TELEMETRY-FIRST (log the fossil's carried contribution; confirm fossils retain accuracy) → fold into the reward → coupled floor-fix → GPU rerun with the hard floor + critic architecture held fixed. Co-requisite: the causal retained/transferred-value instrumentation (host-only/all-off baselines, pre/post-event, terminal, seed identity) the epic already named — do NOT collapse to "pay more at FOSSILIZE" without it.

## Before-a-run check
Existing data cannot measure a fossil's contribution (fossils excluded from ablation; no checkpoints). The confirmatory read (fossils retain contribution) and the fix both require the measurement change + a rerun — hence telemetry-first, drl-reviewed, before GPU spend. The "measurement gap CAUSES commitment-avoidance" claim is a strong hypothesis, not intervention-confirmed (policy gradient-frozen on 94% of commits) — the rerun is the confirmation.

## Reversal trigger
If the drl-expert design finds no valid measure cheaper than a full retrain AND settle-at-fossilize proves invalid (e.g. the last-HOLDING contribution is not a defensible proxy for retained value) → the instrument-first framing needs rework and the transfer-instrumentation track leads instead. If telemetry-first shows fossils do NOT retain accuracy contribution → the −113 forfeit is not a real loss and the diagnosis (PDR-0074) reverses.
