# PDR-0098 — Owner rules F1 = settle-partial and F2 = ADD the pending bit; F3 simulation executed → S9 (span-10 window, LCB z=1.5) PROPOSED on the data

Date: 2026-07-14   Status: accepted (F1/F2 owner-ruled verbatim below; the F3 mechanism is PROPOSED from the simulation — owner ratifies at/before freeze per their "can't yolo" directive)
Follows PDR-0092/0093/0091(#4). Owner ruling (verbatim): "F1, A, 2 A yes more telemetry is always good, F3 ok, lets try both, I can't yolo that one". Artifact: `docs/analysis/2026-07-14-f3-gate-simulation.md`.

## What does this buy?  (REQUIRED — PDR-0068)
The last two design forks close on owner authority and the third gets a measured answer instead of a guess: both
pre-registered gate forms fail exactly as the noise read predicted, and the replacement is selected off an
efficient frontier (windfall bound met at BOTH measured noise levels, at 2.5–3× less legitimate-denial than the
alternatives) rather than by intuition.

## Decisions
1. **F1 RULED: `abort_payment_mode = "settle_partial"`** for the experiment (claude-prime's construction: a
   safety-abort settles at the scheduled boundary on q_settle-as-measured — the liability survives; outcome class
   SETTLEMENT-INTERRUPTED, pre-registered censoring). The `void` branch stays implemented and replay-tested (the
   config field remains required-explicit; the ruling selects the experiment's value).
2. **F2 RULED: ADD the pending bit.** One per-slot `pending_settlement` dimension in the obs (obs-v4 path); arm A
   emits constant 0 (schema identical across arms); explicit obs-schema version stamp; re-warm accepted.
   **Recorded distinction (the owner's phrasing said "telemetry"):** this is an OBSERVATION-schema change — the
   policy/critic network sees it — not merely telemetry; the aliasing telemetry + symmetry check ALSO ship, as
   verification that the bit closes the measured gap. Consequences enacted: pre-reg §2.5.8 pin amended
   (supersedes "no new obs dim"); G-OBSERVABILITY joins the replay acceptance areas; the plan gains an obs
   increment (features.py per-slot block + get_feature_size + schema version + distinguishability test).
3. **F3 SIMULATED (owner-directed):** nine candidate gates × two measured noise levels (σ = 0.82 P50 / 1.87 P90),
   20k seeded draws per cell. Both pre-registered forms FAIL (raw: windfall 0.449; LCB z=1: 0.172). **PROPOSED:
   S9 — `min_window = 10` valid measurements, span-10 EWMA, `LCB(z=1.5) ≥ 1.0`** — windfall 0.077/0.091 (PASS
   both σ), denial 0.118 (vs 0.300/0.342 for the other passers). Costs stated in the artifact: ~2× window →
   lock burden ~3% (still ≪ 25%), earlier late-mask cutoff, and the named structural residual — ~33% denial at
   μ=1.5 is a DESIGNED anti-commit force on marginal seeds (GATE-DOM interpretation note). Pre-reg §2.3/§2.4
   carry S9 as PROPOSED parameters; the owner's ratification lands with the freeze signature.
4. Standing prohibition unchanged: never a `q_decision` gate.

## Reversal trigger
- F1: if SETTLEMENT-INTERRUPTED events exceed ~2% of requests in the screen (aborts are supposed to be rare
  protocol failures) → the censoring class is too fat for the P2 cohort; revisit toward gpt's ITT handling before
  the claim tier.
- F2: if the A-arm-with-bit baseline shifts materially vs prior A-arm telemetry (the re-warm confound check,
  PDR-0093) → the schema change itself becomes a suspect; the A-comparison read gates the screen.
- F3/S9: if the K=4 screen's in-regime near-cliff σ differs materially from 0.82 (either direction) → re-run the
  simulation table (minutes) and re-select before the claim tier; the gate form is frozen per-experiment, never
  retuned mid-run.
