# PDR-0095 — Lock-burden overlay BANKED: 1.75% locked-epoch fraction — the Option-B escalation trigger does NOT fire; the Option-A conditional ratification (PDR-0090 #5) is DISCHARGED

Date: 2026-07-14   Status: accepted
Follows PDR-0090 (#5 condition) / PDR-0091 (#8 task creation). Artifact: `docs/analysis/2026-07-14-r9-lock-burden-overlay.md` (+ scripts/JSON under `docs/analysis/scripts/2026-07-14-r9-lock-burden/`). Tracker esper-lite-8af424bb66 closed.

## What does this buy?  (REQUIRED — PDR-0068)
The last conditional in the round-21 ratification set resolves on data: Option A's cost is measured at ~1/14th of the escalation threshold, so the simple, construction-safe lock stands and the materially harder Option B (shadow-ensemble quote) stays deferred. Every freeze READ is now complete — the freeze queue contains only owner rulings, the build, and reviews.

## Decisions
1. **Burden BANKED (r9 overlay, both seeds on-regime):** locked-epoch fraction **1.75% pooled** (1.72/1.78);
   per-episode max 18.7%, P99 9.3%, P50 0 — **the >25% Option-B escalation trigger does NOT fire under either
   the gated (epoch-fraction) or the stricter per-episode reading.** Axis discipline recorded: 26.85% of
   episodes are TOUCHED by ≥1 lock — episode incidence, a different denominator; never read against the 25%
   epoch gate.
2. **Suppression profile is small and non-pathological:** GERMINATE 1.74% / PRUNE 2.10% / SET_ALPHA 1.19% of
   each type's totals; 52 queued second-FOSSILIZE requests (1.25%); 100 late-masked requests (2.41%); **max
   queue depth 1** (zero windows with ≥2 queued — serialization is a non-issue at r9 rates); suppressed
   PRUNE→HOLDING overlap 0.93%.
3. **Cross-instrument reconciliation banked:** 4,151 FOSSILIZE decisions = 3,999 window-openers + 52 queued +
   100 late-masked — exact, and equal to the phase-1 PBRS scan's independent commit count. Two instruments,
   one total.
4. **Option-A conditional ratification DISCHARGED:** PDR-0090 #5 ratified Option A subject to burden
   quantification; the condition is met. B−A remains labeled "the complete LOCKED settlement protocol"; the
   estimand telemetry (lock-burden counters live in the B arm, PDR-0094 N-r2c) still ships — the overlay is a
   pre-freeze estimate, the counters are the in-experiment measurement.
5. **Caveat carried:** first-order overlay on unlocked r9 behavior — it measures how much realized activity the
   lock would intercept, not the locked policy's adjusted dynamics; the in-experiment counters are the truth.
6. Process observation filed (esper-lite-obs-3811e62ea1): dispatched agents share the session scratchpad with
   generic filenames — the lock agent silently overwrote the IQR read's same-named outputs (harmless only by
   persist-before-next-dispatch ordering). Future dispatches specify task-prefixed filenames.

## Reversal trigger
- If the B-arm's LIVE lock-burden counters read materially above the overlay (e.g., >3× the 1.75% estimate, or
  any queue depth ≥3) → the first-order caveat bit; re-open the Option-B question at the screen gate, before
  the claim tier.
- If commit RATES change materially under the settlement (the overlay assumed r9's floor-forced ~1.9%/epoch
  request rate) → re-estimate before the n=10 wave; the overlay's arithmetic is rate-linear.
