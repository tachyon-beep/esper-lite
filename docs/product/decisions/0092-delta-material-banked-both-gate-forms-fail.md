# PDR-0092 — Δ_material BANKED (0.0075 request-probability per pp; IQR_q = 13.32 pp, not degenerate); near-cliff quote noise σ ≈ 0.82 pp → BOTH pre-registered threshold-gate forms FAIL — new owner fork F3 (qualification mechanism redesign at freeze)

Date: 2026-07-14   Status: accepted (reading banked within standing authority; the F3 mechanism choice is OWNER-OPEN at freeze)
Follows PDR-0089/0091. Artifact: `docs/analysis/2026-07-14-r9-iqr-noise-read.md` (+ scripts/JSON under `docs/analysis/scripts/2026-07-14-r9-iqr-read/`). Tracker esper-lite-e12e2d1543 closed. Read executed spec-first (spec durable before computation); reconstruction validated bit-exact against the trainer's own values (186,786 matches, |Δ|=0); regime stamps pass both seeds; the dispatched agent API-stalled only after all artifacts were written.

## What does this buy?  (REQUIRED — PDR-0068)
The last unfrozen constant in the certification chain is now a number, and the threshold-variance question got answered with data one round before it would have been ratified blind: the pre-registered fallback chain (raw EWMA → LCB z=1) was designed against a hypothesized noise scale that reality exceeds by ~4×. Discovering that at freeze-time costs a design round; discovering it after runs would have cost the gate's meaning.

## Decisions
1. **Δ_material BANKED: 0.0075 request-probability per pp** (pooled IQR_q = 13.32 pp; per-seed 13.72/12.98;
   sensitivity ≥3-prior-measurements: 13.57 — stable). The PDR-0086 degenerate-IQR trigger does NOT fire. The
   blinded certification rule is now concrete: **n=10 is certified for a metric iff UCB₈₀(σ̃_d) ≤ Δ_material /
   d_z*(π)** (with the 80%-UCB-primary posture of PDR-0091 #6), and the gate bounds read UCB₉₅(H) < Δ_material.
2. **Context fact worth carrying:** 95% of a seed's fresh measurements occur PRE-HOLDING, and the median
   HOLDING-decision quote is ≈6.4 pp — far above the 1.0 pp cliff. The quote is a pre-HOLDING-weighted history;
   near-cliff traffic is a tail (which is also why the noise sample is small).
3. **NEW OWNER FORK F3 — the ≥1.0 threshold-qualification mechanism:** measured near-cliff per-measurement noise
   σ P50 ≈ 0.82 pp (98.5% of windows ≥ 0.2, 96% ≥ 0.3) vs the pre-registered thresholds: raw gate fails at
   σ ≳ 0.15–0.2; the LCB(z=1) fallback holds only to σ ≈ 0.3. **Both pre-registered forms are exceeded ~3–5×.**
   Under ΔP<0.10, neither confines windfall pay-probability near the cliff. Design options for the freeze package
   (enumerated, not chosen): (a) longer confirmation window (min_window 5→10+ halves EWMA noise ~√2 — interacts
   with W_settle and the annuity-horizon arithmetic); (b) larger z on the LCB (confines windfall but denies more
   legitimate above-cliff premiums — the option cuts both ways); (c) a margin (qualify at q_settle ≥ 1.0 + k·σ̂);
   (d) accept the windfall at the measured rate and price it in Δ_designed (it is symmetric-ish in expectation
   across arms only if commit-rate near the cliff matches — needs an argument). The pre-registered prohibition
   STANDS regardless: never back to a `q_decision` gate.
4. **Caveats carried, not buried:** IQR_q is dwell-weighted/autocorrelated (effective n ≪ 26,703; n_lifecycles
   13,103); the σ verdict rests on 132 pooled tail windows (direction robust at ~4× over threshold; magnitude
   approximate); K=1/obs-v3 anchors the scale for a K=4/obs-v4 target regime.

## Reversal trigger
- If the K=4/obs-v4 screen shows in-regime near-cliff σ materially below ~0.2 pp → the raw/LCB chain re-enters
  (F3 relaxes); the screen's blinded read should include this measurement.
- If the owner picks option (a) (longer window) → the r9 measurement-cadence result (PDR-0087) still guarantees
  feasibility (100% cadence), but the annuity-horizon and late-request-mask arithmetic recompute; the replay's
  missed-measurement case re-parameterizes.
- If Δ_material is later judged mis-scaled because the dwell-weighted IQR misrepresents the decision-relevant
  spread → the fallback is the per-lifecycle (deduplicated) quote distribution, computable from the preserved
  arrays JSON without touching r9 again.
