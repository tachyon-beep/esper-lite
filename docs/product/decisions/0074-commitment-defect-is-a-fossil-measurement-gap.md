# PDR-0074 — The commitment defect is a MEASUREMENT gap: fossilized seeds are excluded from the counterfactual ablation, so the reward can only pay a seed while it is provisional

Date: 2026-07-14   Status: accepted (diagnostic finding, within grant; offline zero-GPU reads)
Supersedes (specific claims): **PDR-0069 "the gate passes" / "the reward signal for committing is already correct"** and **PDR-0071 §4 "signal survives the 9-pt spread"** — RETRACTED (below). Qualifies PDR-0066 (adds a confirmed main-critic calibration failure on the FOSSILIZE contrast). Related: `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md` (Read A/B, G1/G2/G3, round-10 H1/H2), PDR-0072/0073. Rounds 8-10 reviewers: drl-expert, pytorch-expert, gpt-prime, claude-prime, advisor.

## Context
The read-before-build reads (PDR-0073) resolved. Read A returned INVERTED (per-decision advantage favours SET_ALPHA over FOSSILIZE, PROXY tier). Round-9 G1/G2/G3 traced it: rent negligible (~0.3%), critic overstates the main-stream penalty 4-6×, and the realized total-return penalty for committing (−113/−102, GROUND TRUTH) is ~99% a forfeited `bounded_attribution` (counterfactual) stream. Round-10 decomposed that collapse and read the code.

## What does this buy?  (REQUIRED — PDR-0068)
It replaces a mis-targeted fix (the floor ρ-sweep) with the actual binding constraint, and prevents a contraindicated one (ESCROW). Measured: the fix's target is now the fossil's unmeasured contribution, not the op-head gradient; building the floor-fix on the old premise would drive fossilization to ~0 (see §Package).

## The finding (both seeds, code + ground-truth confirmed)
- **Fire-rate, not magnitude (claude-prime; arithmetic on the measured numbers).** HOLDING `bounded_attribution` = fire-rate 0.875 × magnitude 3.60 = 3.15/epoch; FOSSILIZED = 0.115 × 2.78 = 0.32. The ~10× collapse is **rate 7.6× (79-89% of it)** × magnitude 1.3×. The credit did not shrink at permanence — it STOPPED FIRING.
- **Mechanism (code-confirmed, H2): FOSSILIZED seeds are excluded from the counterfactual ablation by design** (`vectorized_trainer.py:956-975`, skipped unless `drip_fraction>0`; run is SHAPED/drip=0). Rationale in-code and CORRECT: ablating a baked-in fossil measures *host damage*, not the seed's marginal contribution. So a fossil's `seed_contribution` is structurally `None` → `bounded_attribution` cannot fire → **the reward pays a seed for its contribution ONLY while provisional (not at α=0 birth, not at permanence).** This is a MEASUREMENT/INSTRUMENT gap, not a reward specification. It is the same `counterfactual=None` missingness that caused analysis over-reads #6/#7/#8 — but in the TRAINING signal.
- **The gate FAILS by ~75× (retraction).** The −0.7→+8.2 that PDR-0069 read as "the gate passes" was a 9-pt spread on the REVENUE line; a −113 cost line, scaling the same way with LOO, sat UNMEASURED (bonus ~1.5 vs forfeited stream −113 ≈ 75:1; worse for better seeds).

## Facts banked (scoped: seeds 41/42, `reward_mode="shaped"`, current policy/state distribution)
- G1 coefficients: fossilize bonus `0.5+0.1c` one-shot; `fossilized_maintenance_cost=0.002`/step; occupancy unchanged by fossilizing. Rent ≈0.3% of the realized penalty (≈5% of the earlier 2.6-unit continuation estimate).
- Realized total-return contrast (FOSS−SET_ALPHA) −113/−102 = cf-stream ~99% / accuracy ~0.8% / rent ~0.3%.
- **Main-critic calibration failure (subject to MC/value-target parity):** V_main asserts −6.6/−5.4, realized main-stream is −1.0/−1.2 → overstates the FOSSILIZE-vs-SET_ALPHA main-stream disadvantage ~4-6×. Real, SECONDARY, and NOT "the critic is broken" in general.
- τ_main ≈ 0 across strata (the premise "commitment is bad" is UNMEASURED once the cf phantom is stripped, not refuted).
- G3 bound: no statistically reliable positive commitment region **in the cohort this floored policy produced** — NOT "no policy could benefit from committing."

## RETRACTIONS (banked)
PDR-0069 "gate passes"/"reward already correct"; PDR-0071 §4 "signal survives"; the round-9 "reward-specification / commitment-avoidance is optimal / vindicates the EV-stab epic" reading; "ESCROW as the fix." Un-retract PDR-0071's co-requisite (settlement/transfer): it is the missing revenue line, plausibly dead for the same measurement reason (H3, unconfirmed).

## Package, not sequence
FOSSILIZE is floor-bound 93-94% → the ~17% fossilization rate is the FLOOR's number, not the policy's. Fixing the op-head gradient under THIS reward would drive fossilization → ~0 (the floor is accidentally the morphogenesis pipeline). So the floor-fix and the measurement fix are a **package**; shipping the floor-fix alone stops the product producing structures.

## Reversal trigger
Under a reward that correctly MEASURES or SETTLES a fossil's contribution, if FOSSILIZE still shows a large negative realized-return contrast attributable to NON-counterfactual streams → the measurement gap was not the binding cause and the specification/critic lines return. And: if the MC/value-target parity check fails, the 4-6× critic ratio must be recomputed (sign/stream diagnosis still holds).
