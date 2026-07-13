# ON-leg Gram diagnostic (n=2) — PDR-0064 read

Date: 2026-07-13 · Runs: `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/`
Commit: `92d998bd` · Completion: both seeds 600/600 rounds, rc=0, 0 crashes.
Reviewed: advisor, 2026-07-13. Diagnostic (not a gate); rejected HRA re-used as substrate.

## Headline: the pre-committed A′/B discriminator does NOT discriminate — A′-vs-B is underdetermined

The diagnostic killed the lightest-touch fix and localized the problem, but it CANNOT
pick A′ vs B, because the held-out affine calibration-rescue instrument (PDR-0060/0064)
is near-tautological for well-trained component heads. Applying its pre-registered label
would report a verdict the instrument is structurally unable to support.

### Why the discriminator fails (the algebra — flag this up front next time)
Under Objective A, `V_main` is trained toward `E[G_main|s]`, `V_cf` toward `E[G_cf|s]`,
and `returns_total = returns_main + returns_cf` **by construction**. So for decently-
trained heads `V_main + V_cf ≈ E[G_total|s]` — which is exactly the L2-optimal linear
predictor of `G_total`. Therefore **(1,1,0) is optimal by construction**, and an affine
recalibration `a·V_main + b·V_cf + c` can only improve on it to the extent the heads are
*grossly miscalibrated* (wrong scale/offset). "No rescue" is the EXPECTED outcome under
**both** A′ and B — it is baked into the algebra, not evidence for either.

The instrument is, in effect, a **head-calibration-quality meter**. It reports "the heads
are decently calibrated" — useful — and says almost nothing about A′ vs B. Crucially,
PDR-0060's A′ is **heavy-A′** ("primary `V_total` *directly trained* + stabilized aux
heads"): a directly-trained `V_total` learns *different representations* the frozen
Objective-A heads cannot express, so a projection of the frozen heads is structurally
blind to it. The instrument only refutes **light-A′** (a post-hoc calibration layer on the
existing heads), which is not the A′ on the table.

## The numbers (post-anneal window ≥250; 250–425 vs 425–600 medians) — data is sound

### Held-out affine calibration-rescue (fit 250–350, test 400–600) — a calibration meter, not A′/B
| seed | best-fit Ĝ = a·V_main + b·V_cf + c | recal. held-out EV | raw ev_sum | ev_main | lift |
|------|-------------------------------------|--------------------|-----------|---------|------|
| 41 | (0.921, 1.000, −0.18) ≈ (1,1,0) | 0.621 | 0.581 | 0.854 | +0.040 |
| 42 | (1.398, 1.047, +0.20) | 0.610 | 0.563 | 0.835 | +0.047 |

The +0.04 lift is real and held-out but tracks the **normalizer over-estimate** (plateau-3),
seen as seed-42's `a=1.40` main-head reweight — NOT an A′/B signal.

### cf learner
| seed | ev_cf (250–425 → 425–600) | ev_main | ev_sum | Var(e_cf) ratio |
|------|---------------------------|---------|--------|-----------------|
| 41 | 0.591 → 0.596 | 0.82–0.86 | 0.57–0.58 | 0.788 |
| 42 | 0.548 → 0.575 (rising) | 0.78–0.84 | 0.53–0.57 | 1.117 |

### cf target-process (plateau-1) — SPLIT across seeds (the only leg with real A′/B content)
| seed | Var(G_cf) 250–425 → 425–600 | ratio | slope/100 |
|------|-----------------------------|-------|-----------|
| 41 | 601.5 → 510.9 | 0.849 | −53.2 (plateaus) |
| 42 | 419.5 → 503.9 | 1.201 | +40.8 (still drifts up) |

### normalizer (plateau-3): inst_cf_scale / running `cf_value_target_scale` ≈ 0.81–0.86 (running ~15–20% high)

## Verdict

**Robust (n=2, both seeds):**
- **Objective A stays rejected** (PDR-0059) — unchanged.
- **The total-fit deficit is real and cf-driven:** `ev_sum` ~0.55 ≪ `ev_main` ~0.84; the
  cf component caps the total. `ev_cf` ~0.55–0.60 (up from PDR-0060's 0.44–0.54).
- **It is NOT a gross-calibration artifact:** the heads are decently calibrated (best
  affine map ≈ (1,1,0); only a mild ~15–20% normalizer over-estimate). A post-hoc
  calibration fix (**light-A′**) will not help — that much IS refuted.

**NOT established (underdetermined):**
- **B over heavy-A′.** The diagnostic cannot separate them. Low `ev_cf` is equally
  consistent with "cf intrinsically hard → B" and "cf under-trained by the endogenous
  Objective-A bootstrap, fixable by directly training `V_total` → heavy-A′." `ev_cf` is
  still *rising* on seed 42 (ratio 1.049), which if anything leans mildly AGAINST a hard
  ceiling.
- **cf-target intrinsic non-stationarity** — plateau-1 is split (41 plateaus, 42 drifts
  +20%); the pre-committed disagreement-⇒-extend-n trigger fired on this, the only leg
  with genuine discriminating content. But plateau-1 speaks to non-stationarity, not to
  the `ev_cf` ceiling.

## What would actually discriminate A′ vs B (the frozen-head Gram data cannot)

- **A bounded A′ pilot** — actually train a direct `V_total` head and see whether total EV
  breaks the ~0.61 affine ceiling. "To know if retraining helps, retrain." This is the
  only *direct* test and the cheapest thing that genuinely moves the A′/B decision.
- **A Path-C state-conditional information probe** for the true `Var(G_total|s)` ceiling —
  the vg scalars do not contain this.
- n-extension on plateau-1 firms the non-stationarity leg but will NOT resolve the ceiling.

## Process note (the recurring failure this session)

Third time a tidy directional verdict over-read a measurement (schedule-*caused* →
early-training-transient; then B-*favoured* → underdetermined). Common cause: attributing
more to an instrument than its construction supports. The design-time fix: ask **"what can
this metric STRUCTURALLY not tell me?"** before running the read — here, "what does an
affine fit of two conditional-mean heads mechanically produce?" would have flagged the
(1,1,0)-by-construction tautology before the ~40 GPU-hours.
