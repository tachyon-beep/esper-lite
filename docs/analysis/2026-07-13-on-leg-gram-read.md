# ON-leg Gram diagnostic (n=2) — PDR-0064 read (A′/B discriminator)

Date: 2026-07-13 · Runs: `telemetry/stage2_on_longdiag/seed{41,42}/telemetry_2026-07-12_161217/`
Commit: `92d998bd` · Completion: both seeds 600/600 rounds, rc=0, 0 crashes.
Reading pre-committed in PDR-0064. Diagnostic (not a gate); rejected HRA re-used as substrate.

## Headline: points to Path B, not A′ (primary discriminator consistent across n=2)

The held-out affine calibration-rescue — the pre-committed A′/B discriminator — does NOT
rescue the total EV on either seed. The best linear combination of the two value heads is
already ≈ `V_total`, so there is no miscalibration to exploit; the cf head's own fit
ceiling caps the total. That is an information limit (B), not a calibration problem (A′).

## The numbers (post-anneal window, rounds ≥250; medians of 250–425 vs 425–600)

### A′/B discriminator — held-out affine calibration-rescue (fit 250–350, test 400–600)
| seed | best-fit map Ĝ = a·V_main + b·V_cf + c | recalibrated held-out EV | raw ev_sum | ev_main | rescue lift |
|------|-----------------------------------------|--------------------------|-----------|---------|-------------|
| 41 | (0.921, 1.000, −0.18) ≈ (1,1,0) | 0.621 | 0.581 | 0.854 | **+0.040** |
| 42 | (1.398, 1.047, +0.20) | 0.610 | 0.563 | 0.835 | **+0.047** |

Read: the optimal affine recombination barely beats the trivial `V_total` (a=b=1, c=0) and
lands ~0.61 — far below `ev_main` ~0.84. The heads are NOT merely miscalibrated; the
cf-side residual is irreducible at the affine level. **Calibration cannot rescue the sum →
B favoured, A′ weakened.** Consistent across both seeds (the robust signal).

### Plateau 2 — cf learner (partially learnable, ceiling ~0.57)
| seed | ev_cf (250–425 → 425–600) | ev_main | ev_sum | Var(e_cf) ratio |
|------|---------------------------|---------|--------|-----------------|
| 41 | 0.591 → 0.596 | 0.82–0.86 | 0.57–0.58 | 0.788 (improving) |
| 42 | 0.548 → 0.575 | 0.78–0.84 | 0.53–0.57 | 1.117 (worsening) |

`ev_cf` ~0.55–0.60 (up from PDR-0060's 0.44–0.54 Q4) but well short of `ev_main` ~0.83;
`ev_sum` (~0.55) is dragged far below `ev_main` by the cf component — the structural fact
that a de-shaping design must address.

### Plateau 1 — cf target-process non-stationarity: SPLIT across seeds
| seed | Var(G_cf) 250–425 → 425–600 | ratio | slope/100 | reading |
|------|-----------------------------|-------|-----------|---------|
| 41 | 601.5 → 510.9 | 0.849 | −53.2 | plateaus/declines (like OFF total) |
| 42 | 419.5 → 503.9 | 1.201 | +40.8 | still DRIFTING UP post-anneal |

The seeds DISAGREE on whether the cf target keeps drifting at flat schedules. Per the
PDR-0064 rule, disagreement on this leg ⇒ extend n to firm it up. This is the one
ambiguity; it is a *secondary* leg (B needs EITHER "calibration can't rescue" OR "cf target
non-stationary" — the calibration leg already holds consistently).

### Plateau 3 — normalizer: running cf scale runs ~15–20% above instantaneous
Both seeds: inst_cf_scale / running `cf_value_target_scale` ≈ 0.81–0.86 — a persistent
mild over-estimate/lag. Real but not large; not the dominant effect.

## Verdict (against the PDR-0064 decision rule)

- **B favoured** — the primary discriminator (held-out calibration-rescue) fails to rescue
  on BOTH seeds (lift +0.040 / +0.047; recalibrated EV ~0.61 ≪ ev_main ~0.84). The cf
  residual is irreducible at the affine level; `V_total` is already near-optimal as a
  linear combination. A primary-`V_total` design (A′) cannot help what is already the best
  linear combination.
- **A′ weakened** (not eliminated) — a NONLINEAR or per-regime recalibration is untested;
  the affine test only refutes the "miscalibrated scale/offset" version of A′.
- **Objective A stays rejected** (PDR-0059) — unchanged.

## Caveats (ranked, session-hardened)

- **n=2.** The calibration read is consistent across both seeds (its strength), but n=2 is
  narrow — the n=1→n=5 scar (PDR-0026) applies. Firm B before a large commitment.
- **Split cf-target plateau (seed 41 plateaus, 42 drifts).** The pre-committed
  disagreement-⇒-extend-n trigger fired on this leg. Does not overturn the calibration
  read but is the natural target if extending n.
- **Affine-only calibration.** Tests the linear/affine A′ hypothesis (the relevant one for
  a calibration term); a nonlinear rescue is out of scope of this read.
- Pooled Gram moments are count-weighted (exact for the per-update population); fit and test
  windows are disjoint (held-out).
