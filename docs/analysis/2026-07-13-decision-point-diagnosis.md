# Decision-point diagnosis (CALIBRATED — retract the claim, not the argument)

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}` (n=2). Code + telemetry, no GPU.
This file has now swung twice — over-claimed ("50×, the reward pays Tamiyo not to commit"),
then over-retracted ("the commit gate selects correctly"). Both wrong. Calibration fails in
BOTH directions; after a hard hit, "I was wrong about everything" is as costly a distortion
as the tidy verdict, because it discards true findings. The rule that catches both:
**retract the CLAIM, not the ARGUMENT.**

## The commit gate is PARTIALLY correct — and leaks both ways

Split HOLDING-reachers by terminal fate and terminal counterfactual contribution:

| | seed 41 | seed 42 |
|---|---|---|
| P(FOSSILIZE \| contribution ≥ 1.0) | **62%** | **63%** |
| P(FOSSILIZE \| contribution ≥ 5.0) | 80% | 82% |
| **good-at-terminal (≥1.0) seeds PRUNED** | **977** | **958** |
| P(FOSSILIZE \| 0 ≤ c < 1.0) | 13% | 12% |
| **NEGATIVE-contribution seeds FOSSILIZED (committed harm)** | **185** | **228** |

So the gate mostly commits good seeds (62–80%) and mostly prunes weak ones (13% commit
below threshold) — it is NOT inverted (the peer's selection-inversion is still refuted). BUT
it **leaks ~38% of still-good seeds to pruning** and **commits ~200 actively-harmful seeds**.
The cohort medians (fossils 10.8, prunes 0.12) hid both leaks in the tails. My "selects
correctly / good seeds commit" headline over-corrected.

## Retract the claim, keep the argument

- **RETRACTED (claim, invented input):** "re-blend beats fossilize ~50×." Used a guessed
  `c=0.5` and a constant-`c` model; contribution decays, so the income integral is not `c·T`.
  No magnitude is bankable without the decay curve.
- **NOT RETRACTED (argument, a fact about the equations):** the fossilize bonus is a **one-shot**
  `(0.5 + 0.1c)`; pre-commit attribution is an **income stream** `≈ (decaying c)` per step;
  the two anti-loiter guards are structurally blind to a null-return re-blend (`alpha_shock`
  Δα≈0, `holding_warning` resets on stage exit) at cost −0.005. A one-shot cannot generally
  match an integral. No cohort read touches this. Its *bite* depends on the decay curve
  (unmeasured) — and the Query-1 leak (38% of good seeds pruned) is exactly the footprint an
  incentive-to-delay would leave.

## What survives (n=2, both seeds)

- Selection-inversion REFUTED (fossils high-c, prunes low-c on the median).
- The re-blend loop is real (75% re-blend, ≤7 cycles, 88% of re-blenders never commit) and
  concentrated on low/marginal seeds — but Query 1 shows it (or the same economics) also
  costs ~38% of good seeds.
- Two candidate live defects: **good seeds pruned** (~1000/seed) and **harmful seeds committed**
  (~200/seed), pending the caveats below.

## Structural limits (what these metrics CANNOT tell me — stated before banking)

1. Terminal counterfactual is **post-decay** — catches "pruned a still-good seed," misses
   "loitered a good seed until it decayed, then pruned" (hides as a correct prune). So the 38%
   leak is a LOWER bound on good-seed loss.
2. `SEED_{FOSSILIZED,PRUNED}.counterfactual` may not equal the reward's decision-time
   `seed_contribution` — the "committed harm" and "pruned good" counts need the field identity
   confirmed before banking as defects.
3. **Ransomware prunes** (high counterfactual + negative total_improvement) are *legitimate*
   and inflate "good seeds pruned"; separate them before pricing the leak.

## Corrections that stand (earlier reads, unaffected)

PDR-0026's null ≠ mask topology (43% free surface); the germinate/prune VOLUME is exploration
(74k TRAINING-prunes at α≈0); the α-throttle is refuted (all commits at α≈1.0). Scoreboard:
median-0 fossils is `<1/episode` + LOO-population artifact, not "ships nothing."

## Not banked

`50× / any crossover magnitude` (needs decay curve); `the gate is clean` (it leaks ~38% good +
~200 harmful); `committed harm / pruned good are certain defects` (pending caveat 2/3);
`the reward is the sole cause of the re-blend`.

## Remaining reads that settle it (all free, in order)

3. **Units of `seed_contribution`, then per-step `bounded_attribution` traces for the fossilize
   vs never-commit cohorts** → the **decay curve** → the pricing stops being a debate (measured,
   not assumed). This is the one that turns the surviving argument into a number.
4. Separate ransomware/age/scheduled prunes from the "good seeds pruned" count.
5. Confirm the counterfactual field == decision-time contribution (caveat 2).
6. Time-remaining-by-fate, re-run with the within-episode epoch.
7. Full re-blend nullity over style/speed/curve/output (action-hygiene claim).
