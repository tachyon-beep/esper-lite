# Decision-point diagnosis (CORRECTED after cohort read)

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}` (n=2). Code + telemetry, no GPU.
**This supersedes an earlier version of this file that claimed "the reward pays Tamiyo not
to commit" and "re-blend dominates fossilize ~50×." The cohort read below refutes both.**
5th over-read of the session; the data caught it.

## The decisive measurement: the commit gate selects by contribution, CORRECTLY

For seeds that reached HOLDING, split by terminal fate, median counterfactual contribution:

| cohort | median contribution | mean | n |
|--------|---------------------|------|---|
| **FOSSILIZE** | **10.8 / 10.0** (s41/s42) | ~11 | 2,043 / 2,108 |
| **PRUNE-after-HOLDING** | **0.12 / 0.12** | ~0.8 | 3,757 / 3,849 |

Fossils carry ~**90× the contribution** of the loiter-then-prune cohort, on both seeds.
`DEFAULT_MIN_FOSSILIZE_CONTRIBUTION = 1.0` is the gate: below ~1.0 a seed earns the
`fossilize_noncontributing_penalty` (−0.2), not the bonus. Median HOLDING-reacher contributes
0.6 (below threshold); the top ~35% (above it) are what commit. **Tamiyo commits the good
seeds and prunes the weak ones.** The selection-inversion hypothesis (reward commits junk,
loiters the good) is FALSIFIED.

## What this retracts

- **"The reward pays Tamiyo not to commit" — RETRACTED.** Good seeds (contribution ~11) commit;
  most on first reaching HOLDING (74%). Low-contribution seeds (~0.12) loiter and prune.
- **"Re-blend dominates fossilize ~50×" — RETRACTED.** That used an invented `c=0.5` and,
  worse, a **constant-`c`** model. Contribution is hump-shaped and DECAYS as the host absorbs a
  seed (a seed at 0.12 by prune may have been high earlier). A decaying stream is not the annuity
  the arithmetic assumed, and the behavior proves it: a constant-`c` model says a c=10 seed should
  loiter forever, yet c≈10 seeds overwhelmingly commit. The crossover the peer proposed
  (`T* = (0.5+0.1c)/(c+0.002)`) is directionally better but still assumes constant `c`; do not bank
  a magnitude without modeling decay/freshness and reading per-step `bounded_attribution`.
- **The time-remaining test is UNRUN, not passed** — the epoch field used was the global counter,
  not the within-episode host epoch; results were garbage and are discarded.

## What SURVIVES (do not over-correct the other way)

- **The re-blend loop is real** (75% of HOLDING-reachers re-blend ≥1×; up to 7 cycles; 88% of
  re-blenders never commit) — but it is **concentrated on low/marginal-contribution seeds**
  (the never-committers median 0.12), not on the winners. It is mild waste on marginal seeds,
  NOT commitment-avoidance on good ones.
- **The anti-loiter guards don't see the re-blend** (structural): `alpha_shock` sees Δα≈0 on a
  re-blend that returns to the same amplitude; `holding_warning` resets on the stage exit;
  `set_alpha_target_cost` is only −0.005. This action-hygiene gap is genuine — BUT its blast
  radius is smaller than claimed (it mostly lets marginal seeds cycle before pruning), and full
  nullity is NOT yet verified: `slot.py:1837` says partial targets re-enter BLENDING, so a
  re-blend may involve a transient partial-target excursion. Verify nullity over
  target/style/speed/curve/output before banking it as a pure no-op.

## The scoreboard — corrected interpretation

Median episode: `num_contributing_fossilized` = 0, `param_ratio` 1.03, final acc 52.6%. **This is
NOT "ships nothing":** (a) at <1 fossil/episode the median is 0 by construction; (b)
`num_contributing_fossilized` is structurally 0 for fossils that leave the LOO ablation population
(contribution → None), a retired-metric artifact (see the Committed-J retirement, PDR-0027 arc);
(c) transient scaffolds can improve the host and be correctly shipped near-bare. The low commit
rate reflects a **low yield of above-threshold seeds** (median contribution 0.6 < 1.0), not
avoidance of committing good ones. The right product read is all-on vs all-off / host-alone
terminal accuracy + footprint — needs an all-off arm, not in this run.

## Corrections to the record that DO stand (from the earlier reads, unaffected)

- PDR-0026's null was NOT the mask topology — decision surface is ~43% free.
- The germinate/prune VOLUME is exploration, not farming (74k TRAINING-prunes at α≈0).
- The α-throttle is refuted (all commits at α≈1.0; partial-fossil is a latent bug, not live).

## Not banked

`re-blend always beats fossilise`; `the policy commits worse seeds` (FALSIFIED — it commits
better ones); `the reward is the sole cause of every re-blind`; `zero contributing fossils =
no useful result`; any crossover magnitude, pending a decay-aware model on measured units.

## Honest residual — what's actually still open

1. Is the loitering on marginal seeds costly enough to matter (attribution paid + delayed prune)?
   Unmeasured.
2. The full reward economics with contribution DECAY (the missing variable in every version of
   the arithmetic so far).
3. Time-remaining-at-HOLDING by fate (re-run with the within-episode epoch).
4. Full re-blend nullity over the whole schedule (style/speed/curve/output), for the
   action-hygiene claim.
5. The real product read (all-on vs all-off), which needs a new arm.
