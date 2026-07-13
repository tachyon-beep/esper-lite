# Decision-point diagnosis — calibrated status (gpt-prime synthesis)

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}`. Code + telemetry, no GPU.
Multi-model diagnostic (owner-orchestrated: advisor + ChatGPT + Claude). This file swung
many times; every miss was **a load-bearing input assumed instead of read** (invented `c`,
global epoch counter, unfiltered SET_ALPHA population, two different "contribution" fields,
a bonus computed at c≲1 when actual c≈10, "unpayable" stated as absolute). The reasoning was
consistently fine. The rule: **read the variable in the sentence before you write it.**

## READ B — CORRECTED: absorption/"scaffold retirement" REFUTED; ~30% of HOLDING-prunes discard causally-load-bearing seeds

An earlier version of this section (over-read #6, comforting direction, caught by claudeweb +
the single-seed trace) claimed "productive scaffold retirement": the prune cohort peaked high
then DECAYED to a 0.12 removal-cost as the host absorbed the value, so the gate "selects on the
right signal." **That is refuted.** Two artifacts produced it:

1. **The "0.12" was a `None`-filter artifact.** `SEED_PRUNED.counterfactual` is `None` for ~7/8
   pruned-after-HOLDING seeds (single-seed trace). The "prune cohort median 0.12" was computed
   over the small non-`None` subset — never representative. There is NO gliding decay to 0.12.
   The single-seed traces do not glide; several seeds are pruned at HIGH, RISING decision-time
   c (e.g. 4.6→13.6, 8.9→**23.8**), not decayed.
2. **The ransomware/leak split by `total_improvement` was invalid (over-read #7).**
   `total_improvement = current_val_acc − initial_val_acc` at germination (slot.py:217) — HOST
   PROGRESS over residency, explicitly warned NON-CAUSAL (slot.py:251-256, "conflates host
   training gains with seed impact"). It cannot separate ransomware from leak from anything.

**The clean read — causal decision-time contribution AT the commit/discard decision**
(`counterfactual_contribution` = `seed_contribution` at the FOSSILIZE / PRUNE decision row;
NOT peak, NOT the `None`-riddled event field, NOT `total_improvement`), seeds 41/42:

| decision | n | median c | mean | %≥5 | %1–5 | %0–1 | %<0 |
|---|---|---|---|---|---|---|---|
| **FOSSILIZE** (commit) | 2043 / 2108 | **10.4** | 11.2 | **66%** | 12% | 12% | 10% |
| **PRUNE-after-HOLD** (discard) | 11.4k / 12.3k | **0.84** | 4.3 | **30%** | 18% | 30% | 22% |

Reading, calibrated both ways:
- **FOSSILIZE selection is genuinely positive** — Tamiyo commits strongly, causally
  load-bearing seeds (median c≈10, 66% ≥5, only ~10% net-negative). Not inverted, not a defect.
- **Prune-after-HOLDING is MAJORITY-correct by the causal metric** — median c≈0.8; ~51% are
  <1 (spent) including ~22% net-negative (harmful). The CENTER of the cohort is correct retirement.
- **BUT ~30% (~3,600/run) of HOLDING-seed prunes discard a seed causally contributing ≥5
  accuracy points** — removing it costs ≥5 pts AT the prune decision. This is claudeweb's
  "load-bearing discard," now on a causal metric that survives scrutiny. It is real, substantial,
  and NOT the wholesale inversion (30%, not 68%) and NOT explained away by absorption.

**Cause of the ~30% tail is UNRESOLVED and NOT cleanly measurable with current fields.**
Candidate benign explanations — ransomware/dependency (high self-LOO, net-harmful),
param-budget/compute pressure (freeing a slot for a better seed), turntable-retirement — cannot
be separated from genuine waste because the only "net ensemble value" field (`total_improvement`)
is non-causal. A CAUSAL ransomware/leak discriminator does not currently exist in telemetry;
building one is the prerequisite to pricing this tail. Do NOT bank the tail as a defect OR as
benign.

## The calibrated diagnosis (authoritative)

> Tamiyo preferentially fossilises seeds that are currently important to the forward network
> (terminal-current-LOO selection is strongly POSITIVE, not inverted). But current LOO is NOT
> a measure of historical developmental value, so whether the seeds she later prunes were
> failed experiments or successful developmental modulators is UNRESOLVED. Re-blending after
> HOLDING is common; whether it is productive turntabling, recovery, or waste is UNMEASURED
> because HOLDING-origin excursions and their downstream effects have not been isolated.

## BANK (solid)

- **Terminal current-LOO selection is strongly positive, not inverted.** At the causal
  decision moment: FOSSILIZE median c≈10.4 (66% ≥5, ~10% <0) vs PRUNE-after-HOLD median c≈0.84
  (51% <1, ~22% <0). P(FOSSILIZE|c≥1)=62–63%, |c≥5=80–82%, |0≤c<1=12–13%. n=2.
  (NOTE: the old "prune cohort 0.12" was a `SEED_PRUNED.counterfactual` `None`-filter artifact —
  that field is `None` ~7/8 of the time; use decision-time `seed_contribution`, median 0.84.)
- **~30% of prune-after-HOLDING decisions discard a seed causally contributing ≥5 acc-pts.**
  Real on the causal metric; cause (ransomware / budget-pressure / waste) UNRESOLVED because no
  causal net-ensemble-value field exists (`total_improvement` is host-progress, non-causal).
- Current selection is IMPERFECT: substantial above-threshold non-fossilisation and some
  negative-current-LOO fossilisation both exist.
- Re-blending after HOLDING is common (75%); most re-blenders never fossilise (88%).
- Raw germinate/prune churn is largely EXPLORATION, not farming (74k TRAINING-prunes at α≈0).
- **Current LOO (`seed_contribution`) cannot measure historical modulation or seed→seed
  transfer** — a Phase-0-documented limitation, now operational not theoretical.
- Field identity (Read A): `seed_contribution` = LOO marginal in **accuracy points** (0–100).
  Fossilize bonus = `0.5 + 0.1·c` on RAW c → **~1.5 for c≈10** (NOT ≤0.6). `bounded_attribution`
  = discounted `sqrt(progress·c)`-type function × attribution_discount × timing_discount — NOT
  raw c. Decision surface ~43% free; α-throttle refuted; Objective-A rejected, PPO baseline is
  op-independent `V(s)`.

## DO NOT BANK (over-reads, incl. mine this round)

- "38% of above-threshold seeds pruned = a 38% product-quality failure" — it's
  above-threshold-current-LOO non-fossilisation; can include ransomware prunes (positive
  self-LOO, negative ensemble), field-mismatch, deliberate post-help removal, stale telemetry.
- "185–228 negative-LOO fossils = certainly harmful commits" — correct label is
  **negative-current-LOO fossilisation**; a seed can have negative instantaneous LOO yet be a
  developmental enabler / synergistic / noisy. Needs the per-cohort reward + downstream read.
- **(#6, comforting direction) "Productive scaffold retirement / gate selects on the right
  signal" — REFUTED.** Rested on the "0.12 decay" (a `None`-filter artifact) and a two-point
  interpolation across the prune boundary. The lesson repeated in the reassuring direction: a
  tidy benign story is as much a red flag as a tidy alarming one.
- **(#7) The ransomware-vs-leak split by `total_improvement` — INVALID.** `total_improvement`
  is `current_val_acc − initial_val_acc` (host progress since germination), explicitly non-causal
  (slot.py:251-256). Read as "seed's net help to the ensemble" it manufactured a false 68%/32%
  split. Any ransomware/leak claim needs a CAUSAL discriminator, which telemetry lacks today.
- "Re-blending is mostly loitering" AND "mostly useful turntabling" — both unresolved.
- "The successful modulator receives NO temporal credit" (my over-claim) — TOO STRONG. Direct
  per-seed LOO doesn't pay historical modulation, but an **indirect RL channel exists**:
  upstream action → future downstream/terminal reward → GAE return assigned to the earlier
  action. It is indirect, shared, delayed, discounted, high-variance, and weakened by
  truncation/bootstrapping — NOT absent. An explicit scaffold-hindsight term may still help,
  but the system is not mathematically incapable of learning turntabling. (`hindsight_credit`
  fires 0.005% of the time — so the EXPLICIT settlement channel is de facto inert, which is the
  real gap; ordinary future-return credit remains.)
- Any fixed multiplier for the commitment economics; any constant-`c` crossover.

## Turntabling (owner domain knowledge) + the measured screen

Turntabling is intended design: a HOLDING seed drops to partial α to modulate a downstream
seed's blend, then ramps back to 1.0 and may fossilise; some seeds exist only to modulate. So
the HOLDING excursion may be designed modulation, not loiter. **Effect-size screen
(observational, seed 41):** turntabling is RARE (2% of decisions have ≥2 active seeds — masks
enforce sequential dev); modulated fossils are NOT higher-LOO (co-resident 7.4 / upstream 2.3
vs solo 11.6 / none 11.9) — but this is confounded by LOO-dilution AND LOO's blindness to
transfer, so it is a **weak null, not a refutation**. Clean test = a scripted turntable arm
measuring the DOWNSTREAM seed's outcome.

## The decisive zero-GPU packet (gpt-prime), then stop theorising

- **A. Field/unit identity** — DONE (above): contribution in accuracy points; bonus on raw c.
  Still open: whether extreme c (±45) is a meaningful marginal or an off-manifold break;
  confirm event `counterfactual` == decision-time `seed_contribution`.
- **B. First-HOLDING-to-fate trajectories** — per seed: c at first HOLDING, peak c, c at each
  excursion, c at fate, ∫bounded_attribution after first HOLDING, time-remaining, re-blend
  count/type, fate, ransomware/auto/scheduled status. **Decides whether the 0.12 prune cohort
  was ALWAYS marginal or DECAYED after being useful** — the load-bearing unknown.
- **C. Turntable exposure vs outcome** — downstream seeds' outcomes by upstream partial-α
  exposure, matched on slot/blueprint/round/host-acc/param-budget. Positive association →
  licenses a scripted causal arm.
- **D. Reward settlement paycheck** — hindsight_credit & synergy_bonus by fate; cumulative
  reward after first HOLDING by fate; bonus/warnings/shocks/costs/rent.

## Decision table (after B–D)

| Finding | Implication |
|---|---|
| Prune cohort low FROM first HOLDING | marginal-seed exploration/recovery (benign) |
| Prune cohort high early, downstream/host benefit retained | productive scaffolding; LOO under-credits history |
| Prune cohort high early, no retained value, large reward accrued | attribution/timing problem |
| Turntable-exposed downstream materially outperform controls | build explicit history-aware scaffold credit |
| Turntable exposure no outcome association | close the modulation-credit branch |
| Same-target/no-downstream excursions common | action-hygiene fix |
| Negative-current-LOO fossils lack future benefit | commitment-gate defect |
| hindsight_credit never pays PRUNE | explicit settlement gap |

## Product baseline

Morphogenesis-improves-a-host is NOT in doubt (established, incl. degraded hosts). The open
product comparison is **Tamiyo's learned policy vs a competent scripted turntable/scaffold
controller**, primary metric terminal accuracy contribution under parameter + developmental
-compute cost. all-disabled (resident-seed effect, host learning retained) ≠ host-only control
(total developmental effect); both useful.
