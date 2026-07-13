# The decision-point diagnosis: the reward pays Tamiyo NOT to commit

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}` (n=2, both confirm). Code + telemetry,
no GPU. Supersedes the EV-stabilization framing of the low-commitment problem.

## The finding, localized to one transition

Everything about the "churn" that looked pathological is either healthy or refuted (below).
What survives is a single transition: **at HOLDING (the fossilize-eligible state), seeds
re-blend 3:1 over committing, and re-blending is loitering, not retuning.**

### Behavioral discriminator (per-seed, both seeds)
| | seed 41 | seed 42 |
|---|---|---|
| seeds reaching HOLDING | 6,489 | 6,672 |
| …that EVER fossilize | 31% | 32% |
| …that re-blend ≥1× | 75% | 76% |
| **of re-blenders, that EVER fossilize** | **11%** | **12%** |
| re-blend cycles/seed (mode; max) | 1; 7 | 1; 7 |
| ≥3 re-blend cycles (loiter tail) | 6% | 8% |

**Re-blending is a null-target action.** Every HOLDING entry and every fossilize is at
**α≈1.0** (8,923/8,923; 2,043/2,043) — so a re-blend from HOLDING returns to HOLDING at the
*same* α=1.0 it already had. The target never changes; the only effect of the re-blend is to
exit HOLDING and reset the `holding_warning` penalty. **88% of re-blenders never commit.** By
the user's own dispositive criterion (re-blend to the target you're already at = loiter), this
is loitering. Seeds that DO commit (~2/3 of the ~2,043 fossils) mostly commit on first reaching
HOLDING, without re-blending.

## The root: the reward *pays* for the loop (arithmetic, from config)

For a seed in HOLDING at contribution `c` with `T` epochs left in the episode
(`contribution_weight=1.0`, `fossilize_base_bonus=0.5`, `fossilize_contribution_scale=0.1`,
`fossilized_maintenance_cost=0.002`, `holding_warning` 0.1–0.3/epoch but **resets on exit**):

| Choice | Payoff |
|--------|--------|
| **FOSSILIZE now** | one-shot `(0.5 + 0.1c)·legitimacy ≤ 0.6`, then **income → 0** and **−0.002·T** rent forever; tiny terminal-acc share |
| **RE-BLEND (loop)** | `≈ c` per step × `T` attribution, **no rent**, penalty reset each cycle |

Worked: c=0.5, T=50 → **fossilize ≈ 0.45** vs **re-blend ≈ 25** (~50× for re-blend). Even
c=0.1, T=20 → 0.47 vs 2.0 (~4×). **Re-blend dominates fossilize for every plausible (c, T).**
The reward is not merely permitting the loop — it is the optimal policy for the objective as
written. This is December 2025's `+0.8 vs +28` (`fossilize-incentive-fix.md`) in exact
structural form; the P0 fix (2026-01-11) changed *when* the bonus is paid, not *what it is
proportional to* — a one-shot cannot outbid an integral over the remaining horizon.

**Coefficient or redesign?** Not "slightly short" (50×), so no coefficient on the current FORM
closes it (the gap scales with `T`). But the structural fix can be small: the exact lever is
the `not seed_is_fossilized` guard in `contribution.py` that zeros post-fossilize attribution —
paying a post-commit attribution drip (the mechanism `BASIC_PLUS` already implements) or pricing
the bonus as `c·T_remaining` would flip the inequality.

## The scoreboard (the product number the program never had)

Per episode (n=7,200): **median `num_contributing_fossilized` = 0**, **`param_ratio` = 1.03**,
final val-acc median 52.6% [19–65]. The median episode commits nothing and ships near-bare
metal. Whatever morphogenesis contributes arrives via **transient blending** it then discards.
Committed-capability-at-episode-end is the metric everything downstream should be scored on —
NOT `ev_sum`/`ev_cf`. (A clean vs-baseline read needs an all-off arm; not in this run.)

## Corrections to the record (owed)

- **PDR-0026's null was NOT the mask topology.** The decision surface is ~43% free
  (`forced_step_ratio` median 0.57, decision_density 0.43, ~773 usable actor-timesteps/update)
  — large, not a sliver. Whatever bounds coverage, it isn't the masks.
- **The germinate/prune volume is exploration, not farming.** 74,469 TRAINING→PRUNED at α≈0 —
  culled before entering the forward pass or earning attribution. Germinate-and-screen working
  roughly as designed. The pathology is NOT in the churn.
- **The α-throttle is refuted.** All commits at α≈1.0 despite the code permitting 0.5; the
  half-amplitude-fossil concern is a latent bug, not a live mechanism.

## Epilogue on the critic epic

`ev_cf ≈ 0.57` was a critic accurately fitting a reward stream (`bounded_attribution`) that pays
for host *dependence* and *delay*. 66 PDRs, ~40 GPU-hours, and three over-reads were all
downstream of a target nobody had audited. The instrument-validity lesson has a bigger sibling:
**audit what you are paying for before optimizing how well you predict it.**
