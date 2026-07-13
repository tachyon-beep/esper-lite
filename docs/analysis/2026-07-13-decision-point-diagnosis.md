# Decision-point diagnosis (read-the-variable pass)

Date: 2026-07-13 · Runs: `stage2_on_longdiag/seed{41,42}` (READ 3 & the reward_components
reads are seed 41; Q1/Q2 behaviour is n=2). Code + telemetry, no GPU.

**Meta (the one durable lesson): every over-read AND over-retraction this session —
five of them — was a load-bearing input ASSUMED, not read.** `c=0.5`, the global epoch
counter, the unfiltered SET_ALPHA population, `num_contributing_fossilized`, and comparing
two different "contribution" fields. Never the reasoning. This pass reads the variables from
`ANALYTICS_SNAPSHOT.reward_components` (which carries decision-time `seed_contribution`,
`bounded_attribution`, `seed_stage`, `hindsight_credit`, within-episode `epoch` — it was in
the schema the whole time).

## Read 1 — `hindsight_credit` is inert (NEW solid finding)

Non-zero `hindsight_credit` fires on **58 of ~1,079,932 decisions (0.005%)**, spread across
ALL actions (WAIT 29, GERMINATE 5, PRUNE 11, SET_ALPHA 11, FOSSILIZE 2). The scaffold-
retroactive-credit mechanism is **effectively dead**. A seed that germinates, teaches the
host, is absorbed, and is correctly pruned receives its residency attribution but ~zero
credit for the durable value it locked in. This is a strong candidate cause for any
"good seeds pruned for nothing" pattern — and it is unrelated to the one-shot/integral argument.

## Read 3 — the re-blend loop is real and HOLDING-originated (Q3 "refuted" RETRACTED)

`SET_ALPHA_TARGET` conditioned on `seed_stage == 6` (HOLDING), the CORRECT population
(n=6,429): targets chosen 0.5 (40%) / 0.7 (44%) / 1.0 (17%). HOLDING seeds overwhelmingly set
**partial** targets, triggering re-entry to BLENDING (`slot.py:1837`). My prior "re-blends are
real retunes, loophole refuted" used all 34,811 SET_ALPHA decisions (mostly BLENDING ramp
tweaks) — wrong population, retracted. Note also `alpha_shock` at the SET_ALPHA decision is
~−1.5e-5 (negligible; the SLOW ramp — 70% of speeds — keeps per-step Δα tiny), and
`holding_warning` resets on the stage exit. So the guards-evasion concern is **supported, not
refuted**; whether the amplitude cycle (1.0→partial→1.0, all HOLDING *entries* still α=1.0) is
gaming vs legitimate re-tuning is not yet settled.

## Reopened — the commit gate does NOT cleanly select by the reward's contribution

Using **decision-time** `seed_contribution` from `reward_components`:

| cohort (from HOLDING) | median seed_contribution | ≥1.0 | n |
|---|---|---|---|
| FOSSILIZE | 10.79 | 78% | 2,043 |
| PRUNE-from-HOLDING | **10.14** | 76% | 1,434 |

By the reward's own contribution measure the two cohorts are **indistinguishable** (~10, both
~77% above the 1.0 threshold). My earlier "the gate selects correctly (fossils 10.8 vs prunes
0.12)" **conflated two fields**: nominal `seed_contribution` (≈10) vs `SEED_PRUNED.counterfactual`
(≈0.12, a *cost-to-remove* measure). They diverge ~85×. So both my "selects correctly" AND the
peer's "inverts selection" are **unestablished** — the gate is roughly *indifferent* to the
reward's contribution at the commit choice. The two readings imply different worlds:
- prune-cohort seeds are **redundant scaffolds** (nominal 10, cost-to-remove 0.12 → host
  absorbed them → pruning is CORRECT, and attribution was over-paid on the nominal figure), OR
- they are **reward-good seeds being pruned** (a real leak).
Disambiguating needs: does `bounded_attribution` track nominal contribution even after the host
has absorbed the seed? (The reward formula pays on nominal `seed_contribution`, so provisionally
yes — the "over-pay for absorbed contribution" hypothesis — but confirm per-seed.)

## Read 4 — 185 harmful fossils confirmed (solid)

185 seeds fossilized at **negative** decision-time `seed_contribution` (sample −0.36…−2.04),
median `total_reward` −0.57 — the reward PENALIZED the commit (`action_shaping` −0.5…−0.9) and
the policy committed anyway. Permanent, at full param rent, in the shipping product. Survives
every mechanism story.

## Read 2 — attribution income is ~flat ≈3/step (population, not per-seed decay)

`bounded_attribution` by within-episode epoch: 0.11 (ep 0–15) → rises to ~3.2 (ep 45–60) →
flat ~3.0 to episode end. So per-step attribution is substantial and sustained at the
population level (3/step ≫ the ~0.5 one-shot fossilize bonus). This does NOT give the per-seed
decay curve (population conflates seed-ages); the true decay curve needs per-seed age tracking
through `reward_components` — the remaining open read.

## Status — solid vs open (no theory banked as headline)

SOLID: hindsight_credit inert (0.005%); HOLDING sets partial targets 83% (re-blend real);
185 harmful fossils; the two "contribution" fields diverge ~85× so the earlier cohort verdict
is void; attribution ~3/step sustained. Earlier corrections that stand: PDR-0026 null ≠ mask
topology; volume is exploration; α-throttle refuted at HOLDING *entry*.

OPEN (do these deliberately, then write the way-forward): (1) per-seed decay curve + whether
attribution is paid on nominal contribution after absorption (the over-pay hypothesis);
(2) disambiguate redundant-scaffold vs leaked-good-seed for the prune-from-HOLDING cohort
using cost-to-remove; (3) what paid for the 185 harmful fossils (per-component); (4) real
product read (all-on vs all-off). Anchor the way-forward on the two things that survive every
variable: **hindsight_credit is inert** and **185 harmful commits ship**.
