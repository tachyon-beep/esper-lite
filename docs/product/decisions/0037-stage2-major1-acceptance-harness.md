# PDR-0037 — Stage-2 MAJOR-1 acceptance harness: designed, scorer built + review-hardened, NOT-yet-freezable

Date: 2026-07-06   Status: accepted (within grant: spec / dispatch / build acceptance code for the active bet)
Author: Claude (agent)   Related: PDR-0029 (MAJOR-1/-3 folded into Stage-2 acceptance),
task esper-lite-2a4b56e719, epic esper-lite-f25b71c165.

## Context

Stage-2 HRA is **code-complete for a valid ON run** — verified against current code (not
the 12-day-stale memory): the ON-leg cf value + reward streams are wired end-to-end
(`vectorized_trainer.py:2092` → `rollout_buffer.py` per-stream GAE → `ppo_agent.py`
per-stream EV), with a loud guard against the "cf trained on zeros" failure the memory
feared. So Stage-2 is an **acceptance** task, not a finish-the-build task. Acceptance
needs a **falsifiable, pre-registered gate BEFORE any A/B** (MAJOR-1 `ev_sum` hard floor +
downstream non-hollow signal + MAJOR-3 provenance, per PDR-0029) — otherwise `EV_main`
improvement is trivially over-read (`V_main` regresses a definitionally-easier target).

## Options considered

1. **Design + build the pre-registered acceptance harness now; freeze before the A/B.** —
   pro: pre-registration is the only thing that makes the A/B falsifiable; catches the
   hollow win by contract. con: upfront work before any empirical result.
2. Run the paired A/B first, judge `EV_main` post-hoc. — pro: fast. con: exactly the
   build-trap / over-read failure the whole epic is trying to avoid; rejected.

## The call

**(1).** This session: (a) designed the MAJOR-1 rubric via `drl-expert`; (b) merged an
independent external (ChatGPT) critique under a take-good/dump-bad filter — **adopted 4**
(explicit actor-objective declaration A-vs-B, a validity-gate first layer, per-stream
target-scale diagnostics, churn/guard-channel safety gates) and **rejected 2** on
code/algebra grounds (advantage-std *level* downstream signal — algebraically entangled
with the ev_sum floor; paired-ΔEV_main primary gate — `ev_main` is ON-leg-only, no
`EV_main_OFF` exists); (c) wrote the frozen-candidate gate doc
`docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md`; (d) TDD-built the **pure
scorer** `src/esper/simic/telemetry/stage2_acceptance.py` (42 tests, ruff + mypy clean);
(e) ran two adversarial reviews.

- **python-code-reviewer:** 3 critical + 4 warnings — ALL FIXED (tri-state composite bug,
  n/length validation, empty-corpus guard, NaN finiteness, 2^m enumeration cap, typed rng).
- **drl-expert:** independently VERIFIED the core (exact Wilcoxon vs a subset-sum-DP
  reference; the §4 identity `pre_norm_advantage_std² = (1−ev_sum)·Var(returns)` traced
  end-to-end; §8a/b comparability; canonical hollow win caught). Found a **compound
  ACCEPT route** (H2 tier-blindness + H3 mech absolute-count + M1 differential-flooring).
  H2/H3/M3 FIXED in the scorer (tier-aware composite with `SCREEN_PASS`; mech `≥⌈0.8n⌉`;
  required G3/G4). H1/M1/M2/L1 + a citation SYNCED into the doc.

Delivery landed on `feat/ev-stab-stage2-hra` @ `2fad1fc4` (3 files, 42 tests green).

## Rationale

Pre-registration before data is the whole point of a falsifiable acceptance gate; the
scorer freezes the statistics, the doc freezes the criteria. The dangerous failure mode
here is emergent (no single gate broke; three near-misses compounded) — caught only by an
*adversarial* review of the composite path, now closed/documented. The core math is
independently verified, so confidence in the statistics is high.

## The NOT-yet-freezable boundary (drl-expert H1)

Only the **pure statistical predicates** are built + tested. The **telemetry wrapper does
not exist yet** and owns the other half of the gate: §1 validity gates, §0 provenance
block, burn-in `W` discard, §8B floored-update exclusion (a SCORED precondition — the
differential-flooring M1 route), §8E `Var(returns)`-volatility covariate (M2), G3/G4
computation, and the tier-gated packet CLI + §9 diagnostic scalars. **The gate doc is
marked NOT-FREEZABLE until that wrapper exists and its tests encode those predicates.**

## Reversal trigger

- If the telemetry-wrapper build shows the pure-scorer / wrapper predicate split is
  unworkable, or the §8B floored-exclusion cannot be computed from emitted telemetry →
  revisit the harness design before freeze.
- If a Stage-2 ON run (post-freeze) shows `ev_sum` degrade below the frozen δ floor →
  Stage-2 acceptance FAILS (MAJOR-1 hard override), regardless of `ev_main`.
- If the `cifar_baseline` `ev_return_variance` distribution sits near the 1.0 EV floor,
  the M1 differential-flooring risk is live → §8B exclusion is load-bearing and must gate
  LEG-A/MECH before any reading is trusted.
