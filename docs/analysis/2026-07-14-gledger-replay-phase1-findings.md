# G-LEDGER executable replay — Phase 1 findings

Date: 2026-07-14. Freeze blocker #3 (PDR-0083 A6 / PDR-0084 A7), Phase 1 scope: S/A
comparator paths on shipping code, PBRS whole-path telescoping, threshold-variance
formula test, park-vs-fossil residual. Zero-GPU, test-only; no production code touched.
Tracker: esper-lite-e60c03cf28. Phase 2 (B path) remains owner-gated.

**Instrument:** `tests/simic/rewards/replay/` — a path driver that feeds synthetic
single-seed trajectories through the PRODUCTION `compute_reward` dispatch and produces
per-component discounted-PV ledgers. The driver's SeedInfo sequencing is pinned against
the real kasmina state-machine MECHANICS (`test_kasmina_fidelity.py`); the trainer
phase ORDERING itself rests on the code citations in R1 — mirrored by both driver and
walker, so not proven by that test — and was independently re-verified in review plus
confirmed empirically by the r9 population signature below. Two timing conventions are
modeled, because they disagree — and that disagreement is the headline finding.

**Review:** drl-expert **APPROVE-WITH-NITS** (2026-07-14); all five nits closed same
day (fidelity-scope wording, production-derived PRUNE cull with fail-loud desync
detector, proxy-path coverage added for the stale-epoch case, constants-pin test, EWMA
convention note). The reviewer independently re-verified the R1 ordering chain and the
FOSSILIZE-synchronicity premise (`handlers/fossilize.py:192` → `slot.py:1543`, inline
transition) against production. Reviewer caveat, carried honestly: the r9 population
numbers below were produced by this session's scans and were not independently re-run
by the reviewer.

## R1 — The live epoch ordering skips EVERY cross-stage PBRS transition delta

Code chain (each link cited in the harness docstrings): `record_accuracy` ticks
`epochs_in_current_stage` for every slot with state during the metrics phase
(`vectorized_trainer.py:1870-1875`), the reward is computed on the pre-action state
(`action_execution.py:226-230, :973-977, :1089`), the mutation and `step_epoch` run
after (`action_execution.py` dispatch phase, `:1643`), and `transition()` zeroes the
stage clock (`slot.py:520-531`). The first post-transition reward call therefore sees
`epochs_in_stage == 1`, and the PBRS `eis == 0` branch (`contribution.py:1235-1246`)
— the ONLY place cross-stage potential deltas pay, and per its own warning message the
designed-for case — is **unreachable for every transition type on the live path**.

**Empirical confirmation (sealed r9, BOTH seeds, 2,159,882 decisions):** across 4,151
FOSSILIZE commits, **zero** `pbrs_bonus` readings in the forfeiture band [−0.50, −0.25]
(the audit-convention prediction is one per commit). The fossil-stage histogram is
exactly the live-convention signature: −0.012 × 92,782/96,597 rows (s41/s42 — the
post-cap drag `0.3·(0.995·8.0 − 8.0)`) and +0.078…0.081 climb steps.

Consequences:
- **Audit F5 is corrected:** the dwell-graded commit-time PBRS forfeiture
  (−0.31…−0.46, PDR-0082 D2, priced into the ledger and Δ_designed) **never fires**.
  Its "economic classification OPEN" question (A2, PDR-0083 #3) is settled: not a
  commit-time penalty; not a telescoping refund either — the delta is simply skipped.
- **PBRS stage-progression shaping is largely inert as designed:** stage-entry bonuses
  pay only through GERMINATE's action-embedded bonus and PRUNE's action-embedded cull;
  ADVANCE/auto-forward/FOSSILIZE entries pay nothing. The Ng telescoping identity holds
  only WITHIN stages (verified to 1e-9 by closed form); whole-path invariance does not
  hold as designed.
- Matched-control across arms A/B/C is NOT violated — all arms share the trainer
  ordering — but Δ_designed and G-MATCHED-CONTROL's "PBRS semantics" leg must price
  the skip-semantics, not the designed semantics.

## R2 — Fossils re-accrue progress potential to Φ = 8.0 (missed by the audit entirely)

`_contribution_pbrs_bonus` adds the capped dwell-progress term for ANY stage, and
`record_accuracy` keeps ticking fossilized slots (they retain state). A fossil's
potential climbs 6.0 → 8.0 over ~7 post-commit epochs (+0.078…+0.081/epoch, ≈ +0.57
undiscounted), then pays −0.012/epoch drag at the cap. r9 confirms both regimes at
population scale (histograms above). Note 8.0 > 7.5 = Φ(HOLDING, capped): a settled
fossil out-potentials a parked seed.

Incidental: ~9.5% of ALL r9 decisions target a fossilized slot with a non-terminal op,
each paying the −0.012 drag — fossil-targeted decision traffic is material.

## R3 — The park-vs-fossil shaping residual is PRO-FOSSIL, not pro-park (PDR-0084 #4 corrected)

A7 recorded "the shaping itself pays ≈ +0.31 for PARKING" from Φ(H,cap)=7.5 >
Φ(F)=6.0. That arithmetic assumed the forfeiture fires (it does not, R1) and the
fossil potential stays 6.0 (it does not, R2). Measured A−S PBRS PV from the commit
epoch (c=3, cf=3, T=150; positive = shaping favors fossilizing):

| dwell at commit | live convention | audit convention (counterfactual) |
|---|---|---|
| 1 | +0.019 | +0.079 |
| 5 | +0.379 | +0.081 |
| 7 | +0.529 | +0.082 |

(T=75: live +0.055/+0.416/+0.566.) Even paying the forfeiture, re-accrual nets
positive. The GATE-DOM interpretation note that named the park residual as a suppressor
suspect reverses direction: the shaping term is a (small) commit-side subsidy.

## R4 — The full shipping S/A ledger, per-component (c=3, cf=3, dwell 5, T=150, from commit)

| component (A − S) | PV |
|---|---|
| forfeited provisional attribution (to S) | **−274.7** |
| holding_warning relief (to A) | +27.7 |
| flat prior + cost (`0.5+3·tanh(1/3)` × leg − 0.01) | +1.755 |
| PBRS residual | +0.38 |
| fossil maintenance | −0.18 |
| **net** | **≈ −245** |

The measurement-gap diagnosis is untouched: the forfeited stream dominates by two
orders of magnitude, and the annuity remains the fix's revenue line. New precision:
**holding_warning relief is the second-largest pro-commit term (16× the flat prior)**
— F6 under-weighted it; B-path G-MATCHED-CONTROL's warning-cessation parity leg is
load-bearing, not a nicety.

**F8 measured:** committing a bleeding seed (spot c=−0.5, lifetime cf=+2, dwell 5)
beats holding it by **+45.8 PV** — the escape hatch is a first-order incentive, not a
corner case. (Visible in r9: commit-row `action_shaping` mode = −0.21, the
noncontributing branch, 293/262 commits s41/s42.)

## R5 — Threshold-variance machinery (G-THRESHOLD-VARIANCE)

Implemented per §2.3 (span-5 EWMA, ≥5 valid measurements, zero-slack ValueError).
Structural results (Monte-Carlo, seeded): the ≥1.0 cliff is option-like — at μ=0.9 the
expected premium goes 0 → >0.2×premium as σ goes 0.01 → 0.5 (monotone in σ); above the
cliff noise DENIES deserved premiums; the LCB variant reduces the below-cliff windfall
at every grid point (materially at high σ). The materiality threshold that would force
the LCB fallback is a freeze-time owner decision; the surface table is printed by
`test_threshold_variance.py::TestSurfaceReport`.

## Pre-registration clauses affected (proposed round-21 fold-in — body NOT yet edited)

1. **§2.5.9 / PDR-0084 #3 (Φ(PENDING) default):** the parity argument — "realizes the
   forfeiture at the request instant, matching arm A's immediate semantics" — has a
   false premise: arm A realizes NO forfeiture. The PENDING potential choice must be
   re-adjudicated against arm A's ACTUAL semantics (no transition delta paid; re-accrual
   restarts at the boundary). **Owner-pending decision; new information.**
2. **Δ_designed(x):** replace the F5 forfeiture term with the measured skip-semantics
   PBRS path (R3 table); the park residual haircut (~25%, GATE-DOM suspect) reverses
   sign.
3. **G-PBRS gate:** rewrite from "whole-path telescoping" to the two-part form the
   harness now enforces: (a) within-stage closed-form identity ≈ 0 at 1e-9; (b) the
   cross-stage skips priced explicitly in Δ_designed — never asserted away.
4. **G-MATCHED-CONTROL:** the "PBRS semantics" leg must compare B against A's real
   skip-semantics (and B's PENDING window must not accidentally re-introduce a paid
   transition A doesn't have).
5. **Audit doc:** F5/A2/A7-park-residual corrected inline (marker discipline: this doc
   is the executed read).
6. **Separate design question (owner-gated, NOT part of the experiment):** whether to
   FIX the eis==0 skip in production. Fixing it changes arm A's shipping semantics —
   contraindicated mid-arc by matched-control discipline; recommend pricing it now,
   deciding the fix after the experiment.

## Status

- Phase-1 scope of freeze blocker #3: **built and green** (82 tests). Phase 2 (B path,
  7-gate contract on the flagged settlement implementation) awaits the owner gate.
- The finding does NOT weaken the epic's premise (the forfeited measurement stream
  still dominates the commit ledger); it corrects the pricing layer the verdict gates
  read, before anything froze — which is precisely what the replay blocker existed to do.
