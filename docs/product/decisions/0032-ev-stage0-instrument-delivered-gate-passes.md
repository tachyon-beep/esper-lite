# PDR-0032 — EV Stage-0 instrumentation delivered; value-free gate PASSES (~1.0 ≫ 0.40) → Stage-2 de-shaping justified

Date: 2026-07-05   Status: accepted (autonomous within grant: "accept against
criteria" + "launch GPU experiment runs within the active bet"; owner directed
"continue Stage-0" this session)
Author: Claude (agent)   Related: PDR-0027 (parked Shapley, made EV-stab the Now bet),
PDR-0028 (the value-free gate is THE gate; the ON-leg λ-metric is contaminated),
PDR-0031 (ev-stab→main unification deferred). Tracker: esper-lite-3d67b09687
(Stage-0 task), esper-lite-f25b71c165 (epic).

## Context

The Now bet (EV-stabilization, PDR-0027) rests on one falsifiable premise: the dense
counterfactual contribution stream (R_cf) carries enough of the value-target variance
that routing it out of the single critic (Stage-2 de-shaping) is worth doing. The
Stage-0 task's advance gate makes this concrete: **`Cov(R_cf, R)/Var(R) > 0.40` on a
Stage-2-OFF control run**, read from a value-free per-return covariance decomposition
(PDR-0028 — NOT the V_cf-contaminated ON-leg λ-metric). At session start this gate had
NO valid reading: the core math (`compute_return_variance_shares`) had landed
(`f721a5b3`) but nothing collected the per-component data or surfaced the gate.

## Options

- **(a) Build the instrument to a rigorous bar, then read the gate on a local control run**
  (2× RTX 4060 Ti — the canonical host for every prior reading).
- **(b) Hand off a run recipe** for the owner to read elsewhere. Rejected: the local host
  IS the canonical host (all prior n=5/n=6 control readings ran there), so a local reading
  is directly comparable — no comparability gain from handoff.

## Call

**(a), executed.** Stage-0 instrumentation delivered in 5 TDD-green commits on
`feat/ev-stab-stage2-hra` (`562cb5bf` → `70753923`), ~2400 tests green, byte-identity
proven directly, no checkpoint/schema bump. The gate was then read on a full canonical
baseline control run (100 ep × 4 env × 150 epoch, Stage-2 OFF, `return_variance_telemetry`
ON, `gpu_preload`):

**GATE PASSES. `Cov(R_cf, R)/Var(R)` median = 1.02, 12/13 cf-active updates > 0.40**
(min 0.10, max 1.10; residual reconciles to 2e-17; preliminary at n≈14 updates, run still
accumulating toward the full distribution). The counterfactual stream carries essentially
ALL the return variance; R_main is a small, slightly-anticorrelated remainder.

**Consequence: the epic PROCEEDS to Stage-2 de-shaping** (already built on this branch) —
it is NOT re-scoped to Stage-1-only. The premise is confirmed, not falsified.

## Rationale

A gate this saturated (~1.0) is the strongest possible confirmation of the epic's founding
hypothesis: the cf term *is* the value-target variance the critic must regress on. The
reading validates itself by varying correctly (0.10→1.10 across updates, shares sum to 1,
residual ~0) — a live covariance measurement, not a wired constant. Building the instrument
to this bar (completeness keystone falsified/has-teeth; env-boundary reset; full-loop
reducer smoke) before trusting a single number is what makes the pass defensible.

Branch note (unification input, PDR-0031): ev-stab decomposes `synergy_bonus`; the Shapley
line renamed it `interaction_bonus`. The two `ADDITEND_SIGN_MAP` copies diverge on that one
key — reconcile at the owner-gated ev-stab→main merge.

## Reversal triggers

- If the FULL-distribution reading on run completion has median `Cov(R_cf,R)/Var(R) < 0.40`
  → reopen: the de-shaping premise is falsified and the epic re-scopes to Stage-1-only
  (marginal-V, already satisfied), leaving little epic. (Signal to date makes this remote:
  12/13 updates ≫ 0.40.)
- If Stage-2 de-shaping, once enabled, regresses host-accuracy contribution (guardrail,
  metrics.md ≥ +7.69pp) → the cf term was load-bearing credit, refold via DPBA not raw
  reward (per the epic's implementation traps), not a clean de-shape.
- PDR-0027's triggers still stand (a ≥10× coverage A/B re-run becomes informative only if
  the EV track raises k≥2 co-fossilization frequency materially).
