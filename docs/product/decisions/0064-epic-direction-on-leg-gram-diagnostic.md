# PDR-0064 — Epic direction: ON-leg Gram diagnostic first (read the endogeneity before A′/B)

Date: 2026-07-12
Status: accepted (owner DECIDE + run launched under the standing run-authorization grant;
this is a DIAGNOSTIC read, not a gate)
Inputs: the corrected longdiag read (`docs/analysis/2026-07-12-longdiag-s41-read.md`,
commit `92d998bd`, advisor + external-peer reviewed); PDR-0060 endogeneity finding.

## Context

The 600-round OFF diagnostic established that the reward/return-side scale non-stationarity
plateaus post-anneal (structural runaway ruled out), but the DOMINANT unknown — the ON-leg
`V_cf`-bootstrap endogeneity (`returns_cf = A_cf + V_cf`, self-inflicted; the half a
redesign must fix) — is invisible on an OFF run. Advisor + peer converge: do not commit to
a large A′/B experiment on the strength of an OFF read; read the architecture-side first.

## Options considered (owner DECIDE)

- **ON-leg Gram diagnostic first — CHOSEN.** Short instrumented ON run(s) using the vg Gram
  telemetry to read the endogeneity directly, cheaply, before any big bet.
- TIP first — build the general self-defending read machinery; superset of the targeted
  read, higher cost. Deferred (the targeted read answers THIS question more directly).
- Commit to A′/B now — advisor-flagged premature on an OFF-only read.
- Pivot off EV-stab — the corrected read does not force a pivot (structural churn ruled
  out, not confirmed).

## The call

1. Run an **instrumented ON-leg (HRA-on) long diagnostic** on the schedule-fixed code,
   emitting the vg Gram (V_main, V_cf, G_main, G_cf) family + return-variance telemetry.
   **n=2 (seeds 41, 42)** in parallel on the two idle GPUs — same ~20h wall-clock, and it
   blunts the n=1→n=5 scar (PDR-0026) for a direction-informing read. Extensible per the
   grant if the read is ambiguous. 600 rounds each → 350-round post-anneal window.
2. **This resolves the branch-survivor acceptance-harness disposition = KEEP** (EV-stab
   continuation; PDR-0062 step-6 discriminator: A′/B or TIP → keep). The harness is NOT
   excised at the eventual consolidation.
3. The rejected Objective-A HRA stays rejected (PDR-0059); this run re-uses that HRA code
   only as a DIAGNOSTIC substrate to observe the endogeneity — it is NOT re-scored for
   acceptance.

## Pre-committed reading (the A′/B discriminator + the 3 plateaus)

Read over the post-anneal window (rounds ≥250), per seed, from the vg Gram family:

1. **cf target-process plateau** — slope of `Var(G_cf)` and the instantaneous cf target
   scale across 250–600. Plateaus (like the OFF total did) ⇒ the cf target is not
   intrinsically structurally non-stationary. Keeps drifting at flat schedules ⇒
   endogenous/architecture-induced non-stationarity (the V_cf bootstrap), which head-size
   / loss-weight tuning cannot fix.
2. **cf learner plateau** — exact `Var(e_cf) = Var(G_cf − V_cf)` (from the Gram) and
   `ev_cf` across the window. Improving as scale settles ⇒ the cf critic is learnable.
3. **normalizer plateau** — instantaneous cf scale vs running `cf_value_target_scale`
   ratio stabilizes.
4. **Held-out affine calibration-rescue (THE A′/B discriminator)** — fit
   `Ĝ = a·V_main + b·V_cf + c` on an early flat sub-window (250–350), test on a late one
   (400–600). Rescues total EV ⇒ streams individually informative, just miscalibrated ⇒
   **Path A′** credible. Cannot rescue ⇒ cf residual irreducible ⇒ **Path B** favoured.
   (Within-run held-out ⇒ n=1 is meaningfully informative; n=2 adds cross-seed agreement.)

Decision rule (advisory to a later owner DECIDE, not a gate):
- **A′** if: cf target plateaus post-anneal ∧ held-out recalibration rescues total EV ∧
  cf residual manageable/improving after scale settles.
- **B** if: cf target stays non-stationary at flat schedules ∨ calibration cannot rescue
  the sum ∨ cf residual stays dominant.
- Both seeds agreeing strengthens either call; disagreement ⇒ extend n.

## Reversal trigger

- If the exact Gram decomposition contradicts the approximate PDR-0060 pre-DECIDE reads
  (e.g. error covariance proves positive, or Var(e_cf) does NOT dominate), the provisional
  path lean resets to neutral (PDR-0060 trigger).
- If the ON diagnostic fails to reach a stable post-anneal window (cf scale still drifting
  at round 600), that is itself the answer (endogenous non-stationarity → B) — not a run
  failure.
