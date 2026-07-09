# PDR-0053 — Long-horizon OFF diagnostic: transient vs structural non-stationarity

Date: 2026-07-10
Status: accepted — SUPERSEDED IN PART by PDR-0054 (pre-data): the determinism
cross-check claim is retracted and the read window is corrected to rounds 251–451
(the entropy-floor penalty schedule normalizes by horizon; see PDR-0054)
Decider: owner-agent under the explicit run-authorization grant
Relates-to: PDR-0050 (LEG-B demotion), PDR-0052 (cf-plateau demotion) — both caused by
within-run non-stationarity; this diagnostic tests whether that cause is fixable.

## Context

Both Stage-2 leg demotions this session trace to one root: within-run non-stationarity
(EV lifting late, Var(returns) ↑ 10–25×, cf target scale climbing all run). The OFF EV is
still climbing steeply at round 200 (quintile medians 0.005→0.031→0.079→0.344→0.533 —
not a plateau). Crucially, the entropy anneal (0.15→0.08) is scheduled over 250 rounds but
runs stop at 200, so the ENTIRE observed window is during active annealing. Falling entropy
sharpens the policy and makes its own future more predictable — a large candidate driver of
the apparent non-convergence.

## The question

Is the non-stationarity a SCHEDULE TRANSIENT (ends when the entropy anneal completes) or
STRUCTURAL morphogenetic churn (continuous germinate/prune keeps the value target moving
forever)? This determines whether the demoted legs are recoverable:
- transient → score a window AFTER round 250 → LEG-B/cf-plateau become well-posed;
- structural → no horizon fixes it → need a detrended/relative metric (not more rounds).

## The call

Run ONE long-horizon OFF diagnostic: `configs/ablations/stage2-off-longhorizon-diag.json`
= the OFF arm config, `n_episodes` 200→600, seed 41. Design properties:
- **Flat-entropy tail:** anneal completes at round 250 (fixed, horizon-independent), so
  rounds 250–600 (~350 rounds) run at CONSTANT entropy 0.08. Any residual EV/return-scale
  drift there is NOT schedule-driven → isolates the structural component.
- **Determinism cross-check:** seed 41 has an existing 200-round OFF arm; the first 200
  rounds should reproduce it (entropy_anneal_steps is fixed regardless of horizon), a
  built-in "same run, longer" validation.
- **OFF leg** (control, no cf-head confound) — the right leg for a stationarity probe, and
  the OFF-side is where ε_rel spread + stationarity prechecks live.

## Execution / discipline

- Session-proof launcher (`telemetry/stage2_off_longdiag/run_longdiag.sh`, setsid/nohup)
  that WAITS for the gpu1 ON queue to free cuda:1, then runs on cuda:1 — never contends
  with the primary n=5 ON screen. Config-only change; no training-path code touched;
  same commit as the A/B waves (`cd3187ef`, training path == `fe177844`).
- Cost ~18h (600 rounds @ ~3min/round), one GPU, starts ~when gpu1 ON queue finishes.

## What each outcome means (pre-committed reading)

- **EV plateaus + Var(returns) stabilizes in the flat-entropy tail** → non-stationarity was
  schedule-transient. Next: a Stage-2 A/B with a scored window starting after the anneal
  (pre-registered), which would re-power LEG-B and re-pose cf-plateau.
- **EV/return-scale keep drifting at constant entropy** → structural morphogenetic churn.
  No horizon fixes it; the mechanistic (LEG-B) claim needs a detrended/relative metric,
  and the finding feeds the broader EV/TIP understanding (moving target is intrinsic).

## Reversal trigger / scope

This is a diagnostic, NOT part of the frozen A/B — it is scored against no threshold and
changes no gate. It does not gate or delay the n=5 screen (which reads the ON wave under
the amended predicate). If the n=5 screen and this diagnostic disagree on direction, the
screen governs the acceptance decision; the diagnostic governs only the "can the
mechanistic claim be recovered, and how" follow-up.
